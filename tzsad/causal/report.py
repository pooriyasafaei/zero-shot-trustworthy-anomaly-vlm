"""Summary tables, figures and the presentation write-up for a Stage-I run.

Everything here reads the per-image records offline, in keeping with the
repository's rule that evaluation never needs a GPU pass.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from ..eval.metrics import Interval
from ..utils.logging import get_logger
from ..viz.style import PALETTE, SEMANTIC, save_figure, use_paper_style

log = get_logger("causal.report")


def bootstrap_ci(values, stat=np.median, n_boot: int = 2000, alpha: float = 0.05,
                 seed: int = 0) -> Interval:
    """Percentile bootstrap CI for a statistic of one sample."""
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return Interval(float("nan"), float("nan"), float("nan"), 0)
    rng = np.random.default_rng(seed)
    draws = [float(stat(arr[rng.integers(0, arr.size, arr.size)])) for _ in range(n_boot)]
    return Interval(float(stat(arr)), float(np.quantile(draws, alpha / 2)),
                    float(np.quantile(draws, 1 - alpha / 2)), int(arr.size))


def _group_summary(df: pd.DataFrame, name: str, seed: int = 0) -> dict:
    """One summary row for a set of target images."""
    ok = df[df.get("status", "ok") == "ok"] if "status" in df else df
    if not len(ok):
        return {"group": name, "n_targets": 0}
    cee = ok["cee"].to_numpy(dtype=float)
    med = bootstrap_ci(cee, np.median, seed=seed)
    mean = bootstrap_ci(cee, np.mean, seed=seed)
    resolvable = ok[ok["supported_at_005"].notna()] if "supported_at_005" in ok else ok.iloc[:0]
    return {
        "group": name,
        "n_targets": int(len(ok)),
        "delta_defect_median": float(np.nanmedian(ok["delta_defect"])),
        "delta_defect_mean": float(np.nanmean(ok["delta_defect"])),
        "placebo_delta_median": float(np.nanmedian(ok["placebo_delta_median"])),
        "cee_median": med.value, "cee_median_ci_lo": med.lo, "cee_median_ci_hi": med.hi,
        "cee_mean": mean.value, "cee_mean_ci_lo": mean.lo, "cee_mean_ci_hi": mean.hi,
        "frac_cee_positive": float(np.nanmean(cee > 0)),
        "frac_beats_all_placebos": float(ok["defect_beats_all_placebos"].mean()),
        "median_empirical_placebo_p": float(np.nanmedian(ok["empirical_placebo_p"])),
        "median_robust_cee_z": float(np.nanmedian(ok["robust_cee_z"])),
        "n_resolvable_at_005": int(len(resolvable)),
        "supported_at_005_rate": (float(resolvable["supported_at_005"].astype(float).mean())
                                  if len(resolvable) else float("nan")),
        "median_n_valid_placebos": float(np.nanmedian(ok["n_valid_placebos"])),
        "frac_original_predicted_yes": float(ok["original_is_correct_for_anomaly"].mean()),
    }


def summary_tables(records: pd.DataFrame, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``(overall, by_category)`` summary frames."""
    overall = pd.DataFrame([_group_summary(records, "ALL", seed)])
    by_cat = pd.DataFrame([_group_summary(g, str(c), seed)
                           for c, g in records.groupby("category", sort=True)])
    return overall, by_cat


def decision_split_table(records: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """CEE split by what the model originally said about the image.

    A detector can be right for the wrong reason, and it can be wrong while still
    responding to the defect. Splitting on the original verdict separates those two.
    """
    ok = records[records.get("status", "ok") == "ok"] if "status" in records else records
    yes_mask = ok["original_is_correct_for_anomaly"].astype(bool)
    rows = [_group_summary(ok[yes_mask], "predicted_YES", seed),
            _group_summary(ok[~yes_mask], "predicted_NO", seed)]
    table = pd.DataFrame(rows)

    yes = ok[yes_mask]["cee"].to_numpy(dtype=float)
    no = ok[~yes_mask]["cee"].to_numpy(dtype=float)
    yes, no = yes[np.isfinite(yes)], no[np.isfinite(no)]
    # Only quantify the gap when both arms have enough observations to bootstrap;
    # below that it stays descriptive rather than becoming a number to over-read.
    if len(yes) >= 3 and len(no) >= 3:
        rng = np.random.default_rng(seed)
        diffs = [float(np.median(rng.choice(yes, yes.size)) - np.median(rng.choice(no, no.size)))
                 for _ in range(2000)]
        table.loc[len(table)] = {
            "group": "DIFF(YES-NO)", "n_targets": len(yes) + len(no),
            "cee_median": float(np.median(yes) - np.median(no)),
            "cee_median_ci_lo": float(np.quantile(diffs, 0.025)),
            "cee_median_ci_hi": float(np.quantile(diffs, 0.975)),
        }
    return table


def artifact_control_table(interventions: pd.DataFrame, records: pd.DataFrame) -> pd.DataFrame:
    """Do real-defect fills look systematically different from placebo fills?

    If they do, a positive CEE could be a reaction to a worse splice rather than to
    the absence of the defect, and the headline has to be qualified accordingly.
    """
    iv = interventions[interventions["intervention_type"].isin(["defect_removed", "placebo"])]
    rows = []
    for kind, g in iv.groupby("intervention_type", sort=True):
        rows.append({
            "intervention_type": kind, "n": int(len(g)),
            "seam_score_mean": float(g["seam_score"].mean()),
            "seam_score_median": float(g["seam_score"].median()),
            "edit_energy_mean": float(g["edit_energy"].mean()),
            "mask_area_mean": float(g["mask_area"].mean()),
            "outside_change_max": float(g["outside_change_max"].max()),
        })
    table = pd.DataFrame(rows)

    # Is the per-image effect merely tracking how bad that image's splice was?
    ok = records[records.get("status", "ok") == "ok"] if "status" in records else records
    valid = ok.dropna(subset=["cee"])
    rho = float("nan")
    if "seam_ratio" in valid.columns:
        valid = valid[np.isfinite(valid["seam_ratio"].to_numpy(dtype=float))]
        # Spearman is undefined when either input is constant, which happens on
        # uniform synthetic data; report NaN rather than emitting a warning and a nan.
        if len(valid) >= 4 and valid["cee"].nunique() > 1 and valid["seam_ratio"].nunique() > 1:
            from scipy import stats
            rho = float(stats.spearmanr(valid["cee"], valid["seam_ratio"]).statistic)
    table.loc[len(table)] = {"intervention_type": "SPEARMAN(cee, seam_ratio)",
                             "n": int(len(valid)), "seam_score_mean": rho}
    return table


# ---------------------------------------------------------------- figures
def plot_defect_vs_placebo(records: pd.DataFrame, out: Path) -> list[Path]:
    """The headline figure: real-defect effect against matched-placebo effect."""
    use_paper_style()
    ok = records[records.get("status", "ok") == "ok"]
    fig, ax = plt.subplots(figsize=(3.8, 3.5))
    x = ok["placebo_delta_median"].to_numpy(dtype=float)
    y = ok["delta_defect"].to_numpy(dtype=float)
    finite = np.concatenate([x[np.isfinite(x)], y[np.isfinite(y)]]) if len(ok) else np.array([0.0])
    pad = 0.5 if finite.size else 1.0
    lim = [float(finite.min()) - pad, float(finite.max()) + pad]
    ax.plot(lim, lim, ls="--", lw=1.0, color=SEMANTIC["random"], label="no causal advantage")
    for i, (cat, g) in enumerate(ok.groupby("category", sort=True)):
        ax.scatter(g["placebo_delta_median"], g["delta_defect"], s=24, alpha=0.85,
                   color=PALETTE[i % len(PALETTE)], label=str(cat))
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("median placebo effect  $\\Delta_{placebo}$")
    ax.set_ylabel("defect-removal effect  $\\Delta_{defect}$")
    ax.set_title("Does removing the real defect matter more?")
    ax.legend(fontsize=6, loc="upper left")
    return save_figure(fig, out)


def plot_cee_by_category(records: pd.DataFrame, out: Path) -> list[Path]:
    """CEE distribution per category."""
    use_paper_style()
    ok = records[records.get("status", "ok") == "ok"]
    cats = sorted(ok["category"].unique())
    fig, ax = plt.subplots(figsize=(0.95 * max(len(cats), 3) + 1.4, 3.0))
    data = [ok[ok.category == c]["cee"].dropna().to_numpy() for c in cats]
    ax.axhline(0, color=SEMANTIC["random"], ls="--", lw=1.0)
    if any(len(d) for d in data):
        bp = ax.boxplot(data, tick_labels=cats, showfliers=False, widths=0.55, patch_artist=True)
        for patch, colour in zip(bp["boxes"], PALETTE):
            patch.set_facecolor(colour)
            patch.set_alpha(0.35)
    for i, d in enumerate(data, 1):
        if len(d):
            jitter = np.random.default_rng(i).normal(0, 0.05, len(d))
            ax.scatter(np.full(len(d), i) + jitter, d, s=15, zorder=3,
                       color=PALETTE[(i - 1) % len(PALETTE)])
    ax.set_ylabel("CEE  (nats of log-odds)")
    ax.set_title("Causal evidence effect by category")
    return save_figure(fig, out)


def plot_artifact_control(interventions: pd.DataFrame, out: Path) -> list[Path]:
    """Seam severity of real-defect fills vs placebo fills."""
    use_paper_style()
    iv = interventions[interventions["intervention_type"].isin(["defect_removed", "placebo"])]
    groups = [("defect_removed", SEMANTIC["ours"]), ("placebo", SEMANTIC["baseline"])]
    data = [iv[iv.intervention_type == k]["seam_score"].dropna().to_numpy() for k, _ in groups]
    fig, ax = plt.subplots(figsize=(3.2, 2.9))
    if any(len(d) for d in data):
        bp = ax.boxplot([d if len(d) else np.array([np.nan]) for d in data],
                        tick_labels=[g[0] for g in groups], showfliers=False,
                        widths=0.5, patch_artist=True)
        for patch, (_, colour) in zip(bp["boxes"], groups):
            patch.set_facecolor(colour)
            patch.set_alpha(0.35)
    ax.set_ylabel("seam score")
    ax.set_title("Reconstruction artifact control")
    return save_figure(fig, out)


def plot_cee_vs_decision(records: pd.DataFrame, out: Path) -> list[Path]:
    """CEE where the model said YES vs where it missed the anomaly."""
    use_paper_style()
    ok = records[records.get("status", "ok") == "ok"]
    yes_mask = ok["original_is_correct_for_anomaly"].astype(bool)
    yes = ok[yes_mask]["cee"].dropna().to_numpy()
    no = ok[~yes_mask]["cee"].dropna().to_numpy()
    fig, ax = plt.subplots(figsize=(3.2, 2.9))
    ax.axhline(0, color=SEMANTIC["random"], ls="--", lw=1.0)
    data = [yes if len(yes) else np.array([np.nan]), no if len(no) else np.array([np.nan])]
    bp = ax.boxplot(data, tick_labels=[f"said YES (n={len(yes)})", f"said NO (n={len(no)})"],
                    showfliers=False, widths=0.5, patch_artist=True)
    for patch, colour in zip(bp["boxes"], (SEMANTIC["good"], SEMANTIC["danger"])):
        patch.set_facecolor(colour)
        patch.set_alpha(0.35)
    for i, d in enumerate((yes, no), 1):
        if len(d):
            ax.scatter(np.full(len(d), i), d, s=15, zorder=3,
                       color=SEMANTIC["good"] if i == 1 else SEMANTIC["danger"])
    ax.set_ylabel("CEE")
    ax.set_title("Causal support vs original verdict")
    return save_figure(fig, out)


def plot_qualitative_grid(records: pd.DataFrame, interventions: pd.DataFrame,
                          out: Path, n: int = 4) -> list[Path]:
    """Examples spanning high, middling and low CEE, plus a missed anomaly if present."""
    from PIL import Image as PILImage

    use_paper_style(base_font=7.0)
    ok = records[(records.get("status", "ok") == "ok") & records["cee"].notna()]
    if not len(ok):
        return []
    ordered = ok.sort_values("cee", ascending=False)
    picks = [ordered.iloc[0]]
    if len(ordered) > 2:
        picks.append(ordered.iloc[len(ordered) // 2])
    if len(ordered) > 1:
        picks.append(ordered.iloc[-1])
    missed = ok[~ok["original_is_correct_for_anomaly"].astype(bool)]
    if len(missed):
        picks.append(missed.sort_values("cee", ascending=False).iloc[0])

    seen, rows = set(), []
    for rec in picks:
        if rec["image_id"] not in seen:
            seen.add(rec["image_id"])
            rows.append(rec)
    rows = rows[:n]

    fig, axes = plt.subplots(len(rows), 5, figsize=(14.5, 2.9 * len(rows)))
    axes = np.atleast_2d(axes)
    for i, rec in enumerate(rows):
        iv = interventions[interventions.image_id == rec["image_id"]]
        original = np.asarray(PILImage.open(rec["path"]).convert("RGB"))
        mask = np.asarray(PILImage.open(rec["mask_path"]).convert("L").resize(
            (original.shape[1], original.shape[0]), PILImage.NEAREST)) > 127

        axes[i, 0].imshow(original)
        axes[i, 0].set_title(f"{rec['category']}/{rec['defect_type']}", fontsize=7)
        tint = original.copy()
        tint[mask] = [255, 0, 0]
        axes[i, 1].imshow((0.55 * original + 0.45 * tint).astype(np.uint8))
        axes[i, 1].set_title("GT defect mask", fontsize=7)

        for col, kind, label in ((2, "defect_removed", "defect removed"),
                                 (3, "placebo", "matched placebo")):
            sub = iv[iv.intervention_type == kind]
            sub = sub[sub["image_output_path"].astype(str).str.len() > 0]
            if len(sub) and Path(str(sub.iloc[0]["image_output_path"])).exists():
                axes[i, col].imshow(PILImage.open(str(sub.iloc[0]["image_output_path"])))
                axes[i, col].set_title(f"{label}\nP(YES)={sub.iloc[0]['audit_p_yes']:.3f}",
                                       fontsize=7)
            else:
                axes[i, col].text(0.5, 0.5, f"{label}\n(image not saved)",
                                  ha="center", va="center", fontsize=7)
        for j in range(4):
            axes[i, j].axis("off")
        axes[i, 4].axis("off")
        axes[i, 4].text(0, 1, (
            f"P(YES) original      : {rec['audit_p_yes_original']:.4f}\n"
            f"P(YES) defect removed: {rec['audit_p_yes_defect_removed']:.4f}\n\n"
            f"delta_defect         : {rec['delta_defect']:+.4f}\n"
            f"median placebo delta : {rec['placebo_delta_median']:+.4f}\n"
            f"CEE                  : {rec['cee']:+.4f}\n"
            f"robust CEE z         : {rec['robust_cee_z']:+.3f}\n"
            f"empirical placebo p  : {rec['empirical_placebo_p']:.4f}\n"
            f"valid placebos       : {int(rec['n_valid_placebos'])}\n"
            f"original verdict     : {rec['original_direct_prediction']}\n"
            f"seam ratio def/plac  : {float(rec.get('seam_ratio', float('nan'))):.3f}"),
            fontsize=7, family="monospace", va="top")
    fig.tight_layout()
    return save_figure(fig, out)


def build_figures(records: pd.DataFrame, interventions: pd.DataFrame,
                  fig_dir: Path) -> list[str]:
    """Produce every presentation figure, skipping any the data cannot support.

    A figure that cannot be drawn must not take the run down with it: the tables are
    the primary artefact and a partially-plotted run is still useful.
    """
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    made: list[str] = []
    jobs = (
        ("defect_vs_placebo", lambda: plot_defect_vs_placebo(records, fig_dir / "defect_vs_placebo")),
        ("cee_by_category", lambda: plot_cee_by_category(records, fig_dir / "cee_by_category")),
        ("artifact_control", lambda: plot_artifact_control(interventions,
                                                           fig_dir / "artifact_control")),
        ("cee_vs_decision", lambda: plot_cee_vs_decision(records, fig_dir / "cee_vs_decision")),
        ("qualitative_grid", lambda: plot_qualitative_grid(records, interventions,
                                                           fig_dir / "qualitative_grid")),
    )
    for name, fn in jobs:
        try:
            if fn():
                made.append(name)
        except Exception:                                  # noqa: BLE001
            log.exception("figure %s failed; continuing", name)
    return made
