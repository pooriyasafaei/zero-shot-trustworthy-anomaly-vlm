"""Summary tables and paired comparisons for CERM-AD Stage I+.

Every target is scored under every arm, so comparisons between arms are **paired**:
the unit of resampling is the target image, and a difference between two arms is
bootstrapped as the per-image difference - never as two independent intervals.

Nothing here is a significance test in the classical sense at N~40; the bootstrap
intervals are effect-size uncertainty, and the Wilcoxon signed-rank p-values are
reported as secondary descriptors only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .report import bootstrap_ci

#: Display names of the necessity arms, in the order of the ablation story.
NECESSITY_METHODS = (
    ("naive", "naive (raw defect deletion, no control)"),
    ("random", "random-translation control"),
    ("context", "context-matched control (CERM v1)"),
    ("artifact", "context + artifact matched control (CERM v2)"),
    ("occlusion", "occlusion-style baseline"),
)

N_BOOT = 2000


def _ok(records: pd.DataFrame) -> pd.DataFrame:
    return records[records["status"] == "ok"] if "status" in records else records


def _finite(values) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return arr[np.isfinite(arr)]


def paired_bootstrap(a, b, stat=np.median, n_boot: int = N_BOOT, seed: int = 0) -> dict:
    """Bootstrap CI of ``stat(a_i - b_i)`` over images with both values finite."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    keep = np.isfinite(a) & np.isfinite(b)
    d = a[keep] - b[keep]
    out = {"n_pairs": int(d.size), "stat": float("nan"), "ci_lo": float("nan"),
           "ci_hi": float("nan"), "frac_positive": float("nan"), "wilcoxon_p": float("nan")}
    if d.size == 0:
        return out
    ci = bootstrap_ci(d, stat, n_boot=n_boot, seed=seed)
    out.update({"stat": ci.value, "ci_lo": ci.lo, "ci_hi": ci.hi,
                "frac_positive": float((d > 0).mean())})
    if d.size >= 6 and np.any(d != 0):
        from scipy.stats import wilcoxon
        try:
            out["wilcoxon_p"] = float(wilcoxon(d).pvalue)
        except ValueError:
            pass
    return out


def independent_bootstrap(a, b, stat=np.median, n_boot: int = N_BOOT, seed: int = 0) -> dict:
    """CI of ``stat(a) - stat(b)`` for two independent groups (e.g. YES vs NO)."""
    a, b = _finite(a), _finite(b)
    out = {"n_a": int(a.size), "n_b": int(b.size), "stat": float("nan"),
           "ci_lo": float("nan"), "ci_hi": float("nan")}
    if a.size < 3 or b.size < 3:
        return out
    rng = np.random.default_rng(seed)
    draws = [float(stat(rng.choice(a, a.size)) - stat(rng.choice(b, b.size)))
             for _ in range(n_boot)]
    out.update({"stat": float(stat(a) - stat(b)), "ci_lo": float(np.quantile(draws, 0.025)),
                "ci_hi": float(np.quantile(draws, 0.975))})
    return out


def necessity_summary(records: pd.DataFrame, arm: str, group: str = "ALL",
                      seed: int = 0) -> dict:
    """One row of the necessity table for one arm and one group of targets."""
    ok = _ok(records)
    row = {"method": arm, "group": group, "N": int(len(ok))}
    if not len(ok):
        return row
    if arm == "naive":
        d = ok["delta_defect"].to_numpy(float)
        ci = bootstrap_ci(d, seed=seed)
        row.update({"median_delta_defect": ci.value, "median_control_delta": 0.0,
                    "median_cee": ci.value, "cee_ci_lo": ci.lo, "cee_ci_hi": ci.hi,
                    "frac_cee_positive": float(np.mean(_finite(d) > 0))})
        return row
    cee = ok[f"{arm}_cee"].to_numpy(float)
    valid = np.isfinite(cee)
    ci = bootstrap_ci(cee, seed=seed)
    row.update({
        "N_with_controls": int(valid.sum()),
        "median_delta_defect": float(np.nanmedian(ok[f"{arm}_delta_defect"])),
        "median_control_delta": float(np.nanmedian(ok[f"{arm}_placebo_delta_median"])),
        "median_cee": ci.value, "cee_ci_lo": ci.lo, "cee_ci_hi": ci.hi,
        "frac_cee_positive": float(np.mean(cee[valid] > 0)) if valid.any() else float("nan"),
        "frac_beats_all_controls": float(ok.loc[valid, f"{arm}_defect_beats_all_placebos"]
                                         .astype(bool).mean()) if valid.any() else float("nan"),
        "median_empirical_tail": float(np.nanmedian(ok[f"{arm}_empirical_placebo_p"])),
        "median_robust_z": float(np.nanmedian(ok[f"{arm}_robust_cee_z"])),
        "median_n_controls": float(np.nanmedian(ok[f"{arm}_n_valid_placebos"])),
        "median_seam_ratio": float(np.nanmedian(ok[f"{arm}_seam_ratio"])),
        "median_edit_ratio": float(np.nanmedian(ok[f"{arm}_edit_ratio"])),
        "median_artifact_imbalance": float(np.nanmedian(ok[f"{arm}_artifact_imbalance"])),
    })
    return row


def sufficiency_summary(records: pd.DataFrame, group: str = "ALL", seed: int = 0) -> dict:
    """One row of the sufficiency table."""
    ok = _ok(records)
    row = {"group": group, "N": int(len(ok))}
    if not len(ok):
        return row
    cee = ok["cee_s"].to_numpy(float)
    valid = np.isfinite(cee)
    ci = bootstrap_ci(cee, seed=seed)
    row.update({
        "N_with_shams": int(valid.sum()),
        "median_delta_insert_defect": float(np.nanmedian(ok["delta_insert_defect"])),
        "median_sham_delta": float(np.nanmedian(ok["sufficiency_placebo_delta_median"])),
        "median_cee_s": ci.value, "cee_s_ci_lo": ci.lo, "cee_s_ci_hi": ci.hi,
        "frac_cee_s_positive": float(np.mean(cee[valid] > 0)) if valid.any() else float("nan"),
        "frac_insert_beats_all_shams": float(ok.loc[valid, "defect_insert_beats_all_shams"]
                                             .astype(bool).mean()) if valid.any() else float("nan"),
        "median_empirical_sham_tail": float(np.nanmedian(ok["empirical_sham_tail"])),
        "median_robust_z": float(np.nanmedian(ok["robust_cee_s_z"])),
        "median_seam_ratio": float(np.nanmedian(ok["sufficiency_seam_ratio"])),
        "median_edit_ratio": float(np.nanmedian(ok["sufficiency_edit_ratio"])),
        "median_artifact_imbalance": float(np.nanmedian(ok["sufficiency_artifact_imbalance"])),
        "frac_recipient_predicted_yes": float((ok["audit_logit_recipient"] >= 0).mean()),
    })
    return row


def _groups(records: pd.DataFrame):
    ok = _ok(records)
    yield "ALL", ok
    for cat, g in ok.groupby("category", sort=True):
        yield str(cat), g


def necessity_tables(records: pd.DataFrame, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    overall = pd.DataFrame([necessity_summary(records, m, "ALL", seed)
                            for m, _ in NECESSITY_METHODS])
    by_cat = pd.DataFrame([necessity_summary(g, m, name, seed)
                           for name, g in _groups(records) if name != "ALL"
                           for m, _ in NECESSITY_METHODS])
    return overall, by_cat


def sufficiency_tables(records: pd.DataFrame, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    overall = pd.DataFrame([sufficiency_summary(records, "ALL", seed)])
    by_cat = pd.DataFrame([sufficiency_summary(g, name, seed)
                           for name, g in _groups(records) if name != "ALL"])
    return overall, by_cat


def bidirectional_table(records: pd.DataFrame) -> pd.DataFrame:
    """Quadrant fractions overall, by category and by original verdict."""
    ok = _ok(records)
    rows = []

    def add(name: str, g: pd.DataFrame) -> None:
        q = g["quadrant"].value_counts()
        det = g[g["quadrant"] != "undetermined"]
        row = {"group": name, "N": int(len(g)), "N_determined": int(len(det))}
        for label in ("bidirectional", "necessity_only", "sufficiency_only", "neither",
                      "undetermined"):
            row[f"n_{label}"] = int(q.get(label, 0))
            row[f"frac_{label}"] = (float(q.get(label, 0)) / len(det)
                                    if len(det) and label != "undetermined" else float("nan"))
        rows.append(row)

    add("ALL", ok)
    for cat, g in ok.groupby("category", sort=True):
        add(f"category={cat}", g)
    for verdict, g in ok.groupby("original_direct_prediction", sort=True):
        add(f"original_verdict={verdict}", g)
    return pd.DataFrame(rows)


def artifact_balance_table(records: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Per-arm artifact balance plus the paired context-vs-artifact improvement."""
    ok = _ok(records)
    rows = []
    for arm in ("random", "context", "artifact", "occlusion", "sufficiency"):
        sr = ok[f"{arm}_seam_ratio"].to_numpy(float)
        er = ok[f"{arm}_edit_ratio"].to_numpy(float)
        imb = ok[f"{arm}_artifact_imbalance"].to_numpy(float)
        rows.append({
            "arm": arm, "N": int(np.isfinite(imb).sum()),
            "median_seam_ratio": float(np.nanmedian(sr)),
            "median_edit_ratio": float(np.nanmedian(er)),
            "median_abs_log_seam_ratio": float(np.nanmedian(np.abs(np.log(sr)))),
            "median_abs_log_edit_ratio": float(np.nanmedian(np.abs(np.log(er)))),
            "median_artifact_imbalance": float(np.nanmedian(imb)),
            "mean_artifact_imbalance": float(np.nanmean(imb)),
            "frac_seam_ratio_within_1.25x": float(np.nanmean(
                (sr[np.isfinite(sr)] >= 0.8) & (sr[np.isfinite(sr)] <= 1.25))),
            "frac_edit_ratio_within_1.25x": float(np.nanmean(
                (er[np.isfinite(er)] >= 0.8) & (er[np.isfinite(er)] <= 1.25))),
        })
    table = pd.DataFrame(rows)
    pb = paired_bootstrap(ok["context_artifact_imbalance"], ok["artifact_artifact_imbalance"],
                          seed=seed)
    table.loc[len(table)] = {
        "arm": "PAIRED(context - artifact) imbalance", "N": pb["n_pairs"],
        "median_artifact_imbalance": pb["stat"],
        "mean_artifact_imbalance": float("nan"),
    }
    table["paired_ci_lo"] = np.nan
    table["paired_ci_hi"] = np.nan
    table["paired_frac_improved"] = np.nan
    table["paired_wilcoxon_p"] = np.nan
    last = len(table) - 1
    table.loc[last, ["paired_ci_lo", "paired_ci_hi", "paired_frac_improved",
                     "paired_wilcoxon_p"]] = [pb["ci_lo"], pb["ci_hi"], pb["frac_positive"],
                                              pb["wilcoxon_p"]]
    return table


def paired_comparisons(records: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """The predeclared comparisons A-E (see docs/CERM_AD_V2.md, section 'Statistics')."""
    ok = _ok(records)
    rows = []

    def add(name: str, kind: str, res: dict, note: str = "") -> None:
        rows.append({"comparison": name, "type": kind, **res, "note": note})

    add("A: artifact imbalance, context-only minus artifact-matched", "paired",
        paired_bootstrap(ok["context_artifact_imbalance"], ok["artifact_artifact_imbalance"],
                         seed=seed), "> 0 means artifact matching reduced imbalance")
    add("B: CEE_N, context-only minus artifact-matched", "paired",
        paired_bootstrap(ok["cee_n_context"], ok["cee_n_artifact"], seed=seed),
        "> 0 means the effect shrank under stronger controls")
    add("B2: CEE_N, random minus context-only", "paired",
        paired_bootstrap(ok["cee_n_random"], ok["cee_n_context"], seed=seed), "")
    add("C1: delta_defect minus median artifact-matched placebo delta (= CEE_N)", "paired",
        paired_bootstrap(ok["artifact_delta_defect"], ok["artifact_placebo_delta_median"],
                         seed=seed), "")
    add("C2: delta_insert_defect minus median sham delta (= CEE_S)", "paired",
        paired_bootstrap(ok["delta_insert_defect"], ok["sufficiency_placebo_delta_median"],
                         seed=seed), "")
    yes = ok["original_direct_prediction"] == "YES"
    add("D: CEE_N(artifact), original YES minus original NO", "independent groups",
        independent_bootstrap(ok.loc[yes, "cee_n_artifact"], ok.loc[~yes, "cee_n_artifact"],
                              seed=seed), "difference of medians")
    add("E: CEE_S, original YES minus original NO", "independent groups",
        independent_bootstrap(ok.loc[yes, "cee_s"], ok.loc[~yes, "cee_s"], seed=seed),
        "difference of medians")
    return pd.DataFrame(rows)


def probe_agreement_table(records: pd.DataFrame, structured: pd.DataFrame) -> pd.DataFrame:
    """Direct probe vs structured scorer on the original target images (P0)."""
    from scipy.stats import pearsonr, spearmanr

    ok = _ok(records)
    st = structured[structured["role"] == "original"]
    if "structured_status" in st:
        st = st[st["structured_status"].isna()]
    merged = ok.merge(st[["image_id", "structured_audit_p_yes", "structured_audit_logit",
                          "structured_generated_vote", "structured_parse_ok"]],
                      on="image_id", how="inner")
    parsed = merged[merged["structured_parse_ok"].astype(bool)
                    & merged["structured_audit_p_yes"].notna()]
    d = parsed["audit_p_yes_original"].to_numpy(float)
    s = parsed["structured_audit_p_yes"].to_numpy(float)
    row = {"N_targets": int(len(ok)), "N_structured": int(len(merged)),
           "N_parsed": int(len(parsed)),
           "structured_parse_rate": float(merged["structured_parse_ok"].astype(bool).mean())
           if len(merged) else float("nan")}
    if len(parsed) >= 3:
        row["spearman_p_yes"] = float(spearmanr(d, s).statistic) if np.ptp(s) > 0 else float("nan")
        row["pearson_p_yes"] = float(pearsonr(d, s).statistic) if np.ptp(s) > 0 else float("nan")
        row["verdict_agreement"] = float(((d >= 0.5) == (s >= 0.5)).mean())
    row["direct_sensitivity"] = float((ok["audit_p_yes_original"] >= 0.5).mean())
    row["structured_sensitivity"] = float((s >= 0.5).mean()) if len(parsed) else float("nan")
    row["note"] = ("all targets are anomalous, so 'sensitivity' is recall on anomalies, "
                   "not accuracy")
    return pd.DataFrame([row])
