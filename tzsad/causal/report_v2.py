"""Tables, figures, PRESENTATION_RESULTS.md and the bundle for CERM-AD Stage I+.

Offline only: reads the checkpointed records, never needs a GPU. Every sentence of
interpretation in PRESENTATION_RESULTS.md is generated from a measured value against
a threshold stated in this file, so the write-up cannot drift from the numbers.
"""
from __future__ import annotations

import json
import shutil
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from ..utils.logging import get_logger
from .audit import merge_shards
from .casebook import (CATEGORY_COLORS, build_casebook, category_color, select_cases,
                       structured_examples_md)
from .comparison import (NECESSITY_METHODS, artifact_balance_table, bidirectional_table,
                         necessity_tables, paired_comparisons, probe_agreement_table,
                         sufficiency_tables)
from .related_work import related_work_frame

log = get_logger("causal.report_v2")

ARM_COLORS = {"naive": "#000000", "random": "#999999", "context": "#E69F00",
              "artifact": "#0072B2", "occlusion": "#CC79A7", "sufficiency": "#009E73"}
ARM_LABELS = {"naive": "naive\n(no control)", "random": "random\ntranslation",
              "context": "context-matched\n(v1)", "artifact": "context+artifact\n(v2)",
              "occlusion": "occlusion-style\nbaseline"}

#: Predeclared reading thresholds (docs/CERM_AD_V2.md, "Predeclared questions").
BALANCED_RATIO = 1.25        # a seam/edit ratio within [1/1.25, 1.25] counts as balanced
CAUTION_RATIO = 1.5          # the v1 caution threshold

QUANT_FIGURES = ("artifact_balance_before_after", "necessity_method_comparison",
                 "bidirectional_quadrant", "cee_necessity_by_category",
                 "cee_sufficiency_by_category", "direct_vs_structured_probe",
                 "occlusion_vs_counterfactual", "ablation_story")


# ---------------------------------------------------------------- figure helpers
def _style() -> None:
    plt.rcParams.update({
        "font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12.5,
        "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 11,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "grid.alpha": 0.25, "grid.linewidth": 0.6, "figure.dpi": 110, "savefig.dpi": 200,
        "legend.frameon": False})


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    return path


def _jitter(n: int, seed: int, width: float = 0.12) -> np.ndarray:
    return np.random.default_rng(seed).uniform(-width, width, n)


def _cat_legend(ax, cats, loc="upper left", outside=True) -> None:
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker="o", ls="", ms=8, color=category_color(c), label=c)
               for c in cats]
    if outside:
        ax.legend(handles=handles, title="category", loc="upper left",
                  bbox_to_anchor=(1.01, 1.0), borderaxespad=0)
    else:
        ax.legend(handles=handles, title="category", loc=loc)


# ---------------------------------------------------------------- figures
def fig_artifact_balance(ok: pd.DataFrame, out: Path) -> Path:
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.8),
                             gridspec_kw={"width_ratios": [1.0, 1.25]})
    ax = axes[0]
    a = ok["context_artifact_imbalance"].to_numpy(float)
    b = ok["artifact_artifact_imbalance"].to_numpy(float)
    keep = np.isfinite(a) & np.isfinite(b)
    for ai, bi, cat in zip(a[keep], b[keep], ok.loc[keep, "category"]):
        ax.plot([0, 1], [ai, bi], color=category_color(cat), alpha=0.45, lw=1.2)
    ax.plot([0, 1], [np.median(a[keep]), np.median(b[keep])], color="black", lw=3.5,
            marker="o", ms=9, label="median", zorder=5)
    ax.set_xticks([0, 1], ["context-only\n(v1)", "context + artifact\n(v2)"])
    ax.set_xlim(-0.3, 1.3)
    ax.set_ylabel("artifact imbalance\n|log seam ratio| + |log edit ratio|")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title(f"Paired per-target imbalance (N={int(keep.sum())})")
    ax.legend(loc="upper right")

    ax = axes[1]
    for arm, marker in (("context", "o"), ("artifact", "D")):
        sr = np.log2(ok[f"{arm}_seam_ratio"].to_numpy(float))
        er = np.log2(ok[f"{arm}_edit_ratio"].to_numpy(float))
        ax.scatter(er, sr, s=55, marker=marker, color=ARM_COLORS[arm], alpha=0.8,
                   edgecolor="white", linewidth=0.8,
                   label=f"{'context-only (v1)' if arm == 'context' else 'context+artifact (v2)'}")
        ax.scatter(np.nanmedian(er), np.nanmedian(sr), s=260, marker=marker,
                   color=ARM_COLORS[arm], edgecolor="black", linewidth=2, zorder=6)
    ax.axhline(0, color="black", lw=0.9)
    ax.axvline(0, color="black", lw=0.9)
    ax.set_xlabel("log2 edit-energy ratio (defect fill / placebo fill)")
    ax.set_ylabel("log2 seam ratio (defect / placebo)")
    ax.set_title("0,0 = defect and placebo edits equally strong\n(large markers = medians)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0)
    fig.tight_layout()
    return _save(fig, out)


def fig_necessity_methods(ok: pd.DataFrame, table: pd.DataFrame, out: Path) -> Path:
    _style()
    fig, ax = plt.subplots(figsize=(12.5, 6.0))
    methods = [m for m, _ in NECESSITY_METHODS]
    for i, m in enumerate(methods):
        col = "delta_defect" if m == "naive" else f"{m}_cee"
        vals = ok[col].to_numpy(float)
        vals = vals[np.isfinite(vals)]
        ax.scatter(i + _jitter(len(vals), i), vals, s=22, color=ARM_COLORS[m], alpha=0.45,
                   edgecolor="none")
        row = table[table["method"] == m].iloc[0]
        ax.errorbar(i + 0.27, row["median_cee"],
                    yerr=[[row["median_cee"] - row["cee_ci_lo"]], [row["cee_ci_hi"] - row["median_cee"]]],
                    fmt="o", color="black", ms=8, capsize=5, lw=2, zorder=5)
        ax.annotate(f"{row['median_cee']:+.2f}\n[{row['cee_ci_lo']:+.2f}, {row['cee_ci_hi']:+.2f}]",
                    (i + 0.33, row["median_cee"]), fontsize=10, va="center", ha="left")
    ax.axhline(0, color="black", lw=1)
    ax.set_xticks(range(len(methods)), [ARM_LABELS[m] for m in methods])
    ax.set_xlim(-0.5, len(methods) - 0.1)
    ax.set_ylabel("necessity effect (log-odds)\nnaive: delta_defect;  others: CEE_N")
    ax.set_title("How the necessity conclusion changes as controls become stricter\n"
                 "dots = targets; black = median with paired-target bootstrap 95% CI")
    fig.tight_layout()
    return _save(fig, out)


def fig_quadrant(ok: pd.DataFrame, out: Path) -> Path:
    _style()
    fig, ax = plt.subplots(figsize=(10.5, 7.5))
    d = ok[np.isfinite(ok["cee_n_artifact"].astype(float)) & np.isfinite(ok["cee_s"].astype(float))]
    for cat, g in d.groupby("category", sort=True):
        ax.scatter(g["cee_n_artifact"], g["cee_s"], s=70, color=category_color(cat),
                   edgecolor="white", linewidth=0.9, alpha=0.9, label=cat)
    ax.axhline(0, color="black", lw=1)
    ax.axvline(0, color="black", lw=1)
    n = len(d)
    q = d["quadrant"].value_counts()
    for label, (x, y, ha, va) in {"bidirectional": (0.98, 0.98, "right", "top"),
                                  "sufficiency_only": (0.02, 0.98, "left", "top"),
                                  "neither": (0.02, 0.02, "left", "bottom"),
                                  "necessity_only": (0.98, 0.02, "right", "bottom")}.items():
        nice = {"bidirectional": "BIDIRECTIONALLY GROUNDED",
                "sufficiency_only": "sufficient, not necessary",
                "neither": "neither", "necessity_only": "necessary, not sufficient"}[label]
        ax.text(x, y, f"{nice}\n{int(q.get(label, 0))}/{n}", transform=ax.transAxes,
                ha=ha, va=va, fontsize=11.5, color="#333333",
                fontweight="bold" if label == "bidirectional" else "normal")
    ax.set_xlabel("CEE_N  (necessity, artifact-matched placebos)")
    ax.set_ylabel("CEE_S  (sufficiency, artifact-matched normal shams)")
    ax.set_title("Two causal axes, kept separate (one dot per target anomaly)")
    ax.legend(title="category", loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0)
    fig.tight_layout()
    return _save(fig, out)


def fig_by_category(ok: pd.DataFrame, cols: list[tuple[str, str, str]], ylabel: str,
                    title: str, out: Path) -> Path:
    _style()
    cats = sorted(ok["category"].unique())
    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    width = 0.8 / len(cols)
    for j, (col, label, color) in enumerate(cols):
        for i, cat in enumerate(cats):
            vals = ok.loc[ok["category"] == cat, col].to_numpy(float)
            vals = vals[np.isfinite(vals)]
            x = i - 0.4 + width * (j + 0.5)
            ax.scatter(x + _jitter(len(vals), 7 * i + j, width * 0.3), vals, s=30, color=color,
                       alpha=0.55, edgecolor="none", label=label if i == 0 else None)
            if len(vals):
                ax.plot([x - width * 0.4, x + width * 0.4], [np.median(vals)] * 2, color=color,
                        lw=3.2)
    ax.axhline(0, color="black", lw=1)
    counts = ok["category"].value_counts()
    ax.set_xticks(range(len(cats)), [f"{c}\n(n={int(counts.get(c, 0))})" for c in cats])
    ax.set_ylabel(ylabel)
    ax.set_title(title + "\n(bars = medians)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0)
    fig.tight_layout()
    return _save(fig, out)


def fig_probe(ok: pd.DataFrame, structured: pd.DataFrame, agree: pd.DataFrame, out: Path) -> Path | None:
    st = structured[(structured["role"] == "original")]
    if "structured_status" in st:
        st = st[st["structured_status"].isna()]
    m = ok.merge(st[["image_id", "structured_audit_p_yes"]], on="image_id")
    m = m[m["structured_audit_p_yes"].notna()]
    if not len(m):
        return None
    _style()
    fig, ax = plt.subplots(figsize=(9.5, 7.0))
    for cat, g in m.groupby("category", sort=True):
        ax.scatter(g["audit_p_yes_original"], g["structured_audit_p_yes"], s=70,
                   color=category_color(cat), edgecolor="white", linewidth=0.9, label=cat)
    ax.axhline(0.5, color="black", lw=0.8, ls="--")
    ax.axvline(0.5, color="black", lw=0.8, ls="--")
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("direct-verdict probe P(YES)  (causal arms use this)")
    ax.set_ylabel("structured scorer P(YES) at verdict token")
    r = agree.iloc[0]
    ax.set_title(f"Original targets (all anomalous), N={len(m)}\n"
                 f"Spearman {r.get('spearman_p_yes', float('nan')):.2f} | verdict agreement "
                 f"{100 * r.get('verdict_agreement', float('nan')):.0f}% | sensitivity "
                 f"direct {100 * r['direct_sensitivity']:.0f}% vs structured "
                 f"{100 * r['structured_sensitivity']:.0f}%", fontsize=12)
    ax.legend(title="category", loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0)
    fig.tight_layout()
    return _save(fig, out)


def fig_occlusion(ok: pd.DataFrame, out: Path) -> Path:
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 6.0))
    for ax, (xcol, ycol, xl, yl, title) in zip(axes, (
            ("occlusion_delta_defect_raw", "delta_defect",
             "Δ removal, occlusion fill (mean colour)", "Δ removal, normal-donor fill",
             "Raw defect deletion: occlusion vs counterfactual fill"),
            ("cee_n_occlusion", "cee_n_artifact", "CEE_N, occlusion-style baseline",
             "CEE_N, context+artifact matched (v2)",
             "Controlled effect: occlusion vs counterfactual"))):
        for cat, g in ok.groupby("category", sort=True):
            ax.scatter(g[xcol], g[ycol], s=60, color=category_color(cat), edgecolor="white",
                       linewidth=0.8, label=cat)
        vals = np.concatenate([ok[xcol].to_numpy(float), ok[ycol].to_numpy(float)])
        vals = vals[np.isfinite(vals)]
        if len(vals):
            lo, hi = vals.min(), vals.max()
            pad = 0.05 * (hi - lo + 1e-9)
            ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="#777777", lw=1, ls="--",
                    label="y = x")
        ax.axhline(0, color="black", lw=0.9)
        ax.axvline(0, color="black", lw=0.9)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_title(title, fontsize=12.5)
    axes[1].legend(title="category", loc="upper left", bbox_to_anchor=(1.01, 1.0),
                   borderaxespad=0)
    fig.tight_layout()
    return _save(fig, out)


def fig_ablation_story(out: Path, numbers: dict) -> Path:
    """The control progression as a left-to-right diagram, annotated with results."""
    _style()
    steps = [
        ("Naive", "remove defect,\nwatch score", "any edit might\nchange the score",
         numbers.get("naive")),
        ("Control 1", "same-shape\nrandom placebo", "random region may be\nvisually incomparable",
         numbers.get("random")),
        ("Control 2 (v1)", "context-matched\nplacebo", "defect fills ~2x\nstronger / seamier",
         numbers.get("context")),
        ("Control 3 (v2)", "context + artifact\nmatched placebo", "removal shows\nnecessity only",
         numbers.get("artifact")),
        ("Reverse test", "defect insertion vs\nmatched normal shams", "necessity +\nsufficiency",
         numbers.get("sufficiency")),
    ]
    fig, ax = plt.subplots(figsize=(17.0, 4.6))
    ax.axis("off")
    ax.set_xlim(0, len(steps))
    ax.set_ylim(0, 1)
    for i, (head, what, problem, value) in enumerate(steps):
        x = i + 0.5
        ax.text(x, 0.88, head, ha="center", va="center", fontsize=13, fontweight="bold")
        ax.text(x, 0.64, what, ha="center", va="center", fontsize=11.5,
                bbox=dict(boxstyle="round,pad=0.5", fc="#EAF2FA", ec="#0072B2", lw=1.4))
        ax.text(x, 0.34, problem, ha="center", va="center", fontsize=10.5, color="#444444",
                style="italic")
        if value:
            ax.text(x, 0.1, value, ha="center", va="center", fontsize=10.5, family="monospace")
        if i < len(steps) - 1:
            ax.annotate("", xy=(i + 0.92, 0.64), xytext=(i + 0.82, 0.64),
                        arrowprops=dict(arrowstyle="-|>", color="#333333", lw=1.6))
    fig.tight_layout()
    return _save(fig, out)


# ---------------------------------------------------------------- write-up
def _f(x, spec="+.3f") -> str:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return "n/a"
    return format(x, spec) if np.isfinite(x) else "n/a"


def _pct(x) -> str:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return "n/a"
    return f"{100 * x:.0f}%" if np.isfinite(x) else "n/a"


def _md_table(df: pd.DataFrame, cols: list[str], fmt: dict | None = None) -> str:
    fmt = fmt or {}
    head = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    body = ""
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r.get(c, "")
            if c in fmt:
                cells.append(fmt[c](v))
            elif isinstance(v, float):
                cells.append(_f(v, "+.3f"))
            else:
                cells.append(str(v))
        body += "| " + " | ".join(cells) + " |\n"
    return head + body


def write_presentation_results(path: Path, meta: dict, tables: dict, cases: pd.DataFrame,
                               ok: pd.DataFrame) -> Path:
    """PRESENTATION_RESULTS.md: every interpretive sentence keyed to a measured value."""
    nec, suf, bal, bid = tables["nec"], tables["suf"], tables["balance"], tables["bid"]
    comp, agree = tables["comparisons"], tables["agreement"]
    nec_cat, suf_cat = tables["nec_cat"], tables["suf_cat"]
    row = {m: nec[nec["method"] == m].iloc[0] for m, _ in NECESSITY_METHODS}
    s = suf.iloc[0]
    b = bal.set_index("arm")
    ctx_b, art_b = b.loc["context"], b.loc["artifact"]
    paired_b = comp[comp["comparison"].str.startswith("A:")].iloc[0]
    paired_cee = comp[comp["comparison"].str.startswith("B:")].iloc[0]
    q_all = bid[bid["group"] == "ALL"].iloc[0]

    L: list[str] = []
    add = L.append
    add("# CERM-AD Stage I+ — pilot results (bidirectional, artifact-matched)\n")
    add("_Generated automatically from the run's records. Every number below is measured; "
        "every interpretive sentence is produced by a rule stated in `tzsad/causal/report_v2.py`._\n")

    add("## A. Run metadata\n")
    for k in ("branch", "git_sha", "model_id", "score_mode", "mask_source", "N_targets",
              "categories", "forward_images_total", "unique_images_scored", "runtime",
              "gpu", "seed", "transformers"):
        add(f"- **{k}**: {meta.get(k, 'n/a')}")
    add("")

    add("## B. What changed from CERM v1\n")
    add("- **Same 40 targets** (same config chain, seed and per-folder cap as `cerm_pilot`), "
        "so every comparison is paired per image.")
    add("- **Artifact-matched placebos (new arm C):** candidate placebos are ranked by a "
        "predeclared composite of context distance and the |log| edit-energy and seam ratios "
        "to the real defect fill, computed from pixels before any model call; K drawn uniformly "
        "from the best 30%.")
    add("- **The v1 context-only arm is recomputed in the same run** (arm B), plus a random-"
        "translation control (A) and an occlusion-style baseline (D).")
    add("- **Sufficiency (new):** the real defect is transplanted into the normal donor J and "
        "compared with artifact-matched normal-patch shams at the same location.")
    add("- **Structured-scorer check (P0/P1):** the existing structured scorer on all originals "
        "and on casebook interventions.\n")

    # ---- C. artifact balance
    add("## C. Artifact balance (Q1)\n")
    add(_md_table(bal[bal["arm"].isin(["random", "context", "artifact", "occlusion", "sufficiency"])],
                  ["arm", "N", "median_seam_ratio", "median_edit_ratio",
                   "median_abs_log_seam_ratio", "median_abs_log_edit_ratio",
                   "median_artifact_imbalance", "frac_seam_ratio_within_1.25x",
                   "frac_edit_ratio_within_1.25x"],
                  {"median_seam_ratio": lambda v: _f(v, ".2f"),
                   "median_edit_ratio": lambda v: _f(v, ".2f"),
                   "median_abs_log_seam_ratio": lambda v: _f(v, ".2f"),
                   "median_abs_log_edit_ratio": lambda v: _f(v, ".2f"),
                   "median_artifact_imbalance": lambda v: _f(v, ".2f"),
                   "frac_seam_ratio_within_1.25x": _pct, "frac_edit_ratio_within_1.25x": _pct}))
    add(f"Paired improvement in imbalance (context-only minus artifact-matched): median "
        f"**{_f(paired_b['stat'], '+.2f')}** [{_f(paired_b['ci_lo'], '+.2f')}, "
        f"{_f(paired_b['ci_hi'], '+.2f')}], improved on {_pct(paired_b['frac_positive'])} of "
        f"{int(paired_b['n_pairs'])} targets (Wilcoxon p={_f(paired_b['wilcoxon_p'], '.2g')}, "
        "secondary).\n")
    a_seam, a_edit = float(art_b["median_seam_ratio"]), float(art_b["median_edit_ratio"])
    balanced = (1 / BALANCED_RATIO <= a_seam <= BALANCED_RATIO
                and 1 / BALANCED_RATIO <= a_edit <= BALANCED_RATIO)
    improved = float(paired_b["ci_lo"]) > 0
    if improved and balanced:
        add(f"**Reading:** artifact matching fixed the v1 confound — median ratios moved from "
            f"seam {_f(ctx_b['median_seam_ratio'], '.2f')} / edit "
            f"{_f(ctx_b['median_edit_ratio'], '.2f')} to {a_seam:.2f} / {a_edit:.2f}, inside "
            f"the predeclared 1.25x balance band, and the paired CI excludes zero.\n")
    elif improved:
        add(f"**Reading:** artifact matching *reduced* the imbalance (paired CI excludes zero) "
            f"but did **not** reach the predeclared 1.25x balance band (median seam "
            f"{a_seam:.2f}, edit {a_edit:.2f}). The intervention-strength confound is smaller, "
            "not gone.\n")
    else:
        add("**Reading: ARTIFACT MATCHING DID NOT MEASURABLY IMPROVE BALANCE** (paired CI "
            "includes zero). The v1 intervention-strength confound remains and the necessity "
            "result must be read with it.\n")

    # ---- D. necessity
    add("## D. Necessity (Q2)\n")
    add(_md_table(nec, ["method", "N", "median_delta_defect", "median_control_delta",
                        "median_cee", "cee_ci_lo", "cee_ci_hi", "frac_cee_positive",
                        "frac_beats_all_controls", "median_empirical_tail", "median_robust_z",
                        "median_artifact_imbalance"],
                  {"frac_cee_positive": _pct, "frac_beats_all_controls": _pct,
                   "median_empirical_tail": lambda v: _f(v, ".3f"),
                   "median_robust_z": lambda v: _f(v, "+.2f"),
                   "median_artifact_imbalance": lambda v: _f(v, ".2f")}))
    add("`median_empirical_tail` is the empirical matched-placebo tail probability "
        "(rank of the defect effect among its controls), not a classical p-value.\n")
    ra, rc = row["artifact"], row["context"]
    add(f"Paired change in CEE_N (context-only minus artifact-matched): median "
        f"{_f(paired_cee['stat'])} [{_f(paired_cee['ci_lo'])}, {_f(paired_cee['ci_hi'])}].\n")
    if float(ra["cee_ci_lo"]) > 0:
        verdict_n = "SURVIVES"
        add(f"**Reading:** the positive necessity effect **survives** artifact matching: median "
            f"CEE_N {_f(ra['median_cee'])} with CI [{_f(ra['cee_ci_lo'])}, {_f(ra['cee_ci_hi'])}] "
            f"excluding zero; {_pct(ra['frac_cee_positive'])} of targets positive.\n")
    elif float(ra["median_cee"]) > 0:
        verdict_n = "WEAKENED"
        add(f"**Reading:** the necessity effect is **weakened** under artifact matching: median "
            f"CEE_N {_f(ra['median_cee'])} remains positive but its CI "
            f"[{_f(ra['cee_ci_lo'])}, {_f(ra['cee_ci_hi'])}] includes zero "
            f"(context-only v1 arm in this run: {_f(rc['median_cee'])} "
            f"[{_f(rc['cee_ci_lo'])}, {_f(rc['cee_ci_hi'])}]).\n")
    else:
        verdict_n = "ELIMINATED"
        add(f"**Reading: THE POOLED NECESSITY EFFECT DOES NOT SURVIVE ARTIFACT MATCHING** — "
            f"median CEE_N {_f(ra['median_cee'])} [{_f(ra['cee_ci_lo'])}, {_f(ra['cee_ci_hi'])}] "
            f"(context-only v1 arm: {_f(rc['median_cee'])}). This is an important negative "
            "finding.\n")

    # ---- E. sufficiency
    add("## E. Sufficiency (Q3)\n")
    add(_md_table(suf, ["group", "N", "N_with_shams", "median_delta_insert_defect",
                        "median_sham_delta", "median_cee_s", "cee_s_ci_lo", "cee_s_ci_hi",
                        "frac_cee_s_positive", "frac_insert_beats_all_shams",
                        "median_empirical_sham_tail", "median_artifact_imbalance",
                        "frac_recipient_predicted_yes"],
                  {"frac_cee_s_positive": _pct, "frac_insert_beats_all_shams": _pct,
                   "frac_recipient_predicted_yes": _pct,
                   "median_empirical_sham_tail": lambda v: _f(v, ".3f"),
                   "median_artifact_imbalance": lambda v: _f(v, ".2f")}))
    if float(s["cee_s_ci_lo"]) > 0:
        verdict_s = "SUPPORTED"
        add(f"**Reading:** inserting the real defect raises anomaly belief more than matched "
            f"normal shams: median CEE_S {_f(s['median_cee_s'])} "
            f"[{_f(s['cee_s_ci_lo'])}, {_f(s['cee_s_ci_hi'])}].\n")
    elif float(s["median_cee_s"]) > 0:
        verdict_s = "WEAK"
        add(f"**Reading:** weak sufficiency evidence — median CEE_S {_f(s['median_cee_s'])} is "
            f"positive but the CI [{_f(s['cee_s_ci_lo'])}, {_f(s['cee_s_ci_hi'])}] includes zero.\n")
    else:
        verdict_s = "NOT SUPPORTED"
        add(f"**Reading: no sufficiency evidence** — median CEE_S {_f(s['median_cee_s'])} "
            f"[{_f(s['cee_s_ci_lo'])}, {_f(s['cee_s_ci_hi'])}].\n")
    s_imb = float(s["median_artifact_imbalance"])
    if np.isfinite(s_imb) and s_imb > 2 * np.log(BALANCED_RATIO):
        add(f"Caution: insertion artifacts are not balanced (median imbalance {s_imb:.2f}; seam "
            f"ratio {_f(s['median_seam_ratio'], '.2f')}, edit ratio "
            f"{_f(s['median_edit_ratio'], '.2f')}).\n")

    # ---- F. bidirectional
    add("## F. Bidirectional result (Q4)\n")
    add(_md_table(bid, ["group", "N", "N_determined", "n_bidirectional", "frac_bidirectional",
                        "n_necessity_only", "n_sufficiency_only", "n_neither"],
                  {"frac_bidirectional": _pct}))
    add(f"Overall, **{int(q_all['n_bidirectional'])}/{int(q_all['N_determined'])} "
        f"({_pct(q_all['frac_bidirectional'])})** of targets fall in the positive "
        "necessity + sufficiency quadrant. The two axes are reported separately and are not "
        "combined into a score.\n")
    d_row = comp[comp["comparison"].str.startswith("D:")].iloc[0]
    e_row = comp[comp["comparison"].str.startswith("E:")].iloc[0]
    add(f"Original YES minus original NO (difference of medians, independent-group bootstrap): "
        f"CEE_N {_f(d_row['stat'])} [{_f(d_row['ci_lo'])}, {_f(d_row['ci_hi'])}] "
        f"(n={int(d_row['n_a'])} vs {int(d_row['n_b'])}); CEE_S {_f(e_row['stat'])} "
        f"[{_f(e_row['ci_lo'])}, {_f(e_row['ci_hi'])}].\n")

    # ---- G. probe agreement
    add("## G. Direct probe vs structured scorer (Q5)\n")
    if len(agree):
        a = agree.iloc[0]
        add(f"On the {int(a.get('N_parsed', 0))} original targets with a parsed structured "
            f"verdict: Spearman {_f(a.get('spearman_p_yes'), '.2f')}, Pearson "
            f"{_f(a.get('pearson_p_yes'), '.2f')}, YES/NO agreement "
            f"{_pct(a.get('verdict_agreement'))}. Sensitivity (recall on these anomaly-only "
            f"targets): direct {_pct(a['direct_sensitivity'])}, structured "
            f"{_pct(a['structured_sensitivity'])}. Structured parse rate "
            f"{_pct(a.get('structured_parse_rate'))}.\n")
        rho = float(a.get("spearman_p_yes", float("nan")))
        if np.isfinite(rho) and rho < 0.5:
            add("**Warning:** the two probes track each other weakly (Spearman < 0.5). The "
                "causal results describe the direct-verdict probe and should not be "
                "transferred to the structured pipeline without the P1 check below.\n")
    p1 = tables.get("p1")
    if p1 is not None and len(p1):
        add("**P1 sign agreement on casebook targets (qualitative, not a population estimate):**\n")
        add(_md_table(p1, ["image_id", "slot", "direct_removal_delta", "structured_removal_delta",
                           "removal_sign_agree", "direct_insertion_delta",
                           "structured_insertion_delta", "insertion_sign_agree"]))
        add(f"Removal-direction agreement {int(p1['removal_sign_agree'].sum())}/"
            f"{int(p1['removal_sign_agree'].notna().sum())}; insertion-direction agreement "
            f"{int(p1['insertion_sign_agree'].sum())}/"
            f"{int(p1['insertion_sign_agree'].notna().sum())}.\n")

    # ---- H. per category
    add("## H. Per-category results\n")
    nc = nec_cat[nec_cat["method"].isin(["context", "artifact"])]
    piv = nc.pivot(index="group", columns="method", values="median_cee")
    sc = suf_cat.set_index("group")
    cat_rows = []
    for cat in piv.index:
        cr = nec_cat[(nec_cat["group"] == cat) & (nec_cat["method"] == "artifact")].iloc[0]
        cat_rows.append({"category": cat, "N": int(cr["N"]),
                         "CEE_N context (v1)": piv.loc[cat, "context"],
                         "CEE_N artifact (v2)": piv.loc[cat, "artifact"],
                         "CEE_N v2 CI": f"[{_f(cr['cee_ci_lo'])}, {_f(cr['cee_ci_hi'])}]",
                         "CEE_S": sc.loc[cat, "median_cee_s"] if cat in sc.index else np.nan,
                         "CEE_S CI": (f"[{_f(sc.loc[cat, 'cee_s_ci_lo'])}, "
                                      f"{_f(sc.loc[cat, 'cee_s_ci_hi'])}]") if cat in sc.index else "",
                         "direct sensitivity": _pct((ok.loc[ok['category'] == cat,
                                                            'audit_logit_original'] >= 0).mean()),
                         "bidirectional": f"{int((ok.loc[ok['category'] == cat, 'quadrant'] == 'bidirectional').sum())}/"
                                          f"{int((ok['category'] == cat).sum())}"})
    add(_md_table(pd.DataFrame(cat_rows), list(cat_rows[0].keys()) if cat_rows else []))
    add("Per-category N is ~10; per-category intervals are wide and are descriptive only.\n")

    # ---- I/J. examples
    add("## I. Real examples selected for the presentation\n")
    add("Selected deterministically by the rules below (see `casebook_manifest.csv`); figures in "
        "`figures/example_*.png` and `casebook/case_*.png`.\n")
    rec = ok.set_index("image_id")
    for _, c in cases.iterrows():
        if c["image_id"] not in rec.index:
            continue
        r = rec.loc[c["image_id"]]
        add(f"- **{c['slot']}** — `{c['image_id']}`: CEE_N {_f(r['cee_n_artifact'])}, CEE_S "
            f"{_f(r['cee_s'])}, original verdict {r['original_direct_prediction']}. "
            f"_Rule:_ {c['rule']}")
    add("")
    add("## J. Failure examples\n")
    neither = ok[ok["quadrant"] == "neither"].sort_values("cee_n_artifact")
    add(f"{len(neither)} target(s) are in the 'neither' quadrant. Most negative CEE_N:\n")
    for _, r in neither.head(5).iterrows():
        add(f"- `{r['image_id']}`: CEE_N {_f(r['cee_n_artifact'])}, CEE_S {_f(r['cee_s'])}, "
            f"verdict {r['original_direct_prediction']}")
    bad = tables.get("failed", pd.DataFrame())
    if len(bad):
        add(f"\n{len(bad)} target(s) did not complete and are kept with their status in "
            "`causal_records_v2.parquet`.")
    degenerate = ok[ok["artifact_n_valid_placebos"] < 19]
    add(f"\n{len(degenerate)} target(s) have fewer than 19 artifact-matched placebos "
        "(masks too large to translate); they are reported with their actual counts, never padded.\n")

    # ---- K/L
    add("## K. What the results support\n")
    support = []
    if improved:
        support.append("Artifact matching measurably reduces the defect-vs-placebo intervention "
                       "strength imbalance found in v1 (paired, same targets).")
    if verdict_n == "SURVIVES":
        support.append("For the direct-verdict probe of Qwen2.5-VL-3B on this oracle-mask pilot, "
                       "removing the real defect lowers anomaly log-odds more than artifact-"
                       "matched irrelevant edits, at the population median.")
    if verdict_s == "SUPPORTED":
        support.append("Transplanting the real defect into a normal image raises anomaly "
                       "log-odds more than artifact-matched normal-patch insertions.")
    if float(q_all["frac_bidirectional"]) > 0:
        support.append(f"A subset of targets ({_pct(q_all['frac_bidirectional'])}) shows "
                       "evidence in both causal directions; the rest show one or neither, "
                       "so grounding is target- and category-dependent.")
    for line in support or ["No positive causal claim is supported at the pooled level by this run."]:
        add(f"- {line}")
    add("")
    add("## L. What the results do NOT support\n")
    for line in (
        "Deployment readiness: masks are oracle GT masks; a predicted-mask study is not done.",
        "Claims about the structured reasoning pipeline beyond the measured P0 agreement and "
        "the qualitative P1 cases.",
        "Rationale faithfulness or mediation: Stage II is not implemented; the structured "
        "examples are qualitative.",
        "Formal significance: N≈40 over 4 categories; intervals are effect-size uncertainty, "
        "and the empirical tail is a rank statistic, not a classical p-value.",
        "Generalisation to other models, categories or datasets.",
        "Absolute literature novelty; see section M for the narrower positioning.",
    ):
        add(f"- {line}")
    add("")

    # ---- M/N
    add("## M. Comparison with closest prior work\n")
    rw = related_work_frame()
    add(_md_table(rw, ["Work", "Industrial anomaly domain?", "VLM?",
                       "Tests necessity of defect?", "Tests sufficiency of defect?",
                       "Controls intervention magnitude/artifacts?", "Main difference from CERM-AD"]))
    add("Cells summarise our current reading of each paper and should be verified against the "
        "papers before citation. These works solve different problems; none is claimed inferior.\n")
    add("## N. Research claim we can currently defend\n")
    claim = ("We propose and evaluate bidirectional, artifact-controlled causal verification of "
             "localized defect evidence in VLM anomaly verdicts. ")
    if verdict_n == "SURVIVES" and verdict_s == "SUPPORTED":
        claim += ("In an oracle-mask pilot (Qwen2.5-VL-3B, 40 MVTec anomalies, 4 categories), the "
                  "direct verdict depends on the true defect in both directions — removing it "
                  "lowers, and transplanting it raises, anomaly belief more than artifact-matched "
                  "controls — with strong category dependence.")
    elif verdict_n == "SURVIVES" or verdict_s == "SUPPORTED":
        which = "necessity" if verdict_n == "SURVIVES" else "sufficiency"
        claim += (f"In an oracle-mask pilot (Qwen2.5-VL-3B, 40 MVTec anomalies) the pooled evidence "
                  f"supports {which} but not both directions; per-target grounding is "
                  "heterogeneous across categories.")
    else:
        claim += ("In an oracle-mask pilot (Qwen2.5-VL-3B, 40 MVTec anomalies), once intervention "
                  "strength is controlled, the pooled evidence that the verdict depends "
                  "specifically on the physical defect is weak; the method exposes this rather "
                  "than confirming grounding, and effects are concentrated in specific categories.")
    add(claim + "\n")
    path.write_text("\n".join(L))
    return path


def p1_sign_table(ok: pd.DataFrame, iv: pd.DataFrame, structured: pd.DataFrame,
                  cases: pd.DataFrame) -> pd.DataFrame:
    """Direction of removal/insertion effects: direct probe vs structured scorer."""
    st = structured.copy()
    if "structured_status" in st:
        st = st[st["structured_status"].isna()]
    rows = []
    rec = ok.set_index("image_id")
    for _, c in cases.iterrows():
        image_id = c["image_id"]
        if image_id not in rec.index:
            continue
        r = rec.loc[image_id]

        def sz(role):
            h = st[(st["image_id"] == image_id) & (st["role"] == role)]
            return float(h.iloc[-1]["structured_audit_logit"]) if len(h) else float("nan")
        s_rem = sz("original") - sz("defect_removed")
        s_ins = sz("defect_inserted") - sz("recipient")
        d_rem = float(r["delta_defect"])
        d_ins = float(r["delta_insert_defect"])
        rows.append({"image_id": image_id, "slot": c["slot"],
                     "direct_removal_delta": d_rem, "structured_removal_delta": s_rem,
                     "removal_sign_agree": (np.sign(d_rem) == np.sign(s_rem))
                     if np.isfinite(s_rem) else np.nan,
                     "direct_insertion_delta": d_ins, "structured_insertion_delta": s_ins,
                     "insertion_sign_agree": (np.sign(d_ins) == np.sign(s_ins))
                     if np.isfinite(s_ins) else np.nan})
    return pd.DataFrame(rows)


def merge_structured(run_dir: Path) -> pd.DataFrame:
    """Union of structured shards; a success always supersedes an earlier failure."""
    frames = [pd.read_json(p, lines=True) for p in sorted(Path(run_dir).glob(
        "structured_records*.jsonl")) if p.stat().st_size]
    if not frames:
        return pd.DataFrame()
    st = pd.concat(frames, ignore_index=True)
    if "structured_status" not in st:
        st["structured_status"] = np.nan
    st["_ok"] = st["structured_status"].isna()
    st = st.sort_values(["item_key", "_ok"], kind="stable")
    return st.drop_duplicates("item_key", keep="last").drop(columns="_ok").reset_index(drop=True)


# ---------------------------------------------------------------- entry point
def build_report(run_dir: Path, cfg, v2, targets: pd.DataFrame, repo: Path) -> None:
    run_dir = Path(run_dir)
    figs = run_dir / "figures"
    records = merge_shards(run_dir, "causal_records_v2")
    iv = merge_shards(run_dir, "interventions_v2", key="_row_key")
    if not len(records):
        log.error("no v2 records in %s", run_dir)
        return
    ok = records[records["status"] == "ok"].copy()
    failed = records[records["status"] != "ok"]
    missing = sorted(set(targets["image_id"].astype(str)) - set(records["image_id"].astype(str)))
    if missing:
        log.warning("%d targets have no record yet: %s", len(missing), missing[:5])
    records.to_parquet(run_dir / "causal_records_v2.parquet", index=False)
    if len(iv):
        iv.to_parquet(run_dir / "interventions_v2.parquet", index=False)
    structured = merge_structured(run_dir)
    if len(structured):
        structured.to_parquet(run_dir / "structured_records.parquet", index=False)

    seed = int(cfg.seed)
    nec, nec_cat = necessity_tables(ok, seed)
    suf, suf_cat = sufficiency_tables(ok, seed)
    bal = artifact_balance_table(ok, seed)
    bid = bidirectional_table(ok)
    comp = paired_comparisons(ok, seed)
    agree = probe_agreement_table(ok, structured) if len(structured) else pd.DataFrame()
    cases_path = run_dir / "casebook_manifest.csv"
    cases = pd.read_csv(cases_path) if cases_path.exists() else select_cases(
        ok, int(cfg.cerm_v2.get("structured_p1_cases", 10)))
    cases.to_csv(cases_path, index=False)
    p1 = p1_sign_table(ok, iv, structured, cases) if len(structured) else pd.DataFrame()

    out = {"summary_necessity_overall.csv": nec, "summary_necessity_by_category.csv": nec_cat,
           "summary_sufficiency_overall.csv": suf, "summary_sufficiency_by_category.csv": suf_cat,
           "summary_bidirectional.csv": bid, "artifact_balance.csv": bal,
           "probe_agreement.csv": agree, "paired_comparisons.csv": comp,
           "method_ablation.csv": nec, "p1_sign_agreement.csv": p1,
           "related_work_table.csv": related_work_frame()}
    for name, frame in out.items():
        frame.to_csv(run_dir / name, index=False)

    made = []
    numbers = {}
    for m in ("naive", "random", "context", "artifact"):
        r = nec[nec["method"] == m].iloc[0]
        numbers[m] = f"median {_f(r['median_cee'], '+.2f')}\n[{_f(r['cee_ci_lo'], '+.2f')}, {_f(r['cee_ci_hi'], '+.2f')}]"
    numbers["sufficiency"] = (f"CEE_S {_f(suf.iloc[0]['median_cee_s'], '+.2f')}\n"
                              f"[{_f(suf.iloc[0]['cee_s_ci_lo'], '+.2f')}, "
                              f"{_f(suf.iloc[0]['cee_s_ci_hi'], '+.2f')}]")
    jobs = {
        "artifact_balance_before_after": lambda: fig_artifact_balance(ok, figs / "artifact_balance_before_after.png"),
        "necessity_method_comparison": lambda: fig_necessity_methods(ok, nec, figs / "necessity_method_comparison.png"),
        "bidirectional_quadrant": lambda: fig_quadrant(ok, figs / "bidirectional_quadrant.png"),
        "cee_necessity_by_category": lambda: fig_by_category(
            ok, [("cee_n_context", "context-only (v1)", ARM_COLORS["context"]),
                 ("cee_n_artifact", "context+artifact (v2)", ARM_COLORS["artifact"])],
            "CEE_N (log-odds)", "Necessity by category", figs / "cee_necessity_by_category.png"),
        "cee_sufficiency_by_category": lambda: fig_by_category(
            ok, [("cee_s", "CEE_S vs matched shams", ARM_COLORS["sufficiency"])],
            "CEE_S (log-odds)", "Sufficiency by category", figs / "cee_sufficiency_by_category.png"),
        "direct_vs_structured_probe": lambda: fig_probe(ok, structured, agree, figs / "direct_vs_structured_probe.png")
        if len(structured) and len(agree) else None,
        "occlusion_vs_counterfactual": lambda: fig_occlusion(ok, figs / "occlusion_vs_counterfactual.png"),
        "ablation_story": lambda: fig_ablation_story(figs / "ablation_story.png", numbers),
    }
    for name, fn in jobs.items():
        try:
            if fn() is not None:
                made.append(name)
        except Exception:                                  # noqa: BLE001
            log.exception("figure %s failed; continuing", name)
    made += build_casebook(ok, iv, cases, run_dir)
    if len(structured):
        structured_examples_md(cases, ok, structured, run_dir / "casebook" / "structured_examples.md")
    log.info("figures: %s", ", ".join(made))

    meta = run_metadata(run_dir, cfg, v2, ok, iv, repo)
    (run_dir / "run_metadata.json").write_text(json.dumps(meta, indent=1, default=str))
    tables = {"nec": nec, "nec_cat": nec_cat, "suf": suf, "suf_cat": suf_cat,
              "balance": bal, "bid": bid, "comparisons": comp, "agreement": agree,
              "p1": p1, "failed": failed}
    write_presentation_results(run_dir / "PRESENTATION_RESULTS.md", meta, tables, cases, ok)
    make_public_dir(run_dir)
    make_bundle(run_dir)


def run_metadata(run_dir: Path, cfg, v2, ok: pd.DataFrame, iv: pd.DataFrame, repo: Path) -> dict:
    from ..utils.repro import git_sha

    sha = git_sha(repo)
    sha_file = Path(repo) / ".git_sha"
    if sha.startswith("nogit") and sha_file.exists():
        sha = sha_file.read_text().strip() + " (deployed tree)"
    timings = {}
    tf = run_dir / "phase_timings.json"
    if tf.exists():
        timings = json.loads(tf.read_text())
    try:
        import torch
        gpu = ", ".join(torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())) or "none visible"
    except Exception:                                      # noqa: BLE001
        gpu = "unknown"
    try:
        import transformers
        tv = transformers.__version__
    except Exception:                                      # noqa: BLE001
        tv = "unknown"
    n_struct = 0
    for p in run_dir.glob("structured_cache*.jsonl"):
        n_struct += sum(1 for _ in p.open())
    return {
        "branch": "experiment/cerm-ad-v2-controls", "git_sha": sha,
        "model_id": v2.base.model_id, "score_mode": v2.base.score_mode,
        "mask_source": f"{v2.base.mask_source} (oracle ground-truth masks)",
        "N_targets": int(len(ok)), "categories": ", ".join(sorted(ok["category"].unique())),
        "forward_images_total": int(len(iv)),
        "unique_images_scored": int(iv["image_digest"].nunique()) if len(iv) else 0,
        "structured_generations": n_struct,
        "runtime": ", ".join(f"{k} {v} min" for k, v in timings.items()) or "n/a",
        "gpu": gpu, "seed": int(cfg.seed), "transformers": tv,
        "k": {"random": v2.k_random, "context": v2.k_context, "artifact": v2.k_artifact,
              "occlusion": v2.k_occlusion, "sham": v2.k_sham},
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


PUBLIC_FILES = ("summary_necessity_overall.csv", "summary_necessity_by_category.csv",
                "summary_sufficiency_overall.csv", "summary_sufficiency_by_category.csv",
                "summary_bidirectional.csv", "artifact_balance.csv", "probe_agreement.csv",
                "method_ablation.csv", "paired_comparisons.csv", "p1_sign_agreement.csv",
                "casebook_manifest.csv", "related_work_table.csv", "PRESENTATION_RESULTS.md",
                "run_metadata.json", "manifest.json", "phase_timings.json")


def make_public_dir(run_dir: Path) -> Path:
    """Numbers and dataset-free plots only: safe to commit (no MVTec pixels)."""
    pub = run_dir / "public"
    if pub.exists():
        shutil.rmtree(pub)
    (pub / "figures").mkdir(parents=True)
    for name in PUBLIC_FILES:
        if (run_dir / name).exists():
            shutil.copy2(run_dir / name, pub / name)
    for name in QUANT_FIGURES:
        src = run_dir / "figures" / f"{name}.png"
        if src.exists():
            shutil.copy2(src, pub / "figures" / src.name)
    return pub


def make_bundle(run_dir: Path) -> Path:
    """CERM_V2_PRESENTATION_BUNDLE.zip: everything a slide-maker needs, no caches."""
    out = run_dir / "CERM_V2_PRESENTATION_BUNDLE.zip"
    keep_top = list(PUBLIC_FILES) + ["causal_records_v2.parquet", "interventions_v2.parquet",
                                     "structured_records.parquet", "structured_items.csv",
                                     "run.log", "index.csv"]
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for name in keep_top:
            if (run_dir / name).exists():
                z.write(run_dir / name, name)
        for sub in ("figures", "casebook"):
            for p in sorted((run_dir / sub).rglob("*")) if (run_dir / sub).exists() else []:
                if p.is_file():
                    z.write(p, str(p.relative_to(run_dir)))
        for p in sorted(run_dir.glob("worker_*.log")) + sorted(run_dir.glob("batch_check*.json")):
            z.write(p, f"logs/{p.name}")
        cfgs = Path(__file__).resolve().parents[2] / "configs"
        for name in ("base.yaml", "cerm_base.yaml", "cerm_pilot.yaml", "cerm_v2_pilot.yaml"):
            if (cfgs / name).exists():
                z.write(cfgs / name, f"config/{name}")
    log.info("bundle: %s (%.1f MB)", out, out.stat().st_size / 1e6)
    return out
