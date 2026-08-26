"""The paper's figures.

The corruption figure is the single most persuasive one - accuracy falling while
uncertainty stays flat is the silent-failure claim made visible - so it gets the
most care here.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from ..eval.metrics import reliability_curve
from ..eval.selective import risk_coverage_curve
from .style import PALETTE, SEMANTIC, save_figure, use_paper_style


def plot_risk_coverage(records: pd.DataFrame, correct_col: str, signals: list[str],
                       out_path: str | Path, title: str = "Risk-coverage") -> list[Path]:
    """Risk-coverage curves for several uncertainty signals, plus the oracle."""
    use_paper_style()
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    correct = records[correct_col].to_numpy()
    abstain = (~records["parse_ok"].to_numpy(dtype=bool)) if "parse_ok" in records else None

    oracle = risk_coverage_curve(correct, 1.0 - correct.astype(float), None)
    ax.plot(oracle.coverage, oracle.risk, color=SEMANTIC["oracle"], ls=":", lw=1.2,
            label=f"oracle ({oracle.aurc:.3f})")
    for i, s in enumerate(signals):
        rc = risk_coverage_curve(correct, records[s].to_numpy(dtype=float), abstain)
        ax.plot(rc.coverage, rc.risk, color=PALETTE[i % len(PALETTE)],
                label=f"{s.removeprefix('u_')} ({rc.aurc:.3f})")
    ax.set_xlabel("coverage")
    ax.set_ylabel("selective risk")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=6)
    return save_figure(fig, out_path)


def plot_reliability(records: pd.DataFrame, out_path: str | Path, n_bins: int = 12,
                     title: str = "Reliability") -> list[Path]:
    """Reliability diagram with adaptive bins and a bin-count rug."""
    use_paper_style()
    fig, ax = plt.subplots(figsize=(3.0, 2.8))
    conf, freq, cnt = reliability_curve(records["label"].to_numpy(),
                                        records["anomaly_score"].to_numpy(dtype=float), n_bins)
    ax.plot([0, 1], [0, 1], color=SEMANTIC["random"], ls="--", lw=1.0, label="perfect")
    ax.plot(conf, freq, "o-", color=SEMANTIC["ours"], ms=3.5, label="observed")
    for c, f, n in zip(conf, freq, cnt):
        ax.vlines(c, min(c, f), max(c, f), color=SEMANTIC["danger"], alpha=0.35, lw=1.0)
    ax.set_xlabel("mean predicted P(anomaly)")
    ax.set_ylabel("empirical frequency")
    ax.set_title(title)
    ax.legend(loc="upper left")
    return save_figure(fig, out_path)


def plot_coverage_vs_delta(coverage_df: pd.DataFrame, out_path: str | Path) -> list[Path]:
    """Empirical vs nominal conformal coverage, pooled, with bootstrap CIs."""
    use_paper_style()
    fig, ax = plt.subplots(figsize=(3.2, 2.6))
    d = coverage_df[coverage_df["group"] == "POOLED"].sort_values("nominal")
    ax.plot([0, 1], [0, 1], color=SEMANTIC["random"], ls="--", lw=1.0, label="nominal")
    ax.errorbar(d["nominal"], d["empirical_coverage"],
                yerr=[d["empirical_coverage"] - d["ci_lo"], d["ci_hi"] - d["empirical_coverage"]],
                fmt="o-", color=SEMANTIC["ours"], ms=3.5, capsize=2, label="empirical")
    lo = float(min(d["nominal"].min(), d["empirical_coverage"].min())) - 0.05
    ax.set_xlim(lo, 1.02)
    ax.set_ylim(lo, 1.02)
    ax.set_xlabel("nominal coverage $1-\\delta$")
    ax.set_ylabel("empirical coverage on normals")
    ax.set_title("Conformal coverage")
    ax.legend(loc="upper left")
    return save_figure(fig, out_path)


def plot_corruption_sweep(sweep: pd.DataFrame, out_path: str | Path,
                          uncertainty_signals: list[str] | None = None) -> list[Path]:
    """**The silent-failure figure**: AUROC and uncertainty vs corruption severity.

    Left panel: detection AUROC falling with severity. Right panel: mean
    uncertainty per signal. A signal whose curve stays flat while the left panel
    collapses is not detecting its own failure - the headline negative result.

    ``sweep`` must have columns ``corruption``, ``severity``, ``auroc`` and one
    ``mean_<signal>`` column per uncertainty signal.
    """
    use_paper_style()
    signals = uncertainty_signals or [c.removeprefix("mean_") for c in sweep.columns
                                      if c.startswith("mean_u_")]
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.7))

    for i, (name, g) in enumerate(sweep.groupby("corruption", sort=True)):
        g = g.sort_values("severity")
        axes[0].plot(g["severity"], g["auroc"], "o-", ms=3,
                     color=PALETTE[i % len(PALETTE)], label=name)
    axes[0].set_xlabel("severity")
    axes[0].set_ylabel("image AUROC")
    axes[0].set_title("Detection degrades")
    axes[0].legend(fontsize=5.5, ncol=2)

    pooled = sweep.groupby("severity", as_index=False).mean(numeric_only=True)
    for i, s in enumerate(signals):
        col = f"mean_{s}"
        if col not in pooled:
            continue
        v = pooled[col].to_numpy(dtype=float)
        rng = np.nanmax(v) - np.nanmin(v)
        norm = (v - np.nanmin(v)) / rng if rng > 1e-12 else np.zeros_like(v)
        rho = _spearman(pooled["severity"].to_numpy(dtype=float), v)
        axes[1].plot(pooled["severity"], norm, "o-", ms=3, color=PALETTE[i % len(PALETTE)],
                     label=f"{s.removeprefix('u_')} ($\\rho$={rho:+.2f})")
    axes[1].set_xlabel("severity")
    axes[1].set_ylabel("mean uncertainty (min-max scaled)")
    axes[1].set_title("Does uncertainty notice?")
    axes[1].legend(fontsize=5.5)
    fig.tight_layout()
    return save_figure(fig, out_path)


def plot_error_prediction(table: pd.DataFrame, out_path: str | Path) -> list[Path]:
    """Forest plot of error-prediction AUROC per signal with CIs and the 0.5 line."""
    use_paper_style()
    d = table[table["scope"] == "POOLED"].sort_values("err_auroc")
    fig, ax = plt.subplots(figsize=(3.6, 0.32 * max(len(d), 4) + 1.0))
    y = np.arange(len(d))
    colors = [SEMANTIC["good"] if inf else (SEMANTIC["danger"] if mis else SEMANTIC["baseline"])
              for inf, mis in zip(d["informative"], d["misleading"])]
    # One errorbar call per row: matplotlib's ecolor takes a single colour, and
    # the colour is what distinguishes an informative signal from a misleading one.
    for yi, (_, row), color in zip(y, d.iterrows(), colors):
        ax.errorbar(row["err_auroc"], yi,
                    xerr=[[row["err_auroc"] - row["ci_lo"]], [row["ci_hi"] - row["err_auroc"]]],
                    fmt="none", ecolor=color, capsize=2, lw=1.2)
    ax.scatter(d["err_auroc"], y, c=colors, s=18, zorder=3)
    ax.axvline(0.5, color=SEMANTIC["random"], ls="--", lw=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels([s.removeprefix("u_") for s in d["signal"]])
    ax.set_xlabel("AUROC for predicting misclassification")
    ax.set_title("Does uncertainty predict error?")
    return save_figure(fig, out_path)


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rho, NaN-safe, without dragging scipy in for one number."""
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return float("nan")
    rx = pd.Series(x[ok]).rank().to_numpy()
    ry = pd.Series(y[ok]).rank().to_numpy()
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = np.sqrt((rx**2).sum() * (ry**2).sum())
    return float((rx * ry).sum() / denom) if denom else float("nan")
