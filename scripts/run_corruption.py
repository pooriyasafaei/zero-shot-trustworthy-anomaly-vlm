#!/usr/bin/env python3
"""Phase 3: the corruption sweep.

Reports, for every corruption and severity, the detection AUROC *and* the mean of
every uncertainty signal, plus the Spearman correlation between severity and each
signal. A signal whose correlation is near zero while AUROC collapses is the
silent-failure mode the paper is about.

Usage::

    python scripts/run_corruption.py --config configs/corruption.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from tzsad import pipeline as P
from tzsad.eval.bootstrap import bootstrap_metric
from tzsad.eval.metrics import safe_auroc
from tzsad.eval.selective import risk_coverage_curve
from tzsad.records import uncertainty_columns
from tzsad.utils.config import load_config, to_dict
from tzsad.utils.logging import setup_logging
from tzsad.utils.repro import start_run
from tzsad.viz.plots import _spearman, plot_corruption_sweep


def evaluate_setting(cfg, index, embedders, corruption: str, severity: int) -> dict:
    """Score one (corruption, severity) cell and summarise it."""
    P.stage_embed(cfg, index, corruption, severity)
    per_backbone = P.stage_score_clip(cfg, index, embedders, corruption, severity)
    records = per_backbone[P.backbones_of(cfg)[0].cache_tag]
    test, _, coverage = P.stage_calibrate(cfg, records)

    # Calibration stays fixed at the clean thresholds: that is the deployment
    # scenario. Recalibrating under corruption would hide the failure we measure.
    test = P.stage_uncertainty_clip(cfg, test, per_backbone, None)
    au = bootstrap_metric(test["label"].to_numpy(), test["anomaly_score"].to_numpy(),
                          safe_auroc, int(cfg.eval.n_boot), seed=int(cfg.seed))
    row = {"corruption": corruption, "severity": severity, "n": len(test),
           "auroc": au.value, "auroc_ci_lo": au.lo, "auroc_ci_hi": au.hi,
           "accuracy": float(test["correct"].mean()),
           "coverage_on_normals": float(coverage.iloc[-1]["empirical_coverage"])}
    for col in uncertainty_columns(test):
        row[f"mean_{col}"] = float(np.nanmean(test[col].to_numpy(dtype=float)))
        rc = risk_coverage_curve(test["correct"].to_numpy(), test[col].to_numpy(dtype=float))
        row[f"aurc_{col}"] = rc.aurc
    # Prediction-set behaviour gets its own columns: the size-0 rate is the
    # undecidable-gap indicator, and it is the one expected to climb with severity
    # while a scalar confidence stays flat.
    if "set_size" in test:
        row["frac_set_size0"] = float((test.set_size == 0).mean())
        row["frac_set_size1"] = float((test.set_size == 1).mean())
        row["frac_set_size2"] = float((test.set_size == 2).mean())
        row["mean_set_size"] = float(test.set_size.mean())
    return row


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args(argv)

    cfg = load_config(args.config, args.overrides)
    run_dir = P.run_dir_for(cfg)
    log = setup_logging(run_dir)
    start_run(run_dir, to_dict(cfg), int(cfg.seed))

    index = P.stage_index(cfg)
    embedders = P.stage_embed(cfg, index)

    rows = [evaluate_setting(cfg, index, embedders, "none", 0)]
    log.info("clean AUROC %.4f", rows[0]["auroc"])
    for corruption in cfg.corruption.corruptions:
        for severity in cfg.corruption.severities:
            row = evaluate_setting(cfg, index, embedders, str(corruption), int(severity))
            rows.append(row)
            log.info("%-18s s=%d  AUROC %.4f  acc %.3f", corruption, severity,
                     row["auroc"], row["accuracy"])

    sweep = pd.DataFrame(rows)
    report_dir = run_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    sweep.to_csv(report_dir / "corruption_sweep.csv", index=False)

    # Does uncertainty rise monotonically with severity? The trustworthy question.
    signal_cols = [c.removeprefix("mean_") for c in sweep.columns if c.startswith("mean_u_")]
    # The size-0 rate is tracked as its own row: no scalar function of a single
    # one-sided p-value can express "both hypotheses rejected".
    extra_cols = [c for c in ("frac_set_size0", "mean_set_size") if c in sweep.columns]
    mono_rows = []
    for corruption, g in sweep[sweep.corruption != "none"].groupby("corruption"):
        base = sweep[sweep.corruption == "none"].iloc[0]
        for s in signal_cols + extra_cols:
            key = f"mean_{s}" if s in signal_cols else s
            sev = np.concatenate([[0], g["severity"].to_numpy(dtype=float)])
            val = np.concatenate([[base[key]], g[key].to_numpy(dtype=float)])
            auroc = np.concatenate([[base["auroc"]], g["auroc"].to_numpy(dtype=float)])
            mono_rows.append({
                "corruption": corruption, "signal": s,
                "spearman_severity_vs_uncertainty": _spearman(sev, val),
                "spearman_severity_vs_auroc": _spearman(sev, auroc),
                "delta_auroc_at_s5": float(g["auroc"].iloc[-1] - base["auroc"]),
                "delta_uncertainty_at_s5": float(g[key].iloc[-1] - base[key]),
            })
    mono = pd.DataFrame(mono_rows)
    mono["silent_failure"] = (mono["spearman_severity_vs_uncertainty"] < 0.5) & \
                             (mono["delta_auroc_at_s5"] < -0.02)
    mono.to_csv(report_dir / "corruption_monotonicity.csv", index=False)

    plot_corruption_sweep(sweep[sweep.corruption != "none"], run_dir / "figures" / "corruption_sweep",
                          signal_cols)
    n_silent = int(mono["silent_failure"].sum())
    log.info("silent-failure cells: %d/%d (uncertainty flat while AUROC drops)", n_silent, len(mono))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
