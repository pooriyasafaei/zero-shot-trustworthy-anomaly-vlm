#!/usr/bin/env python3
"""Phase 2: the uncertainty bake-off.

Scores every uncertainty estimator and every conformal p-value mode head-to-head
on identical images, and reports error-prediction AUROC with bootstrap CIs.

Three controls, without which this table reports artifacts as results:

1. **Score-monotone baselines.** When one error type dominates, "is this an
   error" collapses into "is this a low-scoring anomaly", so anything monotone in
   the score scores well for free. ``BASELINE:score`` / ``BASELINE:1-score``
   measure that floor and every signal is scored against it.
2. **Decision conditioning.** ``pred=0`` / ``pred=1`` scopes repeat the test where
   the errors are homogeneous, removing the base-rate shortcut entirely.
3. **Selection/report split.** The winner is chosen on one half and quoted from
   the other, so the best of fifteen candidates is not reported at its selection
   optimum.

Everything runs offline from cached records: no GPU forward pass.

Usage::

    python scripts/run_bakeoff.py --config configs/clip_full.yaml \
        --records results/clip_full/records_clip.parquet
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from tzsad import pipeline as P
from tzsad.eval.report import error_prediction_table
from tzsad.eval.selective import selective_table
from tzsad.records import uncertainty_columns
from tzsad.utils.config import load_config, to_dict
from tzsad.utils.logging import setup_logging
from tzsad.utils.repro import start_run
from tzsad.viz.plots import plot_error_prediction, plot_risk_coverage


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--records", required=True, help="scored records (train + test splits)")
    ap.add_argument("--synthetic", default=None,
                    help="synthetic-anomaly records, to enable conformal prediction sets")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args(argv)

    cfg = load_config(args.config, args.overrides)
    run_dir = P.run_dir_for(cfg)
    log = setup_logging(run_dir)
    start_run(run_dir, to_dict(cfg), int(cfg.seed))

    records = P.load_stage(Path(args.records))
    index = P.stage_index(cfg)

    # -- calibrate, then attach every uncertainty channel --------------------
    test, calib, coverage = P.stage_calibrate(cfg, records)
    coverage.to_csv(run_dir / "coverage.csv", index=False)
    test = P.stage_uncertainty_modes(cfg, test)
    test = P.stage_uncertainty_clip(cfg, test, {P.backbones_of(cfg)[0].cache_tag: records}, None)

    if args.synthetic:
        synth = P.load_stage(Path(args.synthetic))
        normal_cal = records[(records.split == "train") & (records.label == 0)]
        test, size_rep = P.stage_prediction_sets(cfg, test, normal_cal, synth)
        size_rep.to_csv(run_dir / "report" / "set_size.csv", index=False)
        pooled = size_rep.iloc[-1]
        log.info("prediction sets: size0 %.3f  size1 %.3f  size2 %.3f  singleton acc %.3f",
                 pooled.frac_size0, pooled.frac_size1, pooled.frac_size2, pooled.singleton_accuracy)
    else:
        log.warning("no --synthetic pool given; conformal prediction sets are skipped")

    test = P.assign_bakeoff_split(test, seed=int(cfg.seed))
    P.save_stage(test, run_dir / "records_bakeoff.parquet")

    signals = uncertainty_columns(test)
    log.info("bake-off over %d signals on %d test images: %s", len(signals), len(test),
             ", ".join(s.removeprefix("u_") for s in signals))

    report_dir = run_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    n_boot = int(cfg.eval.n_boot)

    frames = []
    for half in ("select", "report", "all"):
        sub = test if half == "all" else test[test.bakeoff_half == half]
        sub = sub.reset_index(drop=True)
        tbl = error_prediction_table(sub, "correct", n_boot, int(cfg.seed))
        frames.append(tbl.assign(half=half))
    full = pd.concat(frames, ignore_index=True)
    full.to_csv(report_dir / "bakeoff_error_prediction.csv", index=False)

    sel = selective_table(test, "correct", signals).assign(half="all")
    sel.to_csv(report_dir / "bakeoff_selective.csv", index=False)

    # -- the honest headline: pick on select, quote from report ---------------
    pooled = full[(full.scope == "pred=0") & (~full.signal.str.startswith("BASELINE:"))]
    picks = pooled[pooled.half == "select"].sort_values("err_auroc", ascending=False)
    if len(picks):
        winner = picks.iloc[0]["signal"]
        held = pooled[(pooled.half == "report") & (pooled.signal == winner)].iloc[0]
        log.info("selected on 'select' half: %s (%.4f)", winner, picks.iloc[0]["err_auroc"])
        log.info("held-out 'report' half:    %s = %.4f [%.4f, %.4f]",
                 winner, held.err_auroc, held.ci_lo, held.ci_hi)
        pd.DataFrame([{"selected_signal": winner,
                       "select_auroc": float(picks.iloc[0]["err_auroc"]),
                       "report_auroc": float(held.err_auroc),
                       "report_ci_lo": float(held.ci_lo), "report_ci_hi": float(held.ci_hi),
                       "selection_optimism": float(picks.iloc[0]["err_auroc"] - held.err_auroc)}]
                     ).to_csv(report_dir / "bakeoff_selected.csv", index=False)

    fig_dir = run_dir / "figures"
    all_pooled = full[(full.half == "all") & (full.scope == "pred=0")]
    if len(all_pooled):
        plot_error_prediction(all_pooled.assign(scope="POOLED"), fig_dir / "bakeoff_error_prediction")
    plot_risk_coverage(test, "correct", signals, fig_dir / "bakeoff_risk_coverage",
                       "Risk-coverage: uncertainty bake-off")
    log.info("bake-off written to %s", report_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
