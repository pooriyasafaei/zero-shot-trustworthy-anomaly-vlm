#!/usr/bin/env python3
"""Phase 1/2 CLIP pipeline: index -> embed -> score -> conformal -> uncertainty -> report.

Usage::

    python scripts/run_clip.py --config configs/clip_full.yaml
    python scripts/run_clip.py --config configs/smoke.yaml data.root=/path/to/mvtec
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tzsad import pipeline as P
from tzsad.eval.report import build_report
from tzsad.records import uncertainty_columns
from tzsad.utils.config import load_config, to_dict
from tzsad.utils.logging import setup_logging
from tzsad.utils.repro import start_run
from tzsad.viz.plots import (plot_coverage_vs_delta, plot_error_prediction, plot_reliability,
                             plot_risk_coverage)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("overrides", nargs="*", help="dotlist overrides, e.g. data.root=/x seed=7")
    args = ap.parse_args(argv)

    cfg = load_config(args.config, args.overrides)
    run_dir = P.run_dir_for(cfg)
    log = setup_logging(run_dir)
    start_run(run_dir, to_dict(cfg), int(cfg.seed))
    log.info("run dir: %s", run_dir)

    index = P.stage_index(cfg)
    embedders = P.stage_embed(cfg, index)
    per_backbone, text_embeddings = P.stage_score_clip(cfg, index, embedders, return_text=True)
    primary_tag = P.backbones_of(cfg)[0].cache_tag
    records = per_backbone[primary_tag]
    P.save_stage(records, run_dir / "records_clip.parquet")

    test, calib, coverage = P.stage_calibrate(cfg, records)
    coverage.to_csv(run_dir / "coverage.csv", index=False)
    log.info("pooled coverage %.4f (nominal %.2f)",
             coverage.iloc[-1]["empirical_coverage"], 1 - float(cfg.conformal.delta))
    P.stage_conformal_sweeps(cfg, records, run_dir / "report")

    tta = P.stage_tta_scores(cfg, index, embedders) if cfg.uncertainty.tta.enabled else None
    normal_ref = records[(records["split"] == "train") & (records["label"] == 0)]
    test = P.stage_uncertainty_clip(cfg, test, per_backbone, tta, reference=normal_ref)
    P.save_stage(test, run_dir / "records_test_scored.parquet")

    report_dir = run_dir / "report"
    paths = build_report(test, report_dir, "correct", int(cfg.eval.n_boot), int(cfg.seed))

    if cfg.clip.windows.enabled:
        maps = P.stage_heatmaps(cfg, index, embedders, text_embeddings=text_embeddings)
        pixel = P.stage_pixel_eval(cfg, index, maps)
        pixel.to_csv(report_dir / "pixel.csv", index=False)
        for kind in sorted(pixel["map"].unique()):
            row = pixel[(pixel["map"] == kind) & (pixel.category == "POOLED")].iloc[0]
            log.info("pixel %-5s AUROC %.4f  PRO %.4f", kind, row.pixel_auroc, row.pro)

    if not args.no_figures:
        signals = uncertainty_columns(test)
        fig_dir = run_dir / "figures"
        plot_risk_coverage(test, "correct", signals, fig_dir / "risk_coverage")
        plot_reliability(test, fig_dir / "reliability")
        import pandas as pd

        plot_coverage_vs_delta(pd.read_csv(report_dir / "coverage_sweep.csv"),
                               fig_dir / "conformal_coverage")
        plot_error_prediction(pd.read_csv(paths["error_prediction"]), fig_dir / "error_prediction")
        log.info("figures written to %s", fig_dir)

    det = __import__("pandas").read_csv(paths["detection"])
    pooled = det[det.group == "POOLED"].iloc[0]
    log.info("POOLED AUROC %.4f [%.4f, %.4f] over %d images",
             pooled.auroc, pooled.auroc_ci_lo, pooled.auroc_ci_hi, int(pooled.n))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
