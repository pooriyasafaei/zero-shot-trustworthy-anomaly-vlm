#!/usr/bin/env python3
"""Cross-dataset shift: calibrate conformal on MVTec, test on VisA/BTAD.

The point is to make coverage *break* on purpose. If empirical coverage on the
target dataset's normal images falls well below the nominal 1-δ, exchangeability
has failed, which is the motivation for weighted conformal under covariate shift.
A method that keeps its nominal coverage here would be the surprising result.

Usage::

    python scripts/run_cross_dataset.py --config configs/clip_full.yaml \
        --target-root /path/to/visa --target-name visa
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from tzsad import pipeline as P
from tzsad.calibration.conformal import MondrianConformal, coverage_report
from tzsad.data.mvtec import SubsetSpec, build_index
from tzsad.eval.report import detection_table
from tzsad.features.cache import EmbeddingCache
from tzsad.scorers.base import ScoringContext
from tzsad.scorers.clip_scorer import ClipScorer, ClipScorerConfig
from tzsad.utils.config import load_config, to_dict
from tzsad.utils.logging import setup_logging
from tzsad.utils.repro import start_run


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--target-root", required=True, help="root of the shifted dataset")
    ap.add_argument("--target-name", default="target")
    ap.add_argument("--pool-calibration", action="store_true",
                    help="pool all source categories into one calibration set, for target "
                         "categories with no source counterpart")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args(argv)

    cfg = load_config(args.config, args.overrides)
    run_dir = P.run_dir_for(cfg) / f"cross_{args.target_name}"
    log = setup_logging(run_dir)
    start_run(run_dir, to_dict(cfg), int(cfg.seed))

    # --- source: calibrate on MVTec train/good -------------------------------
    src_index = P.stage_index(cfg)
    embedders = P.stage_embed(cfg, src_index)
    src_records = P.stage_score_clip(cfg, src_index, embedders)[P.backbones_of(cfg)[0].cache_tag]
    src_cal = src_records[(src_records.split == "train") & (src_records.label == 0)]

    # --- target: embed and score with the SAME frozen backbone and prompts ----
    tgt_index = build_index(args.target_root, None, ("train", "test"),
                            SubsetSpec.parse(str(cfg.data.subset), seed=int(cfg.seed)))
    tgt_cfg = OmegaConf.merge(cfg, OmegaConf.create({"data": {"root": args.target_root}}))
    P.stage_embed(tgt_cfg, tgt_index)
    spec = P.backbones_of(cfg)[0]
    cache = EmbeddingCache(cfg.paths.cache)
    scorer = ClipScorer(spec, cache, ClipScorerConfig(prompt_set=str(cfg.clip.prompt_set),
                                                     tau=float(cfg.clip.tau),
                                                     aggregate=str(cfg.clip.aggregate)))
    scorer.ensure_text(sorted(tgt_index["category"].unique()), embedders[spec.cache_tag])
    tgt_records = scorer.score_index(tgt_index, ScoringContext(run_id=f"cross_{args.target_name}"))
    tgt_test = tgt_records[tgt_records.split == "test"].reset_index(drop=True)

    if args.pool_calibration:
        src_cal = src_cal.assign(category="POOL")
        tgt_test = tgt_test.assign(_orig_category=tgt_test["category"], category="POOL")

    report_dir = run_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for delta in cfg.conformal.deltas:
        calib = MondrianConformal(delta=float(delta), seed=int(cfg.seed)).fit(src_cal)
        shared = [c for c in tgt_test["category"].unique() if str(c) in calib.cal_scores_]
        if not shared:
            log.error("no target category shares a name with a source category; "
                      "re-run with --pool-calibration")
            return 1
        sub = tgt_test[tgt_test["category"].isin(shared)].reset_index(drop=True)

        src_cov = coverage_report(src_records[src_records.split == "test"], calib,
                                  n_boot=int(cfg.eval.n_boot))
        tgt_cov = coverage_report(sub, calib, n_boot=int(cfg.eval.n_boot))
        s_p = src_cov.iloc[-1]
        t_p = tgt_cov.iloc[-1]
        rows.append({"delta": float(delta), "nominal": 1 - float(delta),
                     "source_coverage": s_p["empirical_coverage"],
                     "source_ci_lo": s_p["ci_lo"], "source_ci_hi": s_p["ci_hi"],
                     "target_coverage": t_p["empirical_coverage"],
                     "target_ci_lo": t_p["ci_lo"], "target_ci_hi": t_p["ci_hi"],
                     "coverage_gap": float(s_p["empirical_coverage"] - t_p["empirical_coverage"]),
                     "target_undercovers": bool(t_p["ci_hi"] < 1 - float(delta))})
        log.info("delta=%.2f  source cov %.3f  target cov %.3f  %s", float(delta),
                 s_p["empirical_coverage"], t_p["empirical_coverage"],
                 "UNDER-COVERS" if rows[-1]["target_undercovers"] else "holds")

    pd.DataFrame(rows).to_csv(report_dir / "cross_dataset_coverage.csv", index=False)
    detection_table(tgt_test, int(cfg.eval.n_boot), int(cfg.seed)).to_csv(
        report_dir / "cross_dataset_detection.csv", index=False)
    src_pooled = detection_table(src_records[src_records.split == "test"],
                                 int(cfg.eval.n_boot), int(cfg.seed))
    delta_auroc = float(
        detection_table(tgt_test, 200, int(cfg.seed)).query("group == 'POOLED'").auroc.iloc[0]
        - src_pooled.query("group == 'POOLED'").auroc.iloc[0])
    log.info("delta AUROC (target - source) = %+.4f", delta_auroc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
