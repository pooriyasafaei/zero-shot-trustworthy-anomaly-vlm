#!/usr/bin/env python3
"""Phase 2 VLM pipeline: Qwen2.5-VL scoring, then the same offline evaluation stack.

The image subset comes from the same index builder as the CLIP run, so a DeLong
comparison between the two branches is legitimate.

Usage::

    python scripts/run_qwen.py --config configs/qwen_full.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from tzsad import pipeline as P
from tzsad.eval.report import build_report
from tzsad.records import uncertainty_columns
from tzsad.scorers.base import ScoringContext
from tzsad.scorers.qwen_scorer import QwenScorer, QwenScorerConfig
from tzsad.uncertainty.semantic_entropy import ClusteringConfig
from tzsad.uncertainty.vlm_side import (AbstentionFlag, PromptPerturbationConsistency,
                                        SemanticEntropy, TokenEntropy, VerbalizedConfidence,
                                        VerdictVariance)
from tzsad.utils.config import load_config, to_dict
from tzsad.utils.logging import setup_logging
from tzsad.utils.repro import start_run
from tzsad.viz.plots import plot_error_prediction, plot_reliability, plot_risk_coverage


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--skip-scoring", action="store_true",
                    help="reuse records_qwen.parquet and only redo the offline stages")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args(argv)

    cfg = load_config(args.config, args.overrides)
    run_dir = P.run_dir_for(cfg)
    log = setup_logging(run_dir)
    start_run(run_dir, to_dict(cfg), int(cfg.seed))

    index = P.stage_index(cfg)
    gen_dir = run_dir / "generations"
    records_path = run_dir / "records_qwen.parquet"

    if args.skip_scoring and records_path.exists():
        records = P.load_stage(records_path)
        log.info("reusing %d records from %s", len(records), records_path)
    else:
        qcfg = QwenScorerConfig(
            model_id=str(cfg.qwen.model_id), mode=str(cfg.qwen.mode),
            n_samples=int(cfg.qwen.n_samples), temperature=float(cfg.qwen.temperature),
            top_p=float(cfg.qwen.top_p), max_new_tokens=int(cfg.qwen.max_new_tokens),
            dtype=str(cfg.qwen.dtype), max_pixels=int(cfg.qwen.max_pixels),
            prompt_variants=tuple(cfg.qwen.prompt_variants), generations_dir=str(gen_dir),
        )
        scorer = QwenScorer(qcfg)
        ctx = ScoringContext(run_id=str(cfg.paths.run_name),
                             subset_tag=str(cfg.data.subset).replace("=", ""))
        records = scorer.score_index(index, ctx)
        P.save_stage(records, records_path)
        log.info("scored %d images; abstention rate %.3f",
                 len(records), 1.0 - records["parse_ok"].mean())

    test, calib, coverage = P.stage_calibrate(cfg, records)
    coverage.to_csv(run_dir / "coverage.csv", index=False)

    wanted = set(cfg.uncertainty.estimators)
    if "token_entropy" in wanted:
        test = TokenEntropy().attach(test)
    if "verdict_var" in wanted:
        test = VerdictVariance().attach(test)
    if "verbalized_conf" in wanted:
        test = VerbalizedConfidence(gen_dir).attach(test)
    if "abstention" in wanted:
        test = AbstentionFlag().attach(test)
    if "prompt_perturb" in wanted and any(c.startswith("variant_pyes__") for c in test.columns):
        test = PromptPerturbationConsistency().attach(test)
    if "semantic_entropy" in wanted:
        se_cfg = ClusteringConfig(backend=str(cfg.uncertainty.semantic_entropy.backend),
                                  embed_threshold=float(cfg.uncertainty.semantic_entropy.embed_threshold),
                                  entail_threshold=float(cfg.uncertainty.semantic_entropy.entail_threshold))
        test = SemanticEntropy(gen_dir, se_cfg).attach(test)

    P.save_stage(test, run_dir / "records_test_scored.parquet")
    report_dir = run_dir / "report"
    paths = build_report(test, report_dir, "correct", int(cfg.eval.n_boot), int(cfg.seed))

    fig_dir = run_dir / "figures"
    signals = uncertainty_columns(test)
    if signals:
        plot_risk_coverage(test, "correct", signals, fig_dir / "risk_coverage", "Risk-coverage (Qwen)")
        plot_error_prediction(pd.read_csv(paths["error_prediction"]), fig_dir / "error_prediction")
    plot_reliability(test, fig_dir / "reliability", title="Reliability (Qwen)")

    det = pd.read_csv(paths["detection"])
    pooled = det[det.group == "POOLED"].iloc[0]
    log.info("POOLED AUROC %.4f [%.4f, %.4f]", pooled.auroc, pooled.auroc_ci_lo, pooled.auroc_ci_hi)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
