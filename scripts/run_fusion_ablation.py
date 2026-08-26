#!/usr/bin/env python3
"""Phase 5: trust-score fusion and the leave-one-out ablation that justifies it.

§4.5's rule is enforced here: every factor must earn its place by improving AURC
when present. Factors with a non-positive delta are printed as unjustified.

Usage::

    python scripts/run_fusion_ablation.py --config configs/qwen_full.yaml \
        --records results/qwen_full/records_hallucination.parquet
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from tzsad import pipeline as P
from tzsad.eval.selective import risk_coverage_curve
from tzsad.fusion.trust import (COMBINERS, FusionConfig, LogisticTrustCombiner,
                                leave_one_out_ablation)
from tzsad.records import uncertainty_columns
from tzsad.utils.config import load_config, to_dict
from tzsad.utils.logging import setup_logging
from tzsad.utils.repro import start_run


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--records", required=True)
    ap.add_argument("--uncertainty-cols", nargs="*", default=None)
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args(argv)

    cfg = load_config(args.config, args.overrides)
    run_dir = P.run_dir_for(cfg)
    log = setup_logging(run_dir)
    start_run(run_dir, to_dict(cfg), int(cfg.seed))

    records = pd.read_parquet(args.records)
    u_cols = tuple(args.uncertainty_cols or [c for c in uncertainty_columns(records)])
    if not u_cols:
        raise ValueError("no uncertainty columns to fuse")
    fcfg = FusionConfig(uncertainty_cols=u_cols, weights=dict(cfg.fusion.weights))
    log.info("fusing factors: %s", list(u_cols) + [fcfg.cmcs_col, fcfg.halluc_col])

    report_dir = run_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    correct = records["correct"].to_numpy()
    abstain = (~records["parse_ok"].to_numpy(dtype=bool)) if "parse_ok" in records else None

    rows = []
    for name, fn in COMBINERS.items():
        trust = fn(records, fcfg)
        rc = risk_coverage_curve(correct, 1.0 - trust, abstain)
        records[f"trust_{name}"] = trust
        rows.append({"combiner": name, "aurc": rc.aurc, "eaurc": rc.eaurc,
                     "acc@cov0.8": 1.0 - rc.risk_at_coverage(0.8),
                     "cov@risk0.05": rc.coverage_at_risk(0.05), "uses_labels": False})

    # Logistic combiner: fitted on normal-only + synthetic anomalies where available,
    # otherwise reported as unavailable rather than silently fitted on test labels.
    synth_path = Path(cfg.paths.results) / str(cfg.paths.run_name) / "records_synthetic.parquet"
    if synth_path.exists():
        synth = pd.read_parquet(synth_path)
        combiner = LogisticTrustCombiner(fcfg, seed=int(cfg.seed)).fit(
            synth, synth["synthetic_anomaly"].to_numpy())
        trust = combiner.predict_trust(records)
        records["trust_logistic"] = trust
        rc = risk_coverage_curve(correct, 1.0 - trust, abstain)
        rows.append({"combiner": "logistic(synthetic)", "aurc": rc.aurc, "eaurc": rc.eaurc,
                     "acc@cov0.8": 1.0 - rc.risk_at_coverage(0.8),
                     "cov@risk0.05": rc.coverage_at_risk(0.05), "uses_labels": False})
        pd.DataFrame([combiner.coefficients]).to_csv(report_dir / "fusion_logistic_coef.csv", index=False)
    else:
        log.warning("no synthetic-anomaly records at %s; skipping the logistic combiner", synth_path)

    # Single-signal references, so fusion has to beat its own ingredients.
    for col in u_cols:
        rc = risk_coverage_curve(correct, records[col].to_numpy(dtype=float), abstain)
        rows.append({"combiner": f"single:{col}", "aurc": rc.aurc, "eaurc": rc.eaurc,
                     "acc@cov0.8": 1.0 - rc.risk_at_coverage(0.8),
                     "cov@risk0.05": rc.coverage_at_risk(0.05), "uses_labels": False})

    comparison = pd.DataFrame(rows).sort_values("aurc").reset_index(drop=True)
    comparison.to_csv(report_dir / "fusion_comparison.csv", index=False)

    ablation = leave_one_out_ablation(records, fcfg, "correct", str(cfg.fusion.combiner))
    ablation.to_csv(report_dir / "fusion_ablation.csv", index=False)
    records.to_parquet(run_dir / "records_fused.parquet", index=False)

    unjustified = ablation[(ablation["dropped"] != "(none)") & (~ablation["justified"])]
    log.info("best combiner: %s (AURC %.4f)", comparison.iloc[0]["combiner"], comparison.iloc[0]["aurc"])
    if len(unjustified):
        log.warning("factors NOT justified by the ablation (drop them): %s",
                    list(unjustified["dropped"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
