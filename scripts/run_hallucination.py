#!/usr/bin/env python3
"""Phase 4: CMCS and the operational hallucination rate.

Consumes a scored record file plus the cached ``bank`` anomaly maps. Runs both
CMCS arms and the IoU sensitivity sweep.

Usage::

    python scripts/run_hallucination.py --config configs/qwen_full.yaml \
        --records results/qwen_full/records_test_scored.parquet \
        --maps-dir results/clip_full/maps
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from tzsad import pipeline as P
from tzsad.features.heatmap import load_maps
from tzsad.hallucination.cmcs import CMCSConfig, compute_cmcs
from tzsad.hallucination.rate import HRConfig, hallucination_flags, hallucination_rate, iou_sensitivity
from tzsad.utils.config import load_config, to_dict
from tzsad.utils.logging import setup_logging
from tzsad.utils.repro import start_run


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--records", required=True)
    ap.add_argument("--maps-dir", required=True)
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args(argv)

    cfg = load_config(args.config, args.overrides)
    run_dir = P.run_dir_for(cfg)
    log = setup_logging(run_dir)
    start_run(run_dir, to_dict(cfg), int(cfg.seed))

    records = P.load_stage(Path(args.records))
    maps: dict = {}
    for npz in sorted(Path(args.maps_dir).glob("*__bank.npz")):
        maps.update(load_maps(npz))
    if not maps:
        raise FileNotFoundError(f"no *__bank.npz maps under {args.maps_dir}; run the CLIP "
                                "pipeline with clip.windows.enabled=true first")
    log.info("loaded %d anomaly maps", len(maps))

    cm_cfg = CMCSConfig(map_size=int(cfg.clip.windows.map_size),
                        spatial_top_frac=float(cfg.hallucination.top_frac),
                        min_localized_iou=float(cfg.hallucination.iou_threshold))
    report_dir = run_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)

    for arm in ("language_grounding", "global_spatial"):
        scored = compute_cmcs(records, maps, cm_cfg, arm=arm)
        scored.to_parquet(run_dir / f"records_cmcs_{arm}.parquet", index=False)
        summary = scored.groupby("category", as_index=False).agg(
            cmcs_mean=("cmcs", "mean"),
            cmcs_normal=("cmcs", lambda s: s[scored.loc[s.index, "label"] == 0].mean()),
            cmcs_anom=("cmcs", lambda s: s[scored.loc[s.index, "label"] == 1].mean()))
        summary.to_csv(report_dir / f"cmcs_{arm}.csv", index=False)
        log.info("CMCS arm %-18s mean %.4f", arm, scored["cmcs"].mean())
        if arm == cfg.hallucination.arm:
            primary = scored

    hr_cfg = HRConfig(iou_threshold=float(cfg.hallucination.iou_threshold),
                      map_size=int(cfg.clip.windows.map_size),
                      top_frac=float(cfg.hallucination.top_frac))
    flagged = hallucination_flags(primary, maps, hr_cfg)
    flagged.to_parquet(run_dir / "records_hallucination.parquet", index=False)
    hr = hallucination_rate(flagged)
    hr.to_csv(report_dir / "hallucination_rate.csv", index=False)
    sens = iou_sensitivity(primary, maps, tuple(cfg.hallucination.iou_sweep), hr_cfg)
    sens.to_csv(report_dir / "hallucination_iou_sensitivity.csv", index=False)

    pooled = hr.iloc[-1]
    log.info("pooled HR %.4f over %d positive predictions (case a %.3f, case b %.3f)",
             pooled["hr"], int(pooled["n_positive"]), pooled["hr_case_a"], pooled["hr_case_b"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
