#!/usr/bin/env python3
"""CERM-AD Stage I: causal visual evidence verification.

    python scripts/run_cerm.py --config configs/cerm_pilot.yaml
    python scripts/run_cerm.py --config configs/cerm_pilot.yaml --workers 2
    python scripts/run_cerm.py --config configs/cerm_pilot.yaml --report-only

Re-running the same command resumes: every intervention score and every completed
target image is checkpointed as it is produced.

With ``--workers N`` (or ``causal.workers: auto``) the target images are split into
N deterministic shards and one subprocess is launched per GPU, each pinned with
``CUDA_VISIBLE_DEVICES``. Each worker owns its own cache and checkpoint shard, so
there is no coordination and no interleaved writing; the parent then merges the
shards and builds the report.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from tzsad.causal.audit import CausalConfig, merge_shards, run_audit, select_targets
from tzsad.causal.presentation import probe_agreement, write_presentation_summary
from tzsad.causal.report import (artifact_control_table, build_figures, decision_split_table,
                                 summary_tables)
from tzsad.data.mvtec import SubsetSpec, build_index
from tzsad.utils.budget import TimeBudget
from tzsad.utils.config import load_config, to_dict
from tzsad.utils.logging import setup_logging
from tzsad.utils.repro import git_sha, start_run

REPO = Path(__file__).resolve().parents[1]


def causal_config(cfg) -> CausalConfig:
    """Build the causal config from the yaml, falling back to dataclass defaults."""
    c = cfg.causal
    return CausalConfig(
        k_placebos=int(c.k_placebos), seed=int(cfg.seed),
        mask_source=str(c.mask_source), score_mode=str(c.score_mode),
        feather_px=int(c.feather_px), dilate_for_donor=int(c.dilate_for_donor),
        max_iou=float(c.max_iou), ring_width=int(c.ring_width),
        min_separation=int(c.min_separation), candidate_stride=int(c.candidate_stride),
        match_quantile=float(c.match_quantile), min_pool=int(c.get("min_pool", 0)),
        use_gradient=bool(c.get("use_gradient", True)),
        max_donor_candidates=int(c.max_donor_candidates),
        max_candidates=int(c.get("max_candidates", 600)),
        max_targets_per_folder=int(c.max_targets_per_folder),
        save_images=bool(c.save_images),
        save_placebo_images=int(c.get("save_placebo_images", 2)),
        model_id=str(cfg.qwen.model_id),
    )


def build_probe(cfg, ccfg: CausalConfig):
    """Return the verdict probe callable for the configured scoring mode.

    Imported lazily so that ``--report-only`` and the unit tests never touch torch
    or the model weights.
    """
    from tzsad.scorers.qwen_scorer import QwenScorer, QwenScorerConfig

    scorer = QwenScorer(QwenScorerConfig(
        model_id=ccfg.model_id, dtype=str(cfg.qwen.dtype),
        max_pixels=int(cfg.qwen.max_pixels), max_new_tokens=int(cfg.qwen.max_new_tokens)))
    if ccfg.score_mode == "direct_verdict":
        return scorer.score_direct_verdict
    if ccfg.score_mode == "generated_verdict":
        return scorer.score_generated_verdict
    raise ValueError(f"unknown causal.score_mode {ccfg.score_mode!r}; "
                     "use direct_verdict or generated_verdict")


def resolve_workers(requested) -> int:
    """How many GPU workers to run. ``auto`` uses every visible device."""
    if requested not in (None, "auto"):
        return max(int(requested), 1)
    try:
        import torch

        return max(torch.cuda.device_count(), 1)
    except Exception:                                      # noqa: BLE001
        return 1


def launch_workers(args, n_workers: int, run_dir: Path) -> int:
    """Run one pinned subprocess per GPU and wait for them all."""
    procs = []
    for shard in range(n_workers):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(shard))
        cmd = [sys.executable, str(Path(__file__).resolve()),
               "--config", args.config, "--shard", str(shard),
               "--num-shards", str(n_workers), "--worker",
               *args.overrides]
        log_path = run_dir / f"worker{shard}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handle = log_path.open("w")
        procs.append((shard, subprocess.Popen(cmd, cwd=REPO, env=env,
                                              stdout=handle, stderr=subprocess.STDOUT), handle))
        print(f"[cerm] worker {shard} -> GPU {shard}, log {log_path}")
    failures = 0
    for shard, proc, handle in procs:
        rc = proc.wait()
        handle.close()
        print(f"[cerm] worker {shard} exited with {rc}")
        failures += int(rc != 0)
    return failures


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--workers", default=None,
                    help="GPU workers; 'auto' uses every visible device")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--worker", action="store_true",
                    help="internal: run one shard and skip the report")
    ap.add_argument("--report-only", action="store_true",
                    help="rebuild tables and figures from existing checkpoints, no GPU")
    ap.add_argument("--budget-minutes", type=float, default=None)
    ap.add_argument("--compare-with", default=None,
                    help="existing qwen records parquet/csv, to check probe agreement")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args(argv)

    cfg = load_config(args.config, list(args.overrides))
    run_dir = Path(cfg.paths.results) / str(cfg.paths.run_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    log = setup_logging(run_dir)
    ccfg = causal_config(cfg)
    started = time.monotonic()

    subset = SubsetSpec.parse(str(cfg.data.subset), seed=int(cfg.seed))
    cal_subset = SubsetSpec.parse(str(cfg.data.get("cal_subset", "full")), seed=int(cfg.seed))
    categories = list(cfg.data.categories) if cfg.data.categories else None
    index = build_index(cfg.data.root, categories, tuple(cfg.data.splits), subset, cal_subset)
    index.to_csv(run_dir / "index.csv", index=False)

    if not args.worker:
        start_run(run_dir, to_dict(cfg), int(cfg.seed))
        targets = select_targets(index, ccfg)
        log.info("CERM-AD Stage I | %d target anomalies | K=%d | mode=%s | mask=%s",
                 len(targets), ccfg.k_placebos, ccfg.score_mode, ccfg.mask_source)
        log.info("expected forward passes: %d", len(targets) * (2 + ccfg.k_placebos))

    # ---- scoring -----------------------------------------------------------
    if not args.report_only:
        n_workers = 1 if args.worker else resolve_workers(
            args.workers if args.workers is not None else cfg.causal.get("workers", 1))
        if not args.worker and n_workers > 1:
            log.info("launching %d GPU workers", n_workers)
            failures = launch_workers(args, n_workers, run_dir)
            if failures:
                log.error("%d worker(s) failed; see worker*.log", failures)
                return 1
        else:
            budget = TimeBudget(args.budget_minutes * 60) if args.budget_minutes else None
            probe = build_probe(cfg, ccfg)
            run_audit(index, probe, ccfg, run_dir, args.shard, args.num_shards, budget)
            if args.worker:
                return 0

    # ---- aggregation and report -------------------------------------------
    records = merge_shards(run_dir, "causal_records")
    interventions = merge_shards(run_dir, "interventions", key="_row_key")
    if not len(records):
        log.error("no causal records found in %s; nothing to report", run_dir)
        return 1
    ok = records[records.get("status", "ok") == "ok"] if "status" in records else records
    failed = len(records) - len(ok)
    if failed:
        log.warning("%d target(s) failed and are excluded from the summaries "
                    "(kept in causal_records.parquet with their status)", failed)

    records.to_parquet(run_dir / "causal_records.parquet", index=False)
    if len(interventions):
        interventions.to_parquet(run_dir / "interventions.parquet", index=False)

    overall, by_cat = summary_tables(ok, seed=int(cfg.seed))
    decision = decision_split_table(ok, seed=int(cfg.seed))
    artifacts = (artifact_control_table(interventions, ok) if len(interventions)
                 else pd.DataFrame())
    overall.to_csv(run_dir / "summary_overall.csv", index=False)
    by_cat.to_csv(run_dir / "summary_by_category.csv", index=False)
    decision.to_csv(run_dir / "summary_by_decision.csv", index=False)
    if len(artifacts):
        artifacts.to_csv(run_dir / "artifact_controls.csv", index=False)

    made = build_figures(ok, interventions, run_dir / "figures")
    log.info("figures: %s", ", ".join(made) if made else "none")

    verdict = ""
    compare = args.compare_with or str(cfg.causal.get("compare_with", "") or "")
    if compare:
        table, verdict = probe_agreement(ok, compare)
        if len(table):
            table.to_csv(run_dir / "probe_agreement.csv", index=False)
        log.info("probe agreement: %s", verdict)

    meta = {
        "git_sha": git_sha(REPO), "model_id": ccfg.model_id,
        "score_mode": ccfg.score_mode, "mask_source": ccfg.mask_source,
        "categories": ", ".join(sorted(ok["category"].unique())) if len(ok) else "",
        "k_placebos": ccfg.k_placebos, "seed": ccfg.seed,
        "forward_passes": int(len(interventions)) if len(interventions) else "n/a",
        "runtime_minutes": f"{(time.monotonic() - started) / 60:.1f}",
    }
    write_presentation_summary(run_dir / "PRESENTATION_SUMMARY.md", overall, by_cat,
                               decision, artifacts, meta, verdict)

    row = overall.iloc[0]
    log.info("=" * 66)
    log.info("N=%d | median CEE %+.4f [%+.4f, %+.4f] | CEE>0 in %.0f%% | beats all placebos %.0f%%",
             int(row["n_targets"]), row["cee_median"], row["cee_median_ci_lo"],
             row["cee_median_ci_hi"], 100 * row["frac_cee_positive"],
             100 * row["frac_beats_all_placebos"])
    log.info("results in %s", run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
