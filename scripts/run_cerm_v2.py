#!/usr/bin/env python3
"""CERM-AD Stage I+: bidirectional, artifact-matched causal verification.

    python scripts/run_cerm_v2.py --config configs/cerm_v2_pilot.yaml --workers auto
    python scripts/run_cerm_v2.py --config configs/cerm_v2_pilot.yaml --phase report

Phases (``--phase all`` runs them in order):

``direct``      every intervention arm, scored with the direct-verdict probe (P0)
``structured``  structured-generation confirmation: all originals (P0) and the
                casebook targets under six roles (P1)
``report``      tables, figures, casebook, PRESENTATION_RESULTS.md, bundle zip

Re-running the same command resumes every phase from its checkpoints. With
``--workers N`` one pinned subprocess per GPU takes a balanced shard of the work.
Stage I (``scripts/run_cerm.py``) is untouched and stays reproducible.
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

from tzsad.causal.audit import ScoreCache, select_targets
from tzsad.causal.audit_v2 import V2Config, run_audit_v2
from tzsad.data.mvtec import SubsetSpec, build_index
from tzsad.utils.budget import TimeBudget
from tzsad.utils.config import load_config, to_dict
from tzsad.utils.logging import setup_logging
from tzsad.utils.repro import start_run

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_cerm import causal_config, resolve_workers  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def v2_config(cfg) -> V2Config:
    v = cfg.cerm_v2
    return V2Config(
        base=causal_config(cfg), k_random=int(v.k_random), k_context=int(v.k_context),
        k_artifact=int(v.k_artifact), artifact_max_candidates=int(v.artifact_max_candidates),
        artifact_match_quantile=float(v.artifact_match_quantile),
        k_occlusion=int(v.k_occlusion), k_sham=int(v.k_sham),
        max_sham_sources=int(v.max_sham_sources),
        sham_match_quantile=float(v.sham_match_quantile), batch_size=int(v.batch_size),
        max_pixels=int(cfg.qwen.max_pixels), dtype=str(cfg.qwen.dtype))


def load_scorer(cfg, log):
    """Load Qwen once, retrying: two workers reading one HF cache can race."""
    from tzsad.scorers.qwen_scorer import QwenScorer, QwenScorerConfig

    q = cfg.qwen
    scorer = QwenScorer(QwenScorerConfig(
        model_id=str(q.model_id), dtype=str(q.dtype), max_pixels=int(q.max_pixels),
        max_new_tokens=int(q.max_new_tokens)))
    for attempt in range(4):
        try:
            scorer.load()
            return scorer
        except Exception:                                  # noqa: BLE001
            log.exception("model load attempt %d failed; retrying", attempt + 1)
            time.sleep(10 + 10 * attempt)
    scorer.load()
    return scorer


def batch_consistency_check(scorer, index, targets, run_dir: Path, shard: int, log) -> None:
    """Batched vs one-at-a-time logits on four same-size images; logged, not assumed."""
    import numpy as np
    from PIL import Image

    from tzsad.scorers.prompts import class_name_for

    row = targets.iloc[0]
    img = Image.open(row["path"]).convert("RGB")
    imgs = [img, img.transpose(Image.FLIP_LEFT_RIGHT), img.transpose(Image.FLIP_TOP_BOTTOM),
            img.transpose(Image.ROTATE_180)]
    name = class_name_for(str(row["category"]))
    single = np.array([scorer.score_direct_verdict(im, name)["audit_logit"] for im in imgs])
    batch = np.array([r["audit_logit"] for r in scorer.score_direct_verdict_batch(imgs, name)])
    diff = float(np.abs(single - batch).max())
    out = {"image_id": str(row["image_id"]), "single": single.tolist(),
           "batched": batch.tolist(), "max_abs_logit_diff": diff}
    (run_dir / f"batch_check.shard{shard}.json").write_text(json.dumps(out, indent=1))
    log.info("batch consistency: max |logit(single) - logit(batched)| = %.5f", diff)
    if diff > 0.05:
        log.warning("batched scoring differs from single-image scoring by %.4f logits", diff)


def launch(args, phase: str, n_workers: int, run_dir: Path) -> int:
    """One pinned worker per GPU; returns the number of failed workers."""
    procs = []
    for shard in range(n_workers):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(shard))
        cmd = [sys.executable, str(Path(__file__).resolve()), "--config", args.config,
               "--phase", phase, "--shard", str(shard), "--num-shards", str(n_workers),
               "--worker", *(["--budget-minutes", str(args.budget_minutes)]
                             if args.budget_minutes else []), *args.overrides]
        log_path = run_dir / f"worker_{phase}{shard}.log"
        handle = log_path.open("a")
        procs.append((shard, subprocess.Popen(cmd, cwd=REPO, env=env, stdout=handle,
                                              stderr=subprocess.STDOUT), handle))
        print(f"[cerm-v2] {phase} worker {shard} -> GPU {shard}, log {log_path}", flush=True)
        if shard == 0 and n_workers > 1:
            time.sleep(20)     # stagger model loads so the workers do not race the HF cache
    failures = 0
    for shard, proc, handle in procs:
        rc = proc.wait()
        handle.close()
        print(f"[cerm-v2] {phase} worker {shard} exited with {rc}", flush=True)
        failures += int(rc != 0)
    return failures


def structured_items(run_dir: Path, targets, cfg, log):
    """P0 items (every original) plus P1 items (casebook targets x six roles)."""
    import pandas as pd

    from tzsad.causal.audit import merge_shards
    from tzsad.causal.casebook import p1_items, select_cases
    from tzsad.causal.structured import p0_items

    parts = [p0_items(targets)] if bool(cfg.cerm_v2.get("structured_p0", True)) else []
    records = merge_shards(run_dir, "causal_records_v2")
    if len(records):
        ok = records[records["status"] == "ok"]
        cases = select_cases(ok, int(cfg.cerm_v2.get("structured_p1_cases", 10)))
        cases.to_csv(run_dir / "casebook_manifest.csv", index=False)
        parts.append(p1_items(ok, cases))
        log.info("structured P1: %d casebook targets", len(cases))
    items = pd.concat(parts, ignore_index=True).drop_duplicates("item_key")
    items.to_csv(run_dir / "structured_items.csv", index=False)
    return items


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--workers", default=None, help="GPU workers; 'auto' = every visible GPU")
    ap.add_argument("--phase", default="all", choices=("all", "direct", "structured", "report"))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--worker", action="store_true", help="internal: run one shard of a phase")
    ap.add_argument("--budget-minutes", type=float, default=None)
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args(argv)

    cfg = load_config(args.config, list(args.overrides))
    run_dir = Path(cfg.paths.results) / str(cfg.paths.run_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    log = setup_logging(run_dir)
    v2 = v2_config(cfg)
    started = time.monotonic()

    subset = SubsetSpec.parse(str(cfg.data.subset), seed=int(cfg.seed))
    cal_subset = SubsetSpec.parse(str(cfg.data.get("cal_subset", "full")), seed=int(cfg.seed))
    categories = list(cfg.data.categories) if cfg.data.categories else None
    index = build_index(cfg.data.root, categories, tuple(cfg.data.splits), subset, cal_subset)
    targets = select_targets(index, v2.base)
    budget = TimeBudget(args.budget_minutes * 60) if args.budget_minutes else None

    # ---- worker mode: one shard of one phase, then exit ---------------------
    if args.worker:
        scorer = load_scorer(cfg, log)
        if args.phase == "direct":
            if v2.batch_size > 1:
                batch_consistency_check(scorer, index, targets, run_dir, args.shard, log)
            probe = (scorer.score_direct_verdict_batch if v2.base.score_mode == "direct_verdict"
                     else _single_as_batch(scorer.score_generated_verdict))
            run_audit_v2(index, probe, v2, run_dir, args.shard, args.num_shards, budget)
        elif args.phase == "structured":
            import pandas as pd

            from tzsad.causal.structured import run_structured, structured_signature
            q = cfg.qwen
            cache = ScoreCache(run_dir / f"structured_cache.shard{args.shard}.jsonl",
                               structured_signature(str(q.model_id), int(q.max_pixels),
                                                    str(q.dtype), int(q.max_new_tokens)),
                               "generated_verdict")
            run_structured(pd.read_csv(run_dir / "structured_items.csv"),
                           scorer.score_generated_verdict, cache,
                           run_dir / f"structured_records.shard{args.shard}.jsonl",
                           args.shard, args.num_shards)
        return 0

    # ---- coordinator ---------------------------------------------------------
    if not (run_dir / "manifest.json").exists() or args.phase in ("all", "direct"):
        start_run(run_dir, to_dict(cfg), int(cfg.seed))
    n_per_target = (2 + v2.k_random + v2.k_context + v2.k_artifact + 1 + v2.k_occlusion
                    + 2 + v2.k_sham)
    log.info("CERM-AD Stage I+ | %d targets | up to %d images/target (%d before dedup)",
             len(targets), n_per_target, len(targets) * n_per_target)
    n_workers = resolve_workers(args.workers if args.workers is not None
                                else cfg.causal.get("workers", 1))
    timings = {}
    for phase in ("direct", "structured"):
        if args.phase not in ("all", phase):
            continue
        t = time.monotonic()
        if phase == "structured":
            items = structured_items(run_dir, targets, cfg, log)
            log.info("structured phase: %d items", len(items))
        failures = launch(args, phase, n_workers, run_dir)
        timings[f"{phase}_minutes"] = round((time.monotonic() - t) / 60, 2)
        if failures:
            log.error("%d %s worker(s) failed; see worker_%s*.log", failures, phase, phase)
            if phase == "direct":
                return 1
    if timings:
        prev = {}
        tf = run_dir / "phase_timings.json"
        if tf.exists():
            prev = json.loads(tf.read_text())
        prev.update(timings)
        tf.write_text(json.dumps(prev, indent=1))

    if args.phase in ("all", "report"):
        from tzsad.causal.report_v2 import build_report
        build_report(run_dir, cfg, v2, targets, REPO)
    log.info("done in %.1f min; results in %s", (time.monotonic() - started) / 60, run_dir)
    return 0


def _single_as_batch(fn):
    def _batch(images, class_name):
        return [fn(im, class_name) for im in images]
    return _batch


if __name__ == "__main__":
    raise SystemExit(main())
