#!/usr/bin/env python3
"""Run the whole TZ-SAD pipeline under a wall-clock budget, resuming where it stopped.

Built for a headless batch run (a scheduled Kaggle notebook), where the process is
killed at a hard limit and the only thing that persists is what has been written
to disk. Every phase is skipped if a previous run already completed it, and the
orchestrator stops launching new phases once the budget is spent, leaving time to
write reports.

    python scripts/run_all.py --budget-minutes 500 --phases clip,synthetic,bakeoff,corruption

Phases, in dependency order:

``clip``         embed + score CLIP, conformal calibration, core report
``synthetic``    CutPaste/NSA pool needed by prediction sets and the fusion combiner
``bakeoff``      the uncertainty bake-off over every estimator and p-value mode
``corruption``   the corruption sweep and its monotonicity table
``qwen``         the VLM branch (slow; safe to omit on a short budget)
``hallucination`` CMCS arms and the operational hallucination rate
``fusion``       trust-score combiners and the leave-one-out ablation
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tzsad.utils.budget import TimeBudget
from tzsad.utils.checkpoint import StageLedger
from tzsad.utils.config import load_config
from tzsad.utils.logging import setup_logging

REPO = Path(__file__).resolve().parents[1]

#: phase -> (script, extra args). ``{results}`` is filled from the resolved config.
PHASES: dict[str, tuple[str, list[str]]] = {
    "clip": ("scripts/run_clip.py", ["--config", "configs/clip_full.yaml"]),
    "synthetic": ("scripts/make_synthetic.py", ["--config", "configs/clip_full.yaml",
                                                "clip.windows.enabled=false"]),
    "bakeoff": ("scripts/run_bakeoff.py", [
        "--config", "configs/clip_ensemble.yaml", "--all-backbones",
        "--records", "{results}/clip_full/records_clip.parquet",
        "--synthetic", "{results}/clip_full/records_synthetic.parquet",
        "paths.run_name=bakeoff_full", "clip.windows.enabled=false", "data.subset=full"]),
    "corruption": ("scripts/run_corruption.py", ["--config", "configs/corruption.yaml",
                                                 "clip.windows.enabled=false"]),
    "qwen": ("scripts/run_qwen.py", ["--config", "configs/qwen_subset.yaml"]),
    "hallucination": ("scripts/run_hallucination.py", [
        "--config", "configs/clip_full.yaml",
        "--records", "{results}/clip_full/records_test_scored.parquet",
        "--maps-dir", "{results}/clip_full/maps"]),
    "fusion": ("scripts/run_fusion_ablation.py", [
        "--config", "configs/clip_full.yaml",
        "--records", "{results}/clip_full/records_test_scored.parquet"]),
}

DEFAULT_ORDER = ["clip", "synthetic", "bakeoff", "corruption", "qwen",
                 "hallucination", "fusion"]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--budget-minutes", type=float, default=480.0,
                    help="total wall clock for the whole run")
    ap.add_argument("--reserve-minutes", type=float, default=10.0,
                    help="held back so the last phase can write its outputs")
    ap.add_argument("--phases", default=",".join(DEFAULT_ORDER))
    ap.add_argument("--config", default="configs/clip_full.yaml",
                    help="used only to resolve the results directory")
    ap.add_argument("--force", action="store_true", help="re-run phases already marked done")
    ap.add_argument("overrides", nargs="*", help="dotlist overrides passed to every phase")
    args = ap.parse_args(argv)

    cfg = load_config(args.config, list(args.overrides))
    results = str(cfg.paths.results)
    log = setup_logging(Path(results) / "run_all")
    budget = TimeBudget(args.budget_minutes * 60, args.reserve_minutes * 60)
    ledger = StageLedger(Path(results) / "run_all" / "phases.json")

    wanted = [p.strip() for p in args.phases.split(",") if p.strip()]
    unknown = [p for p in wanted if p not in PHASES]
    if unknown:
        log.error("unknown phases: %s (available: %s)", unknown, list(PHASES))
        return 1

    log.info("TZ-SAD batch run | results=%s | budget=%.0fm | phases=%s",
             results, args.budget_minutes, wanted)
    if ledger.done_keys and not args.force:
        log.info("already complete from a previous run: %s", ledger.done_keys)

    summary = []
    for phase in wanted:
        if ledger.is_done(phase) and not args.force:
            log.info("[%s] skipped, already complete", phase)
            summary.append((phase, "skipped", 0.0))
            continue
        if budget.should_stop():
            log.warning("[%s] not started: %s", phase, budget.report())
            summary.append((phase, "out-of-budget", 0.0))
            continue

        script, extra = PHASES[phase]
        phase_args = [a.format(results=results) for a in extra]
        # Give each phase what is left, so a slow one cannot starve the reserve.
        per_phase = max(budget.usable / 60.0, 1.0)
        if script.endswith(("run_qwen.py", "run_corruption.py")):
            phase_args += ["--budget-minutes", f"{per_phase:.1f}"]
        cmd = [sys.executable, script, *phase_args, *args.overrides]

        log.info("[%s] starting (%.0fm usable) : %s", phase, per_phase, " ".join(cmd[1:]))
        t0 = time.monotonic()
        proc = subprocess.run(cmd, cwd=REPO)
        took = time.monotonic() - t0

        if proc.returncode == 0:
            ledger.mark(phase, {"seconds": round(took, 1)})
            log.info("[%s] done in %.1fm | %s", phase, took / 60, budget.report())
            summary.append((phase, "done", took))
        elif proc.returncode == 2:
            # Ran out of budget partway; progress is checkpointed, not lost.
            log.warning("[%s] partial after %.1fm - re-run to continue", phase, took / 60)
            summary.append((phase, "partial", took))
        else:
            log.error("[%s] FAILED (exit %d) after %.1fm", phase, proc.returncode, took / 60)
            summary.append((phase, f"failed({proc.returncode})", took))

    log.info("=" * 62)
    for phase, state, took in summary:
        log.info("  %-14s %-16s %6.1fm", phase, state, took / 60)
    log.info("  %s", budget.report("total: "))
    failed = [p for p, s, _ in summary if s.startswith("failed")]
    if failed:
        log.error("failed phases: %s", failed)
        return 1
    incomplete = [p for p, s, _ in summary if s in ("partial", "out-of-budget")]
    if incomplete:
        log.warning("incomplete (re-run to continue): %s", incomplete)
        return 2
    log.info("all requested phases complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
