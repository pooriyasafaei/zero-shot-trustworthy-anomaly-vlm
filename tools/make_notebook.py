#!/usr/bin/env python3
"""Generate a self-contained Kaggle notebook that runs the TZ-SAD pipeline headless.

    tools/make_notebook.py                    # -> notebooks/tzsad_batch.ipynb
    tools/make_notebook.py --phases clip,bakeoff --budget-minutes 200

The notebook embeds the current working tree as a base64 tarball, so it needs no
credentials and no access to the (private) repository - a scheduled run has no
interactive way to authenticate. Re-run this generator after changing the code.

Persistence between scheduled runs
----------------------------------
A committed Kaggle notebook saves /kaggle/working as that version's output. Attach
the previous run's output as an input dataset and the notebook restores it before
starting, so embeddings, checkpoints and the phase ledger all carry over and the
next run continues rather than restarting.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import tarfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
EXCLUDE = {".git", "notebooks", "sources", "results", "results_remote",
           "__pycache__", ".venv", ".pytest_cache", "tools"}


def build_archive() -> str:
    """Base64 tarball of the code, configs, scripts and tests."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for path in sorted(REPO.rglob("*")):
            rel = path.relative_to(REPO)
            if any(part in EXCLUDE for part in rel.parts):
                continue
            if path.is_file() and not path.name.endswith((".pyc", ".ipynb")):
                tar.add(path, arcname=str(rel))
    return base64.b64encode(buf.getvalue()).decode()


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": text.splitlines(keepends=True)}


def build_notebook(payload: str, phases: str, budget: float, dataset: str,
                   overrides: list[str] | None = None) -> dict:
    cells = [
        md("""# TZ-SAD — batch run

Trustworthy zero-shot anomaly detection on MVTec-AD: uncertainty quantification,
conformal calibration, hallucination detection and the evaluation protocol.

**This notebook is self-contained.** The code is embedded below, so it needs no
repository access. Enable **GPU** and **Internet**, then *Save Version →
Save & Run All (Commit)*, or schedule it.

**To continue a previous run**, add that run's notebook output as an input dataset
(*Add data → Your Work → Notebook Output*). Every expensive stage checkpoints, so
the next run resumes instead of restarting.
"""),
        code(f'''# --- 1. Configuration -------------------------------------------------------
PHASES = "{phases}"           # clip,synthetic,bakeoff,corruption,qwen,hallucination,fusion
BUDGET_MINUTES = {budget}     # Kaggle kills the run at 9h (GPU) / 12h (CPU); stay under it
RESERVE_MINUTES = 12          # held back so the final phase can write its outputs
PREV_OUTPUT = "{dataset}"     # input dataset holding a previous run's /kaggle/working, or ""

# Where run outputs go. Override via TZSAD_RESULTS to keep a test run from writing
# into a directory that already holds full-scale results.
import os
RESULTS = os.environ.get("TZSAD_RESULTS", "/kaggle/working/results")

# Extra dotlist overrides applied to every phase, e.g.
# OVERRIDES = ["data.categories=[bottle,carpet]", "data.subset=n_per_folder=10"]
OVERRIDES = {overrides!r}
print("phases:", PHASES, "| budget:", BUDGET_MINUTES, "min")'''),
        code('''# --- 2. Environment ---------------------------------------------------------
import os, subprocess, sys, time, shutil, pathlib
T0 = time.time()

def sh(cmd, check=False):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if r.stdout.strip(): print(r.stdout.strip()[-2000:])
    if check and r.returncode: raise RuntimeError(r.stderr[-2000:])
    return r.returncode

print(sh("nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader") or "")
sh("pip install -q open_clip_torch qwen-vl-utils omegaconf 2>&1 | tail -2")
import torch
print("torch", torch.__version__, "| cuda", torch.cuda.is_available(),
      "| gpus", torch.cuda.device_count())'''),
        code('''# --- 3. Restore any previous run's state -----------------------------------
# A committed notebook saves /kaggle/working as its output. Attaching that output
# to the next run is what makes the pipeline resumable across scheduled runs.
os.makedirs(RESULTS, exist_ok=True)
print("results ->", RESULTS)

restored = 0
if PREV_OUTPUT:
    src = f"/kaggle/input/{PREV_OUTPUT}/results"
    if os.path.isdir(src):
        for entry in os.listdir(src):
            dst = os.path.join(RESULTS, entry)
            if not os.path.exists(dst):
                shutil.copytree(os.path.join(src, entry), dst, dirs_exist_ok=True)
                restored += 1
        print(f"restored {restored} run directories from {src}")
    else:
        print(f"no previous output at {src} (starting fresh)")
else:
    print("PREV_OUTPUT not set - starting fresh")

led = pathlib.Path(RESULTS) / "run_all" / "phases.json"
print("phases already complete:", list(__import__("json").loads(led.read_text()))
      if led.exists() else "none")'''),
        code(f'''# --- 4. Unpack the embedded code -------------------------------------------
import base64, io, tarfile
CODE_B64 = "{payload}"

ROOT = "/kaggle/working/tzsad"
os.makedirs(ROOT, exist_ok=True)
with tarfile.open(fileobj=io.BytesIO(base64.b64decode(CODE_B64))) as t:
    t.extractall(ROOT)          # overwrite in place; never delete a live tree
os.chdir(ROOT)
print("code unpacked ->", ROOT)
print(sorted(os.listdir(ROOT)))'''),
        code('''# --- 5. Locate the dataset --------------------------------------------------
# Attach MVTec-AD (e.g. the "ipythonx/mvtec-ad" dataset) via Add data, or let
# kagglehub fetch it when Internet is enabled.
sys.path.insert(0, ROOT)
from tzsad.utils.env import resolve_data_root

try:
    DATA_ROOT = resolve_data_root("auto")
except FileNotFoundError:
    import kagglehub
    DATA_ROOT = kagglehub.dataset_download("ipythonx/mvtec-ad")
print("data root:", DATA_ROOT)
print(sorted(os.listdir(DATA_ROOT))[:8])'''),
        code('''# --- 6. Self-test before spending GPU quota --------------------------------
# The suite runs on stubbed backbones in about a minute. If it fails, the batch
# run would waste hours producing nothing, so stop here instead.
rc = sh("python -m pytest -q --tb=line -p no:warnings 2>&1 | tail -6")
assert rc == 0, "test suite failed - not starting the batch run"
print("\\nself-test passed")'''),
        code('''# --- 7. Run the pipeline ----------------------------------------------------
elapsed_min = (time.time() - T0) / 60
budget = max(BUDGET_MINUTES - elapsed_min, 5)
cmd = (f"python scripts/run_all.py --budget-minutes {budget:.0f} "
       f"--reserve-minutes {RESERVE_MINUTES} --phases {PHASES} "
       f"data.root={DATA_ROOT} paths.results={RESULTS} paths.cache={RESULTS}/cache "
       + " ".join(OVERRIDES))
print(cmd, flush=True)

proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT, text=True, bufsize=1)
for line in proc.stdout:            # stream, so a killed run still shows progress
    print(line, end="", flush=True)
rc = proc.wait()
print("\\nrun_all exit code:", rc, "(0 complete, 2 partial - re-run to continue)")'''),
        code('''# --- 8. Audit the outputs ---------------------------------------------------
import pandas as pd, glob, json
pd.set_option("display.width", 200)

runs = sorted(d for d in glob.glob(f"{RESULTS}/*") if os.path.isfile(f"{d}/manifest.json"))
print("run directories:", [os.path.basename(r) for r in runs])

for name, pattern in [
    ("detection",        f"{RESULTS}/*/report/detection.csv"),
    ("bake-off",         f"{RESULTS}/*/report/bakeoff_error_prediction.csv"),
    ("corruption",       f"{RESULTS}/*/report/corruption_monotonicity.csv"),
    ("coverage",         f"{RESULTS}/*/report/coverage_sweep.csv"),
]:
    for path in sorted(glob.glob(pattern)):
        df = pd.read_csv(path)
        print(f"\\n=== {name}: {path.replace(RESULTS + '/', '')} ({len(df)} rows) ===")
        if name == "detection":
            print(df[df.group.isin(["POOLED", "MEAN_OF_CATEGORIES"])].to_string(index=False))
        elif name == "bake-off":
            sub = df[(df.scope == "pred=0") & (df.half == "all")]
            print(sub.sort_values("err_auroc", ascending=False)[
                ["signal", "err_auroc", "ci_lo", "ci_hi", "informative", "misleading"]
            ].to_string(index=False))
        elif name == "corruption":
            print(df.groupby("signal").agg(
                silent=("silent_failure", "sum"),
                mean_rho=("spearman_severity_vs_uncertainty", "mean")).to_string())
        else:
            print(df[df.group == "POOLED"][
                ["delta", "nominal", "empirical_coverage", "ci_lo", "ci_hi"]
            ].to_string(index=False))'''),
        code('''# --- 9. Sanity checks -------------------------------------------------------
# Cheap invariants that catch a corrupted run before the numbers are believed.
problems = []
for path in sorted(glob.glob(f"{RESULTS}/*/records_test_scored.parquet")):
    df = pd.read_parquet(path)
    tag = path.replace(RESULTS + "/", "")
    if not df["anomaly_score"].dropna().between(0, 1).all():
        problems.append(f"{tag}: anomaly_score outside [0,1]")
    if df["image_id"].duplicated().any():
        problems.append(f"{tag}: duplicate image_id")
    if "conformal_p" in df and not df["conformal_p"].dropna().between(0, 1).all():
        problems.append(f"{tag}: conformal_p outside [0,1]")
    if "label" in df and not set(df["label"].unique()) <= {0, 1}:
        problems.append(f"{tag}: label not in {{0,1}}")
    print(f"{tag}: {len(df)} rows, {int(df.label.sum())} anomalous, "
          f"abstention {1 - df.parse_ok.mean():.3f}")

print("\\n" + ("PROBLEMS:\\n  " + "\\n  ".join(problems) if problems else "all sanity checks passed"))
print(f"\\ntotal notebook time: {(time.time() - T0) / 60:.1f} min")
print("outputs under /kaggle/working/results are saved as this version's output;")
print("attach them to the next run via PREV_OUTPUT to continue.")'''),
    ]
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
            "accelerator": "GPU",
            "kaggle": {"accelerator": "nvidiaTeslaT4", "dataSources": [],
                       "isInternetEnabled": True, "language": "python",
                       "sourceType": "notebook"},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="notebooks/tzsad_batch.ipynb")
    ap.add_argument("--phases", default="clip,synthetic,bakeoff,corruption")
    ap.add_argument("--budget-minutes", type=float, default=480)
    ap.add_argument("--prev-output", default="",
                    help="input dataset holding a previous run's output")
    ap.add_argument("--override", action="append", default=[],
                    help="dotlist override applied to every phase (repeatable)")
    args = ap.parse_args()

    payload = build_archive()
    nb = build_notebook(payload, args.phases, args.budget_minutes, args.prev_output,
                        args.override)
    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(nb, indent=1))
    print(f"wrote {out.relative_to(REPO)} "
          f"({out.stat().st_size / 1e3:.0f} KB, code payload {len(payload) / 1e3:.0f} KB)")
    print(f"phases={args.phases} budget={args.budget_minutes}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
