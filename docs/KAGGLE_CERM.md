# Running CERM-AD Stage I on Kaggle

Copy-paste cells, in order. Markdown rather than a generated `.ipynb` on purpose:
a notebook that embeds the repo goes stale the moment the code changes, and a
checked-in `.ipynb` is a merge-conflict magnet. These cells always run the branch
as it is right now.

**Before you start:** enable **GPU T4 x2** and **Internet** in the notebook settings
(*Settings → Accelerator / Internet*). The pilot uses both GPUs.

---

## 1. Get the code

Private repo, so use a token. Create a fine-grained PAT with read access to the
repo, put it in *Add-ons → Secrets* as `GITHUB_TOKEN`, then:

```python
from kaggle_secrets import UserSecretsClient
import subprocess, os

token = UserSecretsClient().get_secret("GITHUB_TOKEN")
url = f"https://{token}@github.com/pooriyasafaei/zero-shot-trustworthy-anomaly-vlm.git"
subprocess.run(["git", "clone", "--branch", "experiment/cerm-ad-pilot", "--depth", "1",
                url, "/kaggle/working/tzsad"], check=True)
os.chdir("/kaggle/working/tzsad")
print(subprocess.run(["git", "log", "--oneline", "-1"], capture_output=True, text=True).stdout)
```

If you would rather not use a token, upload the repo as a Kaggle Dataset and
`cp -r /kaggle/input/<your-dataset>/* /kaggle/working/tzsad/` instead.

## 2. Install dependencies

```python
!pip install -q open_clip_torch qwen-vl-utils omegaconf tabulate 2>&1 | tail -2
```

Everything else (torch, transformers, scipy, pandas, matplotlib) is already in the
Kaggle image.

## 3. Dataset

Attach the `ipythonx/mvtec-ad` dataset via *Add data*, or download it:

```python
import kagglehub
DATA = kagglehub.dataset_download("ipythonx/mvtec-ad")
print(DATA)          # e.g. /kaggle/input/datasets/ipythonx/mvtec-ad
```

The repo's path auto-resolution finds the usual Kaggle locations on its own, so you
normally do **not** need to pass `data.root`. Pass it explicitly only if your copy
lives somewhere unusual.

## 4. GPU sanity check

```python
import torch
print("GPUs:", torch.cuda.device_count(),
      [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
!nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
```

Two T4s should be listed. The scorer picks **fp16** on T4 automatically — bf16 is
emulated on that architecture and several times slower.

## 5. Tests first (about 90 seconds)

Cheap insurance: a broken run wastes GPU quota producing nothing.

```python
!cd /kaggle/working/tzsad && python -m pytest -q --tb=short -p no:warnings 2>&1 | tail -5
```

## 6. Smoke run (about 1 minute of GPU)

One category, K=3, a handful of forward passes. Confirms the model loads, the
interventions build and every output file appears.

```python
!cd /kaggle/working/tzsad && python scripts/run_cerm.py --config configs/cerm_smoke.yaml 2>&1 | tail -15
```

## 7. The pilot — this is the real experiment

```python
!cd /kaggle/working/tzsad && python scripts/run_cerm.py \
    --config configs/cerm_pilot.yaml --workers auto 2>&1 | tail -30
```

`--workers auto` shards the target images across every visible GPU and runs one
pinned worker per device. Force a count with `--workers 1` or `--workers 2`.

Watch it from another cell:

```python
!tail -5 /kaggle/working/results/cerm_pilot/worker0.log
!wc -l /kaggle/working/results/cerm_pilot/causal_records.shard*.jsonl
```

## 8. Resume after an interruption

Re-run **the identical command**. Every intervention score and every completed
target is checkpointed as it is produced, so a killed session loses at most the
image in flight:

```python
!cd /kaggle/working/tzsad && python scripts/run_cerm.py \
    --config configs/cerm_pilot.yaml --workers auto 2>&1 | tail -30
```

It logs `resuming: N already audited, M to go` and reuses cached scores.

Rebuild only the tables and figures, with no GPU at all:

```python
!cd /kaggle/working/tzsad && python scripts/run_cerm.py \
    --config configs/cerm_pilot.yaml --report-only 2>&1 | tail -20
```

## 9. Optional: compare the fast probe against the structured scorer

The direct-verdict probe is a *different measurement* from the repo's
generate-then-answer scorer. If you have an existing qwen record file, check how
closely they track:

```python
!cd /kaggle/working/tzsad && python scripts/run_cerm.py \
    --config configs/cerm_pilot.yaml --report-only \
    --compare-with /kaggle/working/results/qwen_subset/records_test_scored.parquet 2>&1 | tail -10
```

Writes `probe_agreement.csv` and puts the verdict in the summary. A weak
correlation produces a loud warning — do not ignore it.

## 10. Where the results are

```
/kaggle/working/results/cerm_pilot/
  PRESENTATION_SUMMARY.md      <- read this first
  summary_overall.csv
  summary_by_category.csv
  summary_by_decision.csv
  artifact_controls.csv
  causal_records.parquet       <- one row per target anomaly
  interventions.parquet        <- one row per original/defect/placebo
  figures/
    defect_vs_placebo.png      <- the headline figure
    cee_by_category.png
    qualitative_grid.png
    artifact_control.png
    cee_vs_decision.png
  intervention_images/
  index.csv, manifest.json, run.log
  causal_records.shard*.jsonl, score_cache.shard*.jsonl   <- resume state
```

Quick look:

```python
from IPython.display import Markdown, Image, display
display(Markdown(open("/kaggle/working/results/cerm_pilot/PRESENTATION_SUMMARY.md").read()))
display(Image("/kaggle/working/results/cerm_pilot/figures/defect_vs_placebo.png"))
display(Image("/kaggle/working/results/cerm_pilot/figures/qualitative_grid.png"))
```

## 11. Download

```python
!cd /kaggle/working/results && zip -qr /kaggle/working/cerm_pilot.zip cerm_pilot \
    -x 'cerm_pilot/intervention_images/*'
!ls -lh /kaggle/working/cerm_pilot.zip
```

Drop the `-x` to include the intervention images (larger). The file appears in the
notebook's *Output* pane. For a scheduled run, commit the notebook (*Save Version →
Save & Run All*) — `/kaggle/working` becomes that version's output, and attaching it
to the next run makes the pipeline resume.

## 12. A larger run later

```python
!cd /kaggle/working/tzsad && python scripts/run_cerm.py \
    --config configs/cerm_full.yaml --workers auto --budget-minutes 400 2>&1 | tail -30
```

`cerm_full.yaml` uses K=31, every category and 8 targets per defect folder.
`--budget-minutes` stops it starting work it cannot finish, leaving time to write
the report.
