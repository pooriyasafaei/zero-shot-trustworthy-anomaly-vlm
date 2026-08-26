# TZ-SAD — Trustworthy Zero-Shot Anomaly Detection with VLMs

Research codebase for the trustworthiness half of a zero-shot anomaly detection
(ZSAD) project on MVTec-AD: uncertainty quantification, calibration, hallucination
detection, and the evaluation protocol.

## Core design rule

**Scoring and evaluation are decoupled.** A scoring run writes a per-image record
table to Parquet; every uncertainty, calibration and evaluation step consumes those
records offline. Adding an uncertainty signal or a metric never costs a GPU forward
pass. CLIP image (and window) embeddings are cached per `(model, category, split,
corruption)`, so multi-backbone ensembles, TTA and the normal-manifold kNN are
array operations.

## Layout

```
configs/            one yaml per experiment; nothing is configured in code
tzsad/
  data/             MVTec index + subset spec, ImageNet-C corruptions, CutPaste/NSA
  features/         CLIP embedder (the only GPU pass), embedding cache, patch heatmaps
  scorers/          Scorer interface, CLIP scorer, Qwen2.5-VL scorer, prompts/parsers
  uncertainty/      UncertaintyEstimator interface + 11 estimators, semantic entropy
  calibration/      Mondrian split conformal, temperature scaling (labelled baseline)
  hallucination/    language-grounding CMCS, operational hallucination rate
  fusion/           trust-score combiners + leave-one-out ablation
  eval/             metrics, bootstrap, DeLong, selective prediction, pixel AUROC/PRO
  viz/              qualitative TP/TN/FP/FN panels, paper figures
scripts/            thin CLI entrypoints
tests/              unit tests + an end-to-end smoke test with a stubbed backbone
results/            gitignored: caches, records, figures, report tables
```

## Per-image record schema

Core columns (stable): `run_id, scorer_name, model_id, category, defect_type, path,
image_id, label, split, subset_tag, corruption, severity, anomaly_score, raw_score,
parse_ok, n_valid_votes`.

`anomaly_score` is always in `[0, 1]`; the scorer's native quantity lives in
`raw_score`. VLM runs add `observation, predicted_defect, predicted_location,
predicted_confidence, vote_fraction, variant_pyes__*`. Uncertainty signals are added
offline as `u_*` columns. Unparseable VLM generations are recorded with
`parse_ok=False` and treated as abstentions — never dropped.

## Running

```bash
pip install -r requirements.txt

# Phase 1/2 — CLIP measurement foundation
python scripts/run_clip.py --config configs/clip_full.yaml data.root=/path/to/mvtec

# Phase 2 — VLM branch on the identical image subset
python scripts/run_qwen.py --config configs/qwen_full.yaml data.root=/path/to/mvtec

# Legitimate comparison (DeLong, same images)
python scripts/compare_scorers.py --a results/clip_full/records_test_scored.parquet \
    --b results/qwen_full/records_test_scored.parquet --name-a clip --name-b qwen

# Phase 3 — corruption sweep (the silent-failure figure)
python scripts/run_corruption.py --config configs/corruption.yaml

# Phase 4 — CMCS + hallucination rate
python scripts/run_hallucination.py --config configs/qwen_full.yaml \
    --records results/qwen_full/records_test_scored.parquet --maps-dir results/clip_full/maps

# Phase 5 — fusion + leave-one-out ablation
python scripts/make_synthetic.py --config configs/clip_full.yaml
python scripts/run_fusion_ablation.py --config configs/qwen_full.yaml \
    --records results/qwen_full/records_hallucination.parquet

pytest                       # unit tests + smoke test, no GPU or weights required
```

Every run writes `manifest.json` (config, seed, git SHA, library versions, GPU names)
and `run.log` into its results directory.

## What was fixed relative to the prototype

| # | Prototype defect | Fix |
|---|---|---|
| 1 | Prompt-ensemble std is not uncertainty | Kept as a labelled baseline; `prompt_std_z` and four real estimators added |
| 2 | `verdict_variance = sqrt(p(1-p))` is circular | Kept, explicitly labelled circular; `token_entropy` replaces it |
| 3 | Median threshold forces a 50% positive rate | Mondrian split conformal on `train/good` |
| 4 | Raw cosine diffs are not probabilities | Softmax `s_sem` with configurable τ |
| 5 | Branches evaluated on different image sets | One `SubsetSpec` shared by every scorer; DeLong refuses unaligned frames |
| 6 | Unparseable generations silently dropped | `parse_ok=False` → first-class abstention in the risk–coverage curves |
| 7 | `û = 1 − |(s − q̂)/q̂|^(-1)` diverges | Not implemented; conformal p-values instead |
| 8 | `HR` undefined | Operational definition against MVTec masks, with IoU sensitivity |
| 9 | Verbalized confidence near-degenerate | Kept as a baseline expected to be reported as a negative result |

## Out of scope

Video anomaly detection and the temporal filter; the explainability/captioning
module; any fine-tuning. Everything stays zero-shot with frozen backbones.
