# TZ-SAD — design decisions, phased plan, disagreements, risks

Answers to §7 of the brief, written against the implemented code rather than in
the abstract. Where I disagreed with the brief, the disagreement is stated and the
brief's version is still implemented (as a comparison arm) unless it is
mathematically broken.

---

## 1. Repo layout and the per-image record schema

The layout follows the brief with three deviations, each justified below.

```
configs/            one yaml per experiment, with a `defaults:` parent chain
tzsad/
  data/             mvtec.py (index + SubsetSpec), corruptions.py, synthetic.py
  features/         clip_embedder.py, cache.py, heatmap.py      <-- deviation 1
  scorers/          base.py, clip_scorer.py, qwen_scorer.py, prompts.py, qwen_parse.py
  uncertainty/      base.py, image_side.py, vlm_side.py, semantic_entropy.py, registry.py
  calibration/      conformal.py, temperature.py
  hallucination/    location.py, cmcs.py, rate.py
  fusion/           trust.py
  eval/             metrics.py, bootstrap.py, selective.py, pixel.py, report.py
  viz/              panels.py, plots.py, style.py
  records.py        the schema itself                            <-- deviation 2
  pipeline.py       stage functions the scripts compose          <-- deviation 3
scripts/            run_clip, run_qwen, run_corruption, run_hallucination,
                    run_fusion_ablation, compare_scorers, make_synthetic
tests/
results/            gitignored
```

**Deviation 1 — `features/` is separate from `scorers/`.** The brief put embedding
and scoring together. Splitting them is what makes the core design rule
enforceable: `features/` is the only place that touches a GPU for the CLIP branch,
and `scorers/clip_scorer.py` is pure numpy over cached arrays. A new uncertainty
signal, a new prompt set, or a different τ costs zero forward passes because the
scorer is downstream of the cache, not upstream of it.

**Deviation 2 — `records.py` at the package root.** The schema is the contract
between every subsystem, so it should not live inside any one of them.
`validate_records` is called on every write and rejects, among other things, an
`anomaly_score` outside `[0, 1]` — which is what stops defect #4 from creeping
back in.

**Deviation 3 — `pipeline.py`.** The scripts in `scripts/` are argument parsing
plus a sequence of stage calls. The stages themselves are importable functions, so
the smoke test can drive the whole pipeline with a stubbed backbone and no weights.

### Per-image record schema

Core columns, stable:

| column | meaning |
|---|---|
| `run_id`, `scorer_name`, `model_id` | provenance; `scorer_name` is e.g. `clip:ViT-B-16__openai:base` |
| `category`, `defect_type`, `path`, `image_id` | identity; `image_id` is the join key everywhere |
| `label` | 1 anomalous, 0 normal |
| `split` | `train` = normal-only conformal calibration pool, `test` = eval |
| `subset_tag` | which `SubsetSpec` produced the row |
| `corruption`, `severity` | `none`/`0` when clean |
| `anomaly_score` | **always in [0, 1]** |
| `raw_score` | the scorer's native quantity (cosine margin, verdict logit, vote fraction) |
| `parse_ok` | `False` = abstention, carried into every metric |
| `n_valid_votes` | 1 for single-pass scorers |

VLM runs add `observation, predicted_defect, predicted_location,
predicted_confidence, vote_fraction, n_generations, variant_pyes__*`.
CLIP runs add `tpl_NN` / `tplraw_NN` per-template columns.
Offline stages add `conformal_p, conformal_pred, correct, u_*, cmcs, s_adj,
halluc, halluc_case, gt_iou, trust_*`.

Uniqueness is enforced on `(scorer_name, image_id, corruption, severity)`.

---

## 2. Phased plan and effort

| Phase | Content | Status | Effort |
|---|---|---|---|
| 1 | Data layer, embedding cache, CLIP scorer with softmax `s_sem`, record schema, Mondrian conformal, AUROC+CI / ECE / AURC / error-prediction AUROC | implemented | ~2 days |
| 2 | All 11 uncertainty estimators behind one interface; semantic entropy by bidirectional entailment; head-to-head bake-off | implemented | ~2 days |
| 3 | Corruption sweep, monotonicity table, the silent-failure figure | implemented | ~1 day |
| 4 | Language-grounding CMCS, Eq. 4/5 flag + `s_adj`, operational HR with IoU sensitivity | implemented | ~1.5 days |
| 5 | Three fusion combiners + leave-one-out ablation | implemented | ~1 day |
| — | Cross-dataset shift (VisA/BTAD) | needs the datasets mounted; the loader already handles VisA-style masks | ~0.5 day |

GPU cost, measured on 2×T4: full 15-category CLIP embedding pass (≈9k images)
about 10 min, plus about 25 min for the 9-window multi-scale grid. Everything
after that is CPU-only. The Qwen branch is the expensive one — see risks.

---

## 3. Disagreements with §2 and §4

**I agree with all nine diagnosed defects in §2.** Two of them I would state more
strongly, and I disagree with three implementation choices in §4.

### 3.1 `u = 1 − 2|p − 0.5|` peaks in the wrong place (§4.2)

The brief derives uncertainty from the conformal p-value as `1 − 2|p − 0.5|`,
maximal at `p = 0.5`. But `p = 0.5` is the *median of the normal calibration
distribution* — a thoroughly typical normal image, which should be the most
confident "normal" call in the dataset. The decision boundary is at `p = δ`.
As specified, the signal is maximal where the model is most certain and low
(`u ≈ 2δ`) exactly at the boundary where errors concentrate.

**Implemented:** `pvalue_uncertainty(..., mode=...)` with `symmetric` (the brief's
form, default for comparability), `entropy`, and `boundary` — which maps `δ → 0.5`
before applying the same rule, so the peak sits at the decision boundary. The
three are compared head-to-head in the bake-off; `configs/base.yaml:
conformal.uncertainty_mode` selects one.

### 3.2 `PromptEnsembleStd` fails for a second reason the brief does not name (§4.3)

The brief attributes its failure to templates being near-paraphrases. True, but
there is a mechanical reason as well: once scores are softmax probabilities, the
spread of a set of probabilities is largest near 0.5 by construction. So
`prompt_std` inherits exactly the circularity that condemns `verdict_variance` —
it is partly a re-encoding of `|p − 0.5|`. This matters because it means
`NormalizedPromptEnsemble` (z-scoring per template) may not rescue it: z-scoring
removes per-template offset and gain, not the score-magnitude coupling.

**Implemented anyway** — it is one line over cached arrays and a clean negative
result is worth having — but I would not budget hope for it. `prompt_std` is
computed on the *raw* margins by default in the z-scored variant, which sidesteps
the softmax coupling.

### 3.3 `TTAVariance` is the estimator I would cut first (§4.3)

MVTec defects are frequently a few dozen pixels (`screw`, `grid`, `pill`). The
augmentations that make TTA informative — crops and scale jitter — are exactly the
ones that delete such defects, so a correct prediction becomes unstable for a
reason unrelated to model uncertainty. Expect it to look informative on textures
(`carpet`, `leather`, `wood`) and actively misleading on small-defect object
categories. It is implemented and cached (each view is one extra shard), so it
costs little to include, but I would report it per-category rather than pooled,
because the pooled number will average two opposite behaviours into a mush.

**Cheaper route to the same evidence:** `BackboneEnsemble` gets at epistemic
uncertainty without perturbing the image at all, and once embeddings are cached
it is free. If effort must be cut, cut TTA and keep the backbone ensemble.

### 3.4 CMCS on MVTec — I agree with the reframing, with one addition (§4.4)

The brief is right that Grounding-DINO has nothing to disagree about on a centred
single object. The language-grounding version is implemented as primary. One
addition: when the VLM's location claim is "surface" / "throughout" / unparseable,
the claim covers the whole image and **cannot be contradicted**. Counting those as
"grounded" would inflate the grounding rate with vacuous claims, so
`LocationClaim.is_localized` is False for them and they fall back to the
*attended* region (top-5% of the anomaly map). The fraction of vacuous claims is
itself reported — I expect it to be large, and that is a finding about the VLM.

### 3.5 The `s_adj` coupling (proposal Eq. 5) deserves a caveat in the paper

§4.2 says conformal calibrates on "the adjusted anomaly score", i.e.
`s_adj = s_sem · CMCS^α`. Implemented (`apply_flag_and_adjust`). But attenuating
the detection score by a trust quantity fuses detection and trust into one number,
so a post-attenuation AUROC drop cannot be distinguished from a trustworthiness
gain. The pipeline therefore keeps `anomaly_score` and `s_adj` as two separate
arms (`conformal.calibrate_on`) rather than overwriting one with the other.

### 3.6 One place the brief is stricter than necessary

`SemanticEntropy` with an NLI model is the methodologically faithful choice, but
`deberta-large-mnli` is 1.6 GB and the clustering is O(k) entailment calls per
image. On a subset of 20 images/folder × 15 categories × 8 samples that is
manageable; on the full test set it is not. The embedding backend
(`all-MiniLM-L6-v2`, symmetric cosine ≥ threshold) is implemented as the default
and the NLI backend as the faithful arm, with the paper reporting both on a
subset to show the approximation is sound.

---

## 4. The three riskiest parts, and the de-risking

### Risk 1 — the headline result may be that *nothing* predicts error well

The paper's central table is "AUROC of the uncertainty signal for predicting
misclassification". It is entirely possible that every signal lands in
`[0.5, 0.6]` with CIs straddling 0.5. That is a real possibility, not a
pessimistic one: the prototype's evidence for its own signal was ~6/15 categories
with gaps of 0.001 on a scale of 0.015.

*De-risking.* (a) `NormalManifoldDistance` is prioritised because it is the one
signal that does not live inside the model's own belief, so it can be informative
where the belief-internal signals are flat. (b) The corruption sweep is designed
so that a flat-uncertainty result is *itself* the publishable finding — the
"silent failure" claim needs uncertainty to stay flat while AUROC collapses, so a
negative bake-off strengthens rather than weakens that figure. (c) Every signal
declares its expected failure mode in its docstring before the numbers come in, so
a negative result is a confirmed prediction rather than a salvage operation.

### Risk 2 — VLM throughput

Qwen2.5-VL is the bottleneck by two orders of magnitude. On 2×T4 (no bf16, fp16
only) a 7B model with multi-sample generation at 8 samples/image is roughly
15–25 s/image; 15 categories × 20 images/folder ≈ 1,700 images ≈ 8–12 hours, which
does not fit a Kaggle session. Prompt-perturbation consistency multiplies that by
the number of variants.

*De-risking.* (a) `mode=logprob` gets a continuous score from **one greedy pass**,
so the primary comparison does not need sampling at all. (b) Sampling-based signals
run on a smaller subset, and the subset is declared in the record via `subset_tag`
so it can never be silently mixed with the full-set numbers. (c) Every generation
is dumped to `generations/*.json`, so semantic entropy, verbalised confidence and
any future text-based signal are recomputed offline with no GPU. (d) The default
config uses the 3B model; the 7B run is a single confirmation pass on a subset.

### Risk 3 — conformal exchangeability between `train/good` and `test/good`

The whole calibration story rests on `train/good` and `test/good` being
exchangeable. MVTec does not guarantee this: some categories' test-good images
were captured in the same session as the defective ones, and per-category
conditional coverage at n≈200 has a standard deviation of about 0.02 anyway
(`test_conditional_coverage_spread_is_large_at_n_200` demonstrates this).
So a category showing 0.83 coverage against a nominal 0.95 may be a real
exchangeability violation or may be one unlucky calibration draw, and the
difference matters for what we claim.

*De-risking.* (a) `coverage_report` reports per-category coverage **with bootstrap
CIs**, never as a point estimate. (b) `n_cal_sensitivity` repeats the calibration
draw and reports the spread, separating calibration-set randomness from a genuine
violation. (c) The δ sweep gives four independent checks of the same guarantee —
a real violation degrades coverage at every δ, an unlucky draw does not. (d) The
cross-dataset arm (calibrate on MVTec, test on VisA/BTAD) is designed to show
coverage breaking on purpose, which validates that the diagnostic has teeth.

---

## 5. Scope discipline

Nothing outside the brief was added. Video/temporal (proposal Module 5), the
explainability module (proposal §5.8), and any fine-tuning are not implemented.
The image pipeline keeps `split`/`corruption`/`severity` in the record key so a
frame index can slot in later without a schema change.
