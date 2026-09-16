# CERM-AD — Counterfactual Evidence & Rationale Mediation for Anomaly Detection

**Status: Stage I implemented and runnable. Stage II is designed here but NOT implemented.**

## 1. Motivation

The trustworthiness machinery already in this repository measures two things well:

* **Confidence and its calibration** — conformal p-values, token entropy, semantic
  entropy, prediction sets. These answer *how sure is the model?*
* **Cross-modal agreement** — CMCS, language-grounding verification. These answer
  *do two views of the model agree with each other?*

Neither answers the question a safety reviewer actually asks:

> Does the model's "defective" verdict **causally depend on the actual defect**?

A model can be confident and agree with itself while keying on something else
entirely — a lighting gradient, a background texture, a category prior. The
existing corruption sweep already showed the failure mode is real: every
conformal-derived uncertainty in this repository moves *down* as inputs degrade,
i.e. the system grows more confident as it gets worse. Confidence is not evidence.

CERM-AD attacks the question by **intervention** rather than by correlation.

## 2. Stage I — causal visual evidence verification

For an anomalous test image `I` with ground-truth defect mask `M`, define the VLM's
anomaly log-odds

```
z(I) = log( P(YES | I) / P(NO | I) )
```

Build a counterfactual in which **only the true defect region** is replaced by
normal visual content, `I_without_M`, and `K` matched **placebo** counterfactuals
`I_without_P_k` in which a mask of *identical shape and area* is applied at
visually similar, non-defective locations.

### Quantities

```
delta_defect      = z(I) - z(I_without_M)
delta_placebo_k   = z(I) - z(I_without_P_k)

CEE               = delta_defect - median_k( delta_placebo_k )

empirical_placebo_p = ( 1 + #{ k : delta_placebo_k >= delta_defect } ) / ( K + 1 )

robust_CEE_z      = ( delta_defect - median(delta_placebo) )
                    / ( 1.4826 * MAD(delta_placebo) + eps )
```

plus `defect_beats_all_placebos` and `defect_placebo_percentile`.

**Interpretation.** `CEE > 0` means neutralising the real defect moved the model's
anomaly belief *more* than making a comparable edit somewhere irrelevant did. That
is evidence the decision was grounded in the defect rather than merely correlated
with it.

### On `empirical_placebo_p`

This is an **empirical matched-placebo tail probability** — the rank of the observed
effect within its own placebo distribution. It is deliberately **not** called a
causal or randomisation p-value.

Reading it as a randomisation p-value would require the placebo placements to be
exchangeable with the true defect placement under the null. Translating the *exact*
mask to context-matched positions makes that plausible, but does not establish it:
defects do not occur uniformly over an object, and a translated mask may land on a
region with different semantics even when its local statistics match.

The `+1` in numerator and denominator counts the observed effect among the
candidates, so the statistic can never be exactly zero — with `K` placebos its floor
is `1/(K+1)`.

### Resolution guard

A claim at `alpha = 0.05` is impossible when `1/(n_valid_placebos + 1) > 0.05`, i.e.
below 19 valid placebos. In that case `supported_at_005` is `None` (NA), **not
`False`** — reporting `False` would read as evidence against the hypothesis when it
is in fact absence of resolution. `tail_resolution` is recorded per image.

## 3. Counterfactual reconstruction: aligned normal donor fill

Deliberately not a diffusion inpainter. The novelty claimed here is the causal
verification, not the filling; a generative model would add a dependency, a
seed-sensitivity and a large runtime cost for no scientific gain, and would make
the defect and placebo arms harder to hold identical.

For each target image:

1. **Donor candidates** come from that category's `train/good` split **only**.
   Using an anomalous image as a donor would paste defect pixels into a
   "defect removed" counterfactual.
2. **One donor per target**, chosen by downsampled RGB L1 distance computed
   **outside a dilated version of the defect mask** — so the donor choice cannot be
   informed by the very region being removed.
3. **The same donor serves every intervention on that image**, real and placebo.
   This controls donor identity as a confounder.
4. **Outside the intervention mask the output is bit-identical to the original.**
   Feathering uses `distance_transform_edt` *inside* the mask only. Asserted at
   runtime and covered by tests.

Recorded per intervention: donor id and distance, mask area and fraction, feather
width, `edit_energy`, `outside_change_max`, `seam_score`.

## 4. Matched placebo masks

A placebo is the **exact GT mask, translated**. Shape and area are therefore
identical by construction; only location varies.

Constraints on a candidate translation:

* every mask pixel stays inside the image (clipping is forbidden — a clipped mask
  would no longer have the defect's area);
* `IoU` with the true defect `<= max_iou` (default 0.01);
* a **context ring** (`dilate(mask) - mask`) whose RGB and gradient statistics are
  similar to the true defect's ring;
* Chebyshev separation `>= min_separation` from placebos already chosen.

Feasible candidates are swept deterministically on a grid, ranked by context
distance, and the most similar `match_quantile` fraction forms the **matched pool**.
`K` placebos are then drawn **uniformly without replacement** from that pool using
the experiment seed.

> Why not just take the K closest matches? Because they would be near-duplicates of
> one another, collapsing the placebo distribution and leaving no meaningful tail to
> compare against. The pool is the control *distribution*, not the single best match.

Too few feasible placements degrades visibly: `n_valid_placebos`,
`placebo_quality`, `placebo_reason` and `tail_resolution` are all recorded, and
masks are never fabricated to reach `K`.

## 5. Artifact controls

A reviewer will ask whether the model reacted to the defect or to a bad splice.
Every intervention reports `edit_energy`, `outside_change_max` (must be 0),
`seam_score` and the donor distance; `artifact_controls.csv` compares the real-defect
arm against the placebo arm, and the Spearman correlation between `CEE` and
`seam_ratio` is reported.

**A large CEE alongside a systematically worse real-mask seam is not causal
evidence**, and the generated summary says so when the ratio exceeds 1.5.

## 6. The direct-verdict causal probe

The repository's structured scorer generates an `OBSERVATION / DEFECT_TYPE /
DEFECT_LOCATION / CONFIDENCE / ANSWER` block before the verdict is readable. Stage I
needs `2 + K` scores per image (21 at `K=19`) and does not need the prose, so
generating ~120 tokens each time would dominate runtime for no scientific gain.

`QwenScorer.score_direct_verdict` asks a short verdict-only question and reads the
**next-token** logits in a single forward pass:

```
log_yes   = logsumexp( logits[YES token variants] )
log_no    = logsumexp( logits[NO token variants]  )
audit_logit = log_yes - log_no
audit_p_yes = sigmoid(audit_logit)
```

**This is a different measurement, not a faster version of the structured scorer.**
A model asked to answer immediately may answer differently from one that reasons
first. `--compare-with <existing qwen records>` therefore measures Spearman,
Pearson and verdict agreement between the two on shared `image_id`s, and emits a
loud warning when the correlation is weak. `causal.score_mode: generated_verdict`
runs the confirmation arm through the existing generation path.

Neither probe changes the behaviour or outputs of the existing scorer; both are
additive public methods.

## 7. Pilot vs deployment

Tonight's pilot is deliberately an **oracle** study: interventions are built from
MVTec ground-truth masks. The mask is used **only** to construct the counterfactual
— never in the prompt, never in the model's input, never in the original score. A
test asserts the probe sees nothing but a 3-channel image and a class name.

The point is to establish whether the *methodology* detects anything at all before
investing in the harder version. Deployment would replace the oracle mask with a
predicted anomaly region. The interface is already in place:

```yaml
causal:
  mask_source: gt        # implemented
  # mask_source: bank_map  # planned
```

**`bank_map` is not implemented.** The intended source is the existing
`features/heatmap.py` `bank_maps()` normal-manifold deviation map, thresholded to a
region of comparable area. Adding it means implementing a `mask_source` resolver in
`audit.py::audit_target` that returns a boolean mask; everything downstream — donor
matching, placebo generation, reconstruction, effects — is already mask-source
agnostic. The honest caveat for that study is that a predicted mask carries its own
errors, so a null result there could reflect bad masks rather than absent causality.

## 8. Stage II — rationale mediation (DESIGNED, NOT IMPLEMENTED)

**Nothing in Stage II is implemented.** This section records the intended design.

Generate, for original `I` and defect-neutralised `I-`, a structured rationale:

```
VISUAL_EVIDENCE:
...
REASONING:
...
ANSWER:
YES/NO
```

giving rationales `R+` (from `I`) and `R-` (from `I-`). Then evaluate **teacher-forced**
answer logits, holding the rationale fixed while swapping the image and vice versa:

```
Z++ = z(I,  R+)
Z+- = z(I,  R-)
Z-- = z(I-, R-)

Total Effect              = Z++ - Z--
Rationale Mediation Effect = Z++ - Z+-
Residual Visual Effect     = Z+- - Z--
```

This decomposes how much of the verdict flows through the *stated reason* versus
through the image directly — separating a model that reasons from the evidence from
one that produces a plausible rationale after the fact.

Open questions before implementing: teacher-forcing a rationale the model did not
generate puts it off its own distribution, and the decomposition assumes the
rationale is the only mediator.

## 9. What this does and does not show

* It measures the **direct-verdict probe** of one VLM on an **oracle** mask study
  over a small, deliberately capped subset.
* A positive CEE is evidence that the verdict depends on the defect region more
  than on comparable irrelevant edits **of this reconstruction type**.
* It is **not** a statement about the structured reasoning pipeline unless the
  agreement check says so, **not** a deployment result, and **not** a
  publication-grade claim at this sample size.
