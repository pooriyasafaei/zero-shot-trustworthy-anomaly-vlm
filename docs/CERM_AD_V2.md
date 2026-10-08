# CERM-AD Stage I+ — bidirectional, artifact-matched causal verification

**Status: implemented and run as a pilot. Stage I (`docs/CERM_AD.md`) is unchanged and
still reproducible with `scripts/run_cerm.py`. Stage II (rationale mediation) is still
NOT implemented.**

## 1. Question and positioning

> Does a VLM anomaly verdict depend specifically on the physical defect, rather than on
> the presence of an image edit, correlated visual context, or another shortcut?

Counterfactual images are not new in anomaly detection, and causal interventions are
not new in VLM-faithfulness work (see `docs/CERM_RELATED_WORK.md`). What we study and
evaluate is their combination for this question: industrial pixel-level defect
annotations, defect-specific counterfactual restoration, matched irrelevant controls,
explicit matching of intervention strength, and a reverse insertion (sufficiency)
test, applied to the VLM's final verdict. To the best of our current literature
review, prior work does not combine these; we do not claim more than that.

## 2. Two axes, never collapsed

```
z(I) = log P(YES | I) / P(NO | I)                         (direct-verdict probe)

Necessity    delta_defect    = z(I) - z(I without M)
             delta_placebo_k = z(I) - z(I without P_k)
             CEE_N           = delta_defect - median_k delta_placebo_k

Sufficiency  delta_insert_defect = z(J with I[M]) - z(J)
             delta_insert_sham_k = z(J with S_k[M]) - z(J)
             CEE_S               = delta_insert_defect - median_k delta_insert_sham_k
```

`J` is the normal train/good donor already chosen by Stage I; `S_k` are other
train/good images of the same category. Both axes also report the robust z
(`(effect - median) / (1.4826 MAD + eps)`), the empirical control tail
`(1 + #{control >= effect}) / (K + 1)` (a rank statistic, not a classical p-value),
and whether the defect beats every control.

They are **not multiplied, averaged or otherwise combined.** The primary picture is
the quadrant plot (`figures/bidirectional_quadrant.png`): x = CEE_N (artifact-matched),
y = CEE_S. The quadrant label (`bidirectional`, `necessity_only`, `sufficiency_only`,
`neither`) is descriptive.

## 3. Necessity arms (all on the same targets, same donor, same defect-removal image)

| arm | placebo placement | purpose |
|---|---|---|
| naive | none: raw `delta_defect` | what "remove it and watch the score" gives |
| A random | translated exact GT mask, uniform over admissible positions | what matching buys |
| B context (v1) | Stage-I context-ring matched pool, **called with the Stage-I code and parameters** | the v1 result, recomputed in the same run |
| C artifact (v2) | context **and** artifact matched pool (below) | the new primary arm |
| D occlusion-style | defect and 9 context positions filled with the image's mean colour | a simple ROI-deletion style baseline; not a reproduction of any specific paper |

Admissible positions (all arms): every mask pixel inside the image (no clipping, so
area is identical), IoU with the defect <= 0.01, pairwise Chebyshev separation >= 8 px
between chosen placebos.

### Artifact matching (arm C)

The v1 pilot found defect fills ~2x stronger than their placebos (paired median seam
ratio 2.08, edit-energy ratio 2.01). For every admissible translation (up to 1000 per
target) the placebo fill is constructed from the same donor and measured **from pixels
only, before any model call**:

```
d_edit = |log((edit_p + eps) / (edit_defect + eps))|
d_seam = |log((seam_p + eps) / (seam_defect + eps))|
match_distance = sqrt( (ctx/s_ctx)^2 + (d_edit/s_edit)^2 + (d_seam/s_seam)^2 )
```

`s_*` is the robust scale (1.4826 x MAD) of that term across the target's candidates.
The terms are **scaled but not centred**: each is a distance with 0 = perfect match,
and centring at the median would give better-than-median candidates a negative z whose
square penalises them as much as worse ones. The matched pool is the best 30% by
`match_distance`; K = 19 are drawn uniformly from it with the seed (never the K
nearest, which would be near-duplicates). These settings were fixed before any Qwen
output of the v2 run existed.

Recorded per placebo: `context_distance`, `candidate_edit_energy`, `candidate_seam_score`,
`edit_ratio_to_defect`, `seam_ratio_to_defect`, `artifact_match_distance`; per target:
`artifact_n_pool`, `artifact_placebo_quality`, `artifact_tail_resolution`.

### Artifact balance as a result

Per arm and target: `seam_ratio = seam(defect fill) / mean seam(control fills)`, same
for edit energy, and the descriptive diagnostic

```
artifact_imbalance = |log seam_ratio| + |log edit_ratio|
```

(0 = perfectly balanced; it is an analysis diagnostic, not a trust score). The paired
context-only vs artifact-matched comparison is the first predeclared question.

## 4. Sufficiency arm

`J_defect` = `J` with the target's pixels inside `M` blended in (inward-only feather,
3 px); outside `M` it is bit-identical to `J` (asserted). Each sham uses the **same
recipient, mask, location and feathering**, with content from another train/good image
of the same category, so only the inserted content differs. Up to 60 sources per
category are measured (edit energy and seam vs the real insertion), the best 30% by
`sqrt((d_edit/s_edit)^2 + (d_seam/s_seam)^2)` form the pool, and K_suff = 9 are drawn
uniformly. The donor itself is excluded as a source. `DonorBank` refuses any
non-normal row, so anomalous images cannot enter the donor or sham banks.

## 5. Engineering that keeps it honest and fast

* **Exact fast path.** Feather weights and the seam band are translation-invariant
  away from the image border, so `MaskTemplate` computes them once per target and
  evaluates each candidate on a small window. Border placements fall back to the
  reference Stage-I `apply_intervention`. Equality is unit-tested, and every real
  target re-runs three interventions through the reference code (`fast_path_drift`,
  which must be 0; an image mismatch raises).
* **One forward pass per distinct image.** Scores are keyed by a SHA-256 of the exact
  pixels plus the scoring signature (model, probe version, resolution, dtype) - not the
  geometry - so byte-identical images in different arms share a pass, and every score
  is appended to a crash-safe cache as it is produced.
* **Batched probe.** All images of one target share a resolution, so they are scored in
  batches of 6 with no padding; `batch_check.shard*.json` records the measured
  difference from unbatched scoring (fp16 noise, ~0.015 logits on the pilot).
* **CPU/GPU overlap.** The next target is constructed in a background thread while the
  current one is scored.
* **Resume.** Every phase resumes from its JSONL checkpoints; failed targets are kept
  with their status and never dropped.

## 6. Structured-scorer confirmation (P0 / P1)

* **P0:** the repository's existing structured scorer (OBSERVATION / DEFECT_TYPE /
  DEFECT_LOCATION / CONFIDENCE / ANSWER, greedy) on every original target. Reported:
  Spearman/Pearson of P(YES), YES/NO agreement, and sensitivity of each probe. All
  targets are anomalous, so sensitivity is recall on anomalies - not accuracy and not
  "chance".
* **P1:** casebook targets under six roles (original, defect removed, best
  artifact-matched placebo, donor J, J_defect, best sham). Reports whether the *sign*
  of the removal and insertion effects agrees between the probes. Qualitative only;
  never used for population estimates. Exact generations are in
  `casebook/structured_examples.md`. This is not rationale mediation.

## 7. Casebook selection rules (deterministic)

See `tzsad/causal/casebook.py::SLOTS`. In short: best bidirectional case among
artifact-balanced targets (max of min robust z), necessity-without-sufficiency, weakest
in both directions, a missed anomaly (original verdict NO), strongest vs weakest
category, and the largest paired artifact-imbalance reduction. P1 is padded with
targets at fixed CEE_N quantiles.

## 8. Predeclared questions

1. **Q1** Did artifact matching improve balance? Strong: ratios move into the 1.25x band
   and the paired imbalance reduction CI excludes 0.
2. **Q2** Does necessity survive? Strong: median CEE_N (artifact) > 0 with CI excluding 0.
   If it disappears, that is reported as a negative finding.
3. **Q3** Is the defect sufficient? Evidence: CEE_S > 0 vs matched shams.
4. **Q4** Do the axes agree? Quadrant plot; no scalar.
5. **Q5** Does the structured scorer broadly agree with the direct probe? P0/P1.

The wording in `PRESENTATION_RESULTS.md` is generated from these rules in
`tzsad/causal/report_v2.py`.

## 9. Statistics

All arm comparisons are paired per target (bootstrap of per-image differences, 2000
resamples); YES-vs-NO comparisons are independent-group bootstraps of the difference
of medians. Wilcoxon signed-rank p-values are secondary. With N = 40 over 4 categories
these are effect-size intervals, not a significance claim.

## 10. Running it

```bash
python scripts/run_cerm_v2.py --config configs/cerm_v2_smoke.yaml --workers 1
python scripts/run_cerm_v2.py --config configs/cerm_v2_pilot.yaml --workers auto
python scripts/run_cerm_v2.py --config configs/cerm_v2_pilot.yaml --phase report   # no GPU
```

See `docs/KAGGLE_CERM.md` for the Kaggle cells.

## 11. Limitations

* Oracle GT masks: an audit tool, not a deployable detector (`mask_source: bank_map`
  remains planned).
* Donor fill and transplantation are not photorealistic; artifact matching controls
  their *strength*, not their *kind*.
* A transplanted defect sits on a different specimen; for objects with variable pose
  (screw) the inserted patch may be geometrically implausible, which the sham arm
  partly but not fully controls.
* The direct-verdict probe is a different measurement from the structured scorer; P0
  quantifies how different.
* One model (Qwen2.5-VL-3B), four categories, ~10 targets per category.
