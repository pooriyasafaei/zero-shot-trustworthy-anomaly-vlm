# CERM-AD Stage I+ — pilot results (bidirectional, artifact-matched)

_Generated automatically from the run's records. Every number below is measured; every interpretive sentence is produced by a rule stated in `tzsad/causal/report_v2.py`._

## A. Run metadata

- **branch**: experiment/cerm-ad-v2-controls
- **git_sha**: 4b13bba234c33fb428e3c8b54272a7ae5447a177 (deployed tree)
- **model_id**: Qwen/Qwen2.5-VL-3B-Instruct
- **score_mode**: direct_verdict
- **mask_source**: gt (oracle ground-truth masks)
- **N_targets**: 40
- **categories**: bottle, carpet, pill, screw
- **forward_images_total**: 3002
- **unique_images_scored**: 2945
- **runtime**: direct_minutes 16.72 min, structured_minutes 2.36 min
- **gpu**: 2 x Tesla T4
- **seed**: 1234
- **transformers**: 5.0.0

## B. What changed from CERM v1

- **Same 40 targets** (same config chain, seed and per-folder cap as `cerm_pilot`), so every comparison is paired per image.
- **Artifact-matched placebos (new arm C):** candidate placebos are ranked by a predeclared composite of context distance and the |log| edit-energy and seam ratios to the real defect fill, computed from pixels before any model call; K drawn uniformly from the best 30%.
- **The v1 context-only arm is recomputed in the same run** (arm B), plus a random-translation control (A) and an occlusion-style baseline (D).
- **Sufficiency (new):** the real defect is transplanted into the normal donor J and compared with artifact-matched normal-patch shams at the same location.
- **Structured-scorer check (P0/P1):** the existing structured scorer on all originals and on casebook interventions.
- **v1 reproduces:** the Stage-I pilot reported median CEE +0.0966 [+0.0002, +0.2013] (N=40); the context-only arm recomputed here gives +0.096 [+0.001, +0.216] (different transformers version and batched fp16 scoring, so small numeric differences are expected).

## C. Artifact balance (Q1)

| arm | N | median_seam_ratio | median_edit_ratio | median_abs_log_seam_ratio | median_abs_log_edit_ratio | median_artifact_imbalance | frac_seam_ratio_within_1.25x | frac_edit_ratio_within_1.25x |
|---|---|---|---|---|---|---|---|---|
| random | 37 | 2.45 | 2.82 | 0.89 | 1.04 | 2.00 | 27% | 22% |
| context | 37 | 1.94 | 2.03 | 0.66 | 0.71 | 1.37 | 35% | 27% |
| artifact | 37 | 1.58 | 1.81 | 0.46 | 0.60 | 0.98 | 38% | 35% |
| occlusion | 37 | 1.01 | 1.06 | 0.14 | 0.16 | 0.34 | 59% | 57% |
| sufficiency | 40 | 1.01 | 1.06 | 0.12 | 0.16 | 0.31 | 57% | 57% |

Paired improvement in imbalance (context-only minus artifact-matched): median **+0.11** [+0.04, +0.23], improved on 81% of 37 targets (Wilcoxon p=0.00019, secondary).

**Reading:** artifact matching *reduced* the imbalance (paired CI excludes zero) but did **not** reach the predeclared 1.25x balance band (median seam 1.58, edit 1.81). The intervention-strength confound is smaller, not gone.

## D. Necessity (Q2)

| method | N | median_delta_defect | median_control_delta | median_cee | cee_ci_lo | cee_ci_hi | frac_cee_positive | frac_beats_all_controls | median_empirical_tail | median_robust_z | median_artifact_imbalance |
|---|---|---|---|---|---|---|---|---|---|---|---|
| naive | 40 | +0.105 | +0.000 | +0.105 | +0.008 | +0.175 | 68% | n/a | n/a | n/a | n/a |
| random | 40 | +0.105 | +0.015 | +0.090 | +0.000 | +0.188 | 68% | 54% | 0.100 | +2.22 | 2.00 |
| context | 40 | +0.105 | +0.015 | +0.096 | +0.001 | +0.216 | 68% | 54% | 0.075 | +2.12 | 1.37 |
| artifact | 40 | +0.105 | -0.000 | +0.082 | -0.014 | +0.220 | 65% | 46% | 0.150 | +1.90 | 0.98 |
| occlusion | 40 | -0.039 | -0.076 | +0.031 | -0.030 | +0.081 | 54% | 27% | 0.550 | +0.27 | 0.34 |

`median_empirical_tail` is the empirical matched-placebo tail probability (rank of the defect effect among its controls), not a classical p-value.

Paired change in CEE_N (context-only minus artifact-matched): median -0.002 [-0.015, +0.001].

**Reading:** the necessity effect is **weakened** under artifact matching: median CEE_N +0.082 remains positive but its CI [-0.014, +0.220] includes zero (context-only v1 arm in this run: +0.096 [+0.001, +0.216]).

Note: the paired per-target change in CEE_N between context-only and artifact-matched controls is small (median -0.002); the change in the pooled interval is a marginal shift, not a collapse of the per-target effects.

## E. Sufficiency (Q3)

| group | N | N_with_shams | median_delta_insert_defect | median_sham_delta | median_cee_s | cee_s_ci_lo | cee_s_ci_hi | frac_cee_s_positive | frac_insert_beats_all_shams | median_empirical_sham_tail | median_artifact_imbalance | frac_recipient_predicted_yes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ALL | 40 | 40 | +0.199 | +0.015 | +0.156 | +0.097 | +0.230 | 82% | 62% | 0.100 | 0.31 | 0% |

**Reading:** inserting the real defect raises anomaly belief more than matched normal shams: median CEE_S +0.156 [+0.097, +0.230].

## F. Bidirectional result (Q4)

| group | N | N_determined | n_bidirectional | frac_bidirectional | n_necessity_only | n_sufficiency_only | n_neither |
|---|---|---|---|---|---|---|---|
| ALL | 40 | 37 | 22 | 59% | 2 | 9 | 4 |
| category=bottle | 6 | 5 | 5 | 100% | 0 | 0 | 0 |
| category=carpet | 10 | 10 | 10 | 100% | 0 | 0 | 0 |
| category=pill | 14 | 12 | 4 | 33% | 0 | 6 | 2 |
| category=screw | 10 | 10 | 3 | 30% | 2 | 3 | 2 |
| original_verdict=NO | 21 | 19 | 9 | 47% | 2 | 5 | 3 |
| original_verdict=YES | 19 | 18 | 13 | 72% | 0 | 4 | 1 |

Overall, **22/37 (59%)** of targets fall in the positive necessity + sufficiency quadrant. The two axes are reported separately and are not combined into a score.

Original YES minus original NO (difference of medians, independent-group bootstrap): CEE_N +0.252 [-0.077, +0.733] (n=18 vs 19); CEE_S +0.134 [+0.002, +0.490].

## G. Direct probe vs structured scorer (Q5)

On the 40 original targets with a parsed structured verdict: Spearman 0.39, Pearson 0.43, YES/NO agreement 48%. Sensitivity (recall on these anomaly-only targets): direct 48%, structured 85%. Structured parse rate 100%.

**Warning:** the two probes track each other weakly (Spearman < 0.5). The causal results describe the direct-verdict probe and should not be transferred to the structured pipeline without the P1 check below.

**P1 sign agreement on casebook targets (qualitative, not a population estimate):**

| image_id | slot | direct_removal_delta | structured_removal_delta | removal_sign_agree | direct_insertion_delta | structured_insertion_delta | insertion_sign_agree |
|---|---|---|---|---|---|---|---|
| carpet/test/metal_contamination/005 | best_bidirectional | +0.943 | +1.625 | True | +0.932 | +2.031 | True |
| screw/test/thread_side/006 | necessity_not_sufficiency | +0.219 | -0.063 | False | -0.013 | -0.031 | True |
| pill/test/scratch/004 | failure_ungrounded | -0.030 | -0.124 | True | -0.029 | +0.015 | False |
| carpet/test/cut/004 | missed_anomaly | +0.540 | -0.370 | False | +0.698 | -0.696 | False |
| carpet/test/thread/008 | category_strong | +1.484 | +2.627 | True | +1.346 | +2.255 | True |
| pill/test/crack/010 | category_weak | -0.560 | +0.099 | False | -0.010 | +0.062 | False |
| screw/test/manipulated_front/000 | artifact_control | +0.173 | -0.048 | False | +0.235 | -0.078 | False |
| bottle/test/broken_small/004 | spread_q50 | +0.177 | +1.674 | True | +0.380 | +0.354 | True |
| pill/test/faulty_imprint/006 | spread_q10 | -0.017 | +0.077 | False | +0.049 | +0.060 | True |
| carpet/test/cut/000 | spread_q90 | +1.059 | +1.776 | True | +0.904 | +0.884 | True |

Removal-direction agreement 5/10; insertion-direction agreement 6/10.

## H. Per-category results

| category | N | CEE_N context (v1) | CEE_N artifact (v2) | CEE_N v2 CI | CEE_S | CEE_S CI | direct sensitivity | bidirectional |
|---|---|---|---|---|---|---|---|---|
| bottle | 6 | +0.145 | +0.092 | [+0.074, +0.302] | +0.127 | [+0.051, +0.269] | 33% | 5/6 |
| carpet | 10 | +0.779 | +0.825 | [+0.538, +1.228] | +0.807 | [+0.468, +1.315] | 80% | 10/10 |
| pill | 14 | -0.033 | -0.043 | [-0.112, +0.031] | +0.102 | [+0.011, +0.164] | 64% | 4/14 |
| screw | 10 | -0.007 | -0.007 | [-0.041, +0.177] | +0.053 | [-0.031, +0.195] | 0% | 3/10 |

Per-category N is ~10; per-category intervals are wide and are descriptive only.

## I. Real examples selected for the presentation

Selected deterministically by the rules below (see `casebook_manifest.csv`); figures in `figures/example_*.png` and `casebook/case_*.png`.

- **best_bidirectional** — `carpet/test/metal_contamination/005`: CEE_N +0.943, CEE_S +0.946, original verdict YES. _Rule:_ among targets whose necessity AND sufficiency artifact imbalance are at or below the run median, maximise min(robust CEE_N z, robust CEE_S z), preferring CEE_N>0 and CEE_S>0
- **necessity_not_sufficiency** — `screw/test/thread_side/006`: CEE_N +0.220, CEE_S -0.061, original verdict NO. _Rule:_ CEE_N>0 and CEE_S<=0, maximise robust CEE_N z - robust CEE_S z (fallback: maximise CEE_N - CEE_S)
- **failure_ungrounded** — `pill/test/scratch/004`: CEE_N -0.138, CEE_S -0.046, original verdict YES. _Rule:_ minimise max(robust CEE_N z, robust CEE_S z): weakest evidence in both directions
- **missed_anomaly** — `carpet/test/cut/004`: CEE_N +0.509, CEE_S +0.712, original verdict NO. _Rule:_ original direct verdict NO; maximise |robust CEE_N z|
- **category_strong** — `carpet/test/thread/008`: CEE_N +1.516, CEE_S +1.315, original verdict YES. _Rule:_ category with the highest median CEE_N; its target with the highest CEE_N
- **category_weak** — `pill/test/crack/010`: CEE_N -0.514, CEE_S -0.011, original verdict NO. _Rule:_ category with the lowest median CEE_N; its target with the lowest CEE_N
- **artifact_control** — `screw/test/manipulated_front/000`: CEE_N +0.188, CEE_S +0.235, original verdict NO. _Rule:_ maximise the paired reduction in artifact imbalance (context-only minus artifact-matched)
- **spread_q50** — `bottle/test/broken_small/004`: CEE_N +0.082, CEE_S +0.381, original verdict YES. _Rule:_ P1 padding: target whose CEE_N is closest to its 50th percentile
- **spread_q10** — `pill/test/faulty_imprint/006`: CEE_N -0.096, CEE_S +0.094, original verdict YES. _Rule:_ P1 padding: target whose CEE_N is closest to its 10th percentile
- **spread_q90** — `carpet/test/cut/000`: CEE_N +1.119, CEE_S +0.902, original verdict YES. _Rule:_ P1 padding: target whose CEE_N is closest to its 90th percentile

## J. Failure examples

4 target(s) are in the 'neither' quadrant. Most negative CEE_N:

- `pill/test/crack/010`: CEE_N -0.514, CEE_S -0.011, verdict NO
- `pill/test/scratch/004`: CEE_N -0.138, CEE_S -0.046, verdict YES
- `screw/test/scratch_head/002`: CEE_N -0.065, CEE_S -0.016, verdict NO
- `screw/test/thread_top/002`: CEE_N -0.031, CEE_S -0.046, verdict NO

3 target(s) have fewer than 19 artifact-matched placebos (masks too large to translate); they are reported with their actual counts, never padded.

## K. What the results support

- Artifact matching measurably reduces the defect-vs-placebo intervention strength imbalance found in v1 (paired, same targets).
- Transplanting the real defect into a normal image raises anomaly log-odds more than artifact-matched normal-patch insertions.
- A subset of targets (59%) shows evidence in both causal directions; the rest show one or neither, so grounding is target- and category-dependent.

## L. What the results do NOT support

- Deployment readiness: masks are oracle GT masks; a predicted-mask study is not done.
- Claims about the structured reasoning pipeline beyond the measured P0 agreement and the qualitative P1 cases.
- Rationale faithfulness or mediation: Stage II is not implemented; the structured examples are qualitative.
- Formal significance: N≈40 over 4 categories; intervals are effect-size uncertainty, and the empirical tail is a rank statistic, not a classical p-value.
- Generalisation to other models, categories or datasets.
- Absolute literature novelty; see section M for the narrower positioning.

## M. Comparison with closest prior work

| Work | Industrial anomaly domain? | VLM? | Tests necessity of defect? | Tests sufficiency of defect? | Controls intervention magnitude/artifacts? | Main difference from CERM-AD |
|---|---|---|---|---|---|---|
| See What's Wrong (Cholopoulou & Iakovidis) | yes (MVTec, VisA) | no (anomaly detectors) | restoration as explanation | not as described | studies restoration fidelity | Counterfactual restoration explains anomaly detectors; it does not test whether a VLM's final verdict depends specifically on the physical defect relative to matched intervention controls. |
| EDCT (Ding, Vasa, Ramadwar) | no (general VLM) | yes | for explanation-mentioned concepts | not as described | not as described | Causal testing of explanation-mentioned concepts; not industrial-anomaly specific and not GT-defect necessity plus sufficiency. |
| Attention Without Grounding (Sadanandan & Behzadan) | no (medical) | yes | occlusion of salient regions | not as described | not as described | Validates saliency with causal patch occlusion; not defect restoration with intervention-dose matching and reverse insertion. |
| Reasoning faithfulness with multimodal perturbations (Moll et al.) | no (medical) | yes | partial | not as described | not as described | Evaluates reasoning faithfulness under perturbation; not localized industrial defect evidence with matched controls. |
| On the Faithfulness of Visual Thinking (Liu et al.) | no | yes | of thought components | not as described | not as described | Interventions on multimodal thought components; not localized industrial defect evidence. |
| Think-to-Detect / ThinkAnomaly | yes | yes | no | no | no | Improves anomaly reasoning / performance; does not causally audit the evidence the verdict uses. |
| IAD-R1 | yes | yes | no | no | no | Improves industrial anomaly reasoning / localization; does not causally audit the evidence the verdict uses. |
| CERM-AD Stage I+ (this work) | yes (MVTec AD, oracle masks) | yes (Qwen2.5-VL) | yes (CEE_N) | yes (CEE_S) | yes: edit-energy and seam matched before scoring | Defect-specific necessity and sufficiency with matched, artifact-controlled controls, auditing the VLM's final verdict. |

Cells summarise our current reading of each paper and should be verified against the papers before citation. These works solve different problems; none is claimed inferior.

## N. Research claim we can currently defend

We propose and evaluate bidirectional, artifact-controlled causal verification of localized defect evidence in VLM anomaly verdicts. In an oracle-mask pilot (Qwen2.5-VL-3B, 40 MVTec anomalies) the pooled evidence supports sufficiency but not both directions; per-target grounding is heterogeneous across categories.
