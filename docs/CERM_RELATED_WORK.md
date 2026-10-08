# CERM-AD and closely related work

**Scope of the claim.** We do *not* claim to be the first to use counterfactual images in
anomaly detection, nor the first to use interventions to test VLM faithfulness. Both
exist. What we study is narrower:

> Bidirectional, artifact-controlled causal verification of localized defect evidence
> in VLM anomaly decisions.

To the best of our current literature review, the works below do not combine
industrial pixel-level defect annotations, defect-specific counterfactual restoration,
matched irrelevant controls, explicit control of intervention magnitude/artifacts, and
reverse defect insertion (sufficiency) to audit an industrial VLM's final verdict.
The table records our current reading of each paper; cells marked "not as described"
or "partial" should be verified against the paper before citation. These methods
solve different problems and none is claimed inferior.

The same table is generated from `tzsad/causal/related_work.py` into
`artifacts/cerm_v2_pilot_public/related_work_table.csv`.

| Work | Industrial anomaly domain? | VLM? | Counterfactual image intervention? | Matched irrelevant controls? | Controls intervention magnitude/artifacts? | Tests necessity of defect? | Tests sufficiency of defect? | Tests final verdict causally? | Main difference from CERM-AD |
|---|---|---|---|---|---|---|---|---|---|
| **See What's Wrong** — Cholopoulou & Iakovidis, IEEE OJSP 2026, doi:10.1109/OJSP.2026.3712191 | yes (MVTec, VisA) | no (anomaly detectors) | yes: energy-guided normal restoration | partial: oracle masks and random same-ratio masking | studies restoration fidelity | restoration as explanation | not as described | not of a VLM verdict | Counterfactual restoration explains anomaly detectors; does not test whether a VLM's final verdict depends specifically on the physical defect relative to matched intervention controls. |
| **EDCT** — Ding, Vasa, Ramadwar, arXiv:2510.00047 | no (general VLM) | yes | yes: edits concepts named in the explanation | not as described | not as described | for explanation-mentioned concepts | not as described | yes (answer/explanation change) | Causal testing of explanation-mentioned concepts; not industrial-anomaly specific and not GT-defect necessity + sufficiency. |
| **Attention Without Grounding** — Sadanandan & Behzadan, arXiv:2607.18577, MICCAI iMIMIC 2026 | no (medical) | yes | yes: 16x16 patch occlusion | yes: random / shifted regions | not as described | occlusion of salient regions | not as described | yes | Validates saliency with causal patch occlusion; not defect restoration with intervention-dose matching and reverse insertion. |
| **Reasoning faithfulness with multimodal perturbations** — Moll et al., PMLR 297, 2026 | no (medical) | yes | yes: multimodal perturbations | not as described | not as described | partial | not as described | partial | Reasoning faithfulness under perturbation; not localized industrial defect evidence with matched controls. |
| **On the Faithfulness of Visual Thinking** — Liu et al., arXiv:2510.23482 | no | yes | intervenes on visual/textual reasoning components | not as described | not as described | of thought components | not as described | yes | Interventions on multimodal thought components; not localized industrial defect evidence. |
| **Think-to-Detect / ThinkAnomaly** — Mathematics 2025, doi:10.3390/math13243920 | yes | yes | no | no | no | no | no | no | Improves anomaly reasoning/performance; does not causally audit the evidence the verdict uses. |
| **IAD-R1** — AAAI 2026, doi:10.1609/aaai.v40i8.37588 | yes | yes | no | no | no | no | no | no | Improves industrial anomaly reasoning/localization; does not causally audit the evidence the verdict uses. |
| **CERM-AD Stage I+ (this work)** | yes (MVTec AD, oracle masks) | yes (Qwen2.5-VL) | yes: aligned normal-donor fill and defect transplantation | yes: translated exact GT mask, context matched | yes: edit-energy and seam matched before scoring | yes (CEE_N) | yes (CEE_S) | yes (direct-verdict log-odds) | Defect-specific necessity and sufficiency with matched, artifact-controlled controls, auditing the VLM's final verdict. |

## Narrow conclusions

* **See What's Wrong:** counterfactual restoration for explaining anomaly detectors; it
  does not test whether a VLM's final verdict depends specifically on the physical defect
  relative to matched intervention controls.
* **EDCT:** causal testing of explanation-mentioned concepts; not industrial-anomaly
  specific and not GT-defect necessity + sufficiency.
* **Attention Without Grounding:** causal patch occlusion and saliency validation; not
  defect restoration with intervention-dose matching and reverse insertion.
* **Visual Thinking faithfulness:** interventions on multimodal thought components; not
  localized industrial defect evidence.
* **ThinkAnomaly / IAD-R1:** improve reasoning and anomaly performance; they do not
  causally audit the evidence used by the verdict.
* **CERM Stage I+:** defect-specific necessity and sufficiency, matched controls,
  artifact-magnitude controls, VLM final-verdict audit.

Our occlusion arm is an *occlusion-style baseline* in the spirit of ROI/patch deletion;
it does not reproduce any specific paper's protocol.
