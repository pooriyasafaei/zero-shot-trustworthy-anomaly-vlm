"""The auto-generated presentation write-up and the probe-agreement check.

Two jobs that both exist to keep the slide deck honest:

* ``write_presentation_summary`` turns the tables into a markdown page whose
  findings section is generated *only* from the numbers actually measured.
* ``probe_agreement`` measures how closely the fast direct-verdict probe tracks the
  repository's existing structured Qwen score. The two are different measurements,
  and a pilot that quietly assumed otherwise would be making an unchecked claim.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..utils.logging import get_logger

log = get_logger("causal.presentation")

#: Below this Spearman correlation the direct probe is not tracking the structured
#: scorer well enough for the pilot to be read as a statement about that pipeline.
AGREEMENT_WARN_RHO = 0.5


def probe_agreement(records: pd.DataFrame, existing_path: str | Path) -> tuple[pd.DataFrame, str]:
    """Compare direct-probe P(YES) against an existing structured Qwen record file.

    Joins strictly on ``image_id``, the repository's stable key. Returns the table
    and a human-readable verdict; a weak correlation produces a loud warning rather
    than being quietly omitted.
    """
    from scipy import stats

    path = Path(existing_path)
    if not path.exists():
        return pd.DataFrame(), f"no existing record file at {path}"
    existing = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    if "anomaly_score" not in existing.columns:
        return pd.DataFrame(), f"{path.name} has no anomaly_score column"

    merged = records.merge(existing[["image_id", "anomaly_score"]], on="image_id", how="inner")
    merged = merged.dropna(subset=["audit_p_yes_original", "anomaly_score"])
    if len(merged) < 3:
        return pd.DataFrame(), f"only {len(merged)} images overlap with {path.name}"

    direct = merged["audit_p_yes_original"].to_numpy(dtype=float)
    structured = merged["anomaly_score"].to_numpy(dtype=float)
    rho = float(stats.spearmanr(direct, structured).statistic)
    r = float(stats.pearsonr(direct, structured).statistic)
    agreement = float(((direct >= 0.5) == (structured >= 0.5)).mean())

    table = pd.DataFrame([{
        "n_overlap": int(len(merged)), "spearman": rho, "pearson": r,
        "decision_agreement": agreement,
        "direct_mean_p_yes": float(direct.mean()),
        "structured_mean_score": float(structured.mean()),
        "source": str(path),
    }])
    verdict = (f"Spearman {rho:+.3f}, Pearson {r:+.3f}, verdict agreement {agreement:.1%} "
               f"on {len(merged)} shared images")
    if rho < AGREEMENT_WARN_RHO:
        verdict = ("WARNING: the direct causal probe does NOT track the structured Qwen "
                   f"scorer closely ({verdict}). Stage-I conclusions describe the direct "
                   "probe and must not be transferred to the structured pipeline without "
                   "re-running in generated_verdict mode.")
        log.warning(verdict)
    return table, verdict


def _fmt(value, spec: str = "+.4f") -> str:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return "n/a" if not np.isfinite(f) else format(f, spec)


def write_presentation_summary(
    path: str | Path,
    overall: pd.DataFrame,
    by_category: pd.DataFrame,
    decision: pd.DataFrame,
    artifacts: pd.DataFrame,
    meta: dict,
    agreement_verdict: str = "",
) -> Path:
    """Write ``PRESENTATION_SUMMARY.md``.

    The findings section is assembled from measured values only. Where the numbers
    do not support a claim - too few placebos to resolve 0.05, a seam ratio that
    could explain the effect - the text says so instead of rounding it away.
    """
    path = Path(path)
    row = overall.iloc[0].to_dict() if len(overall) else {}
    n = int(row.get("n_targets", 0))
    cee_med = float(row.get("cee_median", float("nan")))
    lo = float(row.get("cee_median_ci_lo", float("nan")))
    hi = float(row.get("cee_median_ci_hi", float("nan")))
    frac_pos = float(row.get("frac_cee_positive", float("nan")))
    frac_all = float(row.get("frac_beats_all_placebos", float("nan")))
    med_p = float(row.get("median_empirical_placebo_p", float("nan")))
    med_k = float(row.get("median_n_valid_placebos", float("nan")))
    resolvable = int(row.get("n_resolvable_at_005", 0))
    support_rate = float(row.get("supported_at_005_rate", float("nan")))

    def _artifact(name: str) -> float:
        if not len(artifacts):
            return float("nan")
        row = artifacts[artifacts.intervention_type == name]
        return float(row.iloc[0]["seam_score_mean"]) if len(row) else float("nan")

    seam_ratio = _artifact("PAIRED_median_seam_ratio")
    energy_ratio = _artifact("PAIRED_median_edit_energy_ratio")
    spearman_row = artifacts[artifacts.intervention_type.astype(str).str.startswith("SPEARMAN")] \
        if len(artifacts) else pd.DataFrame()
    seam_rho = float(spearman_row.iloc[0]["seam_score_mean"]) if len(spearman_row) else float("nan")

    findings: list[str] = []
    if n == 0:
        findings.append("- No target images completed; there is nothing to report yet.")
    else:
        direction = "larger" if cee_med > 0 else "smaller"
        findings.append(
            f"- Across {n} anomalous images, removing the true defect changed the model's "
            f"anomaly log-odds by a median of {_fmt(row.get('delta_defect_median'))} nats, "
            f"against {_fmt(row.get('placebo_delta_median'))} for matched placebo edits.")
        findings.append(
            f"- Median CEE is {_fmt(cee_med)} nats (95% bootstrap CI "
            f"[{_fmt(lo)}, {_fmt(hi)}]), i.e. the real-defect effect is {direction} than the "
            f"matched-placebo effect. {frac_pos:.0%} of images have CEE > 0 and "
            f"{frac_all:.0%} beat every one of their placebos.")
        if np.isfinite(lo) and np.isfinite(hi):
            if lo > 0:
                findings.append("- The CI for the median CEE excludes zero, so on this sample the "
                                "verdict does depend on the defect region more than on comparable "
                                "irrelevant edits.")
            elif hi < 0:
                findings.append("- The CI for the median CEE lies below zero: removing the real "
                                "defect moved the model LESS than matched placebo edits did, "
                                "which is evidence against defect-grounded decisions.")
            else:
                findings.append("- The CI for the median CEE spans zero, so this pilot does not "
                                "resolve the question in either direction.")
        findings.append(
            f"- Median empirical matched-placebo tail probability is {_fmt(med_p, '.4f')} "
            f"with a median of {med_k:.0f} valid placebos per image.")
        yes_rate = float(row.get("frac_original_predicted_yes", float("nan")))
        if np.isfinite(yes_rate):
            findings.append(
                f"- **Detector context:** every target here is anomalous, yet the direct probe "
                f"called only {yes_rate:.0%} of them defective. The audit therefore describes a "
                f"decision process that is itself close to chance on this subset, which limits "
                f"how much any causal conclusion can carry.")
        if len(by_category) > 1 and "cee_median" in by_category:
            best = by_category.loc[by_category["cee_median"].idxmax()]
            worst = by_category.loc[by_category["cee_median"].idxmin()]
            findings.append(
                f"- **Strong heterogeneity across categories:** {best['group']} reaches a median "
                f"CEE of {_fmt(best['cee_median'])} while {worst['group']} sits at "
                f"{_fmt(worst['cee_median'])}. The pooled median hides this; the per-category "
                f"table is the one to read.")
        if resolvable:
            findings.append(
                f"- Of {resolvable} images with enough placebos to resolve alpha=0.05, "
                f"{support_rate:.0%} reach that level.")
        else:
            findings.append("- No image had enough valid placebos to resolve alpha=0.05; "
                            "`supported_at_005` is undefined (NA) throughout.")
        if np.isfinite(seam_ratio):
            if seam_ratio > 1.5:
                findings.append(
                    f"- **CAUTION — the central confound.** Real-defect reconstructions are "
                    f"seamier than their own placebos (paired median ratio {seam_ratio:.2f}"
                    + (f", edit-energy ratio {energy_ratio:.2f}" if np.isfinite(energy_ratio) else "")
                    + "). This is partly intrinsic: a defect region is by definition visually "
                    "distinctive, so replacing it necessarily changes more pixels than "
                    "replacing a patch of ordinary texture. Some of the measured effect may "
                    "therefore be a reaction to the edit itself rather than to the absence of "
                    "the defect.")
            else:
                findings.append(
                    f"- Reconstruction artifact severity is comparable between the real-defect "
                    f"and placebo edits (paired median seam ratio {seam_ratio:.2f}), so the "
                    f"effect is not obviously explained by splice quality.")
        if np.isfinite(seam_rho):
            direction = ("does not support" if seam_rho < 0 else "is consistent with")
            findings.append(
                f"- Across images, the Spearman correlation between CEE and the seam ratio is "
                f"{_fmt(seam_rho, '+.3f')}. A positive value would mean the effect grows with "
                f"artifact severity; this value {direction} the artifact explanation.")

    lines = [
        "# CERM-AD Stage I — Causal Visual Evidence Verification",
        "",
        "Pilot result: does the VLM's anomaly verdict causally depend on the actual defect,",
        "or is it merely confident / spatially correlated with it?",
        "",
        "## Run",
        "",
        "| field | value |",
        "|---|---|",
        f"| git SHA | `{meta.get('git_sha', 'unknown')}` |",
        f"| model | `{meta.get('model_id', '')}` |",
        f"| scoring mode | `{meta.get('score_mode', '')}` |",
        f"| mask source | `{meta.get('mask_source', '')}` (oracle GT) |",
        f"| categories | {meta.get('categories', '')} |",
        f"| target anomalies (N) | {n} |",
        f"| placebos requested (K) | {meta.get('k_placebos', '')} |",
        f"| median valid placebos | {med_k:.0f}" + " |",
        f"| seed | {meta.get('seed', '')} |",
        f"| forward passes | {meta.get('forward_passes', 'n/a')} |",
        f"| runtime | {meta.get('runtime_minutes', 'n/a')} min |",
        "",
        "## Headline numbers",
        "",
        "| metric | value |",
        "|---|---|",
        f"| median delta_defect | {_fmt(row.get('delta_defect_median'))} |",
        f"| median placebo delta | {_fmt(row.get('placebo_delta_median'))} |",
        f"| **median CEE** | **{_fmt(cee_med)}** [{_fmt(lo)}, {_fmt(hi)}] |",
        f"| mean CEE | {_fmt(row.get('cee_mean'))} "
        f"[{_fmt(row.get('cee_mean_ci_lo'))}, {_fmt(row.get('cee_mean_ci_hi'))}] |",
        f"| fraction CEE > 0 | {frac_pos:.1%} |" if np.isfinite(frac_pos) else "| fraction CEE > 0 | n/a |",
        f"| fraction beating all placebos | {frac_all:.1%} |" if np.isfinite(frac_all)
        else "| fraction beating all placebos | n/a |",
        f"| median empirical placebo p | {_fmt(med_p, '.4f')} |",
        f"| median robust CEE z | {_fmt(row.get('median_robust_cee_z'), '+.3f')} |",
        f"| model said YES on the original | "
        f"{float(row.get('frac_original_predicted_yes', float('nan'))):.1%} |",
        "",
        "## By category",
        "",
        by_category.to_markdown(index=False) if len(by_category) else "_no categories_",
        "",
        "## Split by the model's original verdict",
        "",
        decision.to_markdown(index=False) if len(decision) else "_not enough data_",
        "",
        "## Artifact controls",
        "",
        artifacts.to_markdown(index=False) if len(artifacts) else "_none_",
        "",
        "## Preliminary findings",
        "",
        *findings,
        "",
    ]
    if agreement_verdict:
        lines += ["## Direct probe vs the existing structured scorer", "",
                  f"{agreement_verdict}", ""]
    lines += [
        "## Important limitations",
        "",
        "- **This is an oracle study.** Interventions are built from MVTec ground-truth",
        "  masks. The masks are used *only* to construct the counterfactual - never in the",
        "  prompt, never in the model's input, never in the original score. A deployment",
        "  version must use a predicted anomaly region, and nothing here shows that",
        "  predicted regions would behave the same way.",
        "- **Donor reconstruction can introduce artifacts.** The fill is an aligned normal",
        "  donor, not a generative inpainter. The artifact-control table exists precisely",
        "  because a model could react to a splice rather than to a missing defect.",
        "- **`empirical_placebo_p` is a matched-placebo tail probability, not a randomisation",
        "  p-value.** Reading it as one assumes the placebo placements are exchangeable with",
        "  the true defect placement under the null. Matched translation of the same mask",
        "  makes that plausible; it does not establish it.",
        "- **The direct-verdict probe is a different measurement** from the repository's",
        "  structured generate-then-answer scorer. Where the two were compared, the",
        "  correlation is reported above; where they were not, no equivalence is claimed.",
        "- **Pilot sample size.** These are preliminary numbers on a small deliberately",
        "  capped subset, intended to test whether the methodology detects anything at all.",
        "- This is evidence about the *method*, not a publication-grade result about the",
        "  model.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
    log.info("wrote %s", path)
    return path
