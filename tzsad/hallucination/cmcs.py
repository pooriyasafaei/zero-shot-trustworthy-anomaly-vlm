"""Cross-Modal Consistency Score, in two arms.

``language_grounding`` (primary, the reframed version)
    Does the *claimed* defect location contain patches that actually deviate from
    the normal manifold? If the VLM asserts a crack in the upper left and no
    window in the upper left is unusual, the claim is ungrounded.

``global_spatial`` (comparison arm, closest to the proposal as written)
    Rank-normalised global semantic score vs rank-normalised spatial evidence
    score. Kept so the paper can report what the original formulation buys.

Both branches are **rank-normalised within category before differencing**: the
semantic score is a softmax probability and the spatial score is a cosine
distance, and averaging or subtracting them raw compares two different units.
The proposal's Eq. 2 also averages ``p_i * s_i`` over N detections, which shrinks
mechanically as N grows; rank normalisation removes that artefact too.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..uncertainty.base import rank_normalize
from .location import LocationClaim, attended_region, parse_location, region_iou


@dataclass
class CMCSConfig:
    """Knobs for both CMCS arms."""

    map_size: int = 128
    spatial_top_frac: float = 0.05      # fraction of the map defining "the evidence"
    evidence_quantile: float = 0.99     # quantile of the map used as the global spatial score
    min_localized_iou: float = 0.05     # IoU below which a claim counts as ungrounded
    edge_width: float = 0.18


def spatial_evidence_score(anomaly_map: np.ndarray, quantile: float = 0.99) -> float:
    """A single scalar of "how much of this image deviates from normal".

    A high quantile rather than the mean: a small defect moves the mean almost not
    at all, which is why mean-pooled patch scores under-detect on MVTec.
    """
    return float(np.quantile(anomaly_map, quantile))


def claim_grounding(anomaly_map: np.ndarray, claim: LocationClaim, config: CMCSConfig) -> dict:
    """How well the claimed region overlaps the actually-deviating region.

    Returns the IoU, the mean map value inside the claim, and the ratio of that to
    the mean outside - the last being the most interpretable number: >1 means the
    claimed region really is more unusual than the rest of the image.
    """
    size = anomaly_map.shape[0]
    evidence = attended_region(anomaly_map, config.spatial_top_frac)
    if claim.is_localized:
        claimed = claim.mask(size, config.edge_width)
    else:
        claimed = np.ones_like(evidence)
    inside = anomaly_map[claimed.astype(bool)]
    outside = anomaly_map[~claimed.astype(bool)]
    inside_mean = float(inside.mean()) if inside.size else float("nan")
    outside_mean = float(outside.mean()) if outside.size else float("nan")
    return {
        "claim_kind": claim.kind,
        "claim_localized": bool(claim.is_localized),
        "claim_iou": region_iou(claimed, evidence),
        "claim_inside_mean": inside_mean,
        "claim_outside_mean": outside_mean,
        "claim_contrast": (inside_mean / outside_mean) if outside_mean and outside_mean > 0 else float("nan"),
    }


def compute_cmcs(records: pd.DataFrame, maps: dict[str, np.ndarray],
                 config: CMCSConfig | None = None, arm: str = "language_grounding",
                 group_col: str = "category") -> pd.DataFrame:
    """Attach CMCS columns to records.

    Parameters
    ----------
    records:
        Per-image records; the VLM arm needs ``predicted_location``.
    maps:
        ``{image_id: HxW}`` normal-manifold deviation maps (the ``bank`` maps).
    arm:
        ``language_grounding`` or ``global_spatial``.

    Returns
    -------
    A copy of ``records`` with ``cmcs``, the arm's intermediate columns, and (for
    the language arm) the parsed claim diagnostics.
    """
    config = config or CMCSConfig()
    out = records.copy()

    spatial = np.array([spatial_evidence_score(maps[i], config.evidence_quantile)
                        if i in maps else np.nan for i in out["image_id"]])
    out["spatial_score"] = spatial
    s_sem_rank = rank_normalize(out["anomaly_score"], out[group_col])
    s_spa_rank = rank_normalize(out["spatial_score"], out[group_col])
    out["s_sem_rank"] = s_sem_rank.to_numpy()
    out["s_spatial_rank"] = s_spa_rank.to_numpy()

    if arm == "global_spatial":
        # Agreement between the two modalities: 1 when the ranks coincide.
        out["cmcs"] = 1.0 - (s_sem_rank - s_spa_rank).abs().to_numpy()
        out["cmcs_arm"] = "global_spatial"
        return out

    if arm != "language_grounding":
        raise KeyError(f"unknown CMCS arm {arm!r}")

    diag = []
    for _, r in out.iterrows():
        m = maps.get(r["image_id"])
        if m is None:
            diag.append({"claim_kind": "missing_map", "claim_localized": False,
                         "claim_iou": np.nan, "claim_inside_mean": np.nan,
                         "claim_outside_mean": np.nan, "claim_contrast": np.nan})
            continue
        claim = parse_location(str(r.get("predicted_location", "")))
        diag.append(claim_grounding(m, claim, config))
    out = pd.concat([out, pd.DataFrame(diag, index=out.index)], axis=1)

    # Grounded agreement: rank-normalised claim contrast, combined with the
    # rank agreement between the semantic verdict and the spatial evidence.
    contrast_rank = rank_normalize(out["claim_contrast"], out[group_col]).fillna(0.5)
    agreement = 1.0 - (s_sem_rank - s_spa_rank).abs()
    out["cmcs"] = (0.5 * contrast_rank + 0.5 * agreement).to_numpy()
    out["cmcs_arm"] = "language_grounding"
    return out


@dataclass
class FlagConfig:
    """Thresholds for the proposal's hallucination flag (Eq. 4) and attenuation (Eq. 5)."""

    theta_cmcs: float = 0.5      # theta_H: CMCS below this is "the branches disagree"
    theta_anom: float = 0.5      # s_sem above this is "the model asserted an anomaly"
    alpha: float = 1.5           # correction strength in s_adj = s_sem * CMCS^alpha
    calibrate_on_normals: bool = True
    cmcs_quantile: float = 0.10  # theta_H = this quantile of CMCS on normal calibration images
    anom_quantile: float = 0.90  # theta_anom = this quantile of s_sem on normal calibration images


def fit_flag_thresholds(cal_records: pd.DataFrame, config: FlagConfig,
                        group_col: str = "category") -> dict[str, tuple[float, float]]:
    """Fit ``(theta_H, theta_anom)`` per category on **normal-only** held-out images.

    The proposal says both thresholds are "determined on a held-out validation set
    of normal-only images" but not how. We fix them as quantiles of the normal
    distribution, which keeps the whole rule zero-shot and gives each threshold an
    interpretable meaning: ``theta_anom`` is the score a normal image exceeds only
    ``1 - anom_quantile`` of the time, and ``theta_H`` is the CMCS that normal
    images fall below only ``cmcs_quantile`` of the time.
    """
    if "label" in cal_records.columns and (cal_records["label"] != 0).any():
        raise ValueError("flag thresholds must be fitted on normal-only images")
    out: dict[str, tuple[float, float]] = {}
    for group, g in cal_records.groupby(group_col, sort=True):
        cmcs = g["cmcs"].to_numpy(dtype=float)
        sem = g["anomaly_score"].to_numpy(dtype=float)
        cmcs = cmcs[np.isfinite(cmcs)]
        sem = sem[np.isfinite(sem)]
        out[str(group)] = (
            float(np.quantile(cmcs, config.cmcs_quantile)) if cmcs.size else config.theta_cmcs,
            float(np.quantile(sem, config.anom_quantile)) if sem.size else config.theta_anom,
        )
    return out


def apply_flag_and_adjust(records: pd.DataFrame, config: FlagConfig | None = None,
                          thresholds: dict[str, tuple[float, float]] | None = None,
                          group_col: str = "category") -> pd.DataFrame:
    """Proposal Eq. 4 and Eq. 5: the hallucination flag and the attenuated score.

    Adds ``halluc_flag_eq4`` and ``s_adj``. ``s_adj`` is the non-conformity score
    the conformal layer should calibrate on, per §4.2 of the brief.

    Caveat worth stating in the paper: attenuating the score by ``CMCS^alpha``
    couples detection and trust into one number, so a drop in AUROC after
    attenuation cannot be told apart from a gain in trustworthiness. We therefore
    report ``anomaly_score`` and ``s_adj`` as two separate arms rather than
    silently replacing one with the other.
    """
    config = config or FlagConfig()
    out = records.copy()
    sem = out["anomaly_score"].to_numpy(dtype=float)
    cmcs = np.clip(out["cmcs"].to_numpy(dtype=float), 0.0, 1.0)

    if thresholds:
        t_cmcs = np.array([thresholds.get(str(c), (config.theta_cmcs, config.theta_anom))[0]
                           for c in out[group_col]])
        t_anom = np.array([thresholds.get(str(c), (config.theta_cmcs, config.theta_anom))[1]
                           for c in out[group_col]])
    else:
        t_cmcs = np.full(len(out), config.theta_cmcs)
        t_anom = np.full(len(out), config.theta_anom)

    flag = (cmcs < t_cmcs) & (sem > t_anom)
    out["halluc_flag_eq4"] = flag
    out["theta_cmcs"] = t_cmcs
    out["theta_anom"] = t_anom
    out["s_adj"] = np.where(flag, sem * np.power(np.maximum(cmcs, 1e-9), config.alpha), sem)
    return out


def cmcs_literal(records: pd.DataFrame, maps: dict[str, np.ndarray],
                 config: CMCSConfig | None = None) -> np.ndarray:
    """Proposal Eq. 3 verbatim: ``CMCS = 1 - |s_sem - s_spatial|``, unnormalised.

    Provided so the paper can quantify what rank normalisation buys. On raw scales
    this quantity is dominated by the offset between a softmax probability and a
    cosine distance, so it is expected to be near-constant within a category -
    which is the argument for the rank-normalised form used everywhere else.
    """
    config = config or CMCSConfig()
    spatial = np.array([spatial_evidence_score(maps[i], config.evidence_quantile)
                        if i in maps else np.nan for i in records["image_id"]])
    return 1.0 - np.abs(records["anomaly_score"].to_numpy(dtype=float) - spatial)
