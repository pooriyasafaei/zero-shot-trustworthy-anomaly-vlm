"""Sufficiency: the reverse causal test, by defect transplantation (CERM-AD v2).

Necessity asks whether *removing* the defect lowers the model's anomaly belief more
than comparable edits do. It cannot tell a model that keys on the defect from one
that merely reacts to any disturbance of a particular region. The reverse question
closes that gap: does *inserting* the real defect into a normal image raise anomaly
belief more than inserting comparable normal content at the same place?

For anomalous image ``I`` with mask ``M`` and its normal donor ``J`` (the same
train/good image Stage I already selected):

    J_defect   = J with I[M] transplanted into M          (inward-only feather)
    J_sham_k   = J with S_k[M] transplanted into M         S_k: other train/good images
    delta_insert_defect = z(J_defect) - z(J)
    delta_insert_sham_k = z(J_sham_k) - z(J)
    CEE_S = delta_insert_defect - median_k(delta_insert_sham_k)

Every sham uses the **same recipient, the same mask, the same location and the same
feathering**, so only the inserted content differs. Sham sources are chosen, before
any model call, to match the real insertion's artifact profile (edit energy and seam
score), then drawn uniformly from the best fraction - never the K nearest.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..utils.logging import get_logger
from .artifact_matching import RATIO_EPS, MaskTemplate, log_ratio_distance, robust_scale
from .donor import DonorBank, aligned_donor_pixels
from .placebo import _stable_rng

log = get_logger("causal.sufficiency")


@dataclass(frozen=True)
class Sham:
    """One normal-patch sham insertion source, with its matching diagnostics."""

    index: int
    source_image_id: str
    source_path: str
    edit_energy: float
    seam_score: float
    edit_ratio_to_defect: float
    seam_ratio_to_defect: float
    artifact_match_distance: float

    def as_dict(self) -> dict:
        """Flat dict for the interventions table."""
        return {"arm": "sham", "placebo_index": self.index,
                "sham_source_image_id": self.source_image_id,
                "sham_source_path": self.source_path,
                "candidate_edit_energy": self.edit_energy,
                "candidate_seam_score": self.seam_score,
                "edit_ratio_to_defect": self.edit_ratio_to_defect,
                "seam_ratio_to_defect": self.seam_ratio_to_defect,
                "artifact_match_distance": self.artifact_match_distance}


@dataclass
class ShamPlan:
    """Shams chosen for one target, and why there may be fewer than requested."""

    shams: list[Sham] = field(default_factory=list)
    n_requested: int = 0
    n_sources: int = 0
    n_pool: int = 0
    quality: str = "ok"
    reason: str = ""

    @property
    def n_valid(self) -> int:
        return len(self.shams)

    def as_dict(self) -> dict:
        return {"sham_n_valid": self.n_valid, "sham_n_requested": self.n_requested,
                "sham_n_sources": self.n_sources, "sham_n_pool": self.n_pool,
                "sham_quality": self.quality, "sham_reason": self.reason,
                "sham_tail_resolution": 1.0 / (self.n_valid + 1)}


class SourceBank:
    """Full-resolution train/good images aligned to a target resolution.

    Built on :class:`DonorBank`, which refuses any non-normal row, so an anomalous
    image can never become a sham source. Keeps one category's aligned arrays in
    memory at a time, which is how the sorted target list is traversed.
    """

    def __init__(self, bank: DonorBank, max_sources: int = 60) -> None:
        self.bank = bank
        self.max_sources = int(max_sources)
        self._key: tuple | None = None
        self._arrays: list[tuple[str, str, np.ndarray]] = []

    def sources(self, category: str, shape: tuple[int, int]) -> list[tuple[str, str, np.ndarray]]:
        """``[(image_id, path, aligned uint8 RGB), ...]`` in deterministic index order."""
        key = (category, tuple(shape))
        if key != self._key:
            pool: pd.DataFrame = self.bank.candidates(category)
            if self.max_sources and len(pool) > self.max_sources:
                pool = pool.iloc[:self.max_sources]
            self._arrays = [(str(r.image_id), str(r.path), aligned_donor_pixels(r.path, shape))
                            for r in pool.itertuples(index=False)]
            self._key = key
        return self._arrays


def select_shams(recipient: np.ndarray, mask: np.ndarray, defect_edit: float,
                 defect_seam: float, sources: list[tuple[str, str, np.ndarray]],
                 recipient_id: str, k: int, seed: int, image_id: str,
                 match_quantile: float = 0.3, feather_px: int = 3,
                 template: MaskTemplate | None = None) -> ShamPlan:
    """Choose ``k`` sham sources whose insertion artifacts resemble the real one.

    The recipient itself is excluded (inserting J into J is the identity, not a
    sham). Pixel metrics only - this function never sees a model score.
    """
    plan = ShamPlan(n_requested=k)
    cands = [(sid, spath, arr) for sid, spath, arr in sources if sid != recipient_id]
    plan.n_sources = len(cands)
    if not cands or not mask.any():
        plan.quality, plan.reason = "degenerate", "no normal sham sources available"
        return plan
    tmpl = template if template is not None else MaskTemplate(mask, feather_px)
    edits = np.empty(len(cands))
    seams = np.empty(len(cands))
    for i, (_, _, arr) in enumerate(cands):
        edits[i], seams[i] = tmpl.metrics(recipient, arr)
    d_edit = np.array([log_ratio_distance(e, defect_edit) for e in edits])
    d_seam = np.array([log_ratio_distance(s, defect_seam) for s in seams])
    composite = np.sqrt((d_edit / robust_scale(d_edit)) ** 2
                        + (d_seam / robust_scale(d_seam)) ** 2)

    order = np.argsort(composite, kind="stable")
    n_keep = max(int(round(len(cands) * match_quantile)), min(k, len(cands)))
    pool = list(order[:n_keep])
    plan.n_pool = len(pool)
    rng = _stable_rng(seed, image_id, int(mask.sum()), k, "sham-arm")
    for i, j in enumerate(rng.permutation(len(pool))[:k]):
        j = int(pool[int(j)])
        sid, spath, _ = cands[j]
        plan.shams.append(Sham(
            i, sid, spath, float(edits[j]), float(seams[j]),
            float((edits[j] + RATIO_EPS) / (defect_edit + RATIO_EPS)),
            float((seams[j] + RATIO_EPS) / (defect_seam + RATIO_EPS)), float(composite[j])))
    if plan.n_valid < k:
        plan.quality = "few_sources"
        plan.reason = f"only {plan.n_valid} sham sources for k={k} (sources={len(cands)})"
        log.warning("%s: %s", image_id, plan.reason)
    return plan


def source_array(sources: list[tuple[str, str, np.ndarray]], source_id: str) -> np.ndarray:
    """Look up an aligned source array by image id."""
    for sid, _, arr in sources:
        if sid == source_id:
            return arr
    raise KeyError(source_id)
