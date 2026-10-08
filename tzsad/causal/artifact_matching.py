"""Context + artifact matched placebos, and the other necessity control arms (v2).

The Stage-I pilot found that removing the real defect produced a fill about 2x
"stronger" than its context-matched placebos (paired median seam ratio 2.08,
edit-energy ratio 2.01). A positive CEE could therefore partly be a response to
*intervention magnitude* rather than to the absence of the defect.

Stage I+ fixes that at selection time, **before any model is queried**: for every
admissible translation of the defect mask we build the placebo fill from the same
donor and measure, from pixels only,

* ``context_distance`` - how unlike the defect's surroundings the location is,
* ``d_edit = |log((edit_p + eps) / (edit_defect + eps))|``,
* ``d_seam = |log((seam_p + eps) / (seam_defect + eps))|``,

then rank candidates by a composite distance and draw K from the best fraction.

Composite distance (predeclared, not tuned on model output)::

    match_distance = sqrt( (ctx / s_ctx)^2 + (d_edit / s_edit)^2 + (d_seam / s_seam)^2 )

where each ``s`` is a robust scale (1.4826 x MAD) of that term over the target's
candidates. Each term is a *distance* (0 = perfect match), so it is scaled but
deliberately **not centred**: centring at the median would give better-than-median
candidates a negative z whose square penalises them exactly as much as worse ones.

Also here: the random-translation control (arm A) and the occlusion-style baseline
fill (arm D).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from ..utils.logging import get_logger
from .placebo import _stable_rng, draw_separated
from .reconstruction import apply_intervention

log = get_logger("causal.artifact_matching")

#: Guards the log-ratios when an edit is (near) zero.
RATIO_EPS = 1e-3

#: Pixels around a translated mask that the exact local metrics need: 3 for the
#: seam band, 1 for the gradient stencil, plus slack.
LOCAL_MARGIN = 6


def robust_scale(values: Sequence[float]) -> float:
    """1.4826 x MAD, falling back to the std and then to 1 for a degenerate sample."""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 1.0
    mad = float(np.median(np.abs(arr - np.median(arr)))) * 1.4826
    if mad > 1e-12:
        return mad
    sd = float(arr.std())
    return sd if sd > 1e-12 else 1.0


def log_ratio_distance(value: float, reference: float, eps: float = RATIO_EPS) -> float:
    """``|log((value + eps) / (reference + eps))|`` - symmetric in over/under-shoot."""
    return float(abs(np.log((float(value) + eps) / (float(reference) + eps))))


def local_metrics(original: np.ndarray, fill: np.ndarray, mask_ys: np.ndarray,
                  mask_xs: np.ndarray, dy: int = 0, dx: int = 0,
                  feather_px: int = 3) -> tuple[float, float]:
    """``(edit_energy, seam_score)`` of filling the translated mask from ``fill``.

    Identical to running :func:`apply_intervention` on the full image, but computed
    on a crop around the translated mask so that scoring a thousand candidates per
    target stays cheap. The crop keeps ``LOCAL_MARGIN`` pixels of context, which
    covers the seam band, the gradient stencil and the distance transform (the
    nearest background pixel of any mask pixel is always within the bounding box
    grown by one). Where the crop meets the image border it is the image border, so
    border handling is unchanged. A unit test checks equality with the full-image
    computation.
    """
    h, w = original.shape[:2]
    ys, xs = mask_ys + int(dy), mask_xs + int(dx)
    y0 = max(int(ys.min()) - LOCAL_MARGIN, 0)
    y1 = min(int(ys.max()) + LOCAL_MARGIN + 1, h)
    x0 = max(int(xs.min()) - LOCAL_MARGIN, 0)
    x1 = min(int(xs.max()) + LOCAL_MARGIN + 1, w)
    crop_mask = np.zeros((y1 - y0, x1 - x0), dtype=bool)
    crop_mask[ys - y0, xs - x0] = True
    iv = apply_intervention(original[y0:y1, x0:x1], fill[y0:y1, x0:x1], crop_mask,
                            "candidate", feather_px)
    return iv.edit_energy, iv.seam_score


class MaskTemplate:
    """Precomputed feather weights and seam band for one mask shape, for exact fast fills.

    Feather weights (a distance transform) and the seam band (dilation minus erosion)
    are translation-invariant as long as the mask's working window - its bounding box
    grown by ``LOCAL_MARGIN`` - lies fully inside the image. They are computed once
    on a zero-padded window and reused for every such placement, which turns each
    candidate into a blend plus two small gradients. A placement whose window touches
    the image border falls back to the reference full computation, so results are
    always identical to :func:`apply_intervention` (unit-tested).
    """

    def __init__(self, mask: np.ndarray, feather_px: int = 3) -> None:
        from scipy import ndimage

        from .reconstruction import _feather_weights

        self.shape = mask.shape
        self.feather_px = int(feather_px)
        ys, xs = np.nonzero(mask)
        self.ys, self.xs = ys, xs
        self.y0, self.x0 = int(ys.min()) - LOCAL_MARGIN, int(xs.min()) - LOCAL_MARGIN
        hh = int(ys.max()) - int(ys.min()) + 1 + 2 * LOCAL_MARGIN
        ww = int(xs.max()) - int(xs.min()) + 1 + 2 * LOCAL_MARGIN
        local = np.zeros((hh, ww), dtype=bool)
        local[ys - self.y0, xs - self.x0] = True
        self.local = local
        self.weights = _feather_weights(local, feather_px)[..., None]
        self.band = (ndimage.binary_dilation(local, iterations=3)
                     & ~ndimage.binary_erosion(local, iterations=3))
        self.h, self.w = hh, ww

    def window(self, dy: int, dx: int) -> tuple[int, int] | None:
        """Top-left of the working window at offset ``(dy, dx)``, or None if clipped."""
        y0, x0 = self.y0 + int(dy), self.x0 + int(dx)
        if y0 < 0 or x0 < 0 or y0 + self.h > self.shape[0] or x0 + self.w > self.shape[1]:
            return None
        return y0, x0

    @staticmethod
    def _grad(img: np.ndarray) -> np.ndarray:
        grey = img.astype(np.float32).mean(axis=2)
        gy, gx = np.gradient(grey)
        return np.hypot(gx, gy)

    def _blend(self, original: np.ndarray, fill: np.ndarray, y0: int, x0: int):
        o = original[y0:y0 + self.h, x0:x0 + self.w]
        f = fill[y0:y0 + self.h, x0:x0 + self.w]
        out = np.clip(np.rint(o.astype(np.float32) * (1.0 - self.weights)
                              + f.astype(np.float32) * self.weights), 0, 255).astype(np.uint8)
        out[~self.local] = o[~self.local]
        return o, out

    def metrics(self, original: np.ndarray, fill: np.ndarray, dy: int = 0,
                dx: int = 0) -> tuple[float, float]:
        """``(edit_energy, seam_score)`` exactly as :func:`apply_intervention` reports."""
        win = self.window(dy, dx)
        if win is None:
            return local_metrics(original, fill, self.ys, self.xs, dy, dx, self.feather_px)
        o, out = self._blend(original, fill, *win)
        edit = float(np.abs(out[self.local].astype(np.int32)
                            - o[self.local].astype(np.int32)).mean())
        seam = float(np.abs(self._grad(out)[self.band] - self._grad(o)[self.band]).mean()) \
            if self.band.any() else 0.0
        return edit, seam

    def apply(self, original: np.ndarray, fill: np.ndarray, kind: str, dy: int = 0,
              dx: int = 0):
        """Full-image :class:`Intervention`, built from the window when possible."""
        from .reconstruction import Intervention, apply_intervention

        win = self.window(dy, dx)
        full_mask = np.zeros(self.shape, dtype=bool)
        full_mask[self.ys + int(dy), self.xs + int(dx)] = True
        if win is None:
            return apply_intervention(original, fill, full_mask, kind, self.feather_px)
        y0, x0 = win
        o, out_win = self._blend(original, fill, y0, x0)
        out = original.copy()
        out[y0:y0 + self.h, x0:x0 + self.w] = out_win
        diff = out != original
        if diff.any() and (diff.any(axis=2) & ~full_mask).any():
            raise AssertionError("intervention modified pixels outside its mask")
        edit = float(np.abs(out_win[self.local].astype(np.int32)
                            - o[self.local].astype(np.int32)).mean())
        seam = float(np.abs(self._grad(out_win)[self.band] - self._grad(o)[self.band]).mean()) \
            if self.band.any() else 0.0
        return Intervention(kind=kind, image=out, mask=full_mask, edit_energy=edit,
                            outside_change_max=0.0, seam_score=seam,
                            mask_area=int(full_mask.sum()),
                            mask_fraction=float(full_mask.mean()), feather_px=self.feather_px)


@dataclass(frozen=True)
class MatchedPlacebo:
    """One placebo of a v2 necessity arm, with its pixel-level matching diagnostics."""

    arm: str
    index: int
    dx: int
    dy: int
    context_distance: float
    iou_with_defect: float
    mask_area: int
    edit_energy: float = float("nan")
    seam_score: float = float("nan")
    edit_ratio_to_defect: float = float("nan")
    seam_ratio_to_defect: float = float("nan")
    artifact_match_distance: float = float("nan")

    def as_dict(self) -> dict:
        """Flat dict for the interventions table."""
        return {"arm": self.arm, "placebo_index": self.index, "dx": self.dx, "dy": self.dy,
                "context_distance": self.context_distance,
                "iou_with_defect": self.iou_with_defect, "mask_area": self.mask_area,
                "candidate_edit_energy": self.edit_energy,
                "candidate_seam_score": self.seam_score,
                "edit_ratio_to_defect": self.edit_ratio_to_defect,
                "seam_ratio_to_defect": self.seam_ratio_to_defect,
                "artifact_match_distance": self.artifact_match_distance}


@dataclass
class ArmPlan:
    """Placebos chosen for one arm on one target, and why there may be fewer than K."""

    arm: str
    placebos: list[MatchedPlacebo] = field(default_factory=list)
    n_requested: int = 0
    n_feasible: int = 0
    n_pool: int = 0
    quality: str = "ok"           # ok | few_feasible | degenerate
    reason: str = ""

    @property
    def n_valid(self) -> int:
        return len(self.placebos)

    def as_dict(self) -> dict:
        """Per-target diagnostics, prefixed with the arm name."""
        p = f"{self.arm}_"
        return {p + "n_valid_placebos": self.n_valid, p + "n_requested": self.n_requested,
                p + "n_feasible": self.n_feasible, p + "n_pool": self.n_pool,
                p + "placebo_quality": self.quality, p + "placebo_reason": self.reason,
                p + "tail_resolution": 1.0 / (self.n_valid + 1)}


def _finish(plan: ArmPlan, k: int, image_id: str, min_separation: int) -> ArmPlan:
    if plan.n_valid < k and plan.quality == "ok":
        plan.quality = "few_feasible"
        plan.reason = (f"only {plan.n_valid} distinct placements for k={k} "
                       f"(feasible={plan.n_feasible}, pool={plan.n_pool}, "
                       f"min_separation={min_separation})")
        log.warning("%s [%s]: %s", image_id or "image", plan.arm, plan.reason)
    return plan


def random_placebos(feasible: Sequence[tuple[float, int, int, float]], k: int, seed: int,
                    image_id: str, area: int, min_separation: int = 8) -> ArmPlan:
    """Arm A: uniform over every admissible translation, no matching at all.

    Same mask shape and area, same IoU and in-bounds constraints, same donor - only
    the similarity filter is removed. It measures what the matching buys.
    """
    plan = ArmPlan("random", n_requested=k, n_feasible=len(feasible), n_pool=len(feasible))
    if not feasible:
        plan.quality, plan.reason = "degenerate", "no admissible translation"
        return plan
    rng = _stable_rng(seed, image_id, area, k, "random-arm")
    for i, (dist, dy, dx, iou) in enumerate(draw_separated(list(feasible), k, rng,
                                                           min_separation)):
        plan.placebos.append(MatchedPlacebo("random", i, int(dx), int(dy), float(dist),
                                            float(iou), area))
    return _finish(plan, k, image_id, min_separation)


def artifact_matched_placebos(
    original: np.ndarray,
    donor_px: np.ndarray,
    mask: np.ndarray,
    feasible: Sequence[tuple[float, int, int, float]],
    defect_edit: float,
    defect_seam: float,
    k: int,
    seed: int,
    image_id: str,
    match_quantile: float = 0.3,
    min_separation: int = 8,
    feather_px: int = 3,
    template: "MaskTemplate | None" = None,
) -> ArmPlan:
    """Arm C: placebos matched on context *and* on intervention artifact profile.

    Uses only pixels: the defect-removal fill's ``edit_energy``/``seam_score`` and
    the same quantities for every candidate fill. No model output is available to,
    or used by, this function.
    """
    area = int(mask.sum())
    plan = ArmPlan("artifact", n_requested=k, n_feasible=len(feasible))
    if not feasible:
        plan.quality, plan.reason = "degenerate", "no admissible translation"
        return plan
    tmpl = template if template is not None else MaskTemplate(mask, feather_px)
    ctx = np.empty(len(feasible))
    edits = np.empty(len(feasible))
    seams = np.empty(len(feasible))
    for i, (dist, dy, dx, _iou) in enumerate(feasible):
        ctx[i] = dist
        edits[i], seams[i] = tmpl.metrics(original, donor_px, dy, dx)
    d_edit = np.array([log_ratio_distance(e, defect_edit) for e in edits])
    d_seam = np.array([log_ratio_distance(s, defect_seam) for s in seams])
    composite = np.sqrt((ctx / robust_scale(ctx)) ** 2 + (d_edit / robust_scale(d_edit)) ** 2
                        + (d_seam / robust_scale(d_seam)) ** 2)

    order = np.argsort(composite, kind="stable")
    n_keep = max(int(round(len(feasible) * match_quantile)), min(k, len(feasible)))
    pool = [(float(composite[j]), int(feasible[j][1]), int(feasible[j][2]), int(j))
            for j in order[:n_keep]]
    plan.n_pool = len(pool)
    rng = _stable_rng(seed, image_id, area, k, "artifact-arm")
    for i, (dist, dy, dx, j) in enumerate(draw_separated(pool, k, rng, min_separation)):
        plan.placebos.append(MatchedPlacebo(
            "artifact", i, dx, dy, float(ctx[j]), float(feasible[j][3]), area,
            float(edits[j]), float(seams[j]),
            float((edits[j] + RATIO_EPS) / (defect_edit + RATIO_EPS)),
            float((seams[j] + RATIO_EPS) / (defect_seam + RATIO_EPS)), dist))
    return _finish(plan, k, image_id, min_separation)


def context_arm_from_stage1(plan, area: int) -> ArmPlan:
    """Wrap a Stage-I :class:`PlaceboPlan` as arm B, unchanged."""
    arm = ArmPlan("context", n_requested=plan.n_requested, n_feasible=plan.n_feasible,
                  n_pool=plan.n_matched_pool, quality=plan.quality, reason=plan.reason)
    for p in plan.placebos:
        arm.placebos.append(MatchedPlacebo("context", p.index, p.dx, p.dy, p.context_distance,
                                           p.iou_with_defect, area))
    return arm


def occlusion_fill(original: np.ndarray) -> np.ndarray:
    """Arm D fill: a constant image of the original's mean colour.

    Deterministic and content-free. The same fill is used for the defect region and
    for every control region of an image, mirroring the same-donor rule. This is an
    occlusion-*style* baseline in the spirit of ROI deletion; it is not a
    reproduction of any specific paper's protocol.
    """
    mean = original.reshape(-1, original.shape[-1]).mean(axis=0)
    return np.broadcast_to(np.rint(mean).astype(np.uint8), original.shape).copy()
