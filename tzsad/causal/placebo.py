"""Matched placebo masks: the control arm of the causal audit.

A placebo has to answer "what would an edit of this size, at a comparable place,
have done?". Random rectangles do not answer that - they differ from the real mask
in shape, in area and in the kind of surface they sit on, so any difference in the
model's response could be attributed to any of those.

So a placebo here is the **exact ground-truth mask, translated**. Shape and area are
therefore identical by construction, and only the location changes. Candidates are
further filtered so that the surroundings resemble the surroundings of the real
defect, using a dilated-minus-mask "context ring" descriptor.

One deliberate choice: we do **not** take the K closest-matching translations. The
K nearest positions would be near-duplicates of each other and would collapse the
placebo distribution, leaving nothing to compare a tail against. Instead we form a
*matched pool* of sufficiently similar positions and sample K of them uniformly with
the experiment seed.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from scipy import ndimage

from ..utils.logging import get_logger

log = get_logger("causal.placebo")


@dataclass(frozen=True)
class PlaceboMask:
    """One translated copy of the defect mask, with its matching diagnostics."""

    index: int
    dx: int
    dy: int
    context_distance: float
    iou_with_defect: float
    mask_area: int

    def as_dict(self) -> dict:
        """Flat dict for the interventions table."""
        return {"placebo_index": self.index, "dx": self.dx, "dy": self.dy,
                "context_distance": self.context_distance,
                "iou_with_defect": self.iou_with_defect, "mask_area": self.mask_area}


@dataclass
class PlaceboPlan:
    """The placebos chosen for one target image, plus why there may be fewer than K."""

    placebos: list[PlaceboMask] = field(default_factory=list)
    n_requested: int = 0
    n_feasible: int = 0          # translations that fit, respect IoU, and are distinct
    n_matched_pool: int = 0      # of those, how many passed the context-similarity cut
    quality: str = "ok"          # ok | few_matched | few_feasible | degenerate
    reason: str = ""

    @property
    def n_valid(self) -> int:
        """Number of placebos actually produced."""
        return len(self.placebos)

    @property
    def tail_resolution(self) -> float:
        """Finest achievable empirical tail probability, ``1 / (n_valid + 1)``.

        Reported so a claim is never made at a resolution the placebo count cannot
        support: with 4 placebos the smallest possible p is 0.2.
        """
        return 1.0 / (self.n_valid + 1)

    def as_dict(self) -> dict:
        """Diagnostics for the per-image causal record."""
        return {"n_valid_placebos": self.n_valid, "n_requested_placebos": self.n_requested,
                "n_feasible_placebos": self.n_feasible, "n_matched_pool": self.n_matched_pool,
                "placebo_quality": self.quality, "placebo_reason": self.reason,
                "tail_resolution": self.tail_resolution}


def context_ring(mask: np.ndarray, width: int) -> np.ndarray:
    """``dilate(mask) - mask``: the band of image immediately surrounding a region."""
    if width <= 0:
        return np.zeros_like(mask, dtype=bool)
    dilated = ndimage.binary_dilation(mask, iterations=int(width))
    return dilated & ~mask


def ring_descriptor_at(image: np.ndarray, grad: np.ndarray | None,
                       ys: np.ndarray, xs: np.ndarray) -> np.ndarray:
    """Descriptor for a ring given as coordinates rather than a full boolean array.

    The placebo ring is always a *translation* of the defect's ring, because the
    placebo mask is a translation of the defect mask. Translating the precomputed
    coordinates is exact and avoids re-running a full-image binary dilation for
    every candidate - which was costing minutes per target image on MVTec-sized
    masks and leaving the GPU idle.
    """
    if ys.size == 0:
        return np.zeros(8 if grad is not None else 6, dtype=np.float32)
    pix = image[ys, xs].astype(np.float32) / 255.0
    parts = [pix.mean(axis=0), pix.std(axis=0)]
    if grad is not None:
        g = grad[ys, xs]
        parts.append(np.array([g.mean(), g.std()], dtype=np.float32))
    return np.concatenate(parts).astype(np.float32)


def gradient_magnitude(image: np.ndarray) -> np.ndarray:
    """Per-pixel gradient magnitude of the greyscale image, computed once."""
    grey = image.astype(np.float32).mean(axis=2) / 255.0
    gy, gx = np.gradient(grey)
    return np.hypot(gx, gy)


def ring_descriptor(image: np.ndarray, ring: np.ndarray,
                    use_gradient: bool = True) -> np.ndarray:
    """Summarise the appearance of a context ring.

    RGB mean and standard deviation, optionally plus gradient-magnitude statistics so
    that a smooth region is not matched to a textured one. Deliberately low
    dimensional: it should capture "similar kind of surface", not identity.
    """
    if ring.sum() == 0:
        return np.zeros(8, dtype=np.float32)
    pix = image[ring].astype(np.float32) / 255.0
    parts = [pix.mean(axis=0), pix.std(axis=0)]
    if use_gradient:
        grey = image.astype(np.float32).mean(axis=2) / 255.0
        gy, gx = np.gradient(grey)
        grad = np.hypot(gx, gy)[ring]
        parts.append(np.array([grad.mean(), grad.std()], dtype=np.float32))
    return np.concatenate(parts).astype(np.float32)


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    union = (a | b).sum()
    return float((a & b).sum() / union) if union else 0.0


def _translate(mask: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Shift a boolean mask without wrapping or clipping away any pixels."""
    out = np.zeros_like(mask)
    h, w = mask.shape
    ys, xs = np.nonzero(mask)
    out[ys + dy, xs + dx] = True
    return out


def _stable_rng(seed: int, *parts: object) -> np.random.Generator:
    """Deterministic generator keyed by the experiment seed and the image identity.

    Python's ``hash`` is randomised per process, so it cannot seed anything whose
    value has to be reproducible from a manifest.
    """
    key = "|".join([str(seed), *(str(p) for p in parts)]).encode()
    return np.random.default_rng(int.from_bytes(hashlib.sha256(key).digest()[:8], "big"))


def generate_placebos(
    image: np.ndarray,
    mask: np.ndarray,
    k: int,
    seed: int,
    image_id: str = "",
    max_iou: float = 0.01,
    ring_width: int = 12,
    min_separation: int = 8,
    candidate_stride: int = 8,
    match_quantile: float = 0.4,
    min_pool: int = 0,
    use_gradient: bool = True,
    max_candidates: int = 600,
) -> PlaceboPlan:
    """Choose up to ``k`` matched placebo positions for one defect mask.

    Parameters
    ----------
    image, mask:
        Target image (H, W, 3) uint8 and its boolean ground-truth defect mask.
    k:
        How many placebos to draw from the matched pool.
    seed, image_id:
        Together they seed the draw, so the same image always yields the same
        placebos and different images do not share a draw.
    max_iou:
        A candidate overlapping the real defect by more than this is rejected - a
        "placebo" that removes part of the defect is not a control.
    ring_width:
        Width of the context ring compared between candidate and defect.
    min_separation:
        Minimum Chebyshev distance between accepted placebo offsets, so the K
        placebos are distinct placements rather than one position jittered.
    candidate_stride:
        Grid step for the deterministic candidate sweep. Smaller is slower.
    match_quantile:
        Fraction of feasible candidates kept as the matched pool, by context
        distance. 0.4 keeps the most similar 40% - similar enough to be a fair
        control, varied enough to form a distribution.
    min_pool:
        Refuse to sample if the matched pool is smaller than this (0 disables).
    max_candidates:
        Upper bound on translations evaluated, subsampled evenly across the grid so
        the cost does not scale with image area.

    Returns
    -------
    A :class:`PlaceboPlan`. Fewer than ``k`` placebos is a reported outcome, never
    silently padded: the tail resolution depends on how many were actually found.
    """
    h, w = mask.shape
    area = int(mask.sum())
    if area == 0:
        return PlaceboPlan(n_requested=k, quality="degenerate", reason="empty defect mask")

    ys, xs = np.nonzero(mask)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    # Offsets that keep every mask pixel inside the image: no clipping is allowed,
    # because a clipped mask would no longer have the same area as the defect.
    dy_lo, dy_hi = -int(y0), int(h - 1 - y1)
    dx_lo, dx_hi = -int(x0), int(w - 1 - x1)
    if dy_hi < dy_lo or dx_hi < dx_lo:
        return PlaceboPlan(n_requested=k, quality="degenerate",
                           reason="defect mask spans the image; no translation fits")

    target_ring = context_ring(mask, ring_width)
    target_desc = ring_descriptor(image, target_ring, use_gradient)

    # Precompute once: the defect's ring coordinates and the image gradient. Every
    # candidate is a translation of this ring, so no further dilation is needed.
    ring_ys, ring_xs = np.nonzero(target_ring)
    grad = gradient_magnitude(image) if use_gradient else None
    target_desc = ring_descriptor_at(image, grad, ring_ys, ring_xs)
    mask_ys, mask_xs = ys, xs

    offsets = [(dy, dx)
               for dy in range(dy_lo, dy_hi + 1, max(int(candidate_stride), 1))
               for dx in range(dx_lo, dx_hi + 1, max(int(candidate_stride), 1))
               if not (dy == 0 and dx == 0)]
    # Bound the work regardless of image size: a deterministic even subsample of the
    # grid keeps the search spread out instead of biased towards one corner.
    if max_candidates and len(offsets) > max_candidates:
        step = len(offsets) / float(max_candidates)
        offsets = [offsets[int(i * step)] for i in range(max_candidates)]

    feasible: list[tuple[float, int, int, float]] = []
    for dy, dx in offsets:
        # IoU between a mask and its own translation, without materialising both:
        # both have area A, so IoU = I / (2A - I).
        shifted = np.zeros_like(mask)
        shifted[mask_ys + dy, mask_xs + dx] = True
        inter = int((shifted & mask).sum())
        iou = inter / float(2 * area - inter) if (2 * area - inter) else 0.0
        if iou > max_iou:
            continue
        rys, rxs = ring_ys + dy, ring_xs + dx
        keep = (rys >= 0) & (rys < h) & (rxs >= 0) & (rxs < w)
        if not keep.any():
            continue
        dist = float(np.linalg.norm(
            ring_descriptor_at(image, grad, rys[keep], rxs[keep]) - target_desc))
        feasible.append((dist, dy, dx, iou))

    if not feasible:
        return PlaceboPlan(n_requested=k, quality="degenerate",
                           reason="no translation satisfies the IoU and bounds constraints")

    feasible.sort(key=lambda t: t[0])
    n_keep = max(int(round(len(feasible) * match_quantile)), min(k, len(feasible)))
    pool = feasible[:n_keep]

    quality, reason = "ok", ""
    if min_pool and len(pool) < min_pool:
        quality, reason = "few_matched", f"matched pool {len(pool)} < min_pool {min_pool}"

    rng = _stable_rng(seed, image_id, area, k)
    order = rng.permutation(len(pool))
    chosen: list[PlaceboMask] = []
    for idx in order:
        if len(chosen) >= k:
            break
        dist, dy, dx, iou = pool[idx]
        if any(max(abs(dy - c.dy), abs(dx - c.dx)) < min_separation for c in chosen):
            continue        # too close to one already taken: not an independent placement
        chosen.append(PlaceboMask(len(chosen), int(dx), int(dy), dist, iou, area))

    if len(chosen) < k and quality == "ok":
        quality = "few_feasible"
        reason = (f"only {len(chosen)} distinct placements found for k={k} "
                  f"(feasible={len(feasible)}, pool={len(pool)}, min_separation={min_separation})")
        log.warning("%s: %s", image_id or "image", reason)

    return PlaceboPlan(chosen, k, len(feasible), len(pool), quality, reason)


def placebo_mask_array(mask: np.ndarray, placebo: PlaceboMask) -> np.ndarray:
    """Materialise a placebo's boolean mask from the defect mask and its offset."""
    return _translate(mask, placebo.dy, placebo.dx)
