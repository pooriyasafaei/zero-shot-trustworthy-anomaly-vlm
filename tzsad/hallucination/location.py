"""Turn the VLM's natural-language ``DEFECT_LOCATION`` into a spatial region.

This is the core of the reframed CMCS (§4.4). The prototype's Grounding-DINO
comparison is weak on MVTec - one centred object means there is little for an
object detector to disagree about - but the VLM *does* volunteer where it thinks
the defect is, in words, and that claim is checkable against the patch map.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

#: Vocabulary -> normalised row/column bands on the unit square.
_ROW_TERMS: dict[str, tuple[float, float]] = {
    "top": (0.0, 0.42), "upper": (0.0, 0.42), "above": (0.0, 0.42), "head": (0.0, 0.42),
    "bottom": (0.58, 1.0), "lower": (0.58, 1.0), "below": (0.58, 1.0), "base": (0.58, 1.0),
    "middle": (0.29, 0.71), "centre": (0.29, 0.71), "center": (0.29, 0.71), "central": (0.29, 0.71),
}
_COL_TERMS: dict[str, tuple[float, float]] = {
    "left": (0.0, 0.42), "right": (0.58, 1.0),
    "middle": (0.29, 0.71), "centre": (0.29, 0.71), "center": (0.29, 0.71), "central": (0.29, 0.71),
}
#: Terms that describe a shape rather than a quadrant.
_EDGE_TERMS = ("edge", "border", "rim", "perimeter", "boundary", "side", "outer", "circumference")
_SURFACE_TERMS = ("surface", "throughout", "overall", "entire", "whole", "all over", "everywhere",
                  "body", "face", "front")
_CORNER_TERMS = ("corner",)
_NONE_TERMS = ("none", "n/a", "na", "not applicable", "no defect", "nowhere", "-", "")


@dataclass(frozen=True)
class LocationClaim:
    """A parsed spatial claim: a mask generator plus what kind of claim it is."""

    kind: str            # 'none' | 'quadrant' | 'edge' | 'surface' | 'corner' | 'unparsed'
    rows: tuple[float, float] = (0.0, 1.0)
    cols: tuple[float, float] = (0.0, 1.0)
    text: str = ""

    @property
    def is_localized(self) -> bool:
        """Whether the claim actually narrows the region down.

        ``surface``/``unparsed`` claims cover everything, so they can never be
        contradicted by the patch map and must not count as verified grounding.
        """
        return self.kind in ("quadrant", "edge", "corner")

    def mask(self, size: int = 128, edge_width: float = 0.18) -> np.ndarray:
        """Binary region mask on a ``size x size`` grid."""
        m = np.zeros((size, size), dtype=np.uint8)
        if self.kind == "none":
            return m
        if self.kind == "edge":
            w = max(int(round(size * edge_width)), 1)
            m[:] = 1
            m[w : size - w, w : size - w] = 0
            return m
        r0, r1 = int(self.rows[0] * size), int(np.ceil(self.rows[1] * size))
        c0, c1 = int(self.cols[0] * size), int(np.ceil(self.cols[1] * size))
        m[r0:r1, c0:c1] = 1
        return m


def parse_location(text: str) -> LocationClaim:
    """Parse a free-text location into a :class:`LocationClaim`.

    Recognises quadrant language ("upper left", "bottom-right"), shape language
    ("along the edge", "across the surface"), and explicit absence ("NONE").
    Anything unrecognised becomes ``unparsed`` - reported, never silently treated
    as a correct or an incorrect claim.
    """
    raw = (text or "").strip()
    low = re.sub(r"[^a-z\s/-]", " ", raw.lower())
    low = re.sub(r"\s+", " ", low).strip()
    if low in _NONE_TERMS or not low:
        return LocationClaim("none", text=raw)

    if any(t in low for t in _CORNER_TERMS):
        rows = _match_band(low, _ROW_TERMS) or (0.0, 0.42)
        cols = _match_band(low, _COL_TERMS) or (0.0, 0.42)
        return LocationClaim("corner", rows, cols, raw)

    rows = _match_band(low, _ROW_TERMS)
    cols = _match_band(low, _COL_TERMS)
    if rows or cols:
        return LocationClaim("quadrant", rows or (0.0, 1.0), cols or (0.0, 1.0), raw)

    if any(t in low for t in _EDGE_TERMS):
        return LocationClaim("edge", text=raw)
    if any(t in low for t in _SURFACE_TERMS):
        return LocationClaim("surface", text=raw)
    return LocationClaim("unparsed", text=raw)


def _match_band(text: str, terms: dict[str, tuple[float, float]]) -> tuple[float, float] | None:
    """First matching band, preferring the longest term so 'upper' beats 'up'."""
    for term in sorted(terms, key=len, reverse=True):
        if re.search(rf"\b{re.escape(term)}\b", text):
            return terms[term]
    return None


def region_iou(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection-over-union of two binary masks (0.0 when both are empty)."""
    a = a.astype(bool)
    b = b.astype(bool)
    union = (a | b).sum()
    return float((a & b).sum() / union) if union else 0.0


def attended_region(anomaly_map: np.ndarray, top_frac: float = 0.05) -> np.ndarray:
    """The model's *de facto* attended region: the top-scoring fraction of the map.

    Used when the VLM gives no usable location, so that hallucination can still be
    tested against where the visual evidence actually is.
    """
    flat = anomaly_map.ravel()
    if flat.size == 0:
        return np.zeros_like(anomaly_map, dtype=np.uint8)
    k = max(int(round(flat.size * top_frac)), 1)
    thr = np.partition(flat, -k)[-k]
    return (anomaly_map >= thr).astype(np.uint8)
