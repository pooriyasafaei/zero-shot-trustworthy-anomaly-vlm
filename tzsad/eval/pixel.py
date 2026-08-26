"""Pixel-level metrics: pixel AUROC and PRO.

Both operate on the cached anomaly maps, so changing the metric never costs a
forward pass.
"""
from __future__ import annotations

import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score

# np.trapz was renamed to np.trapezoid in NumPy 2.0; support both.
_trapz = getattr(np, "trapezoid", None) or np.trapz


def load_mask(mask_path: str, size: int) -> np.ndarray:
    """Load a ground-truth mask as a binary array resized to ``size x size``."""
    m = Image.open(mask_path).convert("L").resize((size, size), Image.NEAREST)
    return (np.asarray(m) > 127).astype(np.uint8)


def pixel_auroc(maps: dict[str, np.ndarray], masks: dict[str, np.ndarray],
                max_pixels: int = 2_000_000, seed: int = 0) -> float:
    """Pixel AUROC pooled over images.

    Subsamples pixels when the pooled set is large - MVTec has ~10^8 test pixels
    per category at full resolution and the estimate is stable long before that.
    """
    ys, ss = [], []
    for image_id, m in maps.items():
        mask = masks.get(image_id)
        if mask is None:
            mask = np.zeros_like(m, dtype=np.uint8)
        ys.append(mask.ravel())
        ss.append(m.ravel())
    if not ys:
        return float("nan")
    y = np.concatenate(ys)
    s = np.concatenate(ss)
    if len(np.unique(y)) < 2:
        return float("nan")
    if y.size > max_pixels:
        rng = np.random.default_rng(seed)
        take = rng.choice(y.size, max_pixels, replace=False)
        y, s = y[take], s[take]
    return float(roc_auc_score(y, s))


def pro_score(maps: dict[str, np.ndarray], masks: dict[str, np.ndarray],
              max_fpr: float = 0.3, n_thresholds: int = 200) -> float:
    """Per-Region Overlap, normalised over FPR in ``[0, max_fpr]``.

    PRO weights each connected ground-truth region equally, so a method that only
    finds large defects cannot hide behind pixel counts.

    The threshold sweep must span the **whole** score range, normal pixels
    included: sweeping only over the scores found inside defect regions leaves the
    false-positive rate pinned at zero and collapses the integral. When the curve
    never reaches ``max_fpr`` (a near-perfect detector), the last overlap value is
    held out to ``max_fpr`` before integrating, so the normalisation stays
    ``/ max_fpr`` rather than dividing by a vanishing endpoint.
    """
    regions: list[np.ndarray] = []          # scores inside each connected region
    normal_scores: list[np.ndarray] = []
    for image_id, m in maps.items():
        mask = masks.get(image_id)
        if mask is None or mask.sum() == 0:
            normal_scores.append(m.ravel())
            continue
        normal_scores.append(m[mask == 0].ravel())
        for comp in _connected_components(mask):
            regions.append(m[comp])
    if not regions:
        return float("nan")

    all_normal = np.concatenate(normal_scores)
    lo = min(float(all_normal.min()), min(float(r.min()) for r in regions))
    hi = max(float(all_normal.max()), max(float(r.max()) for r in regions))
    if hi <= lo:
        return float("nan")
    thresholds = np.linspace(hi, lo, n_thresholds)

    fprs, pros = [], []
    for t in thresholds:
        fpr = float((all_normal >= t).mean())
        pro = float(np.mean([(scores >= t).mean() for scores in regions]))
        if fpr > max_fpr:
            break
        fprs.append(fpr)
        pros.append(pro)
    if not fprs:
        return float("nan")
    if fprs[-1] < max_fpr:                  # hold the last value out to max_fpr
        fprs.append(max_fpr)
        pros.append(pros[-1])
    if len(fprs) < 2:
        return float("nan")
    return float(_trapz(pros, fprs) / max_fpr)


def _connected_components(mask: np.ndarray) -> list[np.ndarray]:
    """4-connected components of a binary mask, as boolean arrays."""
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for i in range(h):
        for j in range(w):
            if mask[i, j] and not seen[i, j]:
                comp = np.zeros_like(mask, dtype=bool)
                stack = [(i, j)]
                seen[i, j] = True
                while stack:
                    y, x = stack.pop()
                    comp[y, x] = True
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                            seen[ny, nx] = True
                            stack.append((ny, nx))
                comps.append(comp)
    return comps
