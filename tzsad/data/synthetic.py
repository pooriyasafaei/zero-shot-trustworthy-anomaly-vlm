"""Synthetic anomalies from `train/good` images (CutPaste / NSA style).

Used only to fit the logistic trust-score combiner (§4.5) without touching test
labels. Keeping this on the normal-only calibration pool is what lets the fused
trust score stay honest about being zero-shot with respect to real defects.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter


def cutpaste(img: Image.Image, rng: np.random.Generator, area: tuple[float, float] = (0.02, 0.15),
             aspect: tuple[float, float] = (0.3, 3.3)) -> tuple[Image.Image, np.ndarray]:
    """Paste a random patch of the image somewhere else. Returns (image, binary mask)."""
    img = img.convert("RGB")
    w, h = img.size
    frac = rng.uniform(*area)
    ar = np.exp(rng.uniform(np.log(aspect[0]), np.log(aspect[1])))
    pw = int(np.clip(np.sqrt(frac * w * h * ar), 8, w - 1))
    ph = int(np.clip(np.sqrt(frac * w * h / ar), 8, h - 1))
    sx, sy = int(rng.integers(0, w - pw)), int(rng.integers(0, h - ph))
    dx, dy = int(rng.integers(0, w - pw)), int(rng.integers(0, h - ph))
    patch = img.crop((sx, sy, sx + pw, sy + ph))
    if rng.random() < 0.5:
        patch = patch.rotate(float(rng.uniform(-45, 45)), expand=False)
    out = img.copy()
    out.paste(patch, (dx, dy))
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[dy : dy + ph, dx : dx + pw] = 1
    return out, mask


def nsa_blend(img: Image.Image, rng: np.random.Generator, area: tuple[float, float] = (0.02, 0.12)) -> tuple[Image.Image, np.ndarray]:
    """Poisson-free NSA approximation: feathered self-blend of a shifted patch."""
    img = img.convert("RGB")
    w, h = img.size
    frac = rng.uniform(*area)
    pw = int(np.clip(np.sqrt(frac * w * h), 8, w - 1))
    ph = pw
    sx, sy = int(rng.integers(0, w - pw)), int(rng.integers(0, h - ph))
    dx, dy = int(rng.integers(0, w - pw)), int(rng.integers(0, h - ph))
    patch = img.crop((sx, sy, sx + pw, sy + ph))
    alpha = Image.new("L", (pw, ph), 0)
    inner = Image.new("L", (max(pw - 8, 2), max(ph - 8, 2)), 255)
    alpha.paste(inner, (4, 4))
    alpha = alpha.filter(ImageFilter.GaussianBlur(3))
    out = img.copy()
    out.paste(patch, (dx, dy), alpha)
    mask = np.zeros((h, w), dtype=np.uint8)
    a = np.asarray(alpha) > 64
    mask[dy : dy + ph, dx : dx + pw] = a.astype(np.uint8)
    return out, mask


def make_synthetic(img: Image.Image, seed: int, method: str = "cutpaste") -> tuple[Image.Image, np.ndarray]:
    """Generate one synthetic anomaly deterministically from ``seed``."""
    rng = np.random.default_rng(seed)
    if method == "cutpaste":
        return cutpaste(img, rng)
    if method == "nsa":
        return nsa_blend(img, rng)
    raise KeyError(f"unknown synthetic method {method!r}")
