"""ImageNet-C style corruptions at 5 severities.

Used for the corruption sweep (§4.6): the question is not only how much AUROC
drops, but whether uncertainty *rises* with severity. A method whose uncertainty
stays flat while accuracy collapses is the silent-failure mode we want to expose.

Implemented with numpy/PIL only so the sweep runs anywhere, with no `imagecorruptions`
dependency. Severity parameters follow Hendrycks & Dietterich (2019) closely enough
for a monotone-degradation study; they are not bit-identical to that reference.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

CORRUPTIONS: tuple[str, ...] = (
    "gaussian_noise", "defocus_blur", "motion_blur", "jpeg_compression",
    "brightness", "contrast", "fog",
)
SEVERITIES: tuple[int, ...] = (1, 2, 3, 4, 5)


def _as_float(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0


def _to_image(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(np.clip(arr * 255.0, 0, 255).astype(np.uint8))


def gaussian_noise(img: Image.Image, severity: int, rng: np.random.Generator) -> Image.Image:
    scale = [0.04, 0.06, 0.09, 0.13, 0.18][severity - 1]
    x = _as_float(img)
    return _to_image(x + rng.normal(size=x.shape, scale=scale).astype(np.float32))


def defocus_blur(img: Image.Image, severity: int, rng: np.random.Generator) -> Image.Image:
    radius = [1.0, 1.8, 2.6, 3.6, 5.0][severity - 1]
    return img.convert("RGB").filter(ImageFilter.GaussianBlur(radius=radius))


def motion_blur(img: Image.Image, severity: int, rng: np.random.Generator) -> Image.Image:
    length = [3, 5, 9, 13, 19][severity - 1]
    x = _as_float(img)
    kernel = np.zeros((length, length), dtype=np.float32)
    kernel[length // 2, :] = 1.0 / length
    out = np.stack([_convolve2d(x[..., c], kernel) for c in range(3)], axis=-1)
    return _to_image(out)


def jpeg_compression(img: Image.Image, severity: int, rng: np.random.Generator) -> Image.Image:
    import io

    quality = [30, 22, 16, 11, 7][severity - 1]
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def brightness(img: Image.Image, severity: int, rng: np.random.Generator) -> Image.Image:
    factor = [1.15, 1.3, 1.45, 1.65, 1.9][severity - 1]
    return ImageEnhance.Brightness(img.convert("RGB")).enhance(factor)


def contrast(img: Image.Image, severity: int, rng: np.random.Generator) -> Image.Image:
    factor = [0.75, 0.6, 0.45, 0.3, 0.18][severity - 1]
    return ImageEnhance.Contrast(img.convert("RGB")).enhance(factor)


def fog(img: Image.Image, severity: int, rng: np.random.Generator) -> Image.Image:
    """Additive low-frequency haze; a plasma-fractal stand-in for ImageNet-C fog."""
    intensity, blend = [(0.2, 0.85), (0.3, 0.78), (0.4, 0.7), (0.5, 0.62), (0.65, 0.55)][severity - 1]
    x = _as_float(img)
    h, w = x.shape[:2]
    haze = _plasma(max(h, w), rng)[:h, :w, None]
    return _to_image(blend * x + (1.0 - blend) * (1.0 - intensity + intensity * haze))


def _plasma(size: int, rng: np.random.Generator) -> np.ndarray:
    """Diamond-square-ish 1/f noise field in [0, 1]."""
    n = 1
    while n < size:
        n *= 2
    field = rng.random((2, 2)).astype(np.float32)
    while field.shape[0] < n:
        k = field.shape[0]
        up = np.repeat(np.repeat(field, 2, axis=0), 2, axis=1)
        up += rng.normal(scale=0.5 / k, size=up.shape).astype(np.float32)
        field = up
    field -= field.min()
    denom = field.max() or 1.0
    return field / denom


def _convolve2d(channel: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Small 'same' convolution with edge padding (kernels here are tiny)."""
    kh, kw = kernel.shape
    padded = np.pad(channel, ((kh // 2, kh // 2), (kw // 2, kw // 2)), mode="edge")
    out = np.zeros_like(channel)
    for i in range(kh):
        for j in range(kw):
            if kernel[i, j]:
                out += kernel[i, j] * padded[i : i + channel.shape[0], j : j + channel.shape[1]]
    return out


_REGISTRY: dict[str, Callable[[Image.Image, int, np.random.Generator], Image.Image]] = {
    "gaussian_noise": gaussian_noise,
    "defocus_blur": defocus_blur,
    "motion_blur": motion_blur,
    "jpeg_compression": jpeg_compression,
    "brightness": brightness,
    "contrast": contrast,
    "fog": fog,
}


def apply_corruption(img: Image.Image, name: str, severity: int, seed: int = 0) -> Image.Image:
    """Apply a named corruption at severity 1-5. ``name='none'`` is the identity."""
    if name in ("none", "", None):
        return img.convert("RGB")
    if name not in _REGISTRY:
        raise KeyError(f"unknown corruption {name!r}; available: {sorted(_REGISTRY)}")
    if severity not in SEVERITIES:
        raise ValueError(f"severity must be in {SEVERITIES}, got {severity}")
    rng = np.random.default_rng(abs(hash((name, severity, seed))) % (2**32))
    return _REGISTRY[name](img.convert("RGB"), severity, rng)
