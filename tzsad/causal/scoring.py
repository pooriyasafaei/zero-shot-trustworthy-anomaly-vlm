"""Deduplicated, batched, cached scoring of intervention images (CERM-AD v2).

Stage I+ scores five intervention arms per target, and several of them can produce
byte-identical images (the original appears in every arm; two arms can pick the same
translation). Every image is therefore keyed by a digest of its exact pixels, scored
at most once per run, and the score is appended to a crash-safe cache the moment it
exists. The key carries the *scoring* signature (model, probe, resolution) but not
the intervention geometry: the pixels already encode the geometry, and leaving it out
is what lets identical images in different arms share one forward pass.
"""
from __future__ import annotations

import hashlib
import json
from typing import Callable, Sequence

import numpy as np
from PIL import Image

from ..utils.logging import get_logger
from .audit import ScoreCache

log = get_logger("causal.scoring")


def array_digest(image: np.ndarray) -> str:
    """SHA-256 of the exact uint8 pixels plus shape.

    Hashing the raw buffer rather than a PNG encoding is ~50x cheaper and just as
    exact: two images share a digest only if every pixel is identical.
    """
    arr = np.ascontiguousarray(image, dtype=np.uint8)
    h = hashlib.sha256()
    h.update(json.dumps(list(arr.shape)).encode())
    h.update(arr.tobytes())
    return h.hexdigest()


def scoring_signature(model_id: str, score_mode: str, probe_version: str,
                      max_pixels: int, dtype: str) -> str:
    """Everything that changes a score *other than the pixels*."""
    payload = json.dumps({"model": model_id, "mode": score_mode, "probe": probe_version,
                          "max_pixels": int(max_pixels), "dtype": str(dtype)}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


BatchProbe = Callable[[Sequence[Image.Image], str], list[dict]]


def as_batch_probe(probe) -> BatchProbe:
    """Adapt a single-image probe to the batch interface (used by the test stubs)."""
    if hasattr(probe, "batch"):
        return probe.batch

    def _batch(images: Sequence[Image.Image], class_name: str) -> list[dict]:
        return [probe(im, class_name) for im in images]
    return _batch


def score_images(images: Sequence[np.ndarray], class_name: str, probe: BatchProbe,
                 cache: ScoreCache | None, batch_size: int = 1) -> tuple[list[dict], list[str], list[bool]]:
    """Score every image once; identical images share one forward pass.

    Returns ``(scores, digests, cache_hit)`` aligned with ``images``. ``cache_hit``
    is True for an image whose score came from the cache or from an identical image
    earlier in the same call - i.e. one that cost no forward pass of its own.
    """
    digests = [array_digest(im) for im in images]
    scores: dict[str, dict] = {}
    hit = [False] * len(images)
    pending: list[tuple[str, int]] = []
    seen: set[str] = set()
    for i, d in enumerate(digests):
        if d in seen:
            hit[i] = True
            continue
        seen.add(d)
        cached = cache.get(cache.key(d, class_name)) if cache is not None else None
        if cached is not None:
            scores[d] = cached
            hit[i] = True
        else:
            pending.append((d, i))

    step = max(int(batch_size), 1)
    for start in range(0, len(pending), step):
        chunk = pending[start:start + step]
        results = probe([Image.fromarray(images[i]) for _, i in chunk], class_name)
        if len(results) != len(chunk):
            raise RuntimeError(f"probe returned {len(results)} scores for {len(chunk)} images")
        for (d, _), res in zip(chunk, results):
            scores[d] = res
            if cache is not None:
                cache.put(cache.key(d, class_name), res)
    return [scores[d] for d in digests], digests, hit
