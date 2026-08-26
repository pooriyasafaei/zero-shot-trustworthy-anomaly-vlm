"""Patch-level anomaly maps built from cached window embeddings.

Two maps are produced from the same cached windows, so neither costs an extra
forward pass:

``text`` map
    WinCLIP-style: each window's softmax anomaly probability against the text
    prompts, painted back onto the image with overlap averaging.
``bank`` map
    PatchCore-lite: each window's kNN cosine distance to the bank of *positionally
    matched* ``train/good`` windows. This is the map the hallucination module
    trusts, because it needs "does anything here deviate from normal?" rather than
    "does CLIP's text encoder like the word crack here?".
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..scorers.clip_scorer import softmax_anomaly_probability
from ..utils.logging import get_logger
from .cache import EmbeddingCache

log = get_logger("features.heatmap")


def load_windows(cache: EmbeddingCache, model_tag: str, category: str, split: str,
                 corruption: str = "none", severity: int = 0) -> tuple[list[str], np.ndarray, dict]:
    """Load window embeddings as ``[n_images, n_windows, dim]`` plus their metadata."""
    ids, flat, meta = cache.load(model_tag=model_tag, category=category, split=split,
                                kind="window", corruption=corruption, severity=severity)
    k, d = int(meta["n_windows"]), int(meta["dim"])
    return ids, flat.reshape(len(ids), k, d), meta


def paint(window_scores: np.ndarray, boxes: list[tuple[int, int, int, int]], size: int,
          out_size: int = 128) -> np.ndarray:
    """Average per-window scores back into a dense ``out_size x out_size`` map."""
    acc = np.zeros((size, size), dtype=np.float64)
    cnt = np.zeros((size, size), dtype=np.float64)
    for s, (l, t, r, b) in zip(window_scores, boxes):
        acc[t:b, l:r] += float(s)
        cnt[t:b, l:r] += 1.0
    dense = acc / np.maximum(cnt, 1e-8)
    return _resize(dense, out_size)


def _resize(arr: np.ndarray, out: int) -> np.ndarray:
    """Bilinear resize without pulling in scipy/cv2."""
    if arr.shape[0] == out and arr.shape[1] == out:
        return arr.astype(np.float32)
    from PIL import Image

    lo, hi = float(arr.min()), float(arr.max())
    scaled = (arr - lo) / (hi - lo) if hi > lo else np.zeros_like(arr)
    img = Image.fromarray((scaled * 65535).astype(np.uint16)).resize((out, out), Image.BILINEAR)
    back = np.asarray(img, dtype=np.float32) / 65535.0
    return (back * (hi - lo) + lo).astype(np.float32)


def text_maps(win: np.ndarray, boxes, image_size: int, t_norm: np.ndarray, t_anom: np.ndarray,
              tau: float, out_size: int = 128) -> np.ndarray:
    """WinCLIP-style anomaly maps, shape ``[n_images, out_size, out_size]``."""
    mn = t_norm.mean(0) / max(np.linalg.norm(t_norm.mean(0)), 1e-8)
    ma = t_anom.mean(0) / max(np.linalg.norm(t_anom.mean(0)), 1e-8)
    scores = softmax_anomaly_probability(win @ mn, win @ ma, tau)     # [n, k]
    return np.stack([paint(s, boxes, image_size, out_size) for s in scores])


def bank_maps(win: np.ndarray, bank: np.ndarray, boxes, image_size: int, k: int = 1,
              out_size: int = 128, position_aware: bool = True) -> np.ndarray:
    """kNN-to-normal-bank deviation maps, shape ``[n_images, out_size, out_size]``.

    Parameters
    ----------
    win:
        Test window embeddings ``[n_images, n_windows, dim]``.
    bank:
        Normal window embeddings ``[n_normal, n_windows, dim]`` from ``train/good``.
    k:
        Neighbours to average over. ``k=1`` is the PatchCore choice and is the
        most sensitive; larger ``k`` is more robust on small banks.
    position_aware:
        Compare a window only against bank windows at the *same* grid position.
        MVTec objects are centred and roughly aligned, so this sharpens the map
        considerably; set ``False`` for textures where alignment is meaningless.
    """
    n_img, n_win, dim = win.shape
    scores = np.zeros((n_img, n_win), dtype=np.float32)
    if bank.shape[0] == 0:
        return np.zeros((n_img, out_size, out_size), dtype=np.float32)
    if position_aware:
        for w in range(n_win):
            sims = win[:, w, :] @ bank[:, w, :].T                   # [n_img, n_bank]
            scores[:, w] = 1.0 - _topk_mean(sims, k)
    else:
        flat_bank = bank.reshape(-1, dim)
        for w in range(n_win):
            sims = win[:, w, :] @ flat_bank.T
            scores[:, w] = 1.0 - _topk_mean(sims, k)
    return np.stack([paint(s, boxes, image_size, out_size) for s in scores])


def _topk_mean(sims: np.ndarray, k: int) -> np.ndarray:
    """Mean of the k largest similarities per row."""
    k = int(min(max(k, 1), sims.shape[1]))
    if k == 1:
        return sims.max(axis=1)
    part = np.partition(sims, -k, axis=1)[:, -k:]
    return part.mean(axis=1)


def save_maps(path: str | Path, image_ids, maps: np.ndarray) -> Path:
    """Persist a batch of maps as float16 npz keyed by image id."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, image_ids=np.array(list(image_ids), dtype=object),
                        maps=maps.astype(np.float16))
    return path


def load_maps(path: str | Path) -> dict[str, np.ndarray]:
    """Load maps written by :func:`save_maps` as ``{image_id: HxW array}``."""
    with np.load(path, allow_pickle=True) as z:
        ids = [str(s) for s in z["image_ids"]]
        maps = z["maps"].astype(np.float32)
    return dict(zip(ids, maps))
