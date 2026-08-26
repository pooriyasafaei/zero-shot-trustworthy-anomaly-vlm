"""On-disk embedding cache keyed by (model, category, split, corruption).

Caching image embeddings is what makes the multi-backbone ensemble, TTA variance
and the normal-manifold kNN cheap array operations instead of extra GPU passes.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from ..utils.logging import get_logger

log = get_logger("features.cache")


def _safe(name: str) -> str:
    return name.replace("/", "_").replace(":", "-")


class EmbeddingCache:
    """Stores ``(image_ids, embeddings)`` shards as compressed ``.npz``.

    Layout::

        <root>/<model_tag>/<category>__<split>__<corruption>_<severity>__<kind>.npz

    ``kind`` distinguishes ``image`` embeddings from ``window`` (patch) embeddings
    and from TTA views.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def shard_path(self, model_tag: str, category: str, split: str, kind: str = "image",
                   corruption: str = "none", severity: int = 0, extra: str = "") -> Path:
        stem = f"{category}__{split}__{corruption}_{severity}__{kind}"
        if extra:
            stem += f"__{_safe(extra)}"
        return self.root / _safe(model_tag) / f"{stem}.npz"

    def has(self, **kw) -> bool:
        """Whether the shard for these keys already exists."""
        return self.shard_path(**kw).exists()

    def save(self, image_ids: Sequence[str], embeddings: np.ndarray, meta: dict | None = None, **kw) -> Path:
        """Persist embeddings as float16 alongside their image ids."""
        path = self.shard_path(**kw)
        path.parent.mkdir(parents=True, exist_ok=True)
        arr = np.asarray(embeddings, dtype=np.float16)
        if len(image_ids) != arr.shape[0]:
            raise ValueError(f"{len(image_ids)} ids vs {arr.shape[0]} embeddings")
        np.savez_compressed(path, image_ids=np.array(list(image_ids), dtype=object),
                            embeddings=arr, meta=json.dumps(meta or {}))
        log.debug("cached %s %s", arr.shape, path.name)
        return path

    def load(self, **kw) -> tuple[list[str], np.ndarray, dict]:
        """Load a shard. Raises ``FileNotFoundError`` if it was never computed."""
        path = self.shard_path(**kw)
        if not path.exists():
            raise FileNotFoundError(f"embedding shard missing: {path}")
        with np.load(path, allow_pickle=True) as z:
            ids = [str(s) for s in z["image_ids"]]
            emb = z["embeddings"].astype(np.float32)
            meta = json.loads(str(z["meta"])) if "meta" in z else {}
        return ids, emb, meta

    def load_as_dict(self, **kw) -> dict[str, np.ndarray]:
        """Load a shard as ``{image_id: embedding}``."""
        ids, emb, _ = self.load(**kw)
        return dict(zip(ids, emb))


def content_tag(*parts: object) -> str:
    """Short stable hash for cache keys built from many small pieces."""
    payload = json.dumps([str(p) for p in parts], sort_keys=True)
    return hashlib.sha1(payload.encode()).hexdigest()[:10]


def l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-8) -> np.ndarray:
    """Row-wise L2 normalisation that tolerates zero rows."""
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(n, eps)
