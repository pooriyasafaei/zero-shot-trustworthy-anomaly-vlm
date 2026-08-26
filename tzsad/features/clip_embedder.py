"""The only GPU pass in the CLIP branch: images (and windows) -> cached embeddings.

Everything downstream - the prompt ensemble, the backbone ensemble, TTA variance,
the normal-manifold kNN, the patch heatmaps - is an array operation over these
caches. Adding an uncertainty signal therefore costs no forward pass.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd
import torch
from PIL import Image

from ..data.corruptions import apply_corruption
from ..utils.logging import get_logger
from .cache import EmbeddingCache, l2_normalize

log = get_logger("features.clip")


@dataclass(frozen=True)
class BackboneSpec:
    """One CLIP-family backbone: ``open_clip.create_model_and_transforms`` arguments."""

    name: str            # e.g. "ViT-B-16"
    pretrained: str      # e.g. "openai", "laion2b_s34b_b88k"
    tag: str = ""        # cache directory name; defaults to name/pretrained

    @property
    def cache_tag(self) -> str:
        return self.tag or f"{self.name}__{self.pretrained}"

    @property
    def model_id(self) -> str:
        return f"{self.name}/{self.pretrained}"


@dataclass(frozen=True)
class WindowSpec:
    """Multi-scale sliding-window grid for WinCLIP-style patch scoring.

    ``scales`` are window side lengths as a fraction of the (square) input image;
    ``stride_frac`` is the step as a fraction of the window side.
    """

    scales: tuple[float, ...] = (0.5, 0.3333)
    stride_frac: float = 0.5
    image_size: int = 336

    def windows(self) -> list[tuple[int, int, int, int]]:
        """Window boxes ``(left, top, right, bottom)`` in resized-image pixels."""
        boxes: list[tuple[int, int, int, int]] = []
        S = self.image_size
        for scale in self.scales:
            w = max(int(round(S * scale)), 16)
            step = max(int(round(w * self.stride_frac)), 1)
            tops = list(range(0, max(S - w, 0) + 1, step))
            if tops and tops[-1] != S - w:
                tops.append(S - w)
            for top in tops:
                for left in tops:
                    boxes.append((left, top, left + w, top + w))
        return boxes


class ClipEmbedder:
    """Wraps an open_clip backbone and writes embeddings into an :class:`EmbeddingCache`."""

    def __init__(self, spec: BackboneSpec, device: str | None = None,
                 cache_dir: str | Path | None = None, allow_download: bool = True,
                 batch_size: int = 32) -> None:
        self.spec = spec
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self._model = None
        self._preprocess: Callable[[Image.Image], torch.Tensor] | None = None
        self._tokenizer = None
        self.cache_dir = str(cache_dir) if cache_dir else None
        self.allow_download = allow_download

    # -- model loading -----------------------------------------------------
    def load(self) -> None:
        """Load weights. Fails loudly rather than downloading when offline."""
        if self._model is not None:
            return
        import open_clip

        kwargs = {}
        if self.cache_dir:
            kwargs["cache_dir"] = self.cache_dir
        try:
            model, _, preprocess = open_clip.create_model_and_transforms(
                self.spec.name, pretrained=self.spec.pretrained, **kwargs
            )
        except Exception as exc:  # noqa: BLE001 - we want the loud, specific message
            raise RuntimeError(
                f"could not load CLIP backbone {self.spec.model_id}. If this environment has no "
                f"internet, pre-cache the weights into {self.cache_dir or '~/.cache'} first. "
                f"Underlying error: {exc}"
            ) from exc
        self._model = model.to(self.device).eval()
        self._preprocess = preprocess
        self._tokenizer = open_clip.get_tokenizer(self.spec.name)
        log.info("loaded %s on %s", self.spec.model_id, self.device)

    @property
    def logit_scale(self) -> float:
        """The backbone's learned inverse temperature (used as the default tau)."""
        self.load()
        return float(self._model.logit_scale.exp().detach().cpu())

    # -- text --------------------------------------------------------------
    @torch.no_grad()
    def encode_text(self, prompts: Sequence[str]) -> np.ndarray:
        """L2-normalised text embeddings, one row per prompt."""
        self.load()
        tokens = self._tokenizer(list(prompts)).to(self.device)
        emb = self._model.encode_text(tokens).float()
        return l2_normalize(emb.cpu().numpy())

    # -- images ------------------------------------------------------------
    @torch.no_grad()
    def encode_images(self, paths: Sequence[str], transform: Callable[[Image.Image], Image.Image] | None = None) -> np.ndarray:
        """L2-normalised image embeddings for a list of paths."""
        self.load()
        out: list[np.ndarray] = []
        for start in range(0, len(paths), self.batch_size):
            chunk = paths[start : start + self.batch_size]
            batch = []
            for p in chunk:
                img = Image.open(p).convert("RGB")
                if transform is not None:
                    img = transform(img)
                batch.append(self._preprocess(img))
            tensor = torch.stack(batch).to(self.device)
            emb = self._model.encode_image(tensor).float()
            out.append(l2_normalize(emb.cpu().numpy()))
        return np.concatenate(out, axis=0) if out else np.zeros((0, 512), np.float32)

    @torch.no_grad()
    def encode_windows(self, paths: Sequence[str], win: WindowSpec,
                       transform: Callable[[Image.Image], Image.Image] | None = None) -> np.ndarray:
        """Window embeddings of shape ``[n_images, n_windows, dim]``.

        Each window crop is resized to the backbone's native input and encoded
        independently - the WinCLIP recipe, without its harmonic aggregation,
        which we do downstream on cached arrays instead.
        """
        self.load()
        boxes = win.windows()
        out = np.zeros((len(paths), len(boxes), self._dim()), dtype=np.float16)
        for i, p in enumerate(paths):
            img = Image.open(p).convert("RGB")
            if transform is not None:
                img = transform(img)
            img = img.resize((win.image_size, win.image_size), Image.BICUBIC)
            crops = [self._preprocess(img.crop(b)) for b in boxes]
            embs: list[np.ndarray] = []
            for start in range(0, len(crops), self.batch_size):
                tensor = torch.stack(crops[start : start + self.batch_size]).to(self.device)
                e = self._model.encode_image(tensor).float()
                embs.append(l2_normalize(e.cpu().numpy()))
            out[i] = np.concatenate(embs, axis=0).astype(np.float16)
        return out

    def _dim(self) -> int:
        self.load()
        return int(self._model.text_projection.shape[-1]) if hasattr(self._model, "text_projection") \
            else int(self._model.visual.output_dim)


def corruption_transform(name: str, severity: int, seed: int = 0) -> Callable[[Image.Image], Image.Image] | None:
    """Build a PIL transform for a corruption, or ``None`` for the clean setting."""
    if name in ("none", "", None) or severity == 0:
        return None
    return lambda img: apply_corruption(img, name, severity, seed=seed)


def embed_index(
    embedder: ClipEmbedder,
    index: pd.DataFrame,
    cache: EmbeddingCache,
    corruption: str = "none",
    severity: int = 0,
    seed: int = 0,
    windows: WindowSpec | None = None,
    overwrite: bool = False,
) -> None:
    """Embed every (category, split) group of ``index`` into ``cache``.

    Skips groups whose shard already exists unless ``overwrite`` is set, so an
    interrupted run resumes for free.
    """
    tag = embedder.spec.cache_tag
    transform = corruption_transform(corruption, severity, seed)
    for (category, split), group in index.groupby(["category", "split"], sort=True):
        group = group.sort_values("image_id")
        keys = dict(model_tag=tag, category=category, split=split,
                    corruption=corruption, severity=severity)
        if overwrite or not cache.has(kind="image", **keys):
            emb = embedder.encode_images(group["path"].tolist(), transform)
            cache.save(group["image_id"].tolist(), emb, meta={"model_id": embedder.spec.model_id},
                       kind="image", **keys)
            log.info("embedded %-12s %-5s %4d images", category, split, len(group))
        if windows is not None and (overwrite or not cache.has(kind="window", **keys)):
            wemb = embedder.encode_windows(group["path"].tolist(), windows, transform)
            n, k, d = wemb.shape
            cache.save(group["image_id"].tolist(), wemb.reshape(n, k * d),
                       meta={"n_windows": k, "dim": d, "boxes": windows.windows(),
                             "image_size": windows.image_size},
                       kind="window", **keys)
            log.info("embedded %-12s %-5s %4d images x %d windows", category, split, n, k)
