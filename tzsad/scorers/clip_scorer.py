"""CLIP anomaly scorer, computed offline from cached embeddings.

Fixes applied relative to the prototype:

* the raw ``cos(anom) - cos(norm)`` difference is turned into the softmax
  probability the proposal specifies, ``s_sem = softmax([cos_norm, cos_anom]/tau)[1]``,
  so scores are bounded, comparable and usable in ECE/Brier/NLL (defect #4);
* per-template scores are retained, so the prompt-ensemble uncertainty baseline
  and its z-scored rescue can both be computed without a second forward pass;
* nothing here decides a threshold - that is the conformal layer's job (defect #3).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..features.cache import EmbeddingCache
from ..features.clip_embedder import BackboneSpec
from ..records import CORE_COLUMNS
from ..utils.logging import get_logger
from .base import Scorer, ScoringContext
from .prompts import PromptSet, class_name_for, get_prompt_set

log = get_logger("scorers.clip")


def softmax_anomaly_probability(cos_normal: np.ndarray, cos_anomaly: np.ndarray, tau: float) -> np.ndarray:
    """``exp(c_a/tau) / (exp(c_n/tau) + exp(c_a/tau))``, computed stably.

    Parameters
    ----------
    cos_normal, cos_anomaly:
        Cosine similarities to the normal / anomaly text embeddings, same shape.
    tau:
        Temperature. Smaller tau sharpens; the CLIP-native choice is
        ``1 / logit_scale`` (about 0.01), which is the default in the config.
    """
    d = (np.asarray(cos_anomaly, dtype=np.float64) - np.asarray(cos_normal, dtype=np.float64)) / tau
    return 1.0 / (1.0 + np.exp(-d))


@dataclass
class ClipScorerConfig:
    """Configuration of an offline CLIP scoring pass."""

    prompt_set: str = "base"
    tau: float = 0.01
    aggregate: str = "mean_embedding"   # 'mean_embedding' (WinCLIP-style) or 'mean_template'
    keep_per_template: bool = True


class ClipScorer(Scorer):
    """Scores a pre-built index using cached embeddings for one backbone.

    Two aggregations are supported:

    ``mean_embedding``
        Average the normal and anomaly text embeddings first, then take one
        softmax. This is the WinCLIP recipe and the primary score.
    ``mean_template``
        Score each template pair separately and average the probabilities. It
        exists so the per-template spread has a coherent definition.
    """

    def __init__(self, spec: BackboneSpec, cache: EmbeddingCache, config: ClipScorerConfig | None = None,
                 text_embeddings: dict[str, tuple[np.ndarray, np.ndarray]] | None = None) -> None:
        self.spec = spec
        self.cache = cache
        self.config = config or ClipScorerConfig()
        self.prompts: PromptSet = get_prompt_set(self.config.prompt_set)
        self._text = text_embeddings or {}
        self.model_id = spec.model_id
        self.scorer_name = f"clip:{spec.cache_tag}:{self.prompts.name}"

    # -- text embeddings ---------------------------------------------------
    def ensure_text(self, categories, embedder) -> None:
        """Compute and memoise text embeddings for each category (a tiny GPU pass)."""
        for category in categories:
            if category in self._text:
                continue
            normal, anomaly = self.prompts.render(class_name_for(category))
            self._text[category] = (embedder.encode_text(normal), embedder.encode_text(anomaly))

    @property
    def text_embeddings(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Per-category ``(normal, anomaly)`` text embeddings, for the heatmap stage."""
        return self._text

    def save_text(self, path: str | Path) -> None:
        """Persist text embeddings so scoring can be re-run with no GPU at all."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {}
        for cat, (n, a) in self._text.items():
            payload[f"{cat}__normal"] = n
            payload[f"{cat}__anomaly"] = a
        np.savez_compressed(path, **payload)

    def load_text(self, path: str | Path) -> None:
        """Load text embeddings written by :meth:`save_text`."""
        with np.load(path) as z:
            cats = sorted({k.split("__")[0] for k in z.files})
            self._text = {c: (z[f"{c}__normal"], z[f"{c}__anomaly"]) for c in cats}

    # -- scoring -----------------------------------------------------------
    def score_index(self, index: pd.DataFrame, ctx: ScoringContext) -> pd.DataFrame:
        """Score every image in ``index`` from the cache. No GPU involved."""
        frames = []
        for (category, split), group in index.groupby(["category", "split"], sort=True):
            group = group.sort_values("image_id")
            ids, emb, _ = self.cache.load(
                model_tag=self.spec.cache_tag, category=category, split=split, kind="image",
                corruption=ctx.corruption, severity=ctx.severity,
            )
            lookup = {i: k for k, i in enumerate(ids)}
            missing = [i for i in group["image_id"] if i not in lookup]
            if missing:
                raise KeyError(
                    f"{len(missing)} images missing from the {category}/{split} embedding cache "
                    f"(first: {missing[0]}). Re-run scoring with --overwrite-cache."
                )
            rows = emb[[lookup[i] for i in group["image_id"]]]
            frames.append(self._score_block(rows, group, category))
        records = pd.concat(frames, ignore_index=True)
        return self.finalize(records, ctx)

    def _score_block(self, emb: np.ndarray, group: pd.DataFrame, category: str) -> pd.DataFrame:
        if category not in self._text:
            raise KeyError(f"no text embeddings for category {category!r}; call ensure_text first")
        t_norm, t_anom = self._text[category]
        tau = float(self.config.tau)

        cos_n = emb @ t_norm.T                       # [n_images, n_templates]
        cos_a = emb @ t_anom.T
        per_template = softmax_anomaly_probability(cos_n, cos_a, tau)   # [n, T]
        raw_per_template = cos_a - cos_n

        if self.config.aggregate == "mean_embedding":
            mn = t_norm.mean(0) / max(np.linalg.norm(t_norm.mean(0)), 1e-8)
            ma = t_anom.mean(0) / max(np.linalg.norm(t_anom.mean(0)), 1e-8)
            score = softmax_anomaly_probability(emb @ mn, emb @ ma, tau)
            raw = (emb @ ma) - (emb @ mn)
        elif self.config.aggregate == "mean_template":
            score = per_template.mean(axis=1)
            raw = raw_per_template.mean(axis=1)
        else:
            raise KeyError(f"unknown aggregate {self.config.aggregate!r}")

        out = pd.DataFrame({
            "category": group["category"].to_numpy(),
            "defect_type": group["defect_type"].to_numpy(),
            "path": group["path"].to_numpy(),
            "image_id": group["image_id"].to_numpy(),
            "label": group["label"].to_numpy(),
            "split": group["split"].to_numpy(),
            "anomaly_score": score,
            "raw_score": raw,
            "parse_ok": True,
            "n_valid_votes": 1,
        })
        if self.config.keep_per_template:
            for t in range(per_template.shape[1]):
                out[f"tpl_{t:02d}"] = per_template[:, t]
                out[f"tplraw_{t:02d}"] = raw_per_template[:, t]
        return out


def template_columns(df: pd.DataFrame, raw: bool = False) -> list[str]:
    """Per-template score columns emitted by :class:`ClipScorer`."""
    prefix = "tplraw_" if raw else "tpl_"
    return sorted(c for c in df.columns if c.startswith(prefix))
