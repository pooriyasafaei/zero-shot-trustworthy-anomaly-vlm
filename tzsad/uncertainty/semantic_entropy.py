"""Semantic entropy over free-text VLM observations (Kuhn et al. 2023; Farquhar et al. 2024).

The prototype computed an "entropy" over ``(vote, canonicalized defect string)``
pairs using a hand-written synonym dictionary. That measures string diversity in
whatever vocabulary the dictionary happens to cover, and collapses to the vote
entropy whenever the dictionary misses a word.

This module implements the real thing: cluster the sampled generations by
**bidirectional entailment**, then take the entropy over clusters. Two generations
mean the same thing iff each entails the other *in the context of the question*.

Two backends:

``nli``
    A small NLI model (default ``microsoft/deberta-large-mnli``). Faithful to the paper.
``embedding``
    Sentence-embedding cosine similarity above a threshold, used symmetrically.
    A cheap fallback when the NLI model will not fit alongside the VLM; it is a
    weaker approximation and is labelled as such in the results.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..utils.logging import get_logger

log = get_logger("uncertainty.semantic_entropy")


@dataclass
class ClusteringConfig:
    """How to decide that two generations mean the same thing."""

    backend: str = "embedding"                     # 'nli' or 'embedding'
    nli_model: str = "microsoft/deberta-large-mnli"
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embed_threshold: float = 0.85
    entail_threshold: float = 0.5
    device: str = "auto"
    batch_size: int = 32


class BidirectionalEntailmentClusterer:
    """Greedy semantic clustering by bidirectional entailment.

    Following Kuhn et al., clusters are built greedily: a generation joins the
    first existing cluster whose representative it mutually entails, otherwise it
    starts a new cluster. Greedy assignment makes the relation an equivalence
    relation by fiat (entailment is not transitive in general), which is the same
    approximation the original work makes.
    """

    def __init__(self, config: ClusteringConfig | None = None) -> None:
        self.config = config or ClusteringConfig()
        self._model = None
        self._tokenizer = None
        self._embedder = None

    # -- backends ----------------------------------------------------------
    def _device(self) -> str:
        if self.config.device != "auto":
            return self.config.device
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def _load_nli(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.config.nli_model)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self.config.nli_model).to(self._device()).eval()
        labels = {v.lower(): k for k, v in self._model.config.id2label.items()}
        self._entail_idx = labels.get("entailment", 2)
        log.info("loaded NLI model %s (entailment index %d)", self.config.nli_model, self._entail_idx)

    def _load_embedder(self) -> None:
        if self._embedder is not None:
            return
        from sentence_transformers import SentenceTransformer

        self._embedder = SentenceTransformer(self.config.embed_model, device=self._device())
        log.info("loaded sentence embedder %s", self.config.embed_model)

    # -- similarity --------------------------------------------------------
    def _entails(self, premises: Sequence[str], hypotheses: Sequence[str]) -> np.ndarray:
        """Entailment probability for each (premise, hypothesis) pair."""
        import torch

        self._load_nli()
        probs: list[float] = []
        for start in range(0, len(premises), self.config.batch_size):
            p = list(premises[start : start + self.config.batch_size])
            h = list(hypotheses[start : start + self.config.batch_size])
            enc = self._tokenizer(p, h, return_tensors="pt", padding=True,
                                  truncation=True, max_length=256).to(self._device())
            with torch.no_grad():
                logits = self._model(**enc).logits
            probs.extend(torch.softmax(logits, dim=-1)[:, self._entail_idx].cpu().tolist())
        return np.asarray(probs)

    def _embed(self, texts: Sequence[str]) -> np.ndarray:
        self._load_embedder()
        emb = self._embedder.encode(list(texts), convert_to_numpy=True,
                                    normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(emb, dtype=np.float32)

    # -- clustering --------------------------------------------------------
    def cluster(self, generations: Sequence[str], context: str = "") -> list[int]:
        """Assign each generation a cluster id.

        Parameters
        ----------
        generations:
            The free-text answers to cluster (the ``OBSERVATION`` fields).
        context:
            The question, prepended to both sides for the entailment check as in
            the original method - "the object is scratched" and "the object is
            fine" are only contradictory *given* the question being asked.
        """
        texts = [g.strip() for g in generations]
        n = len(texts)
        if n == 0:
            return []
        if n == 1:
            return [0]
        if self.config.backend == "embedding":
            return self._cluster_embedding(texts)
        return self._cluster_nli(texts, context)

    def _cluster_embedding(self, texts: Sequence[str]) -> list[int]:
        emb = self._embed(texts)
        sims = emb @ emb.T
        assign = [-1] * len(texts)
        reps: list[int] = []
        for i in range(len(texts)):
            for c, rep in enumerate(reps):
                if sims[i, rep] >= self.config.embed_threshold:
                    assign[i] = c
                    break
            if assign[i] == -1:
                assign[i] = len(reps)
                reps.append(i)
        return assign

    def _cluster_nli(self, texts: Sequence[str], context: str) -> list[int]:
        prefix = f"{context} " if context else ""
        assign = [-1] * len(texts)
        reps: list[int] = []
        for i in range(len(texts)):
            if not reps:
                assign[i] = 0
                reps.append(i)
                continue
            prem = [prefix + texts[reps[c]] for c in range(len(reps))]
            hyp = [prefix + texts[i]] * len(reps)
            fwd = self._entails(prem, hyp)
            bwd = self._entails(hyp, prem)
            mutual = np.minimum(fwd, bwd)
            best = int(np.argmax(mutual))
            if mutual[best] >= self.config.entail_threshold:
                assign[i] = best
            else:
                assign[i] = len(reps)
                reps.append(i)
        return assign


def cluster_entropy(assignments: Sequence[int], normalize: bool = True) -> float:
    """Discrete semantic entropy over cluster assignments.

    ``normalize`` divides by ``log(n_samples)`` so the value is in [0, 1] and is
    comparable across images with different numbers of *valid* generations - which
    matters here because unparseable generations reduce the sample count.
    """
    n = len(assignments)
    if n == 0:
        return float("nan")
    if n == 1:
        return 0.0
    _, counts = np.unique(np.asarray(assignments), return_counts=True)
    p = counts / n
    h = float(-(p * np.log(p)).sum())
    return h / np.log(n) if normalize else h


def joint_signature_entropy(votes: Sequence[int], clusters: Sequence[int], normalize: bool = True) -> float:
    """Entropy over ``(verdict, semantic cluster)`` pairs.

    The verdict is the quantity the task is scored on, so a model that says YES
    every time but tells a different story each time is not as uncertain *about
    the decision* as cluster entropy alone suggests. This joint form keeps the
    prototype's intent while replacing its synonym dictionary with real clustering.
    """
    if len(votes) != len(clusters):
        raise ValueError("votes and clusters must align")
    pairs = list(zip(votes, clusters))
    n = len(pairs)
    if n == 0:
        return float("nan")
    if n == 1:
        return 0.0
    _, counts = np.unique(np.array([f"{v}|{c}" for v, c in pairs]), return_counts=True)
    p = counts / n
    h = float(-(p * np.log(p)).sum())
    return h / np.log(n) if normalize else h
