"""Detection and calibration metrics.

Every headline number here comes with a bootstrap interval, because the prototype's
comparisons (0.8228 vs 0.8095 on different image sets, no CIs) could not support
any claim at all.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


@dataclass(frozen=True)
class Interval:
    """A point estimate with a bootstrap confidence interval."""

    value: float
    lo: float
    hi: float
    n: int = 0

    def __str__(self) -> str:
        return f"{self.value:.4f} [{self.lo:.4f}, {self.hi:.4f}]"

    def as_dict(self, prefix: str = "") -> dict[str, float]:
        """Flatten into columns for a results table."""
        return {f"{prefix}value": self.value, f"{prefix}ci_lo": self.lo,
                f"{prefix}ci_hi": self.hi, f"{prefix}n": self.n}


def safe_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    """AUROC, returning NaN (not raising) when only one class is present."""
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=float)
    ok = np.isfinite(scores)
    if ok.sum() == 0 or len(np.unique(labels[ok])) < 2:
        return float("nan")
    return float(roc_auc_score(labels[ok], scores[ok]))


def safe_aupr(labels: np.ndarray, scores: np.ndarray) -> float:
    """Average precision, NaN when a class is missing."""
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=float)
    ok = np.isfinite(scores)
    if ok.sum() == 0 or len(np.unique(labels[ok])) < 2:
        return float("nan")
    return float(average_precision_score(labels[ok], scores[ok]))


def brier(labels: np.ndarray, probs: np.ndarray) -> float:
    """Brier score. Requires probabilities, which is why raw cosine diffs are unusable."""
    labels = np.asarray(labels, dtype=float)
    probs = np.asarray(probs, dtype=float)
    ok = np.isfinite(probs)
    return float(np.mean((probs[ok] - labels[ok]) ** 2)) if ok.any() else float("nan")


def nll(labels: np.ndarray, probs: np.ndarray, eps: float = 1e-12) -> float:
    """Mean negative log-likelihood of the binary labels under ``probs``."""
    labels = np.asarray(labels, dtype=float)
    p = np.clip(np.asarray(probs, dtype=float), eps, 1 - eps)
    ok = np.isfinite(p)
    return float(-np.mean(labels[ok] * np.log(p[ok]) + (1 - labels[ok]) * np.log(1 - p[ok]))) \
        if ok.any() else float("nan")


def expected_calibration_error(labels: np.ndarray, probs: np.ndarray, n_bins: int = 15,
                               adaptive: bool = True) -> float:
    """ECE with **adaptive** (equal-mass) binning by default.

    Equal-width binning is misleading here: CLIP softmax probabilities pile up in
    a narrow band, so most equal-width bins are empty and ECE silently reports the
    behaviour of two or three bins. Equal-mass bins put the same number of samples
    in each bin and give every region of the score range equal weight.
    """
    labels, probs = _finite_pairs(labels, probs)
    if labels.size == 0:
        return float("nan")
    edges = (np.quantile(probs, np.linspace(0, 1, n_bins + 1)) if adaptive
             else np.linspace(0.0, 1.0, n_bins + 1))
    edges = np.unique(edges)
    if edges.size < 2:
        return float(abs(probs.mean() - labels.mean()))
    idx = np.clip(np.searchsorted(edges, probs, side="right") - 1, 0, edges.size - 2)
    ece = 0.0
    for b in range(edges.size - 1):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(labels[m].mean() - probs[m].mean())
    return float(ece)


def maximum_calibration_error(labels: np.ndarray, probs: np.ndarray, n_bins: int = 15,
                              adaptive: bool = True, min_bin: int = 5) -> float:
    """Worst-bin calibration gap. ``min_bin`` guards against one-sample bins dominating."""
    labels, probs = _finite_pairs(labels, probs)
    if labels.size == 0:
        return float("nan")
    edges = (np.quantile(probs, np.linspace(0, 1, n_bins + 1)) if adaptive
             else np.linspace(0.0, 1.0, n_bins + 1))
    edges = np.unique(edges)
    if edges.size < 2:
        return float(abs(probs.mean() - labels.mean()))
    idx = np.clip(np.searchsorted(edges, probs, side="right") - 1, 0, edges.size - 2)
    gaps = [abs(labels[idx == b].mean() - probs[idx == b].mean())
            for b in range(edges.size - 1) if (idx == b).sum() >= min_bin]
    return float(max(gaps)) if gaps else float("nan")


def reliability_curve(labels: np.ndarray, probs: np.ndarray, n_bins: int = 15,
                      adaptive: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(mean_predicted, empirical_frequency, bin_counts)`` for a reliability diagram."""
    labels, probs = _finite_pairs(labels, probs)
    edges = (np.quantile(probs, np.linspace(0, 1, n_bins + 1)) if adaptive
             else np.linspace(0.0, 1.0, n_bins + 1))
    edges = np.unique(edges)
    if edges.size < 2:
        return np.array([probs.mean()]), np.array([labels.mean()]), np.array([labels.size])
    idx = np.clip(np.searchsorted(edges, probs, side="right") - 1, 0, edges.size - 2)
    conf, freq, cnt = [], [], []
    for b in range(edges.size - 1):
        m = idx == b
        if m.any():
            conf.append(probs[m].mean())
            freq.append(labels[m].mean())
            cnt.append(int(m.sum()))
    return np.array(conf), np.array(freq), np.array(cnt)


def error_prediction_auroc(correct: np.ndarray, uncertainty: np.ndarray) -> float:
    """**The headline test**: does the uncertainty signal rank errors above successes?

    AUROC of ``uncertainty`` as a detector of ``misclassified`` (label 1 = wrong).
    0.5 means the signal carries no information about whether the prediction is
    wrong; below 0.5 means it is anti-correlated, i.e. the model is *more*
    confident when it is wrong. The prototype compared group means with no
    significance test and read the direction backwards.
    """
    wrong = (~np.asarray(correct, dtype=bool)).astype(int)
    return safe_auroc(wrong, np.asarray(uncertainty, dtype=float))


def _finite_pairs(labels: np.ndarray, probs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels, dtype=float)
    probs = np.asarray(probs, dtype=float)
    ok = np.isfinite(probs) & np.isfinite(labels)
    return labels[ok], probs[ok]
