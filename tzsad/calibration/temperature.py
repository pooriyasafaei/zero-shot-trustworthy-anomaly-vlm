"""Temperature scaling - a labelled-data baseline, explicitly NOT zero-shot.

Kept because reviewers will ask how far the zero-shot conformal machinery is from
a method that gets to see anomaly labels. Any table row using this must carry the
``uses_labels=True`` flag so it is never confused with the zero-shot arms.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar


@dataclass
class TemperatureScaler:
    """One-parameter logistic recalibration of a probability-like score.

    Fits ``p' = sigmoid(logit(p) / T)`` by minimising NLL on labelled data.
    """

    temperature: float = 1.0
    uses_labels: bool = True

    def fit(self, scores: np.ndarray, labels: np.ndarray) -> "TemperatureScaler":
        """Fit ``T`` by NLL minimisation over a bounded 1-D search."""
        logits = _logit(np.asarray(scores, dtype=np.float64))
        y = np.asarray(labels, dtype=np.float64)
        if len(np.unique(y)) < 2:
            raise ValueError("temperature scaling needs both classes present")

        def nll(log_t: float) -> float:
            t = float(np.exp(log_t))
            p = 1.0 / (1.0 + np.exp(-logits / t))
            p = np.clip(p, 1e-12, 1 - 1e-12)
            return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

        res = minimize_scalar(nll, bounds=(np.log(1e-2), np.log(1e2)), method="bounded")
        self.temperature = float(np.exp(res.x))
        return self

    def transform(self, scores: np.ndarray) -> np.ndarray:
        """Apply the fitted temperature."""
        return 1.0 / (1.0 + np.exp(-_logit(np.asarray(scores, dtype=np.float64)) / self.temperature))


def _logit(p: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))
