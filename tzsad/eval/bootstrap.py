"""Bootstrap confidence intervals and the DeLong test for paired AUROC comparisons."""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
from scipy import stats

from .metrics import Interval, safe_auroc


def bootstrap_metric(labels: np.ndarray, scores: np.ndarray,
                     metric: Callable[[np.ndarray, np.ndarray], float] = safe_auroc,
                     n_boot: int = 1000, alpha: float = 0.05, seed: int = 0,
                     stratified: bool = True) -> Interval:
    """Percentile bootstrap CI for a metric of (labels, scores).

    ``stratified`` resamples within each class, which keeps every replicate from
    degenerating to a single class - the failure mode on small per-category test
    sets where a category may have only a handful of normal images.
    """
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=float)
    point = metric(labels, scores)
    rng = np.random.default_rng(seed)
    n = labels.size
    if n == 0 or not np.isfinite(point):
        return Interval(point, float("nan"), float("nan"), n)

    idx_by_class = [np.flatnonzero(labels == c) for c in np.unique(labels)] if stratified else None
    draws = []
    for _ in range(n_boot):
        if stratified:
            take = np.concatenate([rng.choice(ix, ix.size, replace=True) for ix in idx_by_class])
        else:
            take = rng.integers(0, n, n)
        v = metric(labels[take], scores[take])
        if np.isfinite(v):
            draws.append(v)
    if not draws:
        return Interval(point, float("nan"), float("nan"), n)
    return Interval(point, float(np.quantile(draws, alpha / 2)),
                    float(np.quantile(draws, 1 - alpha / 2)), n)


def paired_bootstrap_difference(labels: np.ndarray, scores_a: np.ndarray, scores_b: np.ndarray,
                                metric: Callable = safe_auroc, n_boot: int = 1000,
                                alpha: float = 0.05, seed: int = 0) -> tuple[Interval, float]:
    """CI for ``metric(a) - metric(b)`` on the *same* images, plus a two-sided p-value.

    Pairing is what makes the comparison legitimate: both scorers must have been
    run on an identical image subset, which the shared index guarantees.
    """
    labels = np.asarray(labels)
    a, b = np.asarray(scores_a, dtype=float), np.asarray(scores_b, dtype=float)
    point = metric(labels, a) - metric(labels, b)
    rng = np.random.default_rng(seed)
    idx_by_class = [np.flatnonzero(labels == c) for c in np.unique(labels)]
    draws = []
    for _ in range(n_boot):
        take = np.concatenate([rng.choice(ix, ix.size, replace=True) for ix in idx_by_class])
        d = metric(labels[take], a[take]) - metric(labels[take], b[take])
        if np.isfinite(d):
            draws.append(d)
    if not draws:
        return Interval(point, float("nan"), float("nan"), labels.size), float("nan")
    draws_arr = np.asarray(draws)
    p = 2.0 * min((draws_arr <= 0).mean(), (draws_arr >= 0).mean())
    return (Interval(point, float(np.quantile(draws_arr, alpha / 2)),
                     float(np.quantile(draws_arr, 1 - alpha / 2)), labels.size),
            float(min(p, 1.0)))


# --------------------------------------------------------------------------
# DeLong test
# --------------------------------------------------------------------------
def _compute_midrank(x: np.ndarray) -> np.ndarray:
    """Midranks of ``x`` (ties get their average rank), the DeLong helper."""
    order = np.argsort(x)
    sorted_x = x[order]
    n = x.size
    ranks = np.zeros(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n and sorted_x[j] == sorted_x[i]:
            j += 1
        ranks[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n, dtype=float)
    out[order] = ranks
    return out


def _fast_delong(predictions_sorted_transposed: np.ndarray, m: int) -> tuple[np.ndarray, np.ndarray]:
    """Sun & Xu (2014) fast DeLong. Returns (AUCs, covariance matrix)."""
    n = predictions_sorted_transposed.shape[1] - m
    positive = predictions_sorted_transposed[:, :m]
    negative = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty([k, m], dtype=float)
    ty = np.empty([k, n], dtype=float)
    tz = np.empty([k, m + n], dtype=float)
    for r in range(k):
        tx[r] = _compute_midrank(positive[r])
        ty[r] = _compute_midrank(negative[r])
        tz[r] = _compute_midrank(predictions_sorted_transposed[r])
    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, np.atleast_2d(delongcov)


def delong_test(labels: np.ndarray, scores_a: np.ndarray, scores_b: np.ndarray) -> dict[str, float]:
    """DeLong's test for two correlated ROC curves on the same samples.

    Returns ``auc_a``, ``auc_b``, ``delta``, ``z`` and the two-sided ``p_value``.
    Use this - not two independent CIs - to claim one scorer beats another, and
    only on identical image subsets.
    """
    labels = np.asarray(labels)
    a, b = np.asarray(scores_a, dtype=float), np.asarray(scores_b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    labels, a, b = labels[ok], a[ok], b[ok]
    if len(np.unique(labels)) < 2:
        return {"auc_a": float("nan"), "auc_b": float("nan"), "delta": float("nan"),
                "z": float("nan"), "p_value": float("nan"), "n": int(labels.size)}

    order = np.argsort(-labels, kind="mergesort")   # positives first
    labels, a, b = labels[order], a[order], b[order]
    m = int((labels == 1).sum())
    preds = np.vstack([a, b])
    aucs, cov = _fast_delong(preds, m)
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    delta = float(aucs[0] - aucs[1])
    if var <= 0:
        z, p = (0.0, 1.0) if delta == 0 else (float("inf"), 0.0)
    else:
        z = delta / float(np.sqrt(var))
        p = float(2 * stats.norm.sf(abs(z)))
    return {"auc_a": float(aucs[0]), "auc_b": float(aucs[1]), "delta": delta,
            "z": float(z), "p_value": p, "n": int(labels.size)}


def holm_bonferroni(p_values: Sequence[float], alpha: float = 0.05) -> list[bool]:
    """Holm-Bonferroni step-down correction; returns a reject/keep mask.

    Needed because a 15-category comparison run at alpha=0.05 will hand you a
    "significant" category by chance alone.
    """
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    n = p.size
    reject = np.zeros(n, dtype=bool)
    for rank, idx in enumerate(order):
        if p[idx] <= alpha / (n - rank):
            reject[idx] = True
        else:
            break
    return reject.tolist()
