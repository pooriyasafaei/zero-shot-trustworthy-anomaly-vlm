"""Split conformal prediction calibrated on normal-only data.

This replaces the prototype's per-category median threshold (defect #3), which
forced exactly 50% of every category to be predicted anomalous regardless of the
base rate and therefore invalidated every accuracy/TP/FP/FN number derived from it.

MVTec's ``train/good`` gives 200+ clean images per category for free, which is
exactly the regime split conformal needs: a calibration sample from the null
("normal") distribution and no anomaly labels at all. The zero-shot claim survives.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from ..utils.logging import get_logger

log = get_logger("calibration.conformal")


def conformal_quantile(cal_scores: np.ndarray, delta: float) -> float:
    """Finite-sample-corrected conformal threshold.

    Returns the ``ceil((n+1)(1-delta))``-th smallest calibration score, which
    guarantees ``P(s(X) <= q) >= 1 - delta`` for an exchangeable new normal ``X``.
    The plain empirical ``(1-delta)`` quantile does *not* have this guarantee and
    under-covers at small ``n``; when the corrected rank exceeds ``n`` the honest
    answer is ``+inf`` (the calibration set is too small to certify that level).

    Parameters
    ----------
    cal_scores:
        Anomaly scores of normal calibration images. Higher = more anomalous.
    delta:
        Target miscoverage (false-positive rate on normals).
    """
    s = np.sort(np.asarray(cal_scores, dtype=np.float64))
    n = s.size
    if n == 0:
        raise ValueError("empty calibration set")
    if not 0.0 < delta < 1.0:
        raise ValueError(f"delta must be in (0, 1), got {delta}")
    rank = math.ceil((n + 1) * (1.0 - delta))
    if rank > n:
        return float("inf")
    return float(s[rank - 1])


def conformal_pvalue(cal_scores: np.ndarray, test_scores: np.ndarray, randomized: bool = False,
                     rng: np.random.Generator | None = None) -> np.ndarray:
    """Marginal conformal p-values ``p(x) = (1 + #{i : r_i >= s(x)}) / (n + 1)``.

    Under exchangeability these are (super-)uniform on normal test points, which is
    what makes them a usable uncertainty scale and what the coverage diagnostic
    checks. Anomalies push ``p`` towards 0.

    Parameters
    ----------
    randomized:
        Break ties uniformly (smoothed conformal p-values). This restores *exact*
        uniformity when scores have atoms - which they do for the vote-fraction
        VLM scorer, whose scores take only 11 distinct values.
    """
    cal = np.asarray(cal_scores, dtype=np.float64)
    test = np.asarray(test_scores, dtype=np.float64)
    n = cal.size
    if n == 0:
        raise ValueError("empty calibration set")
    order = np.sort(cal)
    ge = n - np.searchsorted(order, test, side="left")     # #{r_i >= s}
    if not randomized:
        return (1.0 + ge) / (n + 1.0)
    gt = n - np.searchsorted(order, test, side="right")     # #{r_i > s}
    ties = ge - gt
    u = (rng or np.random.default_rng(0)).random(test.shape)
    return (1.0 + gt + u * (1.0 + ties)) / (n + 1.0)


def pvalue_uncertainty(p: np.ndarray, mode: str = "boundary", delta: float = 0.05,
                       tau_u: float = 1.0) -> np.ndarray:
    """Turn conformal p-values into an uncertainty in [0, 1].

    Modes
    -----
    ``symmetric``
        ``1 - 2|p - 0.5|`` as specified in the brief. Peaks at ``p = 0.5``.
    ``entropy``
        Bernoulli entropy ``H(p)`` in bits. Same peak location, softer shoulders.
    ``boundary`` (default)
        Peaks at the *decision boundary* ``p = delta`` instead of at 0.5, by
        mapping ``p`` through the piecewise-linear function that sends
        ``delta -> 0.5`` and then applying the symmetric rule.
    ``log``
        ``exp(-|log(p / delta)| / tau_u)``. Also peaks at ``p = delta``, but on a
        log scale: since ``p`` is uniform under the null, all of the discriminative
        action sits in the tail near 0, which a piecewise-linear rescaling
        compresses and a log one does not. ``tau_u`` sets the decay width in
        e-folds of ``p``.

    Note
    ----
    ``symmetric`` and ``entropy`` are maximal at ``p = 0.5``, i.e. at the *median
    of the normal calibration distribution* - a thoroughly typical normal image,
    not an ambiguous one, while the rejection boundary sits at ``p = delta``. They
    are kept as bake-off arms for comparability with the proposal, but they are
    **not** the default and must not feed the fusion stage: a combiner that
    consumes a baseline-grade signal manufactures the "nothing predicts error"
    outcome regardless of whether it is true.
    """
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-12, 1 - 1e-12)
    if mode == "symmetric":
        return 1.0 - 2.0 * np.abs(p - 0.5)
    if mode == "entropy":
        return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
    if mode == "boundary":
        q = np.where(p <= delta, 0.5 * p / delta, 0.5 + 0.5 * (p - delta) / (1.0 - delta))
        return 1.0 - 2.0 * np.abs(q - 0.5)
    if mode == "log":
        if tau_u <= 0:
            raise ValueError(f"tau_u must be positive, got {tau_u}")
        return np.exp(-np.abs(np.log(p / delta)) / tau_u)
    raise KeyError(f"unknown pvalue uncertainty mode {mode!r}")


#: Modes that peak away from the decision boundary. Kept for comparability, never
#: used as a default and never fused - see `pvalue_uncertainty`.
BASELINE_UNCERTAINTY_MODES: frozenset[str] = frozenset({"symmetric", "entropy"})


@dataclass
class MondrianConformal:
    """Class-conditional (per-category) split conformal calibrator.

    Calibrating per category also fixes cross-category score incomparability: a
    p-value is on the same scale for ``screw`` and ``carpet`` even when their raw
    CLIP margins are not.
    """

    delta: float = 0.05
    randomized: bool = False
    seed: int = 0
    cal_scores_: dict[str, np.ndarray] = field(default_factory=dict)
    thresholds_: dict[str, float] = field(default_factory=dict)

    def fit(self, cal: pd.DataFrame, score_col: str = "anomaly_score",
            group_col: str = "category", n_cal: int | None = None) -> "MondrianConformal":
        """Fit per-group thresholds from normal-only calibration records.

        Parameters
        ----------
        cal:
            Records with ``label == 0``. A non-normal row is a bug and raises.
        n_cal:
            Optional cap on calibration set size per group, for the ``n_cal``
            sensitivity study. Sampling is seeded and therefore reproducible.
        """
        if "label" in cal.columns and (cal["label"] != 0).any():
            raise ValueError(
                "conformal calibration must use normal-only data; got anomalous rows. "
                "That is the whole point of the zero-shot regime."
            )
        rng = np.random.default_rng(self.seed)
        self.cal_scores_, self.thresholds_ = {}, {}
        for group, g in cal.groupby(group_col, sort=True):
            s = g[score_col].to_numpy(dtype=np.float64)
            s = s[np.isfinite(s)]
            if n_cal is not None and s.size > n_cal:
                s = s[rng.choice(s.size, size=n_cal, replace=False)]
            if s.size == 0:
                log.warning("no finite calibration scores for %s; skipping", group)
                continue
            self.cal_scores_[str(group)] = s
            self.thresholds_[str(group)] = conformal_quantile(s, self.delta)
        return self

    def pvalues(self, test: pd.DataFrame, score_col: str = "anomaly_score",
                group_col: str = "category") -> np.ndarray:
        """Conformal p-values for test records, grouped by the Mondrian variable."""
        _require_unique_index(test)
        out = np.full(len(test), np.nan)
        rng = np.random.default_rng(self.seed + 1)
        for group, g in test.groupby(group_col, sort=True):
            key = str(group)
            if key not in self.cal_scores_:
                raise KeyError(f"no calibration data for group {key!r}; fit on all groups first")
            idx = test.index.get_indexer(g.index)
            out[idx] = conformal_pvalue(self.cal_scores_[key], g[score_col].to_numpy(),
                                        randomized=self.randomized, rng=rng)
        return out

    def predict(self, test: pd.DataFrame, score_col: str = "anomaly_score",
                group_col: str = "category") -> np.ndarray:
        """Binary anomaly decisions at the fitted per-group thresholds."""
        _require_unique_index(test)
        out = np.zeros(len(test), dtype=int)
        for group, g in test.groupby(group_col, sort=True):
            thr = self.thresholds_[str(group)]
            idx = test.index.get_indexer(g.index)
            out[idx] = (g[score_col].to_numpy() > thr).astype(int)
        return out

    def transform(self, test: pd.DataFrame, score_col: str = "anomaly_score",
                  group_col: str = "category", uncertainty_mode: str = "boundary",
                  tau_u: float = 1.0) -> pd.DataFrame:
        """Attach ``conformal_p``, ``conformal_pred`` and ``u_conformal`` columns."""
        out = test.copy()
        out["conformal_p"] = self.pvalues(test, score_col, group_col)
        out["conformal_pred"] = self.predict(test, score_col, group_col)
        out["u_conformal"] = pvalue_uncertainty(out["conformal_p"].to_numpy(),
                                                uncertainty_mode, self.delta, tau_u)
        out["conformal_mode"] = uncertainty_mode   # not u_*: that prefix means "numeric signal"
        return out


def coverage_report(test: pd.DataFrame, calibrator: MondrianConformal,
                    score_col: str = "anomaly_score", group_col: str = "category",
                    n_boot: int = 1000, seed: int = 0,
                    include_calibration_term: bool = True) -> pd.DataFrame:
    """Empirical coverage on *normal* test images vs the nominal ``1 - delta``.

    Coverage is the fraction of normal test images that fall below the threshold,
    i.e. are correctly *not* flagged. Under exchangeability it should be at least
    ``1 - delta``; systematic under-coverage means the train/good pool and the
    test-good pool are not exchangeable, which is itself a finding.

    Two sources of noise
    --------------------
    Observed coverage varies for two independent reasons, and a CI over only one
    of them is too narrow:

    ``ci_lo``/``ci_hi``
        Bootstrap over the **test** images. This is the binomial term, and on
        MVTec it dominates (bottle has 20 normal test images: sd ~0.049).
    ``calib_sd``
        The **calibration-draw** term. Conditional coverage given one calibration
        set of size ``n`` is ``Beta(rank, n + 1 - rank)``, whose sd is ~0.021 at
        ``n=200, delta=0.1`` and ~0.015 at ``delta=0.05``. Reported separately, and
        folded into ``ci_lo_total``/``ci_hi_total`` when
        ``include_calibration_term`` is set.

    Quoting ``ci_lo``/``ci_hi`` alone answers "would this calibration set cover
    correctly on a fresh test set", not "does the method cover".
    """
    rows = []
    rng = np.random.default_rng(seed)
    normals = test[test["label"] == 0]
    for group, g in normals.groupby(group_col, sort=True):
        thr = calibrator.thresholds_.get(str(group), float("nan"))
        below = (g[score_col].to_numpy() <= thr).astype(float)
        lo, hi = _bootstrap_ci(below, np.mean, n_boot, rng)
        n_cal = calibrator.cal_scores_.get(str(group), np.empty(0)).size
        c_sd = calibration_coverage_sd(n_cal, calibrator.delta)
        rows.append({"group": str(group), "n_normal_test": len(g), "nominal": 1 - calibrator.delta,
                     "empirical_coverage": float(below.mean()), "ci_lo": lo, "ci_hi": hi,
                     "n_cal": n_cal, "calib_sd": c_sd,
                     "ci_lo_total": _widen(lo, float(below.mean()), c_sd, -1, include_calibration_term),
                     "ci_hi_total": _widen(hi, float(below.mean()), c_sd, +1, include_calibration_term),
                     "threshold": thr})
    pooled = (normals[score_col].to_numpy() <= np.array(
        [calibrator.thresholds_.get(str(c), np.nan) for c in normals[group_col]])).astype(float)
    lo, hi = _bootstrap_ci(pooled, np.mean, n_boot, rng)
    sizes = [v.size for v in calibrator.cal_scores_.values()]
    # Pooling averages over independent per-category calibration draws, so the
    # calibration term shrinks by sqrt(number of categories).
    pooled_sd = (float(np.mean([calibration_coverage_sd(n, calibrator.delta) for n in sizes]))
                 / max(np.sqrt(len(sizes)), 1.0)) if sizes else float("nan")
    rows.append({"group": "POOLED", "n_normal_test": len(normals), "nominal": 1 - calibrator.delta,
                 "empirical_coverage": float(pooled.mean()), "ci_lo": lo, "ci_hi": hi,
                 "n_cal": int(np.sum(sizes)), "calib_sd": pooled_sd,
                 "ci_lo_total": _widen(lo, float(pooled.mean()), pooled_sd, -1, include_calibration_term),
                 "ci_hi_total": _widen(hi, float(pooled.mean()), pooled_sd, +1, include_calibration_term),
                 "threshold": np.nan})
    return pd.DataFrame(rows)


def calibration_coverage_sd(n_cal: int, delta: float) -> float:
    """SD of conditional coverage given one calibration set of size ``n_cal``.

    With the conformal threshold at order statistic ``k = ceil((n+1)(1-delta))``,
    coverage given that draw is ``Beta(k, n + 1 - k)``; this returns its sd.
    """
    if n_cal <= 0:
        return float("nan")
    k = math.ceil((n_cal + 1) * (1.0 - delta))
    if k > n_cal:
        return float("nan")            # threshold is +inf; coverage is degenerate at 1
    a, b = float(k), float(n_cal + 1 - k)
    return float(np.sqrt(a * b / ((a + b) ** 2 * (a + b + 1))))


def _widen(bound: float, centre: float, calib_sd: float, sign: int, enabled: bool) -> float:
    """Fold the calibration-draw sd into a bootstrap bound in quadrature."""
    if not enabled or not np.isfinite(calib_sd) or not np.isfinite(bound):
        return bound
    half = abs(bound - centre)
    return float(np.clip(centre + sign * np.sqrt(half**2 + (1.96 * calib_sd) ** 2), 0.0, 1.0))


def _require_unique_index(df: pd.DataFrame) -> None:
    """Guard against a duplicated index, which makes positional lookup silently wrong."""
    if not df.index.is_unique:
        raise ValueError(
            "records index is not unique; call .reset_index(drop=True) first. "
            "A duplicated index makes get_indexer scatter values to the wrong rows."
        )


def _bootstrap_ci(values: np.ndarray, stat, n_boot: int, rng: np.random.Generator,
                  alpha: float = 0.05) -> tuple[float, float]:
    if values.size == 0 or n_boot <= 0:
        return float("nan"), float("nan")
    draws = [stat(values[rng.integers(0, values.size, values.size)]) for _ in range(n_boot)]
    return float(np.quantile(draws, alpha / 2)), float(np.quantile(draws, 1 - alpha / 2))


def n_cal_sensitivity(cal: pd.DataFrame, test: pd.DataFrame, deltas: Iterable[float] = (0.05,),
                      n_cals: Iterable[int | None] = (25, 50, 100, 200, None),
                      score_col: str = "anomaly_score", group_col: str = "category",
                      seed: int = 0, n_repeats: int = 5) -> pd.DataFrame:
    """How coverage degrades as the calibration set shrinks.

    Repeats the calibration draw ``n_repeats`` times per setting so the reported
    coverage spread reflects calibration-set randomness, not just test noise.
    """
    rows = []
    for delta in deltas:
        for n_cal in n_cals:
            covs = []
            for rep in range(n_repeats if n_cal is not None else 1):
                calib = MondrianConformal(delta=delta, seed=seed + rep).fit(
                    cal, score_col, group_col, n_cal=n_cal)
                rep_cov = coverage_report(test, calib, score_col, group_col, n_boot=0)
                covs.append(float(rep_cov.loc[rep_cov.group == "POOLED", "empirical_coverage"].iloc[0]))
            rows.append({"delta": delta, "n_cal": n_cal if n_cal is not None else -1,
                         "nominal": 1 - delta, "coverage_mean": float(np.mean(covs)),
                         "coverage_std": float(np.std(covs)), "n_repeats": len(covs)})
    return pd.DataFrame(rows)
