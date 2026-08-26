"""Two-sided conformal prediction sets over {normal, anomalous}.

A one-sided conformal p-value answers only "is this conforming to the normal
class?". A prediction set answers both hypotheses at once, and its *size* carries
information that no scalar function of a single p-value can express:

============  ==================================================================
``size 1``    exactly one label survives - a confident, committed prediction
``size 2``    both labels survive - genuinely ambiguous, the classic abstention
``size 0``    **both labels rejected** - the input conforms to neither the normal
              manifold nor anything the anomaly calibration knows about. This is
              the out-of-distribution / silent-failure indicator, and it is the
              reason this module exists: it should light up under corruption while
              a scalar confidence stays flat.
============  ==================================================================

What size 0 can and cannot see
------------------------------
Both nonconformity measures are one-sided functions of the same scalar anomaly
score, so the empty set fires exactly when the score lands in the **gap between
the two calibration pools**: too anomalous to be a normal, too normal to be an
anomaly. That is the regime corruption drives inputs into, which is what makes it
a useful silent-failure indicator.

It is *not* a general OOD detector. An input whose score is more extreme than
every calibration point in the conforming direction gets ``p = 1`` on that side by
construction, so a wildly out-of-distribution image that happens to score very
high still yields the confident singleton ``{anomalous}``. Detecting that case
needs a nonconformity measure with its own view of the input - the normal-manifold
kNN distance - rather than a second reading of the same scalar. Report size-0 as
"fell into the undecidable gap", never as "detected OOD".

Honesty caveat
--------------
The normal side is calibrated on real ``train/good`` images, so its coverage
guarantee is sound. The **anomalous side is calibrated on synthetic defects**
(CutPaste/NSA over those same normal images), because the zero-shot regime has no
real anomalies to calibrate on. The anomaly-side p-value is therefore only as
representative as the synthesis is, and set behaviour on real anomalies must be
reported as a *diagnostic*, never as a coverage guarantee on the anomalous class.
Everything that depends on it is labelled ``synthetic_anomaly_calibrated=True``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..utils.logging import get_logger
from .conformal import conformal_pvalue

log = get_logger("calibration.prediction_sets")

#: Set size -> uncertainty in [0, 1]. Size 0 outranks size 2: "I recognise
#: neither hypothesis" is a stronger warning than "I cannot choose between them".
SET_SIZE_UNCERTAINTY: dict[int, float] = {1: 0.0, 2: 0.5, 0: 1.0}


@dataclass
class ConformalPredictionSet:
    """Mondrian two-sided conformal predictor over {normal=0, anomalous=1}.

    Nonconformity for the *normal* hypothesis is the anomaly score itself (a high
    score conforms badly to "normal"); for the *anomalous* hypothesis it is the
    negated score. Each side gets its own per-category calibration pool.
    """

    delta: float = 0.05
    seed: int = 0
    randomized: bool = False
    normal_scores_: dict[str, np.ndarray] = field(default_factory=dict)
    anomaly_scores_: dict[str, np.ndarray] = field(default_factory=dict)
    synthetic_anomaly_calibrated: bool = True

    def fit(self, normal_cal: pd.DataFrame, anomaly_cal: pd.DataFrame,
            score_col: str = "anomaly_score", group_col: str = "category") -> "ConformalPredictionSet":
        """Fit both calibration pools.

        Parameters
        ----------
        normal_cal:
            ``train/good`` records. Must contain no anomalous rows.
        anomaly_cal:
            Records for *synthetic* anomalies produced from those same normals.
            Real MVTec defects must never be used here - that would leak test-time
            label information into a method claiming to be zero-shot.
        """
        if "label" in normal_cal.columns and (normal_cal["label"] != 0).any():
            raise ValueError("the normal calibration pool must contain only normal images")
        self.normal_scores_, self.anomaly_scores_ = {}, {}
        for group, g in normal_cal.groupby(group_col, sort=True):
            v = g[score_col].to_numpy(dtype=float)
            self.normal_scores_[str(group)] = v[np.isfinite(v)]
        for group, g in anomaly_cal.groupby(group_col, sort=True):
            v = g[score_col].to_numpy(dtype=float)
            self.anomaly_scores_[str(group)] = v[np.isfinite(v)]
        missing = sorted(set(self.normal_scores_) - set(self.anomaly_scores_))
        if missing:
            log.warning("no synthetic-anomaly calibration for %s; those categories "
                        "will never reject the anomalous hypothesis", missing)
        return self

    def pvalues(self, test: pd.DataFrame, score_col: str = "anomaly_score",
                group_col: str = "category") -> tuple[np.ndarray, np.ndarray]:
        """``(p_normal, p_anomalous)`` for every test row."""
        if not test.index.is_unique:
            raise ValueError("test index must be unique; call .reset_index(drop=True)")
        p_norm = np.full(len(test), np.nan)
        p_anom = np.full(len(test), np.nan)
        rng = np.random.default_rng(self.seed)
        for group, g in test.groupby(group_col, sort=True):
            key = str(group)
            idx = test.index.get_indexer(g.index)
            s = g[score_col].to_numpy(dtype=float)
            if key in self.normal_scores_ and self.normal_scores_[key].size:
                p_norm[idx] = conformal_pvalue(self.normal_scores_[key], s,
                                               randomized=self.randomized, rng=rng)
            if key in self.anomaly_scores_ and self.anomaly_scores_[key].size:
                # Mirror the scale: low score conforms badly to "anomalous".
                p_anom[idx] = conformal_pvalue(-self.anomaly_scores_[key], -s,
                                               randomized=self.randomized, rng=rng)
        return p_norm, p_anom

    def transform(self, test: pd.DataFrame, score_col: str = "anomaly_score",
                  group_col: str = "category") -> pd.DataFrame:
        """Attach ``p_normal``, ``p_anomalous``, ``set_size``, ``set_label`` and ``u_set_size``."""
        out = test.copy()
        p_norm, p_anom = self.pvalues(test, score_col, group_col)
        keep_norm = p_norm > self.delta
        keep_anom = p_anom > self.delta
        size = keep_norm.astype(int) + keep_anom.astype(int)

        out["p_normal"] = p_norm
        out["p_anomalous"] = p_anom
        out["set_size"] = size
        out["set_label"] = [_set_label(n, a) for n, a in zip(keep_norm, keep_anom)]
        out["u_set_size"] = [SET_SIZE_UNCERTAINTY[int(s)] for s in size]
        out["set_is_ood"] = size == 0
        return out


def _set_label(keep_normal: bool, keep_anomalous: bool) -> str:
    if keep_normal and keep_anomalous:
        return "{normal,anomalous}"
    if keep_normal:
        return "{normal}"
    if keep_anomalous:
        return "{anomalous}"
    return "{}"


def set_size_report(records: pd.DataFrame, group_col: str = "category") -> pd.DataFrame:
    """Set-size distribution per group, split by true label.

    The columns to read: ``frac_size0`` is the OOD/silent-failure rate, and it is
    the one expected to climb under corruption. ``frac_size1_correct`` is the rate
    of confident *and* right, which is what a deployable system maximises.
    """
    rows = []
    for group, g in list(records.groupby(group_col, sort=True)) + [("POOLED", records)]:
        singleton = g[g.set_size == 1]
        correct_singleton = singleton[
            ((singleton.set_label == "{anomalous}") & (singleton.label == 1))
            | ((singleton.set_label == "{normal}") & (singleton.label == 0))]
        rows.append({
            "group": str(group), "n": len(g),
            "frac_size0": float((g.set_size == 0).mean()),
            "frac_size1": float((g.set_size == 1).mean()),
            "frac_size2": float((g.set_size == 2).mean()),
            "mean_set_size": float(g.set_size.mean()),
            "frac_size1_correct": float(len(correct_singleton) / len(g)) if len(g) else float("nan"),
            "singleton_accuracy": float(len(correct_singleton) / len(singleton)) if len(singleton) else float("nan"),
            "frac_size0_on_normal": float((g[g.label == 0].set_size == 0).mean()) if (g.label == 0).any() else float("nan"),
            "frac_size0_on_anomalous": float((g[g.label == 1].set_size == 0).mean()) if (g.label == 1).any() else float("nan"),
        })
    return pd.DataFrame(rows)
