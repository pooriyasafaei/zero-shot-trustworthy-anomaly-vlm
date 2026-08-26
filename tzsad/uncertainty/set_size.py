"""Prediction-set-size uncertainty estimator."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..calibration.prediction_sets import SET_SIZE_UNCERTAINTY
from .base import EstimatorInfo, UncertaintyEstimator


class PredictionSetSize(UncertaintyEstimator):
    """Uncertainty from the size of the two-sided conformal prediction set.

    Kind
    ----
    Distributional + aleatoric. It is the only estimator in the suite that can
    express *both* "these two hypotheses are equally plausible" (size 2) and
    "neither hypothesis is plausible" (size 0), which no scalar function of a
    single one-sided p-value can distinguish.

    Mapping
    -------
    ``size 1 -> 0.0`` (committed), ``size 2 -> 0.5`` (ambiguous),
    ``size 0 -> 1.0`` (conforms to neither class). Size 0 outranks size 2 because
    "I recognise neither label" is a stronger warning than "I cannot choose".

    Failure modes
    -------------
    Coarse: it takes three values, so it cannot rank within a tier and its
    error-prediction AUROC is capped by that granularity - read it alongside a
    continuous signal rather than instead of one.

    More important: the **anomalous side is calibrated on synthetic defects**
    (CutPaste/NSA), because the zero-shot regime has no real anomalies to
    calibrate against. If the synthesis is unrepresentative of real defects, the
    anomalous p-value is miscalibrated and size-0 rates on real anomalies are a
    diagnostic, not a guarantee. Requires
    :class:`~tzsad.calibration.prediction_sets.ConformalPredictionSet` to have been
    applied first.
    """

    name = "set_size"
    info = EstimatorInfo(
        kind="distributional + aleatoric",
        inputs="set_size column from ConformalPredictionSet.transform",
        failure_modes="only three values; anomaly side calibrated on synthetic defects",
    )

    def compute(self, records: pd.DataFrame, **kwargs) -> pd.Series:
        if "set_size" not in records:
            raise KeyError(
                "no 'set_size' column; run ConformalPredictionSet.transform first. "
                "It needs a synthetic-anomaly calibration pool (scripts/make_synthetic.py)."
            )
        return pd.Series([SET_SIZE_UNCERTAINTY[int(s)] for s in records["set_size"]],
                         index=records.index, dtype=float)
