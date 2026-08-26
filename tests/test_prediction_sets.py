"""Two-sided conformal prediction sets, and the size-0 OOD indicator."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tzsad.calibration.prediction_sets import (SET_SIZE_UNCERTAINTY, ConformalPredictionSet,
                                               set_size_report)
from tzsad.uncertainty.set_size import PredictionSetSize


def _pools(n=400, seed=0):
    """Normal scores near 0.3, synthetic-anomaly scores near 0.7."""
    rng = np.random.default_rng(seed)
    normal = pd.DataFrame({"category": ["a"] * n, "label": 0,
                           "anomaly_score": rng.normal(0.3, 0.05, n)})
    synth = pd.DataFrame({"category": ["a"] * n, "synthetic_anomaly": 1,
                          "anomaly_score": rng.normal(0.7, 0.05, n)})
    return normal, synth


def _fitted(delta=0.05, seed=0):
    normal, synth = _pools(seed=seed)
    return ConformalPredictionSet(delta=delta, seed=seed).fit(normal, synth)


def test_confident_normal_yields_the_singleton_normal_set():
    cps = _fitted()
    test = pd.DataFrame({"category": ["a"], "anomaly_score": [0.30], "label": [0]})
    out = cps.transform(test)
    assert out.set_label.iloc[0] == "{normal}"
    assert out.set_size.iloc[0] == 1
    assert out.u_set_size.iloc[0] == 0.0


def test_confident_anomaly_yields_the_singleton_anomalous_set():
    cps = _fitted()
    out = cps.transform(pd.DataFrame({"category": ["a"], "anomaly_score": [0.70], "label": [1]}))
    assert out.set_label.iloc[0] == "{anomalous}"
    assert out.set_size.iloc[0] == 1


def test_overlapping_pools_yield_the_ambiguous_two_element_set():
    """When the two calibration pools overlap, the overlap region keeps both labels."""
    rng = np.random.default_rng(1)
    normal = pd.DataFrame({"category": ["a"] * 400, "label": 0,
                           "anomaly_score": rng.normal(0.45, 0.15, 400)})
    synth = pd.DataFrame({"category": ["a"] * 400, "synthetic_anomaly": 1,
                          "anomaly_score": rng.normal(0.55, 0.15, 400)})
    cps = ConformalPredictionSet(delta=0.05, seed=1).fit(normal, synth)
    out = cps.transform(pd.DataFrame({"category": ["a"], "anomaly_score": [0.50], "label": [1]}))
    assert out.set_size.iloc[0] == 2
    assert out.set_label.iloc[0] == "{normal,anomalous}"
    assert out.u_set_size.iloc[0] == 0.5


def test_a_score_in_the_gap_between_separated_pools_yields_the_empty_set():
    """Size 0 fires in the undecidable gap: too anomalous to be normal, too normal
    to be an anomaly. This is the regime corruption pushes inputs into."""
    cps = _fitted()                       # pools at 0.30 and 0.70, sd 0.05
    out = cps.transform(pd.DataFrame({"category": ["a"], "anomaly_score": [0.50], "label": [0]}))
    assert out.set_size.iloc[0] == 0
    assert out.set_label.iloc[0] == "{}"
    assert out.u_set_size.iloc[0] == 1.0
    assert bool(out.set_is_ood.iloc[0])


@pytest.mark.parametrize("score,expected", [(-2.0, "{normal}"), (5.0, "{anomalous}")])
def test_scores_beyond_a_pool_commit_rather_than_abstain(score, expected):
    """A documented limitation, pinned so it is never mistaken for OOD detection.

    Both nonconformity measures are one-sided in the same scalar score, so a point
    more extreme than every calibration point *in the conforming direction* gets
    p=1 there by construction. Size 0 means "fell into the gap", not "is OOD".
    """
    cps = _fitted()
    out = cps.transform(pd.DataFrame({"category": ["a"], "anomaly_score": [score], "label": [0]}))
    assert out.set_label.iloc[0] == expected
    assert out.set_size.iloc[0] == 1
    assert not bool(out.set_is_ood.iloc[0])


def test_size0_outranks_size2_in_uncertainty():
    """'I recognise neither label' must warn louder than 'I cannot choose'."""
    assert SET_SIZE_UNCERTAINTY[0] > SET_SIZE_UNCERTAINTY[2] > SET_SIZE_UNCERTAINTY[1]


def test_normal_side_keeps_its_conformal_coverage():
    """The normal hypothesis is calibrated on real normals, so coverage must hold."""
    rng = np.random.default_rng(5)
    covered = []
    for seed in range(30):
        cps = _fitted(delta=0.1, seed=seed)
        fresh = pd.DataFrame({"category": ["a"] * 200, "label": 0,
                              "anomaly_score": rng.normal(0.3, 0.05, 200)})
        out = cps.transform(fresh)
        covered.append(float((out.p_normal > 0.1).mean()))
    assert abs(float(np.mean(covered)) - 0.9) < 0.03


def test_size0_rate_rises_as_inputs_drift_into_the_gap():
    """The behaviour the corruption sweep depends on: drift -> undecidable."""
    cps = _fitted()                       # pools at 0.30 and 0.70
    rng = np.random.default_rng(2)
    rates = []
    for centre in (0.30, 0.38, 0.44, 0.50):     # drifting from the normal pool into the gap
        test = pd.DataFrame({"category": ["a"] * 400, "label": 0,
                             "anomaly_score": rng.normal(centre, 0.05, 400)})
        rates.append(float((cps.transform(test).set_size == 0).mean()))
    assert rates == sorted(rates), rates
    assert rates[0] < 0.2 and rates[-1] > 0.9, rates


def test_real_anomalies_never_calibrate_the_anomalous_side():
    normal, synth = _pools()
    bad = normal.copy()
    bad.loc[0, "label"] = 1
    with pytest.raises(ValueError, match="only normal images"):
        ConformalPredictionSet().fit(bad, synth)


def test_estimator_reports_the_synthetic_calibration_caveat():
    assert "synthetic" in PredictionSetSize.info.failure_modes.lower()
    assert ConformalPredictionSet().synthetic_anomaly_calibrated is True


def test_estimator_fails_loudly_without_the_set_size_column():
    with pytest.raises(KeyError, match="ConformalPredictionSet"):
        PredictionSetSize().compute(pd.DataFrame({"anomaly_score": [0.5]}))


def test_set_size_report_columns():
    cps = _fitted()
    rng = np.random.default_rng(3)
    test = pd.DataFrame({"category": ["a"] * 100, "label": rng.integers(0, 2, 100),
                         "anomaly_score": rng.uniform(0.2, 0.8, 100)})
    rep = set_size_report(cps.transform(test))
    assert rep.iloc[-1]["group"] == "POOLED"
    assert {"frac_size0", "frac_size1", "frac_size2", "singleton_accuracy"} <= set(rep.columns)
    row = rep.iloc[-1]
    assert row.frac_size0 + row.frac_size1 + row.frac_size2 == pytest.approx(1.0)
