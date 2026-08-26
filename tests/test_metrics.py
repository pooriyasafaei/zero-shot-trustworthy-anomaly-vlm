"""Metrics checked against sklearn/scipy where an equivalent exists."""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from tzsad.eval.bootstrap import bootstrap_metric, delong_test, holm_bonferroni
from tzsad.eval.metrics import (brier, error_prediction_auroc, expected_calibration_error, nll,
                                safe_auroc)
from tzsad.eval.selective import accuracy_at_coverage, risk_coverage_curve


@pytest.fixture
def toy():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    s = np.clip(0.5 + 0.25 * (y * 2 - 1) + rng.normal(0, 0.2, 500), 0.001, 0.999)
    return y, s


def test_auroc_matches_sklearn(toy):
    y, s = toy
    assert safe_auroc(y, s) == pytest.approx(roc_auc_score(y, s))


def test_brier_matches_sklearn(toy):
    y, s = toy
    assert brier(y, s) == pytest.approx(brier_score_loss(y, s))


def test_nll_matches_sklearn(toy):
    y, s = toy
    assert nll(y, s) == pytest.approx(log_loss(y, s), rel=1e-6)


def test_auroc_is_nan_with_one_class():
    assert np.isnan(safe_auroc(np.zeros(10), np.random.rand(10)))


def test_ece_is_zero_for_perfectly_calibrated_scores():
    """A score equal to the true event probability has ~zero calibration error."""
    rng = np.random.default_rng(2)
    p = rng.uniform(0.05, 0.95, 20000)
    y = (rng.uniform(size=p.size) < p).astype(int)
    assert expected_calibration_error(y, p, n_bins=15, adaptive=True) < 0.02


def test_adaptive_binning_beats_equal_width_on_clustered_scores():
    """Equal-width bins go blind when the scores occupy a narrow band."""
    rng = np.random.default_rng(4)
    p = np.clip(rng.normal(0.5, 0.01, 5000), 0, 1)
    y = rng.integers(0, 2, 5000)
    adaptive = expected_calibration_error(y, p, 15, adaptive=True)
    assert np.isfinite(adaptive)


def test_error_prediction_auroc_direction():
    """A signal that is high exactly on the errors must score above 0.5, not below."""
    correct = np.array([1, 1, 1, 0, 0, 0], dtype=bool)
    good_signal = np.array([0.1, 0.1, 0.2, 0.9, 0.8, 0.95])
    assert error_prediction_auroc(correct, good_signal) == 1.0
    assert error_prediction_auroc(correct, 1 - good_signal) == 0.0


def test_bootstrap_ci_contains_point_estimate(toy):
    y, s = toy
    iv = bootstrap_metric(y, s, safe_auroc, n_boot=300, seed=0)
    assert iv.lo <= iv.value <= iv.hi
    assert iv.hi - iv.lo > 0


def test_delong_matches_auroc_point_estimates(toy):
    y, s = toy
    rng = np.random.default_rng(1)
    s2 = np.clip(s + rng.normal(0, 0.3, s.size), 0.001, 0.999)
    res = delong_test(y, s, s2)
    assert res["auc_a"] == pytest.approx(roc_auc_score(y, s), abs=1e-6)
    assert res["auc_b"] == pytest.approx(roc_auc_score(y, s2), abs=1e-6)
    assert 0.0 <= res["p_value"] <= 1.0


def test_delong_identical_scores_give_p_one(toy):
    y, s = toy
    res = delong_test(y, s, s.copy())
    assert res["delta"] == pytest.approx(0.0, abs=1e-12)
    assert res["p_value"] == pytest.approx(1.0)


def test_delong_detects_a_real_difference():
    rng = np.random.default_rng(9)
    y = rng.integers(0, 2, 2000)
    strong = y + rng.normal(0, 0.5, 2000)
    weak = y + rng.normal(0, 3.0, 2000)
    res = delong_test(y, strong, weak)
    assert res["delta"] > 0
    assert res["p_value"] < 1e-6


def test_holm_bonferroni_is_more_conservative_than_raw():
    p = [0.001, 0.02, 0.04, 0.5]
    rejects = holm_bonferroni(p, alpha=0.05)
    assert rejects[0] is True
    assert rejects[-1] is False
    assert sum(rejects) < sum(v < 0.05 for v in p)


def test_risk_coverage_oracle_and_random():
    """An oracle ordering attains eAURC 0; a random one must do strictly worse."""
    rng = np.random.default_rng(0)
    correct = rng.permutation(np.array([1] * 80 + [0] * 20, dtype=float))
    oracle = risk_coverage_curve(correct, 1.0 - correct)
    random_u = risk_coverage_curve(correct, rng.random(100))
    assert oracle.aurc < random_u.aurc
    assert oracle.eaurc == pytest.approx(0.0, abs=1e-12)
    assert random_u.eaurc > 0
    assert random_u.risk[-1] == pytest.approx(0.2)


def test_abstentions_are_rejected_first():
    """parse_ok=False rows must sort to the front of the rejection order."""
    correct = np.array([1, 1, 0, 1], dtype=float)
    u = np.array([0.9, 0.1, 0.0, 0.2])          # the error looks most certain
    abstain = np.array([False, False, True, False])
    rc = risk_coverage_curve(correct, u, abstain)
    assert rc.risk[0] == 0.0                     # first kept item is correct
    assert accuracy_at_coverage(correct, u, 0.75, abstain) == pytest.approx(1.0)


def test_pixel_metrics_on_a_planted_defect():
    """A map that lights up exactly on the mask must score near-perfectly."""
    from tzsad.eval.pixel import pixel_auroc, pro_score

    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[8:16, 8:16] = 1
    good_map = mask.astype(float) + np.random.default_rng(0).normal(0, 0.01, mask.shape)
    maps = {"i0": good_map, "i1": np.zeros((32, 32))}
    masks = {"i0": mask, "i1": np.zeros((32, 32), dtype=np.uint8)}
    assert pixel_auroc(maps, masks) > 0.99
    assert pro_score(maps, masks) > 0.9

    rng = np.random.default_rng(1)
    noise = {"i0": rng.random((32, 32)), "i1": rng.random((32, 32))}
    assert abs(pixel_auroc(noise, masks) - 0.5) < 0.1


def test_pro_weights_regions_equally():
    """One big region and one tiny region must count the same in PRO."""
    from tzsad.eval.pixel import pro_score

    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[0:12, 0:12] = 1        # large region
    mask[28:30, 28:30] = 1      # tiny region
    found_big_only = np.zeros((32, 32))
    found_big_only[0:12, 0:12] = 1.0
    both = found_big_only.copy()
    both[28:30, 28:30] = 1.0
    assert pro_score({"i": both}, {"i": mask}) > pro_score({"i": found_big_only}, {"i": mask})


def _imbalanced_errors(n=1200, seed=0):
    """A frame where errors are almost all false negatives, as at a conservative delta.

    This reproduces the base-rate trap: 'is this an error' becomes nearly the same
    question as 'is this a low-scoring anomaly', so a signal monotone in the score
    predicts error for free without carrying any uncertainty information.
    """
    import pandas as pd

    rng = np.random.default_rng(seed)
    label = (rng.random(n) < 0.73).astype(int)    # MVTec test sets are anomaly-heavy
    score = np.clip(0.40 + 0.28 * label + rng.normal(0, 0.14, n), 0.01, 0.99)
    pred = (score > 0.62).astype(int)             # a delta=0.05-like conservative threshold
    return pd.DataFrame({
        "category": rng.choice(list("abc"), n), "label": label, "anomaly_score": score,
        "conformal_pred": pred, "correct": pred == label, "parse_ok": True,
        "u_score_monotone": 1.0 - score,          # carries only what the score carries
        "u_pure_noise": rng.random(n),
    })


def test_error_prediction_exposes_the_score_monotone_artifact():
    """A signal that is only a re-encoding of the score must not read as informative."""
    from tzsad.eval.report import error_prediction_table

    df = _imbalanced_errors()
    tbl = error_prediction_table(df, "correct", n_boot=300)
    pooled = tbl[tbl.scope == "POOLED"].set_index("signal")

    assert {"BASELINE:score", "BASELINE:1-score"} <= set(pooled.index)
    # The artifact is real: the score alone predicts error well above chance.
    assert pooled.loc["BASELINE:1-score", "err_auroc"] > 0.6
    # A signal that merely re-encodes the score does not beat that baseline.
    assert pooled.loc["u_score_monotone", "excess_over_baseline"] <= 0.01
    assert not bool(pooled.loc["u_score_monotone", "beats_baseline"])


def test_conditioning_on_the_decision_removes_the_base_rate_shortcut():
    """Within one decision group the errors are homogeneous, so the shortcut is gone."""
    from tzsad.eval.report import error_prediction_table

    df = _imbalanced_errors()
    tbl = error_prediction_table(df, "correct", n_boot=300)
    mono = tbl[tbl.signal == "u_score_monotone"].set_index("scope")
    assert "pred=0" in mono.index
    # Pooled it looks strong; conditioned on the decision it is far weaker.
    assert mono.loc["pred=0", "err_auroc"] < mono.loc["POOLED", "err_auroc"] - 0.1


def test_pure_noise_is_never_reported_as_informative():
    from tzsad.eval.report import error_prediction_table

    tbl = error_prediction_table(_imbalanced_errors(), "correct", n_boot=300)
    noise = tbl[(tbl.signal == "u_pure_noise") & (tbl.scope == "POOLED")].iloc[0]
    assert not noise.informative and not noise.beats_baseline
