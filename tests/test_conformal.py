"""Conformal guarantees on synthetic data, where the answer is checkable analytically."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from tzsad.calibration.conformal import (MondrianConformal, conformal_pvalue, conformal_quantile,
                                         coverage_report, pvalue_uncertainty)


def test_quantile_uses_finite_sample_correction():
    """The threshold is the ceil((n+1)(1-delta))-th order statistic, not the plain quantile."""
    scores = np.arange(1, 101, dtype=float)          # 1..100
    q = conformal_quantile(scores, 0.05)
    assert q == float(scores[math.ceil(101 * 0.95) - 1])
    assert q == 96.0
    assert q > float(np.quantile(scores, 0.95))      # strictly more conservative


def test_quantile_returns_inf_when_calibration_set_too_small():
    """With n=10 you cannot certify 1% miscoverage; say so instead of pretending."""
    assert conformal_quantile(np.arange(10, dtype=float), 0.01) == float("inf")


@pytest.mark.parametrize("delta", [0.01, 0.05, 0.1, 0.2])
def test_marginal_coverage_holds_on_exchangeable_data(delta):
    """P(s <= q) >= 1 - delta for exchangeable normals, averaged over many draws."""
    rng = np.random.default_rng(7)
    covered = []
    for _ in range(400):
        cal = rng.normal(size=200)
        test = rng.normal(size=50)
        q = conformal_quantile(cal, delta)
        covered.append(np.mean(test <= q))
    empirical = float(np.mean(covered))
    assert empirical >= (1 - delta) - 0.02, f"under-coverage: {empirical:.3f} vs {1-delta:.3f}"


def test_pvalues_are_uniform_on_normals():
    """Conformal p-values are (super-)uniform under exchangeability."""
    rng = np.random.default_rng(3)
    cal = rng.normal(size=500)
    test = rng.normal(size=5000)
    p = conformal_pvalue(cal, test)
    assert 0.0 < p.min() and p.max() <= 1.0
    for level in (0.05, 0.1, 0.25, 0.5):
        assert abs(float((p <= level).mean()) - level) < 0.03


def test_pvalues_drop_for_anomalies():
    rng = np.random.default_rng(11)
    cal = rng.normal(size=500)
    anomalies = rng.normal(loc=4.0, size=200)
    p = conformal_pvalue(cal, anomalies)
    assert float(np.median(p)) < 0.01


def test_randomized_pvalues_are_uniform_with_heavy_ties():
    """Vote-fraction scores have atoms; smoothing restores exact uniformity."""
    rng = np.random.default_rng(5)
    cal = rng.integers(0, 11, size=1000).astype(float) / 10.0
    test = rng.integers(0, 11, size=5000).astype(float) / 10.0
    plain = conformal_pvalue(cal, test)
    smooth = conformal_pvalue(cal, test, randomized=True, rng=rng)
    assert abs(float((smooth <= 0.5).mean()) - 0.5) < abs(float((plain <= 0.5).mean()) - 0.5) + 1e-9
    assert abs(float((smooth <= 0.2).mean()) - 0.2) < 0.03


def test_mondrian_rejects_anomalous_calibration_data():
    df = pd.DataFrame({"category": ["a"] * 5, "anomaly_score": [0.1] * 5, "label": [0, 0, 1, 0, 0]})
    with pytest.raises(ValueError, match="normal-only"):
        MondrianConformal().fit(df)


def test_mondrian_is_per_category():
    """Two categories on wildly different score scales both get valid thresholds."""
    rng = np.random.default_rng(1)
    cal = pd.DataFrame({
        "category": ["a"] * 200 + ["b"] * 200,
        "anomaly_score": np.concatenate([rng.normal(0.1, 0.01, 200), rng.normal(0.9, 0.01, 200)]),
        "label": 0,
    })
    calib = MondrianConformal(delta=0.1, seed=0).fit(cal)
    assert calib.thresholds_["a"] < 0.5 < calib.thresholds_["b"]

    # The conformal guarantee is *marginal* over the calibration draw, so the
    # calibration set must be redrawn each repeat. See the conditional-coverage
    # test below for why a fixed calibration set is not enough.
    covs = []
    for _ in range(60):
        cal = pd.DataFrame({
            "category": ["a"] * 200 + ["b"] * 200,
            "anomaly_score": np.concatenate([rng.normal(0.1, 0.01, 200),
                                             rng.normal(0.9, 0.01, 200)]),
            "label": 0,
        })
        calib = MondrianConformal(delta=0.1, seed=0).fit(cal)
        test = pd.DataFrame({
            "category": ["a"] * 100 + ["b"] * 100,
            "anomaly_score": np.concatenate([rng.normal(0.1, 0.01, 100),
                                             rng.normal(0.9, 0.01, 100)]),
            "label": 0,
        })
        cov = coverage_report(test, calib, n_boot=0)
        covs.append(cov[cov.group != "POOLED"]["empirical_coverage"].to_numpy())
    mean_cov = np.mean(covs, axis=0)
    assert (mean_cov > 0.87).all() and (mean_cov < 0.94).all(), mean_cov


def test_conditional_coverage_spread_is_large_at_n_200():
    """Marginal coverage is guaranteed; coverage *given one* calibration set is not.

    With n=200 the conditional coverage at delta=0.1 has a standard deviation of
    roughly 0.02, so a single category can sit 5-8 points below nominal purely
    because its calibration draw happened to be narrow. This is the reason
    `n_cal_sensitivity` exists and why per-category coverage must be reported with
    intervals rather than as a point estimate.
    """
    from scipy.stats import norm

    rng = np.random.default_rng(99)
    cond = np.array([norm.cdf(conformal_quantile(rng.normal(0, 1, 200), 0.1))
                     for _ in range(1000)])
    assert abs(cond.mean() - 0.9) < 0.01           # unbiased on average
    assert 0.015 < cond.std() < 0.03               # but far from tight
    assert cond.min() < 0.85                       # single-set under-coverage is routine


def test_median_threshold_would_flag_exactly_half():
    """Documents the prototype defect the conformal threshold replaces."""
    rng = np.random.default_rng(0)
    scores = rng.normal(size=1000)
    assert abs(float((scores > np.median(scores)).mean()) - 0.5) < 1e-9


@pytest.mark.parametrize("mode,peak", [("symmetric", 0.5), ("entropy", 0.5), ("boundary", 0.05)])
def test_pvalue_uncertainty_peaks_where_documented(mode, peak):
    grid = np.linspace(0.001, 0.999, 999)
    u = pvalue_uncertainty(grid, mode, delta=0.05)
    assert abs(float(grid[int(np.argmax(u))]) - peak) < 0.02
    assert u.min() >= 0.0


def test_log_mode_peaks_at_the_boundary_and_is_bounded():
    grid = np.geomspace(1e-4, 1 - 1e-6, 4000)
    u = pvalue_uncertainty(grid, "log", delta=0.05, tau_u=1.0)
    assert abs(float(grid[int(np.argmax(u))]) - 0.05) < 0.005
    assert u.min() >= 0.0 and u.max() <= 1.0 + 1e-12


def test_log_mode_is_scale_free_in_p():
    """The defining property: equal *ratios* of p give equal ratios of u.

    p is uniform under the null, so all the discriminative action sits in the tail
    near 0. A log-scaled mode treats p=1e-4 vs 1e-3 the same way it treats 1e-3 vs
    1e-2; a piecewise-linear rescaling collapses both into a vanishing difference.
    """
    p = np.array([1e-5, 1e-4, 1e-3, 1e-2])
    u = pvalue_uncertainty(p, "log", delta=0.05, tau_u=1.0)
    ratios = u[1:] / u[:-1]
    assert np.allclose(ratios, ratios[0])


def test_log_mode_tau_u_controls_tail_resolution():
    """tau_u > 1 keeps the deep tail distinguishable instead of crushing it to zero."""
    p = np.array([1e-5, 1e-4, 1e-3])
    sharp = pvalue_uncertainty(p, "log", delta=0.05, tau_u=1.0)
    wide = pvalue_uncertainty(p, "log", delta=0.05, tau_u=2.0)
    assert (wide > sharp).all()
    assert np.diff(wide).min() > np.diff(sharp).min()


def test_log_and_boundary_agree_below_delta_at_tau_one():
    """Documents that tau_u=1 reduces to the boundary rescaling on the lower side.

    Both give u = p/delta there, so tau_u is the only thing that makes 'log' a
    distinct arm rather than a duplicate of 'boundary'.
    """
    p = np.array([1e-4, 1e-3, 1e-2, 0.04])
    assert np.allclose(pvalue_uncertainty(p, "log", 0.05, tau_u=1.0),
                       pvalue_uncertainty(p, "boundary", 0.05))


def test_tau_u_must_be_positive():
    with pytest.raises(ValueError, match="tau_u"):
        pvalue_uncertainty(np.array([0.1]), "log", tau_u=0.0)


def test_default_uncertainty_mode_is_not_a_baseline():
    """The shipped default must never be a mode we diagnosed as peaking wrongly."""
    import inspect

    from tzsad.calibration.conformal import BASELINE_UNCERTAINTY_MODES, MondrianConformal

    default = inspect.signature(pvalue_uncertainty).parameters["mode"].default
    assert default not in BASELINE_UNCERTAINTY_MODES
    assert inspect.signature(MondrianConformal.transform).parameters[
        "uncertainty_mode"].default not in BASELINE_UNCERTAINTY_MODES


def test_shipped_config_default_is_not_a_baseline_mode():
    from omegaconf import OmegaConf

    from tzsad.calibration.conformal import BASELINE_UNCERTAINTY_MODES
    from tzsad.utils.config import CONFIG_ROOT

    cfg = OmegaConf.load(CONFIG_ROOT / "base.yaml")
    assert str(cfg.conformal.uncertainty_mode) not in BASELINE_UNCERTAINTY_MODES


def test_calibration_coverage_sd_matches_the_beta_law():
    from scipy import stats

    from tzsad.calibration.conformal import calibration_coverage_sd

    # n=200, delta=0.10 -> rank 181 -> Beta(181, 20)
    assert calibration_coverage_sd(200, 0.10) == pytest.approx(stats.beta(181, 20).std(), rel=1e-9)
    assert calibration_coverage_sd(200, 0.10) == pytest.approx(0.0211, abs=5e-4)
    assert calibration_coverage_sd(200, 0.05) == pytest.approx(0.0153, abs=5e-4)
    assert np.isnan(calibration_coverage_sd(40, 0.01))      # threshold is +inf there


def test_coverage_report_ci_separates_the_two_noise_terms():
    rng = np.random.default_rng(0)
    cal = pd.DataFrame({"category": ["a"] * 200, "anomaly_score": rng.normal(size=200), "label": 0})
    calib = MondrianConformal(delta=0.1, seed=0).fit(cal)
    test = pd.DataFrame({"category": ["a"] * 20, "anomaly_score": rng.normal(size=20), "label": 0})
    cov = coverage_report(test, calib, n_boot=300)
    row = cov.iloc[0]
    assert row.calib_sd == pytest.approx(0.0211, abs=5e-4)
    # The total interval is wider than the test-only bootstrap interval.
    assert row.ci_hi_total - row.ci_lo_total > row.ci_hi - row.ci_lo


def test_pvalues_reject_a_duplicated_index():
    rng = np.random.default_rng(0)
    cal = pd.DataFrame({"category": ["a"] * 50, "anomaly_score": rng.normal(size=50), "label": 0})
    calib = MondrianConformal().fit(cal)
    dup = pd.DataFrame({"category": ["a", "a"], "anomaly_score": [0.1, 0.2]}, index=[0, 0])
    with pytest.raises(ValueError, match="not unique"):
        calib.pvalues(dup)
