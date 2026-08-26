"""Uncertainty estimators, CMCS, hallucination rate and fusion behaviour."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tzsad.fusion.trust import (FusionConfig, leave_one_out_ablation, proposal_product,
                                weighted_geometric_mean)
from tzsad.hallucination.rate import HRConfig, hallucination_flags, hallucination_rate
from tzsad.uncertainty.base import rank_normalize, zscore
from tzsad.uncertainty.registry import ESTIMATORS, describe_all, get_estimator
from tzsad.uncertainty.semantic_entropy import cluster_entropy, joint_signature_entropy
from tzsad.uncertainty.vlm_side import TokenEntropy, VerdictVariance


def _records(n=40, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "category": ["a"] * (n // 2) + ["b"] * (n - n // 2),
        "image_id": [f"img{i}" for i in range(n)],
        "label": rng.integers(0, 2, n),
        "anomaly_score": rng.uniform(0.01, 0.99, n),
        "parse_ok": True,
        "split": "test",
    })


def test_every_registered_estimator_declares_its_kind_and_failures():
    """Docstring discipline is enforced: no anonymous uncertainty signals."""
    for name, cls in ESTIMATORS.items():
        assert cls.info.kind, name
        assert cls.info.failure_modes, name
        assert cls.__doc__ and len(cls.__doc__) > 80, name
    assert len(describe_all()) == len(ESTIMATORS)


def test_registry_rejects_unknown_names():
    with pytest.raises(KeyError, match="unknown uncertainty estimator"):
        get_estimator("magic_signal")


def test_verdict_variance_is_a_deterministic_function_of_the_score():
    """Documents defect #2: this is max-probability confidence rebranded."""
    df = _records(200)
    u = VerdictVariance().compute(df).to_numpy()
    expected = np.sqrt(df.anomaly_score * (1 - df.anomaly_score))
    assert np.allclose(u, expected)
    # A strictly monotone (not linear) function of -|p - 0.5|, so it carries no
    # information the score does not already carry: rank correlation is exactly 1.
    from scipy.stats import spearmanr

    assert spearmanr(u, -np.abs(df.anomaly_score - 0.5)).statistic == pytest.approx(1.0)


def test_token_entropy_peaks_at_one_half_and_marks_abstentions():
    df = pd.DataFrame({"anomaly_score": [0.5, 0.01, 0.99, 0.5], "parse_ok": [True, True, True, False]})
    u = TokenEntropy().compute(df).to_numpy()
    assert u[0] == pytest.approx(1.0)
    assert u[1] < 0.1 and u[2] < 0.1
    assert u[3] == 1.0


def test_rank_normalize_is_per_group():
    s = pd.Series([1.0, 2.0, 100.0, 200.0])
    g = ["a", "a", "b", "b"]
    r = rank_normalize(s, g)
    assert r.tolist() == [0.5, 1.0, 0.5, 1.0]


def test_zscore_handles_constant_groups():
    out = zscore(pd.Series([5.0, 5.0, 5.0]), ["a", "a", "a"])
    assert np.allclose(out.to_numpy(), 0.0)


def test_cluster_entropy_bounds():
    assert cluster_entropy([0, 0, 0, 0]) == pytest.approx(0.0)
    assert cluster_entropy([0, 1, 2, 3]) == pytest.approx(1.0)
    assert 0 < cluster_entropy([0, 0, 1, 2]) < 1


def test_joint_signature_entropy_separates_verdict_from_wording():
    """Same verdict, different stories -> lower joint entropy than differing verdicts."""
    same_verdict = joint_signature_entropy([1, 1, 1, 1], [0, 1, 2, 3])
    mixed_verdict = joint_signature_entropy([1, 0, 1, 0], [0, 1, 2, 3])
    assert same_verdict == pytest.approx(mixed_verdict)   # both fully distinct pairs
    assert joint_signature_entropy([1, 1, 1, 1], [0, 0, 0, 0]) == 0.0


def test_hallucination_case_a_is_any_positive_on_a_good_image():
    records = pd.DataFrame({
        "category": ["a", "a"], "image_id": ["i0", "i1"], "label": [0, 1],
        "conformal_pred": [1, 0], "mask_path": ["", ""], "predicted_location": ["center", "center"],
    })
    maps = {"i0": np.zeros((16, 16)), "i1": np.zeros((16, 16))}
    flagged = hallucination_flags(records, maps, HRConfig(map_size=16))
    assert flagged.halluc.tolist() == [True, False]
    assert flagged.halluc_case.tolist() == ["a_positive_on_good", "none"]


def test_hallucination_case_b_needs_region_disagreement(tmp_path):
    from PIL import Image

    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[0:10, 0:10] = 255                      # defect in the TOP-LEFT
    mp = tmp_path / "m.png"
    Image.fromarray(mask).save(mp)
    records = pd.DataFrame({
        "category": ["a", "a"], "image_id": ["i0", "i1"], "label": [1, 1],
        "conformal_pred": [1, 1], "mask_path": [str(mp), str(mp)],
        "predicted_location": ["bottom right", "upper left"],
    })
    maps = {"i0": np.zeros((32, 32)), "i1": np.zeros((32, 32))}
    flagged = hallucination_flags(records, maps, HRConfig(map_size=32, iou_threshold=0.05))
    assert flagged.halluc_case.tolist() == ["b_wrong_region", "grounded"]

    hr = hallucination_rate(flagged)
    assert hr.iloc[-1]["hr"] == pytest.approx(0.5)


def test_proposal_product_is_annihilated_by_one_flag():
    """The documented weakness of t = CMCS * (1-u) * 1[H=0]."""
    df = _records(10)
    df["u_x"] = 0.1
    df["cmcs"] = 0.9
    df["halluc"] = [True] + [False] * 9
    cfg = FusionConfig(uncertainty_cols=("u_x",))
    t = proposal_product(df, cfg)
    assert t[0] == 0.0
    assert (t[1:] > 0).all()


def test_weighted_geometric_can_downweight_a_factor():
    df = _records(10)
    df["u_x"] = np.linspace(0.0, 1.0, 10)
    df["cmcs"] = 0.5
    df["halluc"] = False
    strong = weighted_geometric_mean(df, FusionConfig(uncertainty_cols=("u_x",), weights={"u_x": 4.0}))
    weak = weighted_geometric_mean(df, FusionConfig(uncertainty_cols=("u_x",), weights={"u_x": 0.1}))
    assert strong.std() > weak.std()


def test_ablation_marks_a_useless_factor_as_unjustified():
    """A pure-noise factor must not be reported as earning its place."""
    rng = np.random.default_rng(0)
    n = 400
    df = _records(n, seed=1)
    df["correct"] = rng.random(n) < 0.7
    df["u_good"] = np.where(df.correct, rng.uniform(0, 0.4, n), rng.uniform(0.6, 1.0, n))
    df["u_noise"] = rng.random(n)
    df["cmcs"] = 0.5
    df["halluc"] = False
    cfg = FusionConfig(uncertainty_cols=("u_good", "u_noise"))
    ab = leave_one_out_ablation(df, cfg, "correct")
    good = ab[ab.dropped == "u_good"].iloc[0]
    noise = ab[ab.dropped == "u_noise"].iloc[0]
    assert good.delta_aurc > 0 and good.justified
    assert noise.delta_aurc < good.delta_aurc


def test_eq4_flag_fires_only_on_disagreement_plus_assertion():
    """Proposal Eq. 4: both conditions must hold, not either one."""
    from tzsad.hallucination.cmcs import FlagConfig, apply_flag_and_adjust

    df = pd.DataFrame({
        "category": ["a"] * 4,
        "anomaly_score": [0.9, 0.9, 0.1, 0.1],     # asserted, asserted, quiet, quiet
        "cmcs": [0.1, 0.9, 0.1, 0.9],              # disagree, agree, disagree, agree
    })
    out = apply_flag_and_adjust(df, FlagConfig(theta_cmcs=0.5, theta_anom=0.5, alpha=1.5))
    assert out.halluc_flag_eq4.tolist() == [True, False, False, False]


def test_eq5_attenuates_only_flagged_scores():
    from tzsad.hallucination.cmcs import FlagConfig, apply_flag_and_adjust

    df = pd.DataFrame({"category": ["a", "a"], "anomaly_score": [0.9, 0.9], "cmcs": [0.1, 0.9]})
    out = apply_flag_and_adjust(df, FlagConfig(theta_cmcs=0.5, theta_anom=0.5, alpha=1.5))
    assert out.s_adj.iloc[0] == pytest.approx(0.9 * 0.1 ** 1.5)
    assert out.s_adj.iloc[1] == pytest.approx(0.9)          # unflagged passes through


def test_flag_thresholds_are_fitted_on_normals_only():
    from tzsad.hallucination.cmcs import FlagConfig, fit_flag_thresholds

    rng = np.random.default_rng(0)
    cal = pd.DataFrame({"category": ["a"] * 200, "label": 0,
                        "cmcs": rng.uniform(0, 1, 200), "anomaly_score": rng.uniform(0, 1, 200)})
    th = fit_flag_thresholds(cal, FlagConfig(cmcs_quantile=0.1, anom_quantile=0.9))
    assert 0.0 < th["a"][0] < 0.25 and 0.75 < th["a"][1] < 1.0

    cal.loc[0, "label"] = 1
    with pytest.raises(ValueError, match="normal-only"):
        fit_flag_thresholds(cal, FlagConfig())


def test_proposal_uncertainty_formula_is_not_implemented():
    """Defect #7: u = 1 - |(s - q)/q|^-1 diverges to -inf as s -> q. We must not ship it."""
    import tzsad.calibration.conformal as C

    assert not hasattr(C, "proposal_uncertainty")
    # Demonstrate the divergence that motivates using conformal p-values instead.
    q = 0.5
    s = np.array([0.5001, 0.5010, 0.6000])
    broken = 1.0 - np.abs((s - q) / q) ** -1
    assert broken[0] < -1000 and broken[0] < broken[1] < broken[2]
    # The p-value-based replacement stays bounded everywhere.
    u = C.pvalue_uncertainty(np.linspace(1e-6, 1 - 1e-6, 1000), "symmetric")
    assert u.min() >= 0.0 and u.max() <= 1.0


def test_report_declares_every_signal_it_scores():
    """Each u_* column in a report must arrive with its declared failure mode."""
    from tzsad.eval.report import estimator_declarations

    df = _records(20)
    df["u_prompt_std"] = 0.1
    df["u_manifold_knn"] = 0.2
    df["u_conformal"] = 0.3          # not in the registry: derived by the calibrator
    decl = estimator_declarations(df)
    assert set(decl.signal) == {"u_prompt_std", "u_manifold_knn", "u_conformal"}
    assert decl.failure_modes.str.len().gt(20).all()
    assert bool(decl.set_index("signal").loc["u_prompt_std", "is_baseline"]) is True


def _fusion_records(mode: str, n: int = 40):
    df = _records(n)
    df["conformal_mode"] = mode
    df["u_conformal"] = 0.3
    df["cmcs"] = 0.6
    df["halluc"] = False
    return df


@pytest.mark.parametrize("mode", ["symmetric", "entropy"])
def test_fusion_refuses_a_baseline_grade_uncertainty_mode(mode):
    """A combiner fed a signal that peaks in the wrong place manufactures a null result."""
    from tzsad.fusion.trust import FusionConfig, proposal_product

    with pytest.raises(ValueError, match="baseline-grade"):
        proposal_product(_fusion_records(mode), FusionConfig())


@pytest.mark.parametrize("mode", ["boundary", "log"])
def test_fusion_accepts_a_boundary_peaked_mode(mode):
    from tzsad.fusion.trust import FusionConfig, proposal_product

    assert len(proposal_product(_fusion_records(mode), FusionConfig())) == 40


def test_baseline_mode_can_be_fused_deliberately_for_the_ablation():
    from tzsad.fusion.trust import FusionConfig, proposal_product

    cfg = FusionConfig(allow_baseline_signals=True)
    assert len(proposal_product(_fusion_records("symmetric"), cfg)) == 40
