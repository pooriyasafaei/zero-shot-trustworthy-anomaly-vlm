"""Trust-score fusion: three combiners and the ablation that has to justify them.

The proposal's ``t = CMCS * (1 - u) * 1[H = 0]`` is implemented as *one candidate*,
not as the answer. It is a hard, non-tunable product with an indicator, so a single
hallucination flag zeroes the trust score no matter how strong the other evidence
is, and there is no way to trade the factors off. Two tunable alternatives are
implemented alongside it, and §4.5's rule is enforced by
:func:`leave_one_out_ablation`: a factor stays only if dropping it hurts AURC.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd

from ..eval.selective import risk_coverage_curve
from ..uncertainty.base import rank_normalize


@dataclass
class FusionConfig:
    """Which columns feed the trust score and how they are combined."""

    uncertainty_cols: tuple[str, ...] = ("u_conformal",)
    cmcs_col: str = "cmcs"
    halluc_col: str = "halluc"
    group_col: str = "category"
    weights: dict[str, float] = field(default_factory=dict)


def _factors(records: pd.DataFrame, config: FusionConfig, use: Sequence[str]) -> dict[str, np.ndarray]:
    """Assemble the trust *factors* (higher = more trustworthy) named in ``use``."""
    out: dict[str, np.ndarray] = {}
    for col in config.uncertainty_cols:
        if col in use and col in records:
            r = rank_normalize(records[col], records[config.group_col]).fillna(0.5)
            out[col] = (1.0 - r).to_numpy()
    if config.cmcs_col in use and config.cmcs_col in records:
        out[config.cmcs_col] = rank_normalize(
            records[config.cmcs_col], records[config.group_col]).fillna(0.5).to_numpy()
    if config.halluc_col in use and config.halluc_col in records:
        out[config.halluc_col] = (~records[config.halluc_col].to_numpy(dtype=bool)).astype(float)
    return out


def proposal_product(records: pd.DataFrame, config: FusionConfig,
                     use: Sequence[str] | None = None) -> np.ndarray:
    """``t = CMCS * prod(1 - u) * 1[H = 0]`` - the proposal's formula, verbatim.

    Known weakness: the indicator makes the score non-smooth and non-tunable, and
    the product of many factors decays towards zero as factors are added, which
    penalises richer evidence rather than rewarding it.
    """
    use = list(use) if use is not None else _all_names(config)
    factors = _factors(records, config, use)
    t = np.ones(len(records))
    for name, v in factors.items():
        t = t * (v if name == config.halluc_col else np.clip(v, 0.0, 1.0))
    return t


def weighted_geometric_mean(records: pd.DataFrame, config: FusionConfig,
                            use: Sequence[str] | None = None, eps: float = 1e-6) -> np.ndarray:
    """``t = prod(f_i ^ w_i)`` with tunable exponents.

    Reduces to the proposal's product when every weight is 1, but lets a weak
    factor be down-weighted instead of being all-or-nothing, and keeps the
    hallucination flag as a soft penalty rather than an annihilator.
    """
    use = list(use) if use is not None else _all_names(config)
    factors = _factors(records, config, use)
    if not factors:
        return np.ones(len(records))
    total_w = 0.0
    log_t = np.zeros(len(records))
    for name, v in factors.items():
        w = float(config.weights.get(name, 1.0))
        log_t += w * np.log(np.clip(v, eps, 1.0))
        total_w += w
    return np.exp(log_t / max(total_w, eps))


class LogisticTrustCombiner:
    """Logistic combiner fitted on normal-only data plus synthetic anomalies.

    Fitting on ``train/good`` plus CutPaste/NSA corruptions of those same images
    keeps the combiner zero-shot with respect to *real* defects: it never sees a
    genuine MVTec anomaly, so the test-set numbers are not contaminated.

    Failure mode to watch: synthetic anomalies are far easier than real ones, so
    the fitted weights can be over-confident. The ablation reports the fitted
    combiner and the untuned product side by side for exactly that reason.
    """

    def __init__(self, config: FusionConfig, C: float = 1.0, seed: int = 0) -> None:
        self.config = config
        self.C = C
        self.seed = seed
        self.model = None
        self.feature_names_: list[str] = []

    def fit(self, records: pd.DataFrame, is_anomalous: np.ndarray,
            use: Sequence[str] | None = None) -> "LogisticTrustCombiner":
        """Fit on records whose ``is_anomalous`` labels come from synthesis, not MVTec."""
        from sklearn.linear_model import LogisticRegression

        use = list(use) if use is not None else _all_names(self.config)
        factors = _factors(records, self.config, use)
        self.feature_names_ = sorted(factors)
        X = np.column_stack([factors[n] for n in self.feature_names_])
        y = np.asarray(is_anomalous, dtype=int)
        if len(np.unique(y)) < 2:
            raise ValueError("logistic combiner needs both synthetic-anomalous and normal rows")
        self.model = LogisticRegression(C=self.C, max_iter=1000, random_state=self.seed).fit(X, y)
        return self

    def predict_trust(self, records: pd.DataFrame) -> np.ndarray:
        """Predicted probability that the decision is trustworthy."""
        if self.model is None:
            raise RuntimeError("fit the combiner first")
        factors = _factors(records, self.config, self.feature_names_)
        missing = [n for n in self.feature_names_ if n not in factors]
        if missing:
            raise KeyError(f"missing fusion factors at predict time: {missing}")
        X = np.column_stack([factors[n] for n in self.feature_names_])
        return 1.0 - self.model.predict_proba(X)[:, 1]

    @property
    def coefficients(self) -> dict[str, float]:
        """Fitted weights, for the ablation table."""
        if self.model is None:
            return {}
        return dict(zip(self.feature_names_, self.model.coef_[0].tolist()))


COMBINERS = {
    "proposal_product": proposal_product,
    "weighted_geometric": weighted_geometric_mean,
}


def leave_one_out_ablation(records: pd.DataFrame, config: FusionConfig, correct_col: str,
                           combiner: str = "weighted_geometric") -> pd.DataFrame:
    """Drop each factor in turn and report the change in AURC.

    §4.5's rule, made mechanical: a factor earns its place only if removing it
    makes AURC *worse* (delta_aurc > 0). Factors with a non-positive delta are
    reported as unjustified and should be dropped from the paper's fusion.
    """
    fn = COMBINERS[combiner]
    names = _all_names(config)
    correct = records[correct_col].to_numpy()
    abstain = (~records["parse_ok"].to_numpy(dtype=bool)) if "parse_ok" in records else None

    full = 1.0 - fn(records, config, names)          # trust -> uncertainty ordering
    base = risk_coverage_curve(correct, full, abstain)
    rows = [{"dropped": "(none)", "n_factors": len(names), "aurc": base.aurc,
             "eaurc": base.eaurc, "delta_aurc": 0.0, "justified": True}]
    for name in names:
        kept = [n for n in names if n != name]
        if not kept:
            continue
        u = 1.0 - fn(records, config, kept)
        rc = risk_coverage_curve(correct, u, abstain)
        delta = rc.aurc - base.aurc                  # positive = dropping it hurt
        rows.append({"dropped": name, "n_factors": len(kept), "aurc": rc.aurc,
                     "eaurc": rc.eaurc, "delta_aurc": delta, "justified": bool(delta > 0)})
    return pd.DataFrame(rows)


def _all_names(config: FusionConfig) -> list[str]:
    return list(config.uncertainty_cols) + [config.cmcs_col, config.halluc_col]
