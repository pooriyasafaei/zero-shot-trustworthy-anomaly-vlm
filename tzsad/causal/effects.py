"""Turning intervention log-odds into the per-image causal quantities.

For an anomalous image ``I`` with defect mask ``M`` and placebo masks ``P_1..P_K``,
with ``z(.)`` the VLM anomaly log-odds ``log P(YES)/P(NO)``:

    delta_defect    = z(I) - z(I_without_M)
    delta_placebo_k = z(I) - z(I_without_P_k)
    CEE             = delta_defect - median_k(delta_placebo_k)

``CEE > 0`` means neutralising the real defect moved the model's belief more than a
comparable edit somewhere irrelevant did.

Wording matters here. ``empirical_placebo_p`` is a **matched-placebo tail
probability**, i.e. the rank of the real effect within its own placebo distribution.
It is not a randomisation p-value in general: that would require the placebo
placements to be genuinely exchangeable with the defect placement under the null,
which matched translation makes plausible but does not establish. It is reported as
a descriptive rank statistic and documented as one.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

#: Guards the robust z-score when every placebo effect is identical.
EPS = 1e-9

#: A claim at alpha cannot be made when the finest achievable tail exceeds it.
ALPHA = 0.05


@dataclass
class CausalEffect:
    """Per-image Stage-I result."""

    delta_defect: float
    placebo_deltas: list[float] = field(default_factory=list)
    cee: float = float("nan")
    robust_cee_z: float = float("nan")
    empirical_placebo_p: float = float("nan")
    defect_placebo_percentile: float = float("nan")
    defect_beats_all_placebos: bool = False
    supported_at_005: bool | None = None
    placebo_delta_mean: float = float("nan")
    placebo_delta_median: float = float("nan")
    placebo_delta_mad: float = float("nan")
    placebo_delta_max: float = float("nan")
    tail_resolution: float = float("nan")

    def as_dict(self) -> dict:
        """Flat dict for the causal records table."""
        return {
            "delta_defect": self.delta_defect,
            "n_valid_placebos": len(self.placebo_deltas),
            "placebo_delta_mean": self.placebo_delta_mean,
            "placebo_delta_median": self.placebo_delta_median,
            "placebo_delta_mad": self.placebo_delta_mad,
            "placebo_delta_max": self.placebo_delta_max,
            "cee": self.cee,
            "robust_cee_z": self.robust_cee_z,
            "empirical_placebo_p": self.empirical_placebo_p,
            "defect_placebo_percentile": self.defect_placebo_percentile,
            "defect_beats_all_placebos": self.defect_beats_all_placebos,
            "supported_at_005": self.supported_at_005,
            "tail_resolution": self.tail_resolution,
        }


def median_absolute_deviation(values: Sequence[float]) -> float:
    """MAD about the median. Returns 0.0 for a constant or empty sample."""
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0
    return float(np.median(np.abs(arr - np.median(arr))))


def empirical_placebo_p(delta_defect: float, placebo_deltas: Sequence[float]) -> float:
    """``(1 + #{placebo >= defect}) / (K + 1)``.

    The ``+1`` in both places counts the observed effect among the candidates, which
    keeps the statistic from ever being exactly zero - with K placebos the smallest
    value it can take is ``1/(K+1)``.
    """
    arr = np.asarray(list(placebo_deltas), dtype=float)
    arr = arr[np.isfinite(arr)]
    if not np.isfinite(delta_defect):
        return float("nan")
    return float((1 + int((arr >= delta_defect).sum())) / (arr.size + 1))


def compute_effects(delta_defect: float, placebo_deltas: Sequence[float],
                    alpha: float = ALPHA) -> CausalEffect:
    """Assemble every per-image causal quantity from the raw deltas.

    ``supported_at_005`` is deliberately tri-state. With fewer than 19 placebos the
    finest achievable tail probability exceeds 0.05, so the question cannot be
    answered at that level and the answer is ``None`` (NA) rather than ``False`` -
    reporting False would read as evidence against, when it is absence of resolution.
    """
    deltas = [float(d) for d in placebo_deltas if np.isfinite(d)]
    effect = CausalEffect(delta_defect=float(delta_defect), placebo_deltas=deltas)
    n = len(deltas)
    effect.tail_resolution = 1.0 / (n + 1)

    if n == 0 or not np.isfinite(delta_defect):
        effect.empirical_placebo_p = empirical_placebo_p(delta_defect, deltas)
        return effect

    arr = np.asarray(deltas, dtype=float)
    effect.placebo_delta_mean = float(arr.mean())
    effect.placebo_delta_median = float(np.median(arr))
    effect.placebo_delta_mad = median_absolute_deviation(arr)
    effect.placebo_delta_max = float(arr.max())

    effect.cee = float(delta_defect - effect.placebo_delta_median)
    effect.robust_cee_z = float(
        (delta_defect - effect.placebo_delta_median) / (1.4826 * effect.placebo_delta_mad + EPS))
    effect.empirical_placebo_p = empirical_placebo_p(delta_defect, arr)
    effect.defect_placebo_percentile = float((arr < delta_defect).mean())
    effect.defect_beats_all_placebos = bool((arr < delta_defect).all())
    effect.supported_at_005 = (
        None if effect.tail_resolution > alpha else bool(effect.empirical_placebo_p <= alpha))
    return effect


def log_odds(p_yes: float, eps: float = 1e-6) -> float:
    """``log p/(1-p)`` with clipping, for when only a probability is available."""
    p = min(max(float(p_yes), eps), 1.0 - eps)
    return math.log(p / (1.0 - p))
