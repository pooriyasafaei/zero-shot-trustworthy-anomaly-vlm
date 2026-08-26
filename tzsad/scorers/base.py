"""Common Scorer interface.

A scorer's only job is to turn images into per-image records. It must not decide
thresholds, must not compute metrics, and must never silently drop an image: an
input it cannot handle becomes a row with ``parse_ok=False`` (an abstention),
which the selective-prediction evaluation then measures as a first-class outcome.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from ..records import CORE_COLUMNS


@dataclass
class ScoringContext:
    """Everything a scorer needs that is not the images themselves."""

    run_id: str
    subset_tag: str = "full"
    corruption: str = "none"
    severity: int = 0
    extras: dict[str, Any] = field(default_factory=dict)


class Scorer(abc.ABC):
    """Base class for anomaly-scoring backends.

    Subclasses implement :meth:`score_index` and set :attr:`scorer_name` and
    :attr:`model_id`. ``anomaly_score`` must be a probability-like value in [0, 1];
    the native quantity goes in ``raw_score``.
    """

    scorer_name: str = "base"
    model_id: str = "none"

    @abc.abstractmethod
    def score_index(self, index: pd.DataFrame, ctx: ScoringContext) -> pd.DataFrame:
        """Score every row of an image index and return per-image records."""

    def finalize(self, records: pd.DataFrame, ctx: ScoringContext) -> pd.DataFrame:
        """Fill the core columns that are identical for every row of a run."""
        records = records.copy()
        records["run_id"] = ctx.run_id
        records["scorer_name"] = records.get("scorer_name", self.scorer_name)
        records["model_id"] = self.model_id
        records["subset_tag"] = ctx.subset_tag
        records["corruption"] = ctx.corruption
        records["severity"] = int(ctx.severity)
        for col in CORE_COLUMNS:
            if col not in records.columns:
                records[col] = _default_for(col)
        ordered = list(CORE_COLUMNS) + [c for c in records.columns if c not in CORE_COLUMNS]
        return records[ordered]


def _default_for(column: str):
    if column == "parse_ok":
        return True
    if column == "n_valid_votes":
        return 1
    if column in ("anomaly_score", "raw_score"):
        return float("nan")
    return ""
