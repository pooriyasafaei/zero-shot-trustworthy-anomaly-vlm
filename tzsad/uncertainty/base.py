"""Common UncertaintyEstimator interface.

Every estimator declares what *kind* of uncertainty it claims to capture and its
known failure modes, because the paper's headline claim is comparative: which
signal actually predicts error, not which one has the nicest name.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any, ClassVar

import pandas as pd

from ..records import UNCERTAINTY_PREFIX


@dataclass
class EstimatorInfo:
    """Self-description used in the bake-off tables and the docs."""

    kind: str                # aleatoric | epistemic | distributional | mixed
    inputs: str              # what it consumes (cached embeddings, generations, ...)
    failure_modes: str
    is_baseline: bool = False


class UncertaintyEstimator(abc.ABC):
    """Turns per-image records into one uncertainty column in [0, 1] (higher = less certain).

    Estimators run **offline on records**; none of them may trigger a GPU forward
    pass. That is what makes adding a signal cheap.
    """

    name: ClassVar[str] = "base"
    info: ClassVar[EstimatorInfo]

    @abc.abstractmethod
    def compute(self, records: pd.DataFrame, **kwargs: Any) -> pd.Series:
        """Return an uncertainty value per row of ``records`` (same index)."""

    @property
    def column(self) -> str:
        """Column name this estimator writes into the record table."""
        return f"{UNCERTAINTY_PREFIX}{self.name}"

    def attach(self, records: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        """Compute and attach the uncertainty column, returning a new frame."""
        out = records.copy()
        out[self.column] = self.compute(records, **kwargs)
        return out


def rank_normalize(values, groups=None) -> pd.Series:
    """Map values to [0, 1] by rank, optionally within groups.

    Used wherever two signals on different scales are compared or combined - the
    rank-normalisation the brief requires before differencing CMCS branches.
    """
    s = pd.Series(values).astype(float)
    if groups is None:
        return s.rank(pct=True, na_option="keep")
    return s.groupby(pd.Series(list(groups), index=s.index)).rank(pct=True, na_option="keep")


def zscore(values, groups=None) -> pd.Series:
    """Standardise values, optionally within groups (NaN-safe)."""
    s = pd.Series(values).astype(float)
    if groups is None:
        std = s.std(ddof=0)
        return (s - s.mean()) / (std if std > 0 else 1.0)
    g = pd.Series(list(groups), index=s.index)
    return s.groupby(g).transform(lambda x: (x - x.mean()) / (x.std(ddof=0) or 1.0))
