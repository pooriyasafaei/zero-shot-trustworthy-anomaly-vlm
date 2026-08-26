"""Name -> estimator lookup, so configs can list uncertainty signals by string."""
from __future__ import annotations

from typing import Any, Callable

from .base import UncertaintyEstimator
from .image_side import (BackboneEnsemble, NormalManifoldDistance, NormalizedPromptEnsemble,
                         PromptEnsembleStd, TTAVariance)
from .vlm_side import (AbstentionFlag, PromptPerturbationConsistency, SemanticEntropy,
                       TokenEntropy, VerbalizedConfidence, VerdictVariance)

ESTIMATORS: dict[str, type[UncertaintyEstimator]] = {
    cls.name: cls for cls in (
        PromptEnsembleStd, NormalizedPromptEnsemble, BackboneEnsemble, TTAVariance,
        NormalManifoldDistance, TokenEntropy, SemanticEntropy,
        PromptPerturbationConsistency, VerbalizedConfidence, VerdictVariance, AbstentionFlag,
    )
}

#: Which estimators are baselines we expect to under-perform. Reported, not hidden.
BASELINES = tuple(n for n, c in ESTIMATORS.items() if c.info.is_baseline)


def get_estimator(name: str, **kwargs: Any) -> UncertaintyEstimator:
    """Instantiate an estimator by its registry name."""
    if name not in ESTIMATORS:
        raise KeyError(f"unknown uncertainty estimator {name!r}; available: {sorted(ESTIMATORS)}")
    return ESTIMATORS[name](**kwargs)


def describe_all() -> list[dict[str, str]]:
    """Table of every estimator with the uncertainty kind and failure modes it declares."""
    return [
        {"name": name, "kind": cls.info.kind, "inputs": cls.info.inputs,
         "failure_modes": cls.info.failure_modes, "is_baseline": str(cls.info.is_baseline)}
        for name, cls in sorted(ESTIMATORS.items())
    ]
