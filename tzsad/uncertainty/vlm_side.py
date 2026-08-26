"""VLM-side (Qwen) uncertainty estimators.

All of them read the records and the dumped generations, so the whole set can be
recomputed without re-running the VLM.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from ..scorers.qwen_parse import CONFIDENCE_MAP
from ..scorers.qwen_scorer import load_generations
from ..scorers.prompts import class_name_for
from ..utils.logging import get_logger
from .base import EstimatorInfo, UncertaintyEstimator
from .semantic_entropy import (BidirectionalEntailmentClusterer, ClusteringConfig,
                               cluster_entropy, joint_signature_entropy)

log = get_logger("uncertainty.vlm")


class TokenEntropy(UncertaintyEstimator):
    """Binary predictive entropy of ``P(YES)`` from the verdict-token logprobs.

    Kind
    ----
    Aleatoric (the model's own predictive uncertainty about the verdict).

    Why it beats the vote fraction
    ------------------------------
    One greedy forward pass, continuous, and free of the ties that make an
    11-valued vote fraction unrankable. It is *not* independent of the score - it
    is a deterministic function of it - so it is reported as the reference
    "confidence" channel rather than as a second opinion, and any fusion that
    includes it must justify itself against that.

    Failure modes
    -------------
    A confidently wrong model has low token entropy: this signal cannot detect
    silent failure under distribution shift. Pair it with
    :class:`~tzsad.uncertainty.image_side.NormalManifoldDistance`.
    """

    name = "token_entropy"
    info = EstimatorInfo(
        kind="aleatoric",
        inputs="anomaly_score (= P(YES)) from the logprob scorer",
        failure_modes="deterministic in the score; blind to confidently-wrong shift",
    )

    def compute(self, records: pd.DataFrame, **kwargs) -> pd.Series:
        p = np.clip(records["anomaly_score"].to_numpy(dtype=float), 1e-12, 1 - 1e-12)
        h = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
        h = np.where(records["parse_ok"].to_numpy(), h, 1.0)   # abstention = maximal uncertainty
        return pd.Series(h, index=records.index)


class SemanticEntropy(UncertaintyEstimator):
    """Entropy over bidirectional-entailment clusters of the sampled observations.

    Kind
    ----
    Aleatoric + epistemic: it measures whether the model tells a *consistent
    story*, independently of whether that story is the majority verdict.

    Failure modes
    -------------
    Needs multi-sample generation (N forward passes). A model that hallucinates
    the *same* defect every time looks perfectly certain - consistency is not
    correctness, which is precisely why the hallucination module exists separately.
    Cluster granularity is a hyperparameter: too strict and every sample is its
    own cluster, too loose and everything collapses to one.
    """

    name = "semantic_entropy"
    info = EstimatorInfo(
        kind="mixed",
        inputs="dumped generations (OBSERVATION strings) + votes",
        failure_modes="needs N samples; consistent hallucination looks certain",
    )

    def __init__(self, generations_dir: str | Path, config: ClusteringConfig | None = None,
                 joint_with_vote: bool = False) -> None:
        self.generations_dir = Path(generations_dir)
        self.clusterer = BidirectionalEntailmentClusterer(config)
        self.joint_with_vote = joint_with_vote

    def compute(self, records: pd.DataFrame, **kwargs) -> pd.Series:
        values = np.full(len(records), np.nan)
        for pos, (_, r) in enumerate(records.iterrows()):
            samples = load_generations(self.generations_dir, r["image_id"])
            valid = [s for s in samples if s.vote is not None]
            if len(valid) < 2:
                values[pos] = 1.0 if not valid else 0.0
                continue
            context = f"Is this {class_name_for(r['category'])} defective?"
            texts = [s.observation or s.text for s in valid]
            assign = self.clusterer.cluster(texts, context)
            values[pos] = (joint_signature_entropy([s.vote for s in valid], assign)
                           if self.joint_with_vote else cluster_entropy(assign))
        return pd.Series(values, index=records.index)


class PromptPerturbationConsistency(UncertaintyEstimator):
    """Verdict instability under semantically equivalent prompt rewrites and mild blur.

    Kind
    ----
    Epistemic (sensitivity to nuisance changes that should not matter).

    Method
    ------
    Follows VL-Uncertainty (arXiv:2411.11919): perturb both modalities in ways
    that preserve meaning - paraphrase the instruction, blur the image slightly -
    and measure the spread of ``P(YES)`` across variants. A model whose verdict
    flips when the wording changes did not have a grounded verdict.

    Failure modes
    -------------
    Costs one extra forward pass per variant; and a blur strong enough to erase a
    genuinely small defect turns a correct prediction into apparent instability,
    so the blur radius must stay mild and be reported.
    """

    name = "prompt_perturb"
    info = EstimatorInfo(
        kind="epistemic",
        inputs="variant_pyes__* columns written by the Qwen scorer",
        failure_modes="extra forward passes; blur can destroy small defects",
    )

    def __init__(self, stat: str = "std") -> None:
        self.stat = stat

    def compute(self, records: pd.DataFrame, **kwargs) -> pd.Series:
        cols = sorted(c for c in records.columns if c.startswith("variant_pyes__"))
        if not cols:
            raise KeyError(
                "no variant_pyes__* columns; run the Qwen scorer with prompt_variants set"
            )
        base = records["anomaly_score"].to_numpy(dtype=float)[:, None]
        stacked = np.column_stack([records[c].to_numpy(dtype=float) for c in cols])
        stacked = np.hstack([base, stacked])
        u = np.nanstd(stacked, axis=1, ddof=0) if self.stat == "std" else \
            np.nanmax(stacked, axis=1) - np.nanmin(stacked, axis=1)
        return pd.Series(np.nan_to_num(u, nan=1.0), index=records.index)


class VerbalizedConfidence(UncertaintyEstimator):
    """``1 - mean(self-reported HIGH/MEDIUM/LOW)``. **Baseline; expected negative result.**

    Kind
    ----
    Claimed aleatoric; empirically near-degenerate.

    Failure modes
    -------------
    Qwen answers HIGH on almost every image, so the signal has almost no variance
    and cannot rank anything. Reported as a negative result: verbalised confidence
    from an instruction-tuned VLM is not a usable uncertainty channel on MVTec.
    """

    name = "verbalized_conf"
    info = EstimatorInfo(
        kind="aleatoric (claimed)",
        inputs="predicted_confidence column / dumped generations",
        failure_modes="near-degenerate (almost always HIGH); no rank information",
        is_baseline=True,
    )

    def __init__(self, generations_dir: str | Path | None = None) -> None:
        self.generations_dir = Path(generations_dir) if generations_dir else None

    def compute(self, records: pd.DataFrame, **kwargs) -> pd.Series:
        values = np.full(len(records), np.nan)
        for pos, (_, r) in enumerate(records.iterrows()):
            confs: list[float] = []
            if self.generations_dir is not None:
                confs = [CONFIDENCE_MAP[s.confidence] for s in
                         load_generations(self.generations_dir, r["image_id"])
                         if s.confidence in CONFIDENCE_MAP]
            if not confs:
                c = str(r.get("predicted_confidence", "")).upper()
                if c in CONFIDENCE_MAP:
                    confs = [CONFIDENCE_MAP[c]]
            values[pos] = 1.0 - float(np.mean(confs)) if confs else 1.0
        return pd.Series(values, index=records.index)


class VerdictVariance(UncertaintyEstimator):
    """``sqrt(p(1-p))`` over the vote fraction. **Baseline; documented as circular.**

    Kind
    ----
    None. It is a deterministic, monotone-in-|p-0.5| transform of the score.

    Failure modes
    -------------
    Maximal exactly at the 0.5 decision threshold by construction, so its apparent
    ability to "predict error" is an artefact of errors clustering near any
    threshold. It is max-probability confidence rebranded, and is kept only so the
    ablation can show that an independent signal beats it.
    """

    name = "verdict_var"
    info = EstimatorInfo(
        kind="none (circular)",
        inputs="vote_fraction (or anomaly_score)",
        failure_modes="deterministic function of the score; maximal at the threshold",
        is_baseline=True,
    )

    def compute(self, records: pd.DataFrame, **kwargs) -> pd.Series:
        col = "vote_fraction" if "vote_fraction" in records else "anomaly_score"
        p = records[col].to_numpy(dtype=float)
        return pd.Series(np.sqrt(np.clip(p * (1 - p), 0, None)), index=records.index)


class AbstentionFlag(UncertaintyEstimator):
    """Maximal uncertainty for unparseable generations.

    Not a competitor to the other signals - it exists so that abstentions
    (``parse_ok=False``) enter the selective-prediction curves at the top of the
    uncertainty ranking rather than being dropped, which is what defect #6 asks for.
    """

    name = "abstention"
    info = EstimatorInfo(
        kind="operational",
        inputs="parse_ok",
        failure_modes="binary; no ranking information among parsed rows",
        is_baseline=True,
    )

    def compute(self, records: pd.DataFrame, **kwargs) -> pd.Series:
        return pd.Series((~records["parse_ok"].to_numpy(dtype=bool)).astype(float),
                         index=records.index)
