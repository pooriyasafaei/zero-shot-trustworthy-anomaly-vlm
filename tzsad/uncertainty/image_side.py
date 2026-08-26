"""Image-side (CLIP) uncertainty estimators.

All five consume cached embeddings or the per-template columns the CLIP scorer
already wrote, so the whole bake-off costs zero GPU passes after the first one.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from ..features.cache import EmbeddingCache, l2_normalize
from ..scorers.clip_scorer import template_columns
from ..utils.logging import get_logger
from .base import EstimatorInfo, UncertaintyEstimator, zscore

log = get_logger("uncertainty.image")


class PromptEnsembleStd(UncertaintyEstimator):
    """Standard deviation of the per-template anomaly scores. **Baseline; expected to fail.**

    Kind
    ----
    Nominally aleatoric/epistemic; in practice mostly wording jitter.

    Failure modes
    -------------
    The five templates are near-paraphrases occupying nearly the same region of
    CLIP text space, so their spread measures phrasing noise confounded with
    score magnitude: the std of a softmax is mechanically largest near ``p=0.5``,
    which re-imports the same circularity as ``verdict_variance``. Retained so
    the ablation can *show* it fails rather than assert it.
    """

    name = "prompt_std"
    info = EstimatorInfo(
        kind="aleatoric (nominal)",
        inputs="per-template score columns from ClipScorer",
        failure_modes="near-paraphrase templates; spread scales with score magnitude",
        is_baseline=True,
    )

    def __init__(self, use_raw: bool = False) -> None:
        self.use_raw = use_raw

    def compute(self, records: pd.DataFrame, **kwargs) -> pd.Series:
        cols = template_columns(records, raw=self.use_raw)
        if not cols:
            raise KeyError("no per-template columns; run ClipScorer with keep_per_template=True")
        return pd.Series(records[cols].to_numpy().std(axis=1, ddof=0), index=records.index)


class NormalizedPromptEnsemble(UncertaintyEstimator):
    """Prompt-ensemble spread after z-scoring each template.

    Kind
    ----
    Aleatoric (text-side).

    Rationale
    ---------
    Each template carries a constant offset and gain (some phrasings score
    everything higher). Standardising per template within a category removes both,
    so what remains is genuine per-image disagreement rather than template bias.
    This is the cheap rescue of :class:`PromptEnsembleStd`.

    Fit it, do not let it standardise against the test batch
    -------------------------------------------------------
    Call :meth:`fit` with the ``train/good`` records first. Without that the
    estimator falls back to standardising against *whatever frame it is handed*,
    which has two problems: an image's uncertainty then depends on which other
    images share its batch (test-set leakage), and - measured on MVTec - dividing
    by the corrupted batch's own spread cancels almost the whole distribution-shift
    signal. Against a fixed clean reference the mean rises 0.74 -> 2.15 from clean
    to defocus-blur severity 5; transductively it only reaches 0.78.

    Failure modes
    -------------
    Still text-side: it cannot see anything the image encoder is unsure about. And
    an unfitted instance is transductive, which is why that path warns.
    """

    name = "prompt_std_z"
    info = EstimatorInfo(
        kind="aleatoric (text-side)",
        inputs="per-template score columns + train/good reference statistics",
        failure_modes="text-side only; transductive and shift-blind unless fitted",
    )

    def __init__(self, use_raw: bool = True, group_col: str = "category") -> None:
        self.use_raw = use_raw
        self.group_col = group_col
        self.reference_: dict[str, dict[str, tuple[float, float]]] = {}

    def fit(self, reference: pd.DataFrame) -> "NormalizedPromptEnsemble":
        """Store per-category, per-template mean/std from normal-only records."""
        if "label" in reference.columns and (reference["label"] != 0).any():
            raise ValueError("the reference for z-scoring must be normal-only data")
        cols = template_columns(reference, raw=self.use_raw)
        if not cols:
            raise KeyError("reference frame has no per-template columns")
        self.reference_ = {
            str(group): {c: (float(g[c].mean()), float(g[c].std(ddof=0)) or 1.0) for c in cols}
            for group, g in reference.groupby(self.group_col, sort=True)
        }
        return self

    def compute(self, records: pd.DataFrame, **kwargs) -> pd.Series:
        cols = template_columns(records, raw=self.use_raw)
        if not cols:
            raise KeyError("no per-template columns; run ClipScorer with keep_per_template=True")
        if not self.reference_:
            log.warning("%s is unfitted: standardising against the evaluation batch, which "
                        "leaks across images and hides distribution shift. Call fit(train_good).",
                        self.name)
            groups = records[self.group_col] if self.group_col in records else None
            z = np.column_stack([zscore(records[c], groups).to_numpy() for c in cols])
            return pd.Series(z.std(axis=1, ddof=0), index=records.index)

        out = pd.Series(np.nan, index=records.index, dtype=float)
        for group, g in records.groupby(self.group_col, sort=True):
            ref = self.reference_.get(str(group))
            if ref is None:
                raise KeyError(f"no reference statistics for {group!r}; fit on all categories")
            z = np.column_stack([(g[c].to_numpy() - ref[c][0]) / ref[c][1] for c in cols])
            out.loc[g.index] = z.std(axis=1, ddof=0)
        return out


class BackboneEnsemble(UncertaintyEstimator):
    """Disagreement across heterogeneously pretrained backbones. **Real epistemic uncertainty.**

    Kind
    ----
    Epistemic (model uncertainty).

    Method
    ------
    Score the same images with several backbones whose pretraining data and
    objective genuinely differ (OpenAI CLIP, LAION OpenCLIP, SigLIP), rank-normalise
    each backbone's scores within a category so they are comparable, and take the
    dispersion across backbones.

    Failure modes
    -------------
    Backbones sharing pretraining data are correlated and will look falsely
    confident; and this needs one cached embedding pass per backbone, which is the
    only estimator here with a non-trivial one-off cost.
    """

    name = "backbone_ens"
    info = EstimatorInfo(
        kind="epistemic",
        inputs="score columns from >=2 backbones, joined on image_id",
        failure_modes="correlated pretraining hides disagreement; needs N cached passes",
    )

    def __init__(self, score_frames: dict[str, pd.DataFrame] | None = None,
                 group_col: str = "category", stat: str = "std") -> None:
        self.score_frames = score_frames or {}
        self.group_col = group_col
        self.stat = stat

    def compute(self, records: pd.DataFrame, score_frames: dict[str, pd.DataFrame] | None = None,
                **kwargs) -> pd.Series:
        frames = score_frames or self.score_frames
        if len(frames) < 2:
            raise ValueError(f"need >= 2 backbones, got {len(frames)}")
        cols = []
        for tag, frame in sorted(frames.items()):
            sub = frame.set_index("image_id")["anomaly_score"]
            aligned = records["image_id"].map(sub)
            ranks = aligned.groupby(records[self.group_col]).rank(pct=True) \
                if self.group_col in records else aligned.rank(pct=True)
            cols.append(ranks.to_numpy())
        stacked = np.column_stack(cols)
        if self.stat == "std":
            u = np.nanstd(stacked, axis=1, ddof=0)
        elif self.stat == "range":
            u = np.nanmax(stacked, axis=1) - np.nanmin(stacked, axis=1)
        else:
            raise KeyError(f"unknown stat {self.stat!r}")
        return pd.Series(u, index=records.index)


class TTAVariance(UncertaintyEstimator):
    """Variance of the anomaly score under image-side test-time augmentation.

    Kind
    ----
    Mixed aleatoric/epistemic; directly probes sensitivity to nuisance transforms.

    Method
    ------
    Re-embed each image under K augmentations (flips, crops, scale and mild
    photometric jitter), score each view, and take the spread. Unlike text-side
    paraphrase noise this perturbs the modality the defect actually lives in.

    Failure modes
    -------------
    Augmentations that destroy small defects (aggressive crops on ``screw``)
    inflate uncertainty for correct predictions; the augmentation set is a
    hyperparameter the result is sensitive to.
    """

    name = "tta_var"
    info = EstimatorInfo(
        kind="mixed",
        inputs="cached TTA-view embeddings",
        failure_modes="augmentations can destroy small defects; sensitive to the view set",
    )

    def __init__(self, view_frames: Sequence[pd.DataFrame] | None = None, stat: str = "std") -> None:
        self.view_frames = list(view_frames or [])
        self.stat = stat

    def compute(self, records: pd.DataFrame, view_frames: Sequence[pd.DataFrame] | None = None,
                **kwargs) -> pd.Series:
        frames = list(view_frames or self.view_frames)
        if len(frames) < 2:
            raise ValueError(f"need >= 2 TTA views, got {len(frames)}")
        cols = [records["image_id"].map(f.set_index("image_id")["anomaly_score"]).to_numpy()
                for f in frames]
        stacked = np.column_stack(cols)
        u = np.nanstd(stacked, axis=1, ddof=0) if self.stat == "std" else \
            np.nanmax(stacked, axis=1) - np.nanmin(stacked, axis=1)
        return pd.Series(u, index=records.index)


class NormalManifoldDistance(UncertaintyEstimator):
    """kNN distance from the image embedding to the bank of ``train/good`` embeddings.

    Kind
    ----
    Distributional (out-of-distribution / covariate shift).

    Why this one is prioritised
    ---------------------------
    It is the only signal in the suite that answers "is this input unlike anything
    I was calibrated on?". Prompt spread, token entropy and vote variance are all
    computed *inside* the model's own belief and stay small when the model is
    confidently wrong on a shifted input - the silent-failure mode. This one rises
    under corruption by construction, which is exactly what the corruption sweep
    tests.

    Failure modes
    -------------
    A defect that is *semantically* wrong but *visually* typical (a mislabelled
    but in-distribution part) leaves this signal flat; and it is a PatchCore-lite,
    so it inherits PatchCore's sensitivity to bank size and to category alignment.

    **Rank normalisation destroys the shift signal.** ``normalize_per_category=True``
    maps distances to within-category ranks, which is right for ranking images
    *inside* a category and for AURC, but it makes the signal scale-free: the mean
    is 0.5 whatever the input. That erases the property this estimator exists for,
    since under corruption the absolute distance to the normal manifold grows while
    the rank does not move at all. Use ``manifold_knn_raw`` for any question about
    distribution shift, including the corruption sweep.

    **Its sign depends on the decision.** Measured on MVTec (ViT-B/16, delta=0.05)
    it reaches 0.824 error-prediction AUROC among images predicted *normal* - it
    is the strongest signal in the suite for catching a missed anomaly - but 0.145
    among images predicted *anomalous*, i.e. strongly anti-correlated there. That
    is coherent rather than broken: the quantity being measured is "how unlike the
    normal manifold is this", so a flagged image that is far from the manifold is
    probably a *correct* detection. It behaves as a second anomaly detector, not as
    a symmetric uncertainty. Fusing it as if it were symmetric will cancel these
    two regimes out; condition on the predicted class, or use it one-sided.
    """

    name = "manifold_knn"
    info = EstimatorInfo(
        kind="distributional",
        inputs="cached image embeddings for test + train/good",
        failure_modes="blind to in-distribution semantic errors; bank-size sensitive",
    )

    def __init__(self, cache: EmbeddingCache, model_tag: str, k: int = 5,
                 corruption: str = "none", severity: int = 0,
                 normalize_per_category: bool = True) -> None:
        self.cache = cache
        self.model_tag = model_tag
        self.k = k
        self.corruption = corruption
        self.severity = severity
        self.normalize_per_category = normalize_per_category

    def compute(self, records: pd.DataFrame, **kwargs) -> pd.Series:
        out = pd.Series(np.nan, index=records.index, dtype=float)
        # Group by (category, split): a frame holding both splits would otherwise
        # look up every row in whichever shard happened to come first.
        for (category, split), g in records.groupby(["category", "split"], sort=True):
            # The normal-manifold bank is always the *clean* train/good embeddings,
            # even when the test images are corrupted - that is what makes this
            # signal rise under corruption instead of drifting along with it.
            bank_ids, bank, _ = self.cache.load(
                model_tag=self.model_tag, category=category, split="train", kind="image",
                corruption="none", severity=0)
            bank = l2_normalize(bank)
            ids, emb, _ = self.cache.load(
                model_tag=self.model_tag, category=category, split=str(split),
                kind="image", corruption=self.corruption, severity=self.severity)
            lookup = {i: j for j, i in enumerate(ids)}
            rows = l2_normalize(emb[[lookup[i] for i in g["image_id"]]])
            sims = rows @ bank.T
            k = int(min(self.k, sims.shape[1]))
            topk = np.partition(sims, -k, axis=1)[:, -k:].mean(axis=1)
            out.loc[g.index] = 1.0 - topk
        if self.normalize_per_category:
            out = out.groupby(records["category"]).rank(pct=True)
        return out


class RawNormalManifoldDistance(NormalManifoldDistance):
    """kNN distance to the ``train/good`` bank on its **absolute** scale.

    Identical to :class:`NormalManifoldDistance` except that distances are not
    rank-normalised, so the value can grow when the whole input distribution
    shifts. This is the variant the corruption sweep needs: a rank is invariant to
    the shift it is supposed to detect.

    Failure modes
    -------------
    Not comparable across categories - ``screw`` and ``carpet`` sit at different
    absolute distances from their own banks - so it must not be pooled across
    categories without standardising against each category's *clean* distance.
    """

    name = "manifold_knn_raw"
    info = EstimatorInfo(
        kind="distributional",
        inputs="cached image embeddings for test + train/good",
        failure_modes="absolute scale differs per category; do not pool without standardising",
    )

    def __init__(self, cache: EmbeddingCache, model_tag: str, k: int = 5,
                 corruption: str = "none", severity: int = 0) -> None:
        super().__init__(cache, model_tag, k, corruption, severity,
                         normalize_per_category=False)
