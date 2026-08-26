"""CLIP scoring maths and the cache, with a stubbed backbone (no weights needed)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tzsad.features.cache import EmbeddingCache, l2_normalize
from tzsad.features.clip_embedder import BackboneSpec, WindowSpec
from tzsad.features.heatmap import bank_maps, paint
from tzsad.scorers.base import ScoringContext
from tzsad.scorers.clip_scorer import (ClipScorer, ClipScorerConfig, softmax_anomaly_probability,
                                       template_columns)
from tzsad.scorers.prompts import class_name_for, get_prompt_set


def test_softmax_score_is_a_probability_and_monotone():
    """Defect #4: cosine differences are not probabilities; this converts them."""
    cn = np.array([0.20, 0.20, 0.20])
    ca = np.array([0.15, 0.20, 0.30])
    p = softmax_anomaly_probability(cn, ca, tau=0.01)
    assert (p > 0).all() and (p < 1).all()
    assert p[0] < p[1] < p[2]
    assert p[1] == pytest.approx(0.5)


def test_tau_controls_sharpness():
    cn, ca = np.array([0.2]), np.array([0.25])
    sharp = softmax_anomaly_probability(cn, ca, tau=0.005)
    soft = softmax_anomaly_probability(cn, ca, tau=0.5)
    assert sharp[0] > soft[0] > 0.5


def test_prompt_sets_render_class_names():
    normal, anomaly = get_prompt_set("base").render(class_name_for("metal_nut"))
    assert len(normal) == len(anomaly) == 5
    assert all("metal nut" in p for p in normal + anomaly)
    with pytest.raises(KeyError):
        get_prompt_set("nonexistent")


def test_cache_roundtrip(tmp_path):
    cache = EmbeddingCache(tmp_path)
    ids = ["a/1", "a/2"]
    emb = np.random.default_rng(0).normal(size=(2, 8)).astype(np.float32)
    cache.save(ids, emb, model_tag="m/x", category="a", split="test")
    got_ids, got, _ = cache.load(model_tag="m/x", category="a", split="test")
    assert got_ids == ids
    assert np.allclose(got, emb, atol=1e-3)          # float16 storage


def test_cache_missing_shard_fails_loudly(tmp_path):
    with pytest.raises(FileNotFoundError, match="embedding shard missing"):
        EmbeddingCache(tmp_path).load(model_tag="m", category="a", split="test")


def _fake_run(tmp_path, n=8, dim=16, seed=0):
    """Build a cache + index whose embeddings have a planted anomaly direction."""
    rng = np.random.default_rng(seed)
    cache = EmbeddingCache(tmp_path)
    spec = BackboneSpec("Fake-B", "test")
    index = pd.DataFrame({
        "category": ["a"] * n, "split": ["test"] * n,
        "defect_type": ["good"] * (n // 2) + ["scratch"] * (n // 2),
        "path": [f"/x/{i}.png" for i in range(n)], "label": [0] * (n // 2) + [1] * (n // 2),
        "mask_path": "", "image_id": [f"a/test/{i}" for i in range(n)],
    })
    direction = np.zeros(dim, dtype=np.float32)
    direction[0] = 1.0
    emb = l2_normalize(rng.normal(size=(n, dim)).astype(np.float32) * 0.1
                       + index.label.to_numpy()[:, None] * direction)
    cache.save(index.image_id.tolist(), emb, model_tag=spec.cache_tag, category="a", split="test")
    t_anom = l2_normalize(np.tile(direction, (3, 1)) + rng.normal(size=(3, dim)) * 0.01)
    t_norm = l2_normalize(-np.tile(direction, (3, 1)) + rng.normal(size=(3, dim)) * 0.01)
    return cache, spec, index, {"a": (t_norm.astype(np.float32), t_anom.astype(np.float32))}


def test_scorer_produces_valid_records_and_separates_classes(tmp_path):
    cache, spec, index, text = _fake_run(tmp_path)
    scorer = ClipScorer(spec, cache, ClipScorerConfig(tau=0.1), text_embeddings=text)
    rec = scorer.score_index(index, ScoringContext(run_id="t"))
    assert len(rec) == len(index)
    assert rec.anomaly_score.between(0, 1).all()
    assert rec[rec.label == 1].anomaly_score.mean() > rec[rec.label == 0].anomaly_score.mean()
    assert rec.parse_ok.all()
    assert any(c.startswith("tpl_") for c in rec.columns)


def test_scorer_fails_loudly_on_a_missing_image(tmp_path):
    cache, spec, index, text = _fake_run(tmp_path)
    index.loc[0, "image_id"] = "a/test/does_not_exist"
    scorer = ClipScorer(spec, cache, text_embeddings=text)
    with pytest.raises(KeyError, match="missing from the"):
        scorer.score_index(index, ScoringContext(run_id="t"))


def test_window_spec_tiles_the_image():
    win = WindowSpec(scales=(0.5,), stride_frac=0.5, image_size=224)
    boxes = win.windows()
    assert len(boxes) == 9                       # 3 x 3 positions
    assert all(r - l == 112 and b - t == 112 for l, t, r, b in boxes)
    assert (0, 0, 112, 112) in boxes and (112, 112, 224, 224) in boxes


def test_paint_averages_overlapping_windows():
    boxes = [(0, 0, 2, 2), (0, 0, 2, 2)]
    m = paint(np.array([0.0, 1.0]), boxes, size=2, out_size=2)
    assert np.allclose(m, 0.5)


def test_bank_map_is_high_where_the_test_window_is_unlike_normals():
    dim = 8
    normal = np.zeros(dim, dtype=np.float32)
    normal[0] = 1.0
    odd = np.zeros(dim, dtype=np.float32)
    odd[1] = 1.0
    bank = np.tile(normal, (4, 2, 1))            # 4 normals, 2 windows
    test = np.stack([np.stack([normal, odd])])   # 1 image: window 0 normal, window 1 odd
    boxes = [(0, 0, 8, 16), (8, 0, 16, 16)]
    m = bank_maps(test, bank, boxes, image_size=16, k=1, out_size=16)
    assert m[0][:, :8].mean() < m[0][:, 8:].mean()


def test_manifold_distance_looks_up_the_right_shard_per_split(tmp_path):
    """A frame holding both splits must not read every row from one shard."""
    from tzsad.uncertainty.image_side import NormalManifoldDistance

    rng = np.random.default_rng(0)
    cache = EmbeddingCache(tmp_path)
    dim, tag = 8, "M"
    normal = l2_normalize(rng.normal(size=(6, dim)).astype(np.float32) * 0.05 + 1.0)
    cache.save([f"a/train/{i}" for i in range(6)], normal,
               model_tag=tag, category="a", split="train", kind="image",
               corruption="none", severity=0)
    far = l2_normalize(rng.normal(size=(3, dim)).astype(np.float32))
    cache.save([f"a/test/{i}" for i in range(3)], far,
               model_tag=tag, category="a", split="test", kind="image",
               corruption="none", severity=0)

    records = pd.DataFrame({
        "category": ["a"] * 9,
        "split": ["train"] * 6 + ["test"] * 3,
        "image_id": [f"a/train/{i}" for i in range(6)] + [f"a/test/{i}" for i in range(3)],
    })
    u = NormalManifoldDistance(cache, tag, k=2, normalize_per_category=False).compute(records)
    assert u.notna().all()
    # Train images are in the bank, so they sit closer to it than the test images.
    assert u[:6].mean() < u[6:].mean()


# Prototype reference numbers, measured on the full MVTec-AD test set with
# OpenCLIP ViT-B/16 (openai), the 5+5 base prompt set, and the prototype's own
# score: the mean over templates of (cos_anomaly - cos_normal). The colleague's
# notebook reported 0.8228 mean AUROC; this rewrite reproduces 0.8227.
#
# These are pinned so that any future refactor that changes the scoring semantics
# fails loudly instead of drifting.
PROTOTYPE_AUROC = {
    "bottle": 0.9198, "cable": 0.7116, "capsule": 0.6598, "carpet": 0.8864,
    "grid": 0.9449, "hazelnut": 0.8114, "leather": 0.9949, "metal_nut": 0.8915,
    "pill": 0.6803, "screw": 0.5989, "tile": 0.9859, "toothbrush": 0.8167,
    "transistor": 0.7100, "wood": 0.9781, "zipper": 0.7508,
}
PROTOTYPE_MEAN_AUROC = 0.8227


def test_prototype_reference_numbers_are_self_consistent():
    """The pinned per-category values must average to the pinned mean."""
    mean = sum(PROTOTYPE_AUROC.values()) / len(PROTOTYPE_AUROC)
    assert len(PROTOTYPE_AUROC) == 15
    assert mean == pytest.approx(PROTOTYPE_MEAN_AUROC, abs=5e-4)


@pytest.mark.skipif(
    not (Path(__file__).resolve().parents[1] / "results/clip_full/records_clip.parquet").exists(),
    reason="needs a full clip_full scoring run",
)
def test_reproduces_the_prototype_auroc_from_cached_records():
    """Regression test against the prototype, recomputed offline from records.

    The per-template raw columns hold (cos_anomaly - cos_normal) for each template,
    so their row mean *is* the prototype's score. No GPU pass is involved, which is
    the decoupling rule doing its job.
    """
    from tzsad.eval.metrics import safe_auroc

    path = Path(__file__).resolve().parents[1] / "results/clip_full/records_clip.parquet"
    df = pd.read_parquet(path)
    test = df[df["split"] == "test"]
    raw_cols = template_columns(test, raw=True)
    assert len(raw_cols) == 5, "the base prompt set has five template pairs"
    proto = test[raw_cols].to_numpy().mean(axis=1)
    test = test.assign(proto=proto)

    measured = {c: safe_auroc(g.label, g.proto) for c, g in test.groupby("category")}
    assert set(measured) == set(PROTOTYPE_AUROC)
    for category, expected in PROTOTYPE_AUROC.items():
        assert measured[category] == pytest.approx(expected, abs=1e-3), category
    mean = float(np.mean(list(measured.values())))
    assert mean == pytest.approx(PROTOTYPE_MEAN_AUROC, abs=1e-3)


def test_rank_normalisation_destroys_the_distribution_shift_signal(tmp_path):
    """The bug the corruption sweep exposed, pinned so it cannot come back.

    A within-category rank is invariant to a shift of the whole category, which is
    precisely the thing NormalManifoldDistance is meant to detect. The raw variant
    must move when every image drifts away from the bank; the normalised one must
    not.
    """
    from tzsad.uncertainty.image_side import NormalManifoldDistance, RawNormalManifoldDistance

    rng = np.random.default_rng(0)
    cache = EmbeddingCache(tmp_path)
    dim, tag = 16, "M"
    bank = l2_normalize(np.tile(np.eye(1, dim, 0).ravel(), (10, 1)).astype(np.float32)
                        + rng.normal(0, 0.01, (10, dim)).astype(np.float32))
    cache.save([f"a/train/{i}" for i in range(10)], bank, model_tag=tag, category="a",
               split="train", kind="image", corruption="none", severity=0)

    def shard(corruption, drift):
        base = np.tile(np.eye(1, dim, 0).ravel(), (8, 1)).astype(np.float32)
        base[:, 1] += drift                      # push every image off the manifold
        cache.save([f"a/test/{i}" for i in range(8)], l2_normalize(base), model_tag=tag,
                   category="a", split="test", kind="image", corruption=corruption, severity=1)

    records = pd.DataFrame({"category": ["a"] * 8, "split": ["test"] * 8,
                            "image_id": [f"a/test/{i}" for i in range(8)]})
    means = {}
    for corruption, drift in (("mild", 0.05), ("severe", 1.5)):
        shard(corruption, drift)
        means[corruption] = (
            float(NormalManifoldDistance(cache, tag, k=3, corruption=corruption,
                                         severity=1).compute(records).mean()),
            float(RawNormalManifoldDistance(cache, tag, k=3, corruption=corruption,
                                            severity=1).compute(records).mean()),
        )
    (norm_mild, raw_mild), (norm_severe, raw_severe) = means["mild"], means["severe"]
    assert raw_severe > raw_mild * 2, (raw_mild, raw_severe)     # the raw scale reacts
    assert norm_severe == pytest.approx(norm_mild)               # the rank does not


def test_cache_tops_up_a_shard_built_from_a_subset(tmp_path):
    """A shard cached from a subsampled index must not be reused for a full one.

    The cache key records model/category/split/corruption but not *which* images
    went in, so without a coverage check a subset shard is silently reused and
    scoring later dies on a missing id.
    """
    from tzsad.features.clip_embedder import embed_index

    cache = EmbeddingCache(tmp_path)
    spec = BackboneSpec("Fake-B", "test")

    class Counter:
        """Minimal embedder that records how many images it was asked to encode."""

        def __init__(self):
            self.spec = spec
            self.calls = []

        def encode_images(self, paths, transform=None):
            self.calls.append(len(paths))
            return np.tile(np.arange(4, dtype=np.float32), (len(paths), 1))

    index = pd.DataFrame({
        "category": ["a"] * 5, "split": ["test"] * 5,
        "path": [f"/x/{i}.png" for i in range(5)],
        "image_id": [f"a/test/{i}" for i in range(5)],
    })
    emb = Counter()
    embed_index(emb, index.iloc[:2], cache)          # subset first
    assert emb.calls == [2]
    assert not cache.covers(index.image_id.tolist(), model_tag=spec.cache_tag,
                            category="a", split="test", kind="image",
                            corruption="none", severity=0)

    embed_index(emb, index, cache)                   # now the full index
    assert emb.calls == [2, 3], "only the three missing images should be embedded"
    ids, arr, _ = cache.load(model_tag=spec.cache_tag, category="a", split="test",
                             kind="image", corruption="none", severity=0)
    assert sorted(ids) == sorted(index.image_id)
    assert arr.shape[0] == 5


def _template_frame(n, shift=0.0, scale=1.0, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"category": ["a"] * n, "label": 0})
    for t in range(5):
        df[f"tplraw_{t:02d}"] = rng.normal(t * 0.1 + shift, 0.01 * scale, n)
    return df


def test_fitted_prompt_std_z_keeps_the_shift_signal_that_transductive_loses():
    """The bug the corruption sweep exposed for the second time.

    Standardising against the batch under evaluation divides the shift away: the
    corrupted batch's own spread is the denominator. A fixed clean reference keeps
    it visible.
    """
    from tzsad.uncertainty.image_side import NormalizedPromptEnsemble

    clean = _template_frame(400, seed=1)
    shifted = _template_frame(400, shift=0.5, scale=6.0, seed=2)   # drifted and noisier

    fitted = NormalizedPromptEnsemble().fit(clean)
    assert fitted.compute(shifted).mean() > 3 * fitted.compute(clean).mean()

    untransductive = NormalizedPromptEnsemble()
    flat = untransductive.compute(shifted).mean() / untransductive.compute(clean).mean()
    assert flat < 1.5, "transductive z-scoring should have hidden the shift"


def test_prompt_std_z_reference_must_be_normal_only():
    from tzsad.uncertainty.image_side import NormalizedPromptEnsemble

    ref = _template_frame(20)
    ref.loc[0, "label"] = 1
    with pytest.raises(ValueError, match="normal-only"):
        NormalizedPromptEnsemble().fit(ref)


def test_unfitted_prompt_std_z_warns(caplog):
    from tzsad.uncertainty.image_side import NormalizedPromptEnsemble

    with caplog.at_level("WARNING"):
        NormalizedPromptEnsemble().compute(_template_frame(30))
    assert any("unfitted" in r.message for r in caplog.records)
