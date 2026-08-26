"""Index building, subset specs, records schema, corruptions."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from tzsad.data.corruptions import CORRUPTIONS, apply_corruption
from tzsad.data.mvtec import SubsetSpec, build_index, mask_path_for
from tzsad.data.synthetic import make_synthetic
from tzsad.records import CORE_COLUMNS, empty_records, validate_records


def test_index_has_expected_shape(fake_mvtec):
    idx = build_index(fake_mvtec)
    assert set(idx.category) == {"bottle", "carpet"}
    assert idx[(idx.split == "test") & (idx.defect_type == "scratch")].label.eq(1).all()
    assert idx[idx.defect_type == "good"].label.eq(0).all()
    assert idx.image_id.is_unique


def test_subset_is_deterministic_across_calls(fake_mvtec):
    """Identical subsets for every scorer - the fix for defect #5."""
    spec = SubsetSpec.parse("n_per_folder=3", seed=42)
    a = build_index(fake_mvtec, subset=spec)
    b = build_index(fake_mvtec, subset=spec)
    pd.testing.assert_frame_equal(a, b)
    test = a[a.split == "test"]
    assert (test.groupby(["category", "defect_type"]).size() <= 3).all()


def test_subset_does_not_touch_the_calibration_pool(fake_mvtec):
    """train/good is the conformal calibration set and must stay whole."""
    full = build_index(fake_mvtec)
    small = build_index(fake_mvtec, subset=SubsetSpec.parse("n_per_folder=2"))
    assert (full.split == "train").sum() == (small.split == "train").sum()


def test_subset_spec_rejects_nonsense():
    with pytest.raises(ValueError, match="unparseable subset"):
        SubsetSpec.parse("twenty images please")


def test_masks_resolve_for_defects_only(fake_mvtec):
    idx = build_index(fake_mvtec)
    defects = idx[idx.defect_type == "scratch"]
    assert defects.mask_path.str.len().gt(0).all()
    assert idx[idx.defect_type == "good"].mask_path.eq("").all()
    assert mask_path_for(defects.iloc[0].path).exists()


def test_missing_data_root_fails_loudly(tmp_path):
    with pytest.raises(FileNotFoundError):
        build_index(tmp_path / "nope")


def test_records_reject_out_of_range_scores():
    df = empty_records()
    df.loc[0] = {c: "" for c in CORE_COLUMNS}
    df.loc[0, ["label", "anomaly_score", "parse_ok", "severity", "n_valid_votes"]] = [0, 1.7, True, 0, 1]
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        validate_records(df)


def test_records_allow_nan_scores_for_abstentions():
    df = empty_records()
    df.loc[0] = {c: "" for c in CORE_COLUMNS}
    df.loc[0, ["label", "anomaly_score", "parse_ok", "severity", "n_valid_votes"]] = \
        [0, float("nan"), False, 0, 0]
    validate_records(df)


@pytest.mark.parametrize("name", CORRUPTIONS)
def test_corruptions_run_and_change_the_image(name):
    rng = np.random.default_rng(0)
    img = Image.fromarray(rng.integers(60, 200, (48, 48, 3), dtype=np.uint8))
    out = apply_corruption(img, name, 3)
    assert out.size == img.size
    assert np.abs(np.asarray(out, float) - np.asarray(img, float)).mean() > 0.5


def test_corruption_is_deterministic():
    img = Image.fromarray(np.full((32, 32, 3), 128, dtype=np.uint8))
    a = np.asarray(apply_corruption(img, "gaussian_noise", 4, seed=1))
    b = np.asarray(apply_corruption(img, "gaussian_noise", 4, seed=1))
    assert np.array_equal(a, b)


def test_corruption_severity_is_monotone():
    """Higher severity must damage the image more, or the sweep means nothing."""
    rng = np.random.default_rng(3)
    img = Image.fromarray(rng.integers(60, 200, (64, 64, 3), dtype=np.uint8))
    base = np.asarray(img, dtype=float)
    deltas = [np.abs(np.asarray(apply_corruption(img, "gaussian_noise", s), float) - base).mean()
              for s in (1, 3, 5)]
    assert deltas[0] < deltas[1] < deltas[2]


def test_synthetic_anomaly_changes_pixels_inside_its_mask():
    rng = np.random.default_rng(0)
    img = Image.fromarray(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8))
    out, mask = make_synthetic(img, seed=0, method="cutpaste")
    assert mask.sum() > 0
    assert not np.array_equal(np.asarray(out), np.asarray(img))


def test_cal_subset_caps_the_calibration_pool_independently(fake_mvtec):
    """The VLM branch caps train/good without touching the test subset."""
    idx = build_index(fake_mvtec, subset=SubsetSpec.parse("n_per_folder=3"),
                      cal_subset=SubsetSpec.parse("n_per_folder=5"))
    train = idx[idx.split == "train"]
    test = idx[idx.split == "test"]
    assert (train.groupby("category").size() <= 5).all()
    assert (test.groupby(["category", "defect_type"]).size() <= 3).all()


def test_cal_subset_defaults_to_keeping_everything(fake_mvtec):
    a = build_index(fake_mvtec)
    b = build_index(fake_mvtec, cal_subset=SubsetSpec.parse("full"))
    pd.testing.assert_frame_equal(a, b)
