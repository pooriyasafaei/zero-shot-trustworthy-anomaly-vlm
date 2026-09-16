"""CERM-AD Stage I unit tests. None of these require Qwen weights or a GPU."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from PIL import Image
from scipy import ndimage

from tzsad.causal.donor import DonorBank, aligned_donor_pixels, load_mask, load_rgb
from tzsad.causal.effects import (compute_effects, empirical_placebo_p, log_odds,
                                  median_absolute_deviation)
from tzsad.causal.placebo import (context_ring, generate_placebos, placebo_mask_array,
                                  ring_descriptor)
from tzsad.causal.reconstruction import (apply_intervention, identity_intervention,
                                         reconstruction_diagnostics, seam_score)


# --------------------------------------------------------------- fixtures
@pytest.fixture
def toy_image():
    """A textured 96x96 image with a bright square 'defect'."""
    rng = np.random.default_rng(0)
    img = np.clip(rng.normal(120, 10, (96, 96, 3)), 0, 255).astype(np.uint8)
    img[40:52, 40:52] = 250
    return img


@pytest.fixture
def toy_mask():
    mask = np.zeros((96, 96), dtype=bool)
    mask[40:52, 40:52] = True
    return mask


@pytest.fixture
def toy_donor():
    rng = np.random.default_rng(1)
    return np.clip(rng.normal(118, 10, (96, 96, 3)), 0, 255).astype(np.uint8)


# --------------------------------------------------------------- placebos
def test_placebo_preserves_mask_area(toy_image, toy_mask):
    """Shape and area are held fixed so only location varies between arms."""
    plan = generate_placebos(toy_image, toy_mask, k=5, seed=1, image_id="t", candidate_stride=6)
    assert plan.n_valid > 0
    for p in plan.placebos:
        arr = placebo_mask_array(toy_mask, p)
        assert arr.sum() == toy_mask.sum()
        assert p.mask_area == int(toy_mask.sum())


def test_placebo_stays_inside_the_image(toy_image, toy_mask):
    plan = generate_placebos(toy_image, toy_mask, k=8, seed=2, image_id="t", candidate_stride=6)
    for p in plan.placebos:
        arr = placebo_mask_array(toy_mask, p)
        assert arr.shape == toy_mask.shape
        assert arr.sum() == toy_mask.sum(), "a clipped placebo would not match the defect area"


def test_placebo_does_not_overlap_the_defect(toy_image, toy_mask):
    """A 'placebo' that removes part of the defect is not a control."""
    plan = generate_placebos(toy_image, toy_mask, k=8, seed=3, image_id="t",
                             max_iou=0.01, candidate_stride=6)
    for p in plan.placebos:
        arr = placebo_mask_array(toy_mask, p)
        union = (arr | toy_mask).sum()
        assert (arr & toy_mask).sum() / union <= 0.01
        assert p.iou_with_defect <= 0.01


def test_placebo_generation_is_deterministic(toy_image, toy_mask):
    a = generate_placebos(toy_image, toy_mask, k=6, seed=7, image_id="img", candidate_stride=6)
    b = generate_placebos(toy_image, toy_mask, k=6, seed=7, image_id="img", candidate_stride=6)
    assert [(p.dx, p.dy) for p in a.placebos] == [(p.dx, p.dy) for p in b.placebos]

    c = generate_placebos(toy_image, toy_mask, k=6, seed=8, image_id="img", candidate_stride=6)
    assert [(p.dx, p.dy) for p in a.placebos] != [(p.dx, p.dy) for p in c.placebos]


def test_different_images_get_different_placebo_draws(toy_image, toy_mask):
    """The draw is keyed by image id, so two images do not share placements."""
    a = generate_placebos(toy_image, toy_mask, k=6, seed=7, image_id="a", candidate_stride=6)
    b = generate_placebos(toy_image, toy_mask, k=6, seed=7, image_id="b", candidate_stride=6)
    assert [(p.dx, p.dy) for p in a.placebos] != [(p.dx, p.dy) for p in b.placebos]


def test_placebos_are_separated_not_near_duplicates(toy_image, toy_mask):
    """K nearly identical placements would collapse the placebo distribution."""
    plan = generate_placebos(toy_image, toy_mask, k=6, seed=4, image_id="t",
                             min_separation=10, candidate_stride=4)
    offsets = [(p.dx, p.dy) for p in plan.placebos]
    for i, (dx1, dy1) in enumerate(offsets):
        for dx2, dy2 in offsets[i + 1:]:
            assert max(abs(dx1 - dx2), abs(dy1 - dy2)) >= 10


def test_too_few_placebos_is_reported_not_fabricated(toy_image):
    """Asking for more placebos than fit must degrade visibly."""
    mask = np.zeros((96, 96), dtype=bool)
    mask[4:92, 4:92] = True                     # leaves almost no room to translate
    plan = generate_placebos(toy_image, mask, k=19, seed=5, image_id="t", candidate_stride=4)
    assert plan.n_valid < 19
    assert plan.quality != "ok" or plan.n_valid == 19
    assert plan.reason or plan.n_valid == 19
    assert plan.tail_resolution == pytest.approx(1 / (plan.n_valid + 1))


def test_empty_mask_is_degenerate_not_crash(toy_image):
    plan = generate_placebos(toy_image, np.zeros((96, 96), bool), k=5, seed=1)
    assert plan.n_valid == 0 and plan.quality == "degenerate"


def test_context_ring_surrounds_without_including(toy_mask):
    ring = context_ring(toy_mask, 5)
    assert not (ring & toy_mask).any()
    assert ring.sum() > 0
    assert (ndimage.binary_dilation(toy_mask, iterations=5) | toy_mask)[ring].all()


def test_ring_descriptor_separates_different_surfaces(toy_image):
    flat = np.full((96, 96, 3), 128, dtype=np.uint8)
    mask = np.zeros((96, 96), dtype=bool)
    mask[20:30, 20:30] = True
    ring = context_ring(mask, 6)
    assert np.linalg.norm(ring_descriptor(toy_image, ring) - ring_descriptor(flat, ring)) > 0


# --------------------------------------------------------- reconstruction
def test_reconstruction_changes_nothing_outside_the_mask(toy_image, toy_donor, toy_mask):
    """The core guarantee: an outside-mask change makes the counterfactual unreadable."""
    iv = apply_intervention(toy_image, toy_donor, toy_mask, "defect_removed", feather_px=3)
    assert np.array_equal(iv.image[~toy_mask], toy_image[~toy_mask])
    assert iv.outside_change_max == 0


def test_reconstruction_actually_changes_inside_the_mask(toy_image, toy_donor, toy_mask):
    iv = apply_intervention(toy_image, toy_donor, toy_mask, "defect_removed", feather_px=3)
    assert not np.array_equal(iv.image[toy_mask], toy_image[toy_mask])
    assert iv.edit_energy > 1.0
    assert iv.mask_area == int(toy_mask.sum())


def test_feather_ramps_only_inward(toy_image, toy_donor, toy_mask):
    """A feather that extended outwards would modify original pixels."""
    for feather in (0, 3, 6):
        iv = apply_intervention(toy_image, toy_donor, toy_mask, "placebo", feather_px=feather)
        assert np.array_equal(iv.image[~toy_mask], toy_image[~toy_mask])
        assert iv.feather_px == feather


def test_same_donor_serves_every_intervention(toy_image, toy_donor, toy_mask):
    """Donor identity must not differ between the defect arm and the placebo arm."""
    plan = generate_placebos(toy_image, toy_mask, k=4, seed=1, image_id="t", candidate_stride=6)
    ivs = [apply_intervention(toy_image, toy_donor, toy_mask, "defect_removed")]
    for p in plan.placebos:
        ivs.append(apply_intervention(toy_image, toy_donor, placebo_mask_array(toy_mask, p),
                                      "placebo"))
    assert all(iv.outside_change_max == 0 for iv in ivs)
    assert {iv.mask_area for iv in ivs} == {int(toy_mask.sum())}


def test_mismatched_shapes_are_rejected(toy_image, toy_mask):
    with pytest.raises(ValueError, match="donor shape"):
        apply_intervention(toy_image, np.zeros((10, 10, 3), np.uint8), toy_mask, "placebo")
    with pytest.raises(ValueError, match="mask shape"):
        apply_intervention(toy_image, toy_image, np.zeros((10, 10), bool), "placebo")


def test_identity_intervention_is_a_no_op(toy_image):
    iv = identity_intervention(toy_image)
    assert np.array_equal(iv.image, toy_image)
    assert iv.kind == "original" and iv.mask_area == 0 and iv.edit_energy == 0.0


def test_seam_score_is_larger_for_a_harsher_splice(toy_image, toy_mask):
    smooth = apply_intervention(toy_image, toy_image.copy(), toy_mask, "placebo", feather_px=3)
    harsh_donor = np.zeros_like(toy_image)
    harsh = apply_intervention(toy_image, harsh_donor, toy_mask, "placebo", feather_px=0)
    assert harsh.seam_score > smooth.seam_score


def test_reconstruction_diagnostics_compare_arms(toy_image, toy_donor, toy_mask):
    ivs = [identity_intervention(toy_image),
           apply_intervention(toy_image, toy_donor, toy_mask, "defect_removed")]
    plan = generate_placebos(toy_image, toy_mask, k=3, seed=1, image_id="t", candidate_stride=6)
    for p in plan.placebos:
        ivs.append(apply_intervention(toy_image, toy_donor,
                                      placebo_mask_array(toy_mask, p), "placebo"))
    diag = reconstruction_diagnostics(ivs)
    assert diag["max_outside_change"] == 0
    assert np.isfinite(diag["defect_seam_score"])
    assert np.isfinite(diag["placebo_seam_score_mean"])


# ----------------------------------------------------------------- donors
def _fake_index(tmp_path, n_train=4, n_test=2):
    rows = []
    rng = np.random.default_rng(0)
    for split, folder, n, label in (("train", "good", n_train, 0),
                                    ("test", "scratch", n_test, 1)):
        d = tmp_path / "bottle" / split / folder
        d.mkdir(parents=True, exist_ok=True)
        gt = tmp_path / "bottle" / "ground_truth" / folder
        for i in range(n):
            arr = np.clip(rng.normal(120, 8, (64, 64, 3)), 0, 255).astype(np.uint8)
            if label:
                arr[24:34, 24:34] = 250
            path = d / f"{i:03d}.png"
            Image.fromarray(arr).save(path)
            mask_path = ""
            if label:
                gt.mkdir(parents=True, exist_ok=True)
                m = np.zeros((64, 64), np.uint8)
                m[24:34, 24:34] = 255
                mask_path = str(gt / f"{i:03d}_mask.png")
                Image.fromarray(m).save(mask_path)
            rows.append({"category": "bottle", "split": split, "defect_type": folder,
                         "path": str(path), "label": label, "mask_path": mask_path,
                         "image_id": f"bottle/{split}/{folder}/{i:03d}"})
    return pd.DataFrame(rows)


def test_donor_bank_holds_only_train_good(tmp_path):
    """An anomalous donor would put defect pixels into a 'defect removed' image."""
    index = _fake_index(tmp_path)
    bank = DonorBank(index)
    assert (bank.index["split"] == "train").all()
    assert (bank.index["label"] == 0).all()
    assert set(bank.index["image_id"]) == set(
        index[index.split == "train"]["image_id"])


def test_donor_bank_rejects_anomalous_training_rows(tmp_path):
    index = _fake_index(tmp_path)
    index.loc[index.split == "train", "label"] = 1
    with pytest.raises(ValueError):
        DonorBank(index)


def test_donor_bank_needs_a_train_split(tmp_path):
    index = _fake_index(tmp_path)
    with pytest.raises(ValueError, match="train/good"):
        DonorBank(index[index.split == "test"])


def test_donor_selection_is_deterministic_and_excludes_the_defect(tmp_path):
    index = _fake_index(tmp_path)
    bank = DonorBank(index)
    target = index[index.label == 1].iloc[0]
    mask = load_mask(target["mask_path"], (64, 64))
    exclude = ndimage.binary_dilation(mask, iterations=4)
    a = bank.select(target["path"], "bottle", exclude)
    b = bank.select(target["path"], "bottle", exclude)
    assert a.donor_image_id == b.donor_image_id
    assert a.donor_image_id in set(bank.index["image_id"])
    assert 0.0 < a.valid_fraction <= 1.0
    assert a.n_candidates == len(bank.candidates("bottle"))


def test_aligned_donor_is_resized_to_the_target(tmp_path):
    index = _fake_index(tmp_path)
    donor = index[index.split == "train"].iloc[0]
    px = aligned_donor_pixels(donor["path"], (40, 30))
    assert px.shape == (40, 30, 3)


# ---------------------------------------------------------------- effects
def test_cee_formula():
    e = compute_effects(2.0, [0.5, 1.0, 1.5])
    assert e.placebo_delta_median == pytest.approx(1.0)
    assert e.cee == pytest.approx(1.0)
    assert e.defect_beats_all_placebos is True
    assert e.defect_placebo_percentile == pytest.approx(1.0)


def test_empirical_placebo_p_formula():
    # None of the 19 placebos reaches the defect effect -> the floor, 1/20.
    assert empirical_placebo_p(5.0, [0.0] * 19) == pytest.approx(1 / 20)
    # All of them do -> 20/20.
    assert empirical_placebo_p(0.0, [1.0] * 19) == pytest.approx(20 / 20)
    # Exactly four at or above -> 5/20.
    assert empirical_placebo_p(1.0, [1.0] * 4 + [0.0] * 15) == pytest.approx(5 / 20)


def test_empirical_placebo_p_can_never_be_zero():
    """The +1 counts the observed effect among the candidates."""
    assert empirical_placebo_p(99.0, [0.0] * 5) == pytest.approx(1 / 6)


def test_robust_cee_z_uses_scaled_mad():
    deltas = [0.0, 1.0, 2.0, 3.0, 4.0]
    e = compute_effects(6.0, deltas)
    mad = median_absolute_deviation(deltas)
    assert mad == pytest.approx(1.0)
    assert e.robust_cee_z == pytest.approx((6.0 - 2.0) / (1.4826 * 1.0), rel=1e-6)


def test_mad_of_constant_sample_is_zero_and_z_stays_finite():
    e = compute_effects(1.0, [0.5] * 5)
    assert e.placebo_delta_mad == 0.0
    assert np.isfinite(e.robust_cee_z)


def test_significance_resolution_guard():
    """With too few placebos the 0.05 question cannot be answered, so it is NA."""
    coarse = compute_effects(5.0, [0.0] * 4)          # finest tail is 1/5 = 0.2
    assert coarse.tail_resolution == pytest.approx(0.2)
    assert coarse.supported_at_005 is None

    fine = compute_effects(5.0, [0.0] * 19)           # finest tail is 1/20 = 0.05
    assert fine.tail_resolution == pytest.approx(0.05)
    assert fine.supported_at_005 is True

    fine_null = compute_effects(-5.0, [0.0] * 19)
    assert fine_null.supported_at_005 is False


def test_effects_with_no_placebos_are_reported_not_invented():
    e = compute_effects(1.0, [])
    assert e.cee != e.cee                              # NaN
    assert e.supported_at_005 is None
    assert e.as_dict()["n_valid_placebos"] == 0


def test_log_odds_matches_definition():
    assert log_odds(0.5) == pytest.approx(0.0)
    assert log_odds(0.75) == pytest.approx(np.log(3.0), rel=1e-6)
    assert np.isfinite(log_odds(0.0)) and np.isfinite(log_odds(1.0))
