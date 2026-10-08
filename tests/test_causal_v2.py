"""CERM-AD Stage I+ tests: artifact matching, sufficiency, scoring, casebook, report.

No Qwen weights needed: the VLM is replaced by a deterministic stub whose response
to the planted defect is known in advance.
"""
from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf
from PIL import Image

from tzsad.causal.artifact_matching import (MaskTemplate, artifact_matched_placebos,
                                            local_metrics, log_ratio_distance,
                                            occlusion_fill, random_placebos, robust_scale)
from tzsad.causal.audit import CausalConfig, ScoreCache, merge_shards, select_targets
from tzsad.causal.audit_v2 import (V2Config, build_target, quadrant, run_audit_v2,
                                   score_target)
from tzsad.causal.casebook import build_casebook, select_cases
from tzsad.causal.donor import DonorBank
from tzsad.causal.placebo import feasible_candidates, generate_placebos
from tzsad.causal.reconstruction import apply_intervention
from tzsad.causal.scoring import array_digest, as_batch_probe, score_images
from tzsad.causal.sufficiency import SourceBank, select_shams


# ------------------------------------------------------------------ fixtures
@pytest.fixture
def synthetic_dataset(tmp_path):
    """Miniature MVTec-shaped data: bright square defects on textured plates."""
    rng = np.random.default_rng(0)
    rows = []
    for category, base in (("bottle", 120), ("carpet", 90)):
        for split, folder, n, label in (("train", "good", 7, 0), ("test", "scratch", 3, 1),
                                        ("test", "good", 2, 0)):
            d = tmp_path / category / split / folder
            d.mkdir(parents=True, exist_ok=True)
            gt = tmp_path / category / "ground_truth" / folder
            for i in range(n):
                arr = np.clip(rng.normal(base, 9, (80, 80, 3)), 0, 255).astype(np.uint8)
                mask_path = ""
                if label:
                    y, x = 20 + 6 * i, 22 + 5 * i
                    arr[y:y + 10, x:x + 10] = 250
                    gt.mkdir(parents=True, exist_ok=True)
                    m = np.zeros((80, 80), np.uint8)
                    m[y:y + 10, x:x + 10] = 255
                    mask_path = str(gt / f"{i:03d}_mask.png")
                    Image.fromarray(m).save(mask_path)
                path = d / f"{i:03d}.png"
                Image.fromarray(arr).save(path)
                rows.append({"category": category, "split": split, "defect_type": folder,
                             "path": str(path), "label": label, "mask_path": mask_path,
                             "image_id": f"{category}/{split}/{folder}/{i:03d}"})
    return pd.DataFrame(rows)


class StubProbe:
    """High anomaly log-odds when bright pixels are present; counts forward passes."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, image, class_name):
        return self.batch([image], class_name)[0]

    def batch(self, images, class_name):
        out = []
        for image in images:
            self.calls += 1
            arr = np.asarray(image, dtype=np.float32)
            logit = -2.0 + 220.0 * float((arr > 230).mean())
            out.append({"audit_p_yes": float(1 / (1 + np.exp(-logit))), "audit_logit": logit,
                        "probe_version": "stub-v1", "parse_ok": True})
        return out


@pytest.fixture
def v2cfg():
    base = CausalConfig(k_placebos=4, seed=11, candidate_stride=4, min_separation=4,
                        max_targets_per_folder=2, save_images=True, save_placebo_images=1,
                        dilate_for_donor=4, max_donor_candidates=7, model_id="stub")
    return V2Config(base=base, k_random=4, k_context=4, k_artifact=4,
                    artifact_max_candidates=300, artifact_match_quantile=0.3, k_occlusion=2,
                    k_sham=3, max_sham_sources=7, sham_match_quantile=0.6, batch_size=3)


def _square(h=64, w=64, y=20, x=24, s=10):
    m = np.zeros((h, w), bool)
    m[y:y + s, x:x + s] = True
    return m


def _textured(seed=0, h=64, w=64):
    rng = np.random.default_rng(seed)
    return np.clip(rng.normal(120, 25, (h, w, 3)), 0, 255).astype(np.uint8)


# ------------------------------------------------------------------ metrics
@pytest.mark.parametrize("dy,dx", [(0, 0), (10, -12), (-20, 30), (34, -24)])
def test_local_and_template_metrics_equal_full_image(dy, dx):
    original, donor = _textured(1), _textured(2)
    mask = _square()
    ys, xs = np.nonzero(mask)
    shifted = np.zeros_like(mask)
    shifted[ys + dy, xs + dx] = True
    ref = apply_intervention(original, donor, shifted, "placebo", 3)
    edit, seam = local_metrics(original, donor, ys, xs, dy, dx, 3)
    assert edit == pytest.approx(ref.edit_energy, abs=1e-9)
    assert seam == pytest.approx(ref.seam_score, abs=1e-6)
    tmpl = MaskTemplate(mask, 3)
    e2, s2 = tmpl.metrics(original, donor, dy, dx)
    assert e2 == pytest.approx(ref.edit_energy, abs=1e-9)
    assert s2 == pytest.approx(ref.seam_score, abs=1e-6)
    iv = tmpl.apply(original, donor, "placebo", dy, dx)
    assert np.array_equal(iv.image, ref.image)       # border placements included (dy=34 touches the edge)


def test_log_ratio_distance_and_robust_scale():
    assert log_ratio_distance(2.0, 1.0, eps=0) == pytest.approx(np.log(2))
    assert log_ratio_distance(0.5, 1.0, eps=0) == pytest.approx(np.log(2))   # symmetric
    assert robust_scale([1, 1, 1, 1]) == 1.0
    assert robust_scale([0, 1, 2, 3, 100]) == pytest.approx(1.4826 * 1.0)


# ------------------------------------------------------------------ artifact matching
def _artifact_plan(seed=5):
    original, donor = _textured(1), _textured(2)
    mask = _square()
    feasible, _ = feasible_candidates(original, mask, candidate_stride=3, max_candidates=400)
    defect = apply_intervention(original, donor, mask, "defect_removed", 3)
    return original, donor, mask, feasible, artifact_matched_placebos(
        original, donor, mask, feasible, defect.edit_energy, defect.seam_score, 6, seed,
        "img", 0.3, 4, 3)


def test_artifact_matching_is_deterministic_with_seed():
    *_, a = _artifact_plan(5)
    *_, b = _artifact_plan(5)
    *_, c = _artifact_plan(6)
    key = lambda p: [(x.dy, x.dx) for x in p.placebos]  # noqa: E731
    assert key(a) == key(b)
    assert key(a) != key(c)


def test_artifact_matching_never_sees_model_output():
    for fn in (artifact_matched_placebos, select_shams, build_target, random_placebos):
        params = set(inspect.signature(fn).parameters)
        assert not params & {"probe", "scores", "logits", "audit_logit", "cache"}, fn.__name__


def test_artifact_placebos_keep_area_and_iou():
    original, donor, mask, feasible, plan = _artifact_plan()
    assert plan.n_valid == 6
    ys, xs = np.nonzero(mask)
    for p in plan.placebos:
        pm = np.zeros_like(mask)
        pm[ys + p.dy, xs + p.dx] = True
        assert pm.sum() == mask.sum()
        inter = (pm & mask).sum()
        assert inter / (pm | mask).sum() <= 0.01
        assert p.iou_with_defect <= 0.01


def test_artifact_match_metrics_recorded_correctly():
    original, donor, mask, feasible, plan = _artifact_plan()
    defect = apply_intervention(original, donor, mask, "defect_removed", 3)
    ys, xs = np.nonzero(mask)
    for p in plan.placebos:
        pm = np.zeros_like(mask)
        pm[ys + p.dy, xs + p.dx] = True
        ref = apply_intervention(original, donor, pm, "placebo", 3)
        assert p.edit_energy == pytest.approx(ref.edit_energy, abs=1e-9)
        assert p.edit_ratio_to_defect == pytest.approx(
            (ref.edit_energy + 1e-3) / (defect.edit_energy + 1e-3), rel=1e-9)


def test_artifact_pool_improves_known_balance():
    """Donor differs strongly from the original only on the left; the defect sits there.

    A context-only placebo usually lands on the right (weak edit); the artifact arm
    must pick placements whose edit strength matches the defect's.
    """
    rng = np.random.default_rng(3)
    original = np.full((96, 96, 3), 120, np.uint8)
    original = np.clip(original + rng.normal(0, 4, original.shape), 0, 255).astype(np.uint8)
    donor = original.copy()
    donor[:, :30] = np.clip(donor[:, :30].astype(int) + 70, 0, 255).astype(np.uint8)
    mask = _square(96, 96, 40, 8, 10)
    feasible, _ = feasible_candidates(original, mask, candidate_stride=3, max_candidates=900)
    defect = apply_intervention(original, donor, mask, "defect_removed", 3)
    art = artifact_matched_placebos(original, donor, mask, feasible, defect.edit_energy,
                                    defect.seam_score, 8, 1, "x", 0.3, 4, 3)
    ctx = generate_placebos(original, mask, 8, 1, "x", candidate_stride=3, min_separation=4,
                            max_candidates=900)
    ys, xs = np.nonzero(mask)

    def imbalance(placebos):
        vals = []
        for p in placebos:
            e, _ = local_metrics(original, donor, ys, xs, p.dy, p.dx, 3)
            vals.append(abs(np.log((e + 1e-3) / (defect.edit_energy + 1e-3))))
        return float(np.median(vals))
    assert imbalance(art.placebos) < imbalance(ctx.placebos)


def test_random_arm_is_uniform_and_separated():
    original, donor, mask, feasible, _ = _artifact_plan()
    plan = random_placebos(feasible, 6, 3, "img", int(mask.sum()), min_separation=4)
    assert plan.n_valid == 6
    offs = [(p.dy, p.dx) for p in plan.placebos]
    for i, a in enumerate(offs):
        for b in offs[i + 1:]:
            assert max(abs(a[0] - b[0]), abs(a[1] - b[1])) >= 4


def test_occlusion_fill_is_constant_mean():
    img = _textured(4)
    fill = occlusion_fill(img)
    assert fill.shape == img.shape
    assert (fill == fill[0, 0]).all()


# ------------------------------------------------------------------ sufficiency
def test_transplant_changes_nothing_outside_mask():
    recipient, source = _textured(7), _textured(8)
    mask = _square()
    iv = MaskTemplate(mask, 3).apply(recipient, source, "defect_inserted")
    assert np.array_equal(iv.image[~mask], recipient[~mask])
    assert iv.outside_change_max == 0


def test_sham_sources_are_train_good_only(synthetic_dataset):
    bank = DonorBank(synthetic_dataset)
    srcs = SourceBank(bank, 50).sources("bottle", (80, 80))
    good = synthetic_dataset[(synthetic_dataset.split == "train") & (synthetic_dataset.label == 0)]
    assert {s[0] for s in srcs} <= set(good.image_id)
    bad = synthetic_dataset.copy()
    bad.loc[bad.index[0], "label"] = 1            # an anomaly sneaks into train
    bad.loc[bad.index[0], "split"] = "train"
    with pytest.raises(ValueError):
        DonorBank(bad)


def test_same_recipient_for_real_and_sham_insertion(synthetic_dataset, v2cfg):
    bank = DonorBank(synthetic_dataset)
    row = select_targets(synthetic_dataset, v2cfg.base).iloc[0]
    built = build_target(row, bank, SourceBank(bank, 7), v2cfg)
    mask = next(it.iv.mask for it in built.items if it.role == "defect_inserted")
    recipient = next(it.iv.image for it in built.items if it.role == "recipient")
    inserts = [it for it in built.items if it.role in ("defect_inserted", "sham")]
    assert len(inserts) == 1 + v2cfg.k_sham
    for it in inserts:
        assert np.array_equal(it.iv.mask, mask)                  # same location M
        assert np.array_equal(it.iv.image[~mask], recipient[~mask])
    assert built.donor["donor_image_id"] not in {it.extra.get("sham_source_image_id")
                                                 for it in inserts}


# ------------------------------------------------------------------ effects
def test_cee_n_and_cee_s_formulas(synthetic_dataset, v2cfg):
    bank = DonorBank(synthetic_dataset)
    row = select_targets(synthetic_dataset, v2cfg.base).iloc[0]
    built = build_target(row, bank, SourceBank(bank, 7), v2cfg)
    rec, rows = score_target(built, as_batch_probe(StubProbe()), None, 3)
    z = {i: r["audit_logit"] for i, r in enumerate(rows)}
    roles = [(r["role"], r["arm"]) for r in rows]
    z0 = z[roles.index(("original", "shared"))]
    zd = z[roles.index(("defect_removed", "necessity"))]
    art = [z0 - z[i] for i, ra in enumerate(roles) if ra == ("placebo", "artifact")]
    assert rec["delta_defect"] == pytest.approx(z0 - zd)
    assert rec["cee_n_artifact"] == pytest.approx((z0 - zd) - np.median(art))
    zj = z[roles.index(("recipient", "sufficiency"))]
    zi = z[roles.index(("defect_inserted", "sufficiency"))]
    sh = [z[i] - zj for i, ra in enumerate(roles) if ra[0] == "sham"]
    assert rec["delta_insert_defect"] == pytest.approx(zi - zj)
    assert rec["cee_s"] == pytest.approx((zi - zj) - np.median(sh))
    assert rec["empirical_sham_tail"] == pytest.approx(
        (1 + sum(s >= zi - zj for s in sh)) / (len(sh) + 1))
    # The stub responds to the bright defect: removal lowers z, insertion raises it.
    assert rec["cee_n_artifact"] > 0 and rec["cee_s"] > 0
    assert rec["quadrant"] == "bidirectional"
    assert rec["max_outside_change"] == 0
    assert rec["fast_path_drift"] == pytest.approx(0, abs=1e-6)


def test_quadrant_labels():
    assert quadrant(1, 1) == "bidirectional"
    assert quadrant(1, -1) == "necessity_only"
    assert quadrant(-1, 1) == "sufficiency_only"
    assert quadrant(0, 0) == "neither"
    assert quadrant(float("nan"), 1) == "undetermined"


# ------------------------------------------------------------------ scoring
def test_identical_images_across_arms_are_scored_once(tmp_path):
    probe = StubProbe()
    a, b = _textured(1), _textured(2)
    cache = ScoreCache(tmp_path / "c.jsonl", "sig")
    scores, digests, hit = score_images([a, b, a.copy(), b], "x", as_batch_probe(probe),
                                        cache, batch_size=2)
    assert probe.calls == 2
    assert hit == [False, False, True, True]
    assert scores[0] == scores[2]
    again, _, hit2 = score_images([a, b], "x", as_batch_probe(probe),
                                  ScoreCache(tmp_path / "c.jsonl", "sig"), 2)
    assert probe.calls == 2 and all(hit2)              # resumed from the cache file
    assert array_digest(a) != array_digest(b)


# ------------------------------------------------------------------ end to end
def test_run_audit_v2_resume_and_report(synthetic_dataset, v2cfg, tmp_path):
    from tzsad.causal.report_v2 import build_report

    probe = StubProbe()
    run_dir = tmp_path / "run"
    run_audit_v2(synthetic_dataset, probe.batch, v2cfg, run_dir)
    first = probe.calls
    assert first > 0
    run_audit_v2(synthetic_dataset, probe.batch, v2cfg, run_dir)
    assert probe.calls == first                         # everything resumed
    records = merge_shards(run_dir, "causal_records_v2")
    targets = select_targets(synthetic_dataset, v2cfg.base)
    assert set(records.image_id) == set(targets.image_id)
    assert (records.status == "ok").all()
    for col in ("cee_n_random", "cee_n_context", "cee_n_artifact", "cee_n_occlusion", "cee_s",
                "artifact_artifact_imbalance", "context_artifact_imbalance", "quadrant"):
        assert col in records
    assert "anomaly_score" not in records

    cfg = OmegaConf.create({"seed": 11, "cerm_v2": {"structured_p1_cases": 3}})
    build_report(run_dir, cfg, v2cfg, targets, tmp_path)
    for name in ("summary_necessity_overall.csv", "summary_sufficiency_overall.csv",
                 "summary_bidirectional.csv", "artifact_balance.csv", "method_ablation.csv",
                 "casebook_manifest.csv", "PRESENTATION_RESULTS.md",
                 "CERM_V2_PRESENTATION_BUNDLE.zip", "figures/bidirectional_quadrant.png",
                 "figures/artifact_balance_before_after.png", "public/PRESENTATION_RESULTS.md"):
        assert (run_dir / name).exists(), name
    # The public directory must never carry dataset pixels.
    pub_figs = {p.name for p in (run_dir / "public" / "figures").iterdir()}
    assert not any(n.startswith(("example_", "case_")) for n in pub_figs)
    text = (run_dir / "PRESENTATION_RESULTS.md").read_text()
    assert "first ever" not in text.lower()


def test_casebook_selection_is_deterministic(synthetic_dataset, v2cfg, tmp_path):
    probe = StubProbe()
    run_audit_v2(synthetic_dataset, probe.batch, v2cfg, tmp_path / "r")
    records = merge_shards(tmp_path / "r", "causal_records_v2")
    a = select_cases(records, 5)
    b = select_cases(records.sample(frac=1, random_state=3), 5)
    assert a.reset_index(drop=True).equals(b.reset_index(drop=True))
    assert a["image_id"].is_unique


def test_casebook_figures_survive_missing_examples(tmp_path):
    records = pd.DataFrame([{"image_id": "x/test/a/000", "category": "x", "path": "/nope.png",
                             "mask_path": "/nope_mask.png", "status": "ok"}])
    cases = pd.DataFrame([{"slot": "best_bidirectional", "image_id": "x/test/a/000",
                           "category": "x", "rule": "", "criterion_value": 1.0},
                          {"slot": "category_strong", "image_id": "missing", "category": "y",
                           "rule": "", "criterion_value": 0.0}])
    made = build_casebook(records, pd.DataFrame(columns=["image_id"]), cases, tmp_path)
    assert made == []
    assert select_cases(records.assign(cee_n_artifact=np.nan, cee_s=np.nan), 3).empty
