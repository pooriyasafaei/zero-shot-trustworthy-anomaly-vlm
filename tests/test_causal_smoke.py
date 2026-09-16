"""End-to-end CERM-AD Stage I on synthetic images, with the VLM probe stubbed.

Exercises the real audit loop - donor selection, placebo matching, reconstruction,
scoring, caching, checkpointing, sharding, aggregation and reporting - without
needing Qwen weights or a GPU.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from tzsad.causal.audit import (CausalConfig, ScoreCache, assign_shards, audit_target,
                                merge_shards, run_audit, select_targets)
from tzsad.causal.donor import DonorBank
from tzsad.causal.presentation import write_presentation_summary
from tzsad.causal.report import (artifact_control_table, build_figures, decision_split_table,
                                 summary_tables)


@pytest.fixture
def synthetic_dataset(tmp_path):
    """A miniature MVTec-shaped dataset: bright square defects on textured plates."""
    rng = np.random.default_rng(0)
    rows = []
    for category, base in (("bottle", 120), ("carpet", 90)):
        for split, folder, n, label in (("train", "good", 6, 0),
                                        ("test", "scratch", 3, 1),
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
    """Deterministic stand-in for the VLM.

    Reports high anomaly log-odds when bright pixels are present, so removing the
    planted defect genuinely lowers the score and a placebo elsewhere does not.
    That makes the expected sign of CEE known in advance.
    """

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, image: Image.Image, class_name: str) -> dict:
        self.calls += 1
        arr = np.asarray(image, dtype=np.float32)
        brightness = float((arr > 230).mean())
        logit = -2.0 + 220.0 * brightness
        return {"audit_p_yes": float(1 / (1 + np.exp(-logit))), "audit_logit": logit,
                "probe_version": "stub-v1", "parse_ok": True}


@pytest.fixture
def config():
    return CausalConfig(k_placebos=4, seed=11, candidate_stride=6, min_separation=6,
                        max_targets_per_folder=2, save_images=True, save_placebo_images=1,
                        dilate_for_donor=4, max_donor_candidates=6, model_id="stub")


# ------------------------------------------------------------------ audit
def test_only_anomalous_masked_images_are_targets(synthetic_dataset, config):
    targets = select_targets(synthetic_dataset, config)
    assert len(targets) > 0
    assert (targets["label"] == 1).all()
    assert (targets["split"] == "test").all()
    assert targets["mask_path"].str.len().gt(0).all()
    # Capped per (category, defect_type).
    assert (targets.groupby(["category", "defect_type"]).size() <= 2).all()


def test_audit_target_produces_expected_interventions(synthetic_dataset, config, tmp_path):
    bank = DonorBank(synthetic_dataset)
    probe = StubProbe()
    row = select_targets(synthetic_dataset, config).iloc[0]
    result = audit_target(row, bank, probe, config, images_dir=tmp_path / "img")

    kinds = [r["intervention_type"] for r in result.interventions]
    assert kinds[0] == "original" and kinds[1] == "defect_removed"
    assert kinds.count("placebo") == result.record["n_valid_placebos"]
    assert len(result.interventions) == 2 + result.record["n_valid_placebos"]
    assert probe.calls == len(result.interventions)
    # No intervention may touch a pixel outside its own mask.
    assert max(r["outside_change_max"] for r in result.interventions) == 0
    # The same donor is used throughout, so donor identity cannot confound.
    assert len({r["donor_image_id"] for r in result.interventions}) == 1


def test_removing_a_real_defect_beats_placebos_on_a_known_stub(synthetic_dataset, config, tmp_path):
    """With a probe that reacts to the planted defect, CEE must come out positive."""
    bank = DonorBank(synthetic_dataset)
    row = select_targets(synthetic_dataset, config).iloc[0]
    rec = audit_target(row, bank, StubProbe(), config, images_dir=tmp_path / "img").record
    assert rec["status"] == "ok"
    assert rec["delta_defect"] > 0
    assert rec["cee"] > 0
    assert rec["defect_beats_all_placebos"] is True
    assert 0 < rec["empirical_placebo_p"] <= 1


def test_gt_mask_never_reaches_the_probe(synthetic_dataset, config, tmp_path):
    """The oracle mask may shape the intervention and nothing else."""
    seen = []

    def spy(image, class_name):
        seen.append((np.asarray(image).shape, class_name))
        return {"audit_p_yes": 0.5, "audit_logit": 0.0, "probe_version": "spy", "parse_ok": True}

    bank = DonorBank(synthetic_dataset)
    row = select_targets(synthetic_dataset, config).iloc[0]
    audit_target(row, bank, spy, config, images_dir=tmp_path / "img")
    # The probe receives 3-channel images and a bare class name - no mask channel,
    # no mask path, nothing derived from the ground truth.
    assert all(shape[2] == 3 for shape, _ in seen)
    assert {name for _, name in seen} == {"bottle"}


# ------------------------------------------------------------------ cache
def test_score_cache_round_trip_and_reuse(tmp_path):
    cache = ScoreCache(tmp_path / "score_cache.jsonl", "sig123", "direct_verdict")
    img = np.zeros((8, 8, 3), np.uint8)
    key = cache.key(ScoreCache.image_digest(img), "bottle")
    assert cache.get(key) is None and cache.misses == 1
    cache.put(key, {"audit_logit": 1.5})

    reopened = ScoreCache(tmp_path / "score_cache.jsonl", "sig123", "direct_verdict")
    assert reopened.get(key)["audit_logit"] == 1.5
    assert reopened.hits == 1


def test_cache_key_changes_with_signature_and_content(tmp_path):
    """A cached score must never be reused under different settings or pixels."""
    a = ScoreCache(tmp_path / "a.jsonl", "sigA", "direct_verdict")
    b = ScoreCache(tmp_path / "b.jsonl", "sigB", "direct_verdict")
    img = np.zeros((8, 8, 3), np.uint8)
    other = img.copy()
    other[0, 0] = 255
    assert a.key(a.image_digest(img), "bottle") != b.key(b.image_digest(img), "bottle")
    assert a.key(a.image_digest(img), "bottle") != a.key(a.image_digest(other), "bottle")
    assert a.key(a.image_digest(img), "bottle") != a.key(a.image_digest(img), "carpet")


def test_cache_survives_a_torn_final_line(tmp_path):
    path = tmp_path / "score_cache.jsonl"
    cache = ScoreCache(path, "sig", "m")
    key = cache.key("d0", "bottle")
    cache.put(key, {"audit_logit": 2.0})
    with path.open("a") as fh:
        fh.write('{"key": "torn", "sco')
    assert ScoreCache(path, "sig", "m").get(key)["audit_logit"] == 2.0


def test_config_signature_tracks_scoring_relevant_fields():
    base = CausalConfig()
    assert base.signature() == CausalConfig().signature()
    assert base.signature() != CausalConfig(k_placebos=31).signature()
    assert base.signature() != CausalConfig(feather_px=9).signature()
    assert base.signature() != CausalConfig(model_id="other").signature()
    # Purely cosmetic settings must not invalidate an expensive cache.
    assert base.signature() == CausalConfig(save_images=not base.save_images).signature()


# --------------------------------------------------------- run and resume
def test_run_audit_resumes_after_interruption(synthetic_dataset, config, tmp_path):
    probe = StubProbe()
    records, interventions = run_audit(synthetic_dataset, probe, config, tmp_path)
    assert len(records) == len(select_targets(synthetic_dataset, config))
    first_calls = probe.calls
    assert first_calls > 0

    # A second run must reuse everything and score nothing new.
    probe2 = StubProbe()
    records2, _ = run_audit(synthetic_dataset, probe2, config, tmp_path)
    assert probe2.calls == 0
    assert len(records2) == len(records)


def test_partial_resume_only_scores_what_is_missing(synthetic_dataset, config, tmp_path):
    targets = select_targets(synthetic_dataset, config)
    bank = DonorBank(synthetic_dataset)
    cache = ScoreCache(tmp_path / "score_cache.jsonl", config.signature(), config.score_mode)
    from tzsad.utils.checkpoint import RecordCheckpoint

    ckpt = RecordCheckpoint(tmp_path / "causal_records.jsonl")
    first = audit_target(targets.iloc[0], bank, StubProbe(), config, cache)
    ckpt.append(first.record)
    ckpt.flush()

    probe = StubProbe()
    records, _ = run_audit(synthetic_dataset, probe, config, tmp_path)
    assert len(records) == len(targets)
    # The already-done target contributed no new probe calls.
    assert probe.calls < len(targets) * (2 + config.k_placebos)


def test_sharding_is_balanced_and_deterministic(synthetic_dataset, config, tmp_path):
    """An unbalanced split leaves a GPU idle; the assignment must stay even."""
    targets = select_targets(synthetic_dataset, config)
    ids = targets["image_id"].tolist()
    a = assign_shards(ids, 2)
    assert np.array_equal(a, assign_shards(ids, 2))
    counts = np.bincount(a, minlength=2)
    assert counts.max() - counts.min() <= 1
    assert (assign_shards(ids, 1) == 0).all()

    for shard in (0, 1):
        run_audit(synthetic_dataset, StubProbe(), config, tmp_path,
                  shard=shard, num_shards=2)
    merged = merge_shards(tmp_path, "causal_records")
    assert len(merged) == len(targets)
    assert merged["image_id"].is_unique


def test_resume_respects_work_done_by_another_worker(synthetic_dataset, config, tmp_path):
    """Resharding must never cause a target to be audited twice."""
    run_audit(synthetic_dataset, StubProbe(), config, tmp_path, shard=0, num_shards=2)
    probe = StubProbe()
    # Same worker id, but now a single shard: it would otherwise redo shard 1's work
    # and re-audit shard 0's. Only the genuinely missing targets should be scored.
    run_audit(synthetic_dataset, probe, config, tmp_path, shard=0, num_shards=1)
    merged = merge_shards(tmp_path, "causal_records")
    assert merged["image_id"].is_unique
    assert len(merged) == len(select_targets(synthetic_dataset, config))


def test_failed_target_is_recorded_not_dropped(synthetic_dataset, config, tmp_path):
    def exploding(image, class_name):
        raise RuntimeError("probe exploded")

    records, _ = run_audit(synthetic_dataset, exploding, config, tmp_path)
    assert len(records) == len(select_targets(synthetic_dataset, config))
    assert records["status"].str.startswith("failed").all()


# ------------------------------------------------------- outputs / report
def test_output_schema_has_the_documented_columns(synthetic_dataset, config, tmp_path):
    records, interventions = run_audit(synthetic_dataset, StubProbe(), config, tmp_path)
    required_record = {
        "image_id", "category", "defect_type",
        "audit_p_yes_original", "audit_logit_original",
        "audit_p_yes_defect_removed", "audit_logit_defect_removed",
        "delta_defect", "n_valid_placebos", "placebo_delta_mean", "placebo_delta_median",
        "placebo_delta_mad", "placebo_delta_max", "cee", "robust_cee_z",
        "empirical_placebo_p", "defect_placebo_percentile", "defect_beats_all_placebos",
        "supported_at_005", "original_direct_prediction", "original_is_correct_for_anomaly",
        "donor_image_id", "donor_distance", "seam_ratio", "mask_source", "score_mode",
    }
    assert required_record <= set(records.columns), required_record - set(records.columns)

    required_iv = {
        "image_id", "category", "defect_type", "intervention_type", "placebo_index",
        "mask_area", "mask_fraction", "dx", "dy", "context_distance", "donor_image_id",
        "donor_distance", "seam_score", "edit_energy", "outside_change_max",
        "audit_p_yes", "audit_logit", "cache_hit", "image_output_path",
    }
    assert required_iv <= set(interventions.columns), required_iv - set(interventions.columns)
    # CEE must not be smuggled into the repository's standard score field.
    assert "anomaly_score" not in records.columns


def test_intervention_images_are_written(synthetic_dataset, config, tmp_path):
    run_audit(synthetic_dataset, StubProbe(), config, tmp_path)
    saved = list((tmp_path / "intervention_images").glob("*.png"))
    assert saved
    assert any("defect_removed" in p.name for p in saved)
    assert any("placebo" in p.name for p in saved)


def test_summary_and_figures_build(synthetic_dataset, config, tmp_path):
    records, interventions = run_audit(synthetic_dataset, StubProbe(), config, tmp_path)
    overall, by_cat = summary_tables(records)
    assert overall.iloc[0]["n_targets"] == len(records)
    assert set(by_cat["group"]) == set(records["category"].unique())
    assert 0.0 <= overall.iloc[0]["frac_cee_positive"] <= 1.0

    decision = decision_split_table(records)
    assert "predicted_YES" in set(decision["group"])
    artifacts = artifact_control_table(interventions, records)
    assert {"defect_removed", "placebo"} <= set(artifacts["intervention_type"])
    assert float(artifacts[artifacts.intervention_type == "placebo"]["outside_change_max"].iloc[0]) == 0

    made = build_figures(records, interventions, tmp_path / "figures")
    assert "defect_vs_placebo" in made and "cee_by_category" in made
    assert (tmp_path / "figures" / "defect_vs_placebo.png").exists()


def test_presentation_summary_states_its_limitations(synthetic_dataset, config, tmp_path):
    records, interventions = run_audit(synthetic_dataset, StubProbe(), config, tmp_path)
    overall, by_cat = summary_tables(records)
    out = write_presentation_summary(
        tmp_path / "PRESENTATION_SUMMARY.md", overall, by_cat,
        decision_split_table(records), artifact_control_table(interventions, records),
        {"git_sha": "abc123", "model_id": "stub", "score_mode": "direct_verdict",
         "mask_source": "gt", "categories": "bottle", "k_placebos": 4, "seed": 11})
    text = out.read_text()
    assert "abc123" in text
    assert "oracle" in text.lower()
    assert "matched-placebo tail probability" in text
    assert "Preliminary findings" in text and "Important limitations" in text


def test_low_placebo_count_blocks_the_005_claim(synthetic_dataset, tmp_path):
    """K=4 cannot resolve alpha=0.05, so the flag must be NA rather than False."""
    config = CausalConfig(k_placebos=4, seed=3, candidate_stride=8, min_separation=6,
                          max_targets_per_folder=1, save_images=False, dilate_for_donor=4,
                          max_donor_candidates=6)
    records, _ = run_audit(synthetic_dataset, StubProbe(), config, tmp_path)
    ok = records[records.status == "ok"]
    assert (ok["n_valid_placebos"] < 19).all()
    assert ok["supported_at_005"].isna().all()
    assert (ok["tail_resolution"] > 0.05).all()
