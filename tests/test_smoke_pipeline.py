"""End-to-end smoke test: the whole CLIP pipeline on 2 categories x a few images.

Runs with a stubbed backbone so it needs no weights and no GPU, but exercises the
real index -> cache -> score -> conformal -> uncertainty -> report path, including
the parquet round-trip and the figure code.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from tzsad import pipeline as P
from tzsad.eval.report import build_report
from tzsad.records import uncertainty_columns


class FakeEmbedder:
    """Deterministic stand-in for :class:`ClipEmbedder`.

    Embeddings are a hash of the path plus a planted "defect" direction for
    non-good images, so the pipeline sees a signal without loading CLIP.
    """

    def __init__(self, spec, device=None, cache_dir=None, allow_download=True, batch_size=32):
        self.spec = spec
        self.dim = 32

    def load(self):
        return None

    @property
    def logit_scale(self):
        return 100.0

    def _vec(self, key: str) -> np.ndarray:
        h = hashlib.sha256(key.encode()).digest()
        v = np.frombuffer(h * ((self.dim // 32) + 1), dtype=np.uint8)[: self.dim].astype(np.float32)
        return (v / 255.0) - 0.5

    def encode_text(self, prompts):
        out = np.stack([self._vec("T" + p) for p in prompts])
        anomaly_like = np.array([any(w in p for w in ("damaged", "defect", "flawed", "anomalous",
                                                     "broken", "failed", "fault", "bad", "scratch"))
                                 for p in prompts], dtype=np.float32)
        out[:, 0] = anomaly_like * 2.0 - 1.0
        return out / np.linalg.norm(out, axis=1, keepdims=True)

    def encode_images(self, paths, transform=None):
        out = np.stack([self._vec("I" + str(p)) for p in paths])
        defect = np.array([0.0 if "/good/" in str(p) else 1.0 for p in paths], dtype=np.float32)
        out[:, 0] = defect * 1.5 - 0.75
        return out / np.linalg.norm(out, axis=1, keepdims=True)

    def encode_windows(self, paths, win, transform=None):
        boxes = win.windows()
        out = np.zeros((len(paths), len(boxes), self.dim), dtype=np.float16)
        for i, p in enumerate(paths):
            defect = 0.0 if "/good/" in str(p) else 1.0
            for j, _ in enumerate(boxes):
                v = self._vec(f"W{p}{j}")
                v[0] = defect * (1.0 if j == 0 else 0.1)     # defect concentrated in window 0
                out[i, j] = (v / np.linalg.norm(v)).astype(np.float16)
        return out


@pytest.fixture
def smoke_cfg(fake_mvtec, tmp_path, monkeypatch):
    monkeypatch.setattr(P, "ClipEmbedder", FakeEmbedder)
    return OmegaConf.create({
        "seed": 7,
        "data": {"root": str(fake_mvtec), "categories": ["bottle", "carpet"],
                 "subset": "n_per_folder=4", "splits": ["train", "test"]},
        "paths": {"results": str(tmp_path / "results"), "cache": str(tmp_path / "cache"),
                  "run_name": "smoke"},
        "clip": {"backbones": [{"name": "Fake-B", "pretrained": "test"}], "prompt_set": "base",
                 "tau": 0.05, "aggregate": "mean_embedding", "batch_size": 8,
                 "allow_download": False,
                 "windows": {"enabled": True, "scales": [0.5], "stride_frac": 0.5,
                             "image_size": 64, "map_size": 32, "bank_k": 2,
                             "position_aware": True}},
        "conformal": {"delta": 0.1, "deltas": [0.05, 0.1, 0.2], "uncertainty_mode": "symmetric",
                      "randomized": False, "n_cal": None, "n_cal_sweep": [5, 10, None]},
        "uncertainty": {"estimators": ["prompt_std", "prompt_std_z", "manifold_knn"],
                        "manifold_k": 3, "tta": {"enabled": False, "views": []},
                        "semantic_entropy": {"backend": "embedding", "embed_threshold": 0.85,
                                             "entail_threshold": 0.5}},
        "eval": {"n_boot": 50, "n_bins": 5},
    })


def test_pipeline_runs_end_to_end(smoke_cfg, tmp_path):
    index = P.stage_index(smoke_cfg)
    assert len(index) > 0

    embedders = P.stage_embed(smoke_cfg, index)
    per_backbone = P.stage_score_clip(smoke_cfg, index, embedders)
    records = per_backbone["Fake-B__test"]
    assert records.anomaly_score.between(0, 1).all()

    path = P.save_stage(records, P.run_dir_for(smoke_cfg) / "records.parquet")
    assert P.load_stage(path).shape == records.shape

    test, calib, coverage = P.stage_calibrate(smoke_cfg, records)
    assert {"conformal_p", "conformal_pred", "u_conformal", "correct"} <= set(test.columns)
    assert test.conformal_p.between(0, 1).all()
    # Unlike a median threshold, the positive rate is set by the calibrated
    # threshold, not forced to 50%: normals are flagged at close to delta.
    normals = test[test.label == 0]
    assert normals.conformal_pred.mean() <= 0.5
    assert coverage.iloc[-1]["empirical_coverage"] >= 0.0

    test = P.stage_uncertainty_clip(smoke_cfg, test, per_backbone, None)
    signals = uncertainty_columns(test)
    assert {"u_conformal", "u_prompt_std", "u_prompt_std_z", "u_manifold_knn"} <= set(signals)

    report_dir = P.run_dir_for(smoke_cfg) / "report"
    paths = build_report(test, report_dir, "correct", n_boot=50, seed=7)
    assert set(paths) >= {"detection", "calibration", "abstention", "error_prediction", "selective"}

    det = pd.read_csv(paths["detection"])
    pooled = det[det.group == "POOLED"].iloc[0]
    assert 0.0 <= pooled.auroc <= 1.0
    assert pooled.auroc_ci_lo <= pooled.auroc <= pooled.auroc_ci_hi

    err = pd.read_csv(paths["error_prediction"])
    assert set(err.columns) >= {"signal", "err_auroc", "ci_lo", "ci_hi", "informative", "misleading"}


def test_heatmaps_and_hallucination_run(smoke_cfg):
    from tzsad.hallucination.cmcs import CMCSConfig, compute_cmcs
    from tzsad.hallucination.rate import HRConfig, hallucination_flags, hallucination_rate

    index = P.stage_index(smoke_cfg)
    embedders = P.stage_embed(smoke_cfg, index)
    per_backbone, text = P.stage_score_clip(smoke_cfg, index, embedders, return_text=True)
    all_maps = P.stage_heatmaps(smoke_cfg, index, embedders, text_embeddings=text)
    assert set(all_maps) == {"bank", "text"}
    assert all_maps["bank"] and all_maps["text"]
    maps = all_maps["bank"]
    assert next(iter(maps.values())).shape == (32, 32)

    # Pixel metrics run off the same cached maps - no extra forward pass.
    pixel = P.stage_pixel_eval(smoke_cfg, index, all_maps)
    assert set(pixel["map"]) == {"bank", "text"}
    assert (pixel.query("category == 'POOLED'").pixel_auroc.between(0, 1)).all()

    test, _, _ = P.stage_calibrate(smoke_cfg, per_backbone["Fake-B__test"])
    test["predicted_location"] = "upper left"
    scored = compute_cmcs(test, maps, CMCSConfig(map_size=32), arm="language_grounding")
    assert scored.cmcs.between(0, 1).all()
    assert "claim_kind" in scored

    flagged = hallucination_flags(scored, maps, HRConfig(map_size=32))
    hr = hallucination_rate(flagged)
    assert hr.iloc[-1]["group"] == "POOLED"
    assert np.isnan(hr.iloc[-1]["hr"]) or 0.0 <= hr.iloc[-1]["hr"] <= 1.0


def test_figures_are_written(smoke_cfg, tmp_path):
    from tzsad.viz.plots import plot_error_prediction, plot_reliability, plot_risk_coverage

    index = P.stage_index(smoke_cfg)
    embedders = P.stage_embed(smoke_cfg, index)
    per_backbone = P.stage_score_clip(smoke_cfg, index, embedders)
    test, _, _ = P.stage_calibrate(smoke_cfg, per_backbone["Fake-B__test"])
    test = P.stage_uncertainty_clip(smoke_cfg, test, per_backbone, None)

    fig_dir = tmp_path / "figs"
    out = plot_risk_coverage(test, "correct", uncertainty_columns(test), fig_dir / "rc")
    assert all(p.exists() for p in out)
    assert plot_reliability(test, fig_dir / "rel")[0].exists()

    from tzsad.eval.report import error_prediction_table

    tbl = error_prediction_table(test, "correct", n_boot=50)
    assert plot_error_prediction(tbl, fig_dir / "err")[0].exists()
