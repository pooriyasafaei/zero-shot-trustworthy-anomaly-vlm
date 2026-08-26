"""The qualitative TP/TN/FP/FN viewer ported from the prototype notebook."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tzsad.data.mvtec import build_index
from tzsad.viz.panels import display_predictions, prediction_type, wrap_caption


@pytest.fixture
def scored(fake_mvtec):
    """A small scored frame over real image files, with every diagnostic column."""
    idx = build_index(fake_mvtec, ["bottle"], ("test",))
    rng = np.random.default_rng(0)
    df = idx.copy()
    df["anomaly_score"] = np.clip(0.5 + 0.3 * (df.label * 2 - 1) + rng.normal(0, 0.1, len(df)), 0.01, 0.99)
    df["raw_score"] = df["anomaly_score"] - 0.5
    df["parse_ok"] = True
    df["conformal_p"] = 1.0 - df["anomaly_score"]
    df["conformal_pred"] = (df["anomaly_score"] > 0.5).astype(int)
    df["u_conformal"] = 1 - 2 * (df["conformal_p"] - 0.5).abs()
    df["u_manifold_knn"] = rng.random(len(df))
    df["halluc"] = False
    df["halluc_case"] = "none"
    df["gt_iou"] = 0.4
    df["observation"] = "The surface shows a visible scratch near the centre of the object."
    df["predicted_defect"] = "scratch"
    df["predicted_location"] = "centre"
    df["prediction_type"] = prediction_type(df)
    return df


def test_prediction_type_uses_the_calibrated_prediction(scored):
    assert set(scored.prediction_type) <= {"TP", "TN", "FP", "FN"}
    tp = scored[scored.prediction_type == "TP"]
    assert (tp.label == 1).all() and (tp.conformal_pred == 1).all()


def test_prediction_type_refuses_a_missing_prediction_column(scored):
    """Refuses to fall back to a median split - that was the prototype's defect #3."""
    with pytest.raises(KeyError, match="conformal calibration"):
        prediction_type(scored.drop(columns=["conformal_pred"]))


def test_panel_figure_is_written(scored, tmp_path):
    maps = {i: np.random.default_rng(1).random((32, 32)) for i in scored.image_id}
    out = display_predictions(scored, maps, n=3, category="bottle",
                              save_path=tmp_path / "panel")
    assert out and all(p.exists() for p in out)


def test_panel_handles_no_maps_and_no_matches(scored, tmp_path):
    assert display_predictions(scored, None, n=2, save_path=tmp_path / "nomaps")
    assert display_predictions(scored, None, filter_query="category == 'nothing'") is None


def test_panel_renders_abstentions_and_missing_signals(scored, tmp_path):
    """A parse failure must be visible in the panel, not hidden."""
    df = scored.copy()
    df.loc[df.index[0], "parse_ok"] = False
    df.loc[df.index[0], "anomaly_score"] = float("nan")
    out = display_predictions(df, None, n=2, save_path=tmp_path / "abstain")
    assert out and all(p.exists() for p in out)


def test_wrap_caption_truncates():
    long = " ".join(["word"] * 200)
    wrapped = wrap_caption(long, width=20, max_lines=3)
    assert wrapped.count("\n") == 2
    assert wrapped.endswith("...")
    assert wrap_caption(None) == ""
