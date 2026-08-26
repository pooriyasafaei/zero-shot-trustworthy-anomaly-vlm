"""Qualitative TP/TN/FP/FN panel viewer, ported from the prototype notebook and extended.

Beyond the notebook's image | GT mask | overlay | text layout, each row now also
shows the anomaly heatmap, the conformal p-value, every uncertainty signal, the
hallucination flag, and - for the VLM - the observation text and the claimed
defect location.
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from PIL import Image

from ..data.mvtec import mask_path_for
from ..records import uncertainty_columns
from .style import save_figure, use_paper_style


def prediction_type(records: pd.DataFrame, prediction_col: str = "conformal_pred") -> pd.Series:
    """TP/TN/FP/FN labels from a *calibrated* prediction column.

    Deliberately takes the conformal prediction, not a median split: the
    prototype's median threshold forced a 50% positive rate and every TP/FP count
    derived from it was an artefact of the threshold, not of the model.
    """
    if prediction_col not in records:
        raise KeyError(
            f"{prediction_col!r} not found. Run conformal calibration first - "
            "TP/FP counts from a median threshold are not meaningful."
        )
    label = records["label"].to_numpy()
    pred = records[prediction_col].to_numpy()
    return pd.Series(np.select(
        [(label == 1) & (pred == 1), (label == 0) & (pred == 0),
         (label == 1) & (pred == 0), (label == 0) & (pred == 1)],
        ["TP", "TN", "FN", "FP"], default="Unknown"), index=records.index)


def wrap_caption(text: str, width: int = 40, max_lines: int = 7) -> str:
    """Wrap free text for the info panel."""
    lines = textwrap.wrap(str(text or ""), width=width)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1].rstrip() + "..."
    return "\n".join(lines)


def display_predictions(records: pd.DataFrame, maps: dict[str, np.ndarray] | None = None,
                        n: int = 6, category: str | None = None, defect_type: str | None = None,
                        filter_query: str | None = None, sort_by: str | None = None,
                        ascending: bool = True, row_height: float = 2.6,
                        save_path: str | Path | None = None):
    """Render up to ``n`` matching rows as a 5-panel-per-row figure.

    Panels: image | GT mask | heatmap overlay | anomaly heatmap | diagnostics text.
    """
    use_paper_style(base_font=7.0)
    data = records.copy()
    if category is not None:
        data = data[data["category"] == category]
    if defect_type is not None:
        data = data[data["defect_type"] == defect_type]
    if filter_query:
        data = data.query(filter_query)
    if sort_by:
        data = data.sort_values(sort_by, ascending=ascending)
    data = data.head(n)
    if len(data) == 0:
        return None

    fig, axes = plt.subplots(len(data), 5, figsize=(13.5, row_height * len(data)))
    axes = np.atleast_2d(axes)
    u_cols = uncertainty_columns(records)

    for i, (_, row) in enumerate(data.iterrows()):
        img = np.asarray(Image.open(row["path"]).convert("RGB"))
        mask_p = row.get("mask_path") or mask_path_for(row["path"])
        gt = np.asarray(Image.open(mask_p).convert("L")) if mask_p and Path(str(mask_p)).exists() else None
        heat = maps.get(row["image_id"]) if maps else None

        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f"{row['category']}/{row['defect_type']}  GT={row['label']}")

        if gt is not None:
            axes[i, 1].imshow(gt, cmap="gray", vmin=0, vmax=255)
            axes[i, 1].set_title("GT mask")
        else:
            axes[i, 1].imshow(np.zeros(img.shape[:2]), cmap="gray")
            axes[i, 1].set_title("GT mask (normal)")

        axes[i, 2].imshow(img)
        if heat is not None:
            up = np.asarray(Image.fromarray((_scale01(heat) * 255).astype(np.uint8))
                            .resize((img.shape[1], img.shape[0]), Image.BILINEAR))
            axes[i, 2].imshow(up, cmap="inferno", alpha=0.45)
        elif gt is not None:
            axes[i, 2].imshow(gt > 127, cmap="Reds", alpha=0.35)
        axes[i, 2].set_title("overlay")

        if heat is not None:
            axes[i, 3].imshow(heat, cmap="inferno")
            axes[i, 3].set_title("anomaly map")
        else:
            axes[i, 3].text(0.5, 0.5, "no map", ha="center", va="center")
            axes[i, 3].set_title("anomaly map")

        for j in range(4):
            axes[i, j].axis("off")
        axes[i, 4].axis("off")
        axes[i, 4].text(0, 1, _info_text(row, u_cols), fontsize=6, family="monospace", va="top")

    fig.tight_layout()
    if save_path:
        return save_figure(fig, save_path)
    return fig


def _info_text(row: pd.Series, u_cols: list[str]) -> str:
    lines = [f"type       : {row.get('prediction_type', '?')}",
             f"score      : {row['anomaly_score']:.4f}",
             f"raw        : {_fmt(row.get('raw_score'))}"]
    if "conformal_p" in row:
        lines.append(f"conf. p    : {_fmt(row['conformal_p'])}")
    if "conformal_pred" in row:
        lines.append(f"pred       : {int(row['conformal_pred'])}")
    if "parse_ok" in row and not bool(row["parse_ok"]):
        lines.append("ABSTAINED  : parse failed")
    lines.append("")
    for c in u_cols:
        lines.append(f"{c.removeprefix('u_')[:12]:<12}: {_fmt(row.get(c))}")
    if "halluc" in row:
        lines += ["", f"halluc     : {bool(row['halluc'])} ({row.get('halluc_case', '')})",
                  f"gt_iou     : {_fmt(row.get('gt_iou'))}"]
    if str(row.get("predicted_location", "")):
        lines += ["", f"claim loc  : {row['predicted_location']}",
                  f"claim type : {row.get('predicted_defect', '')}"]
    if str(row.get("observation", "")):
        lines += ["", "observation:", wrap_caption(row["observation"])]
    return "\n".join(lines)


def _fmt(v) -> str:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    return "nan" if np.isnan(f) else f"{f:.4f}"


def _scale01(a: np.ndarray) -> np.ndarray:
    lo, hi = float(np.nanmin(a)), float(np.nanmax(a))
    return (a - lo) / (hi - lo) if hi > lo else np.zeros_like(a)
