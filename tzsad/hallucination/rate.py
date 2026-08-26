"""Operational hallucination rate against MVTec pixel masks.

``HR`` is undefined in the prototype (defect #8). Here it is a measurable event
per image, with the two cases the brief specifies:

(a) any positive prediction on a ``good`` image - the model asserted a defect in
    an image with no defect at all;
(b) a positive prediction on an anomalous image whose *claimed or attended*
    region has essentially zero overlap with the ground-truth mask - right answer,
    wrong reason, which for a trustworthiness paper is still a failure.

The IoU threshold is configurable and its sensitivity is reported, because case
(b) is entirely a function of where you draw that line.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..eval.pixel import load_mask
from .location import LocationClaim, attended_region, parse_location, region_iou


@dataclass
class HRConfig:
    """Hallucination-rate definition parameters."""

    iou_threshold: float = 0.05
    map_size: int = 128
    top_frac: float = 0.05
    edge_width: float = 0.18
    use_claim_when_available: bool = True


def hallucination_flags(records: pd.DataFrame, maps: dict[str, np.ndarray],
                        config: HRConfig | None = None,
                        prediction_col: str = "conformal_pred") -> pd.DataFrame:
    """Per-image hallucination flags and the evidence behind each one.

    Returns a copy of ``records`` with ``halluc``, ``halluc_case`` and ``gt_iou``.
    Rows where the model predicted *normal* are never hallucinations here: this
    metric is about unfounded assertions, and a missed defect is a false negative
    measured elsewhere.
    """
    config = config or HRConfig()
    out = records.copy()
    flags, cases, ious = [], [], []

    for _, r in out.iterrows():
        pred = int(r.get(prediction_col, 0) or 0)
        if pred == 0:
            flags.append(False); cases.append("none"); ious.append(np.nan)
            continue
        if int(r["label"]) == 0:
            flags.append(True); cases.append("a_positive_on_good"); ious.append(0.0)
            continue

        mask_path = str(r.get("mask_path", "") or "")
        m = maps.get(r["image_id"])
        if not mask_path or m is None:
            flags.append(False); cases.append("unverifiable"); ious.append(np.nan)
            continue

        gt = load_mask(mask_path, config.map_size)
        claim = parse_location(str(r.get("predicted_location", "")))
        if config.use_claim_when_available and claim.is_localized:
            region = claim.mask(config.map_size, config.edge_width)
        else:
            region = attended_region(m, config.top_frac)
        iou = region_iou(region, gt)
        ious.append(iou)
        if iou < config.iou_threshold:
            flags.append(True); cases.append("b_wrong_region")
        else:
            flags.append(False); cases.append("grounded")

    out["halluc"] = flags
    out["halluc_case"] = cases
    out["gt_iou"] = ious
    return out


def hallucination_rate(flagged: pd.DataFrame, group_col: str = "category") -> pd.DataFrame:
    """HR per group and pooled, split by the two cases.

    HR is computed over *positive predictions* - a system that predicts normal
    for everything hallucinates nothing and is useless, so HR must always be read
    next to the detection metrics, never alone.
    """
    rows = []
    for group, g in flagged.groupby(group_col, sort=True):
        pos = g[g["halluc_case"] != "none"]
        rows.append(_hr_row(str(group), g, pos))
    rows.append(_hr_row("POOLED", flagged, flagged[flagged["halluc_case"] != "none"]))
    return pd.DataFrame(rows)


def _hr_row(name: str, g: pd.DataFrame, pos: pd.DataFrame) -> dict:
    n_pos = len(pos)
    return {
        "group": name, "n": len(g), "n_positive": n_pos,
        "hr": float(pos["halluc"].mean()) if n_pos else float("nan"),
        "hr_case_a": float((pos["halluc_case"] == "a_positive_on_good").mean()) if n_pos else float("nan"),
        "hr_case_b": float((pos["halluc_case"] == "b_wrong_region").mean()) if n_pos else float("nan"),
        "n_unverifiable": int((pos["halluc_case"] == "unverifiable").sum()),
    }


def iou_sensitivity(records: pd.DataFrame, maps: dict[str, np.ndarray],
                    thresholds=(0.0, 0.01, 0.05, 0.1, 0.2, 0.3),
                    config: HRConfig | None = None,
                    prediction_col: str = "conformal_pred") -> pd.DataFrame:
    """How HR moves with the IoU threshold. Case (a) is invariant to it by construction."""
    base = config or HRConfig()
    rows = []
    for t in thresholds:
        cfg = HRConfig(iou_threshold=t, map_size=base.map_size, top_frac=base.top_frac,
                       edge_width=base.edge_width,
                       use_claim_when_available=base.use_claim_when_available)
        flagged = hallucination_flags(records, maps, cfg, prediction_col)
        pooled = hallucination_rate(flagged).iloc[-1]
        rows.append({"iou_threshold": t, "hr": pooled["hr"], "hr_case_a": pooled["hr_case_a"],
                     "hr_case_b": pooled["hr_case_b"], "n_positive": pooled["n_positive"]})
    return pd.DataFrame(rows)
