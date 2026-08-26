"""Assemble the evaluation tables that go into ``results/report/``."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..records import UNCERTAINTY_PREFIX, uncertainty_columns
from ..utils.logging import get_logger
from .bootstrap import bootstrap_metric, delong_test, holm_bonferroni
from .metrics import (brier, error_prediction_auroc, expected_calibration_error,
                      maximum_calibration_error, nll, safe_aupr, safe_auroc)
from .selective import abstention_summary, selective_table

log = get_logger("eval.report")


def detection_table(records: pd.DataFrame, n_boot: int = 1000, seed: int = 0,
                    group_col: str = "category") -> pd.DataFrame:
    """Per-category and pooled AUROC/AUPR with bootstrap CIs. No number without an interval."""
    rows = []
    for group, g in records.groupby(group_col, sort=True):
        rows.append(_detection_row(str(group), g, n_boot, seed))
    rows.append(_detection_row("POOLED", records, n_boot, seed))
    df = pd.DataFrame(rows)
    per_cat = df[df[group_col if group_col in df else "group"] != "POOLED"]
    mean_row = {"group": "MEAN_OF_CATEGORIES", "n": int(per_cat["n"].sum()),
                "n_anomalous": int(per_cat["n_anomalous"].sum()),
                "auroc": float(per_cat["auroc"].mean()),
                "auroc_ci_lo": np.nan, "auroc_ci_hi": np.nan,
                "aupr": float(per_cat["aupr"].mean()), "aupr_ci_lo": np.nan, "aupr_ci_hi": np.nan}
    return pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)


def _detection_row(name: str, g: pd.DataFrame, n_boot: int, seed: int) -> dict:
    y = g["label"].to_numpy()
    s = g["anomaly_score"].to_numpy(dtype=float)
    au = bootstrap_metric(y, s, safe_auroc, n_boot, seed=seed)
    ap = bootstrap_metric(y, s, safe_aupr, n_boot, seed=seed)
    return {"group": name, "n": len(g), "n_anomalous": int((y == 1).sum()),
            "auroc": au.value, "auroc_ci_lo": au.lo, "auroc_ci_hi": au.hi,
            "aupr": ap.value, "aupr_ci_lo": ap.lo, "aupr_ci_hi": ap.hi}


def calibration_table(records: pd.DataFrame, n_bins: int = 15,
                      group_col: str = "category") -> pd.DataFrame:
    """ECE (adaptive bins), MCE, Brier and NLL per category and pooled."""
    rows = []
    for group, g in list(records.groupby(group_col, sort=True)) + [("POOLED", records)]:
        y = g["label"].to_numpy()
        p = g["anomaly_score"].to_numpy(dtype=float)
        rows.append({
            "group": str(group), "n": len(g),
            "ece_adaptive": expected_calibration_error(y, p, n_bins, adaptive=True),
            "ece_equalwidth": expected_calibration_error(y, p, n_bins, adaptive=False),
            "mce": maximum_calibration_error(y, p, n_bins, adaptive=True),
            "brier": brier(y, p), "nll": nll(y, p),
        })
    return pd.DataFrame(rows)


def error_prediction_table(records: pd.DataFrame, correct_col: str, n_boot: int = 1000,
                           seed: int = 0, group_col: str = "category",
                           prediction_col: str = "conformal_pred") -> pd.DataFrame:
    """**The headline table**: AUROC of each uncertainty signal for predicting error.

    A signal at 0.5 carries no information about whether the prediction is wrong.
    Below 0.5 means it is actively misleading - the system is more confident when
    it is wrong, which is the silent-failure signature.

    Two controls are built in, because the naive version of this test reports an
    artefact as a result:

    **Score-monotone baselines.** When one error type dominates - on MVTec at
    delta=0.05 the errors are ~95% false negatives - "is this an error" collapses
    into "is this anomalous and scored low", so *any* signal monotone in the
    anomaly score scores well for free. The ``BASELINE:score`` and
    ``BASELINE:1-score`` rows measure exactly that free signal, and
    ``excess_over_baseline`` reports how much each estimator adds over it. A
    signal whose excess is at or below zero has told us nothing the score did not.

    **Conditioning on the decision.** Rows with scope ``pred=0`` / ``pred=1``
    repeat the test within each decision group, where the errors are homogeneous
    (all false negatives, or all false positives). This removes the base-rate
    shortcut entirely and is the number to quote when the class balance is skewed.
    """
    signals = uncertainty_columns(records)
    if not signals:
        raise ValueError("no u_* uncertainty columns present; run the uncertainty stage first")
    correct = records[correct_col].to_numpy(dtype=bool)
    wrong = (~correct).astype(int)

    baselines: dict[str, np.ndarray] = {}
    if "anomaly_score" in records:
        score = records["anomaly_score"].to_numpy(dtype=float)
        baselines = {"BASELINE:score": score, "BASELINE:1-score": -score}
    baseline_level = max(
        (safe_auroc(wrong, v) for v in baselines.values() if np.isfinite(safe_auroc(wrong, v))),
        default=0.5,
    )

    rows = []
    for name, values in list(baselines.items()) + [(s, records[s].to_numpy(dtype=float)) for s in signals]:
        is_baseline = name.startswith("BASELINE:")
        pooled = bootstrap_metric(wrong, values, safe_auroc, n_boot, seed=seed)
        rows.append({
            "signal": name, "scope": "POOLED", "n": len(records),
            "n_errors": int(wrong.sum()), "err_auroc": pooled.value,
            "ci_lo": pooled.lo, "ci_hi": pooled.hi,
            "informative": bool(pooled.lo > 0.5), "misleading": bool(pooled.hi < 0.5),
            "excess_over_baseline": np.nan if is_baseline else pooled.value - baseline_level,
            "beats_baseline": False if is_baseline else bool(pooled.lo > baseline_level),
        })

        # Conditioned on the decision: errors within a group are all of one type.
        if prediction_col in records:
            for pred_value, sub in records.groupby(prediction_col, sort=True):
                m = records.index.get_indexer(sub.index)
                w = wrong[m]
                if len(np.unique(w)) < 2:
                    continue
                v = bootstrap_metric(w, values[m], safe_auroc, n_boot, seed=seed)
                rows.append({
                    "signal": name, "scope": f"pred={int(pred_value)}", "n": len(sub),
                    "n_errors": int(w.sum()), "err_auroc": v.value, "ci_lo": v.lo, "ci_hi": v.hi,
                    "informative": bool(v.lo > 0.5), "misleading": bool(v.hi < 0.5),
                    "excess_over_baseline": np.nan, "beats_baseline": False,
                })

        if is_baseline:
            continue
        for group, g in records.groupby(group_col, sort=True):
            m = records.index.get_indexer(g.index)
            w = wrong[m]
            if len(np.unique(w)) < 2:
                continue
            v = bootstrap_metric(w, values[m], safe_auroc, max(n_boot // 5, 200), seed=seed)
            rows.append({
                "signal": name, "scope": str(group), "n": len(g), "n_errors": int(w.sum()),
                "err_auroc": v.value, "ci_lo": v.lo, "ci_hi": v.hi,
                "informative": bool(v.lo > 0.5), "misleading": bool(v.hi < 0.5),
                "excess_over_baseline": np.nan, "beats_baseline": False,
            })
    return pd.DataFrame(rows)


def scorer_comparison_table(a: pd.DataFrame, b: pd.DataFrame, name_a: str, name_b: str,
                            group_col: str = "category") -> pd.DataFrame:
    """DeLong comparison of two scorers on the **identical** image subset.

    Raises if the two frames are not aligned on ``image_id`` - which is precisely
    the mistake that made the prototype's CLIP-vs-Qwen comparison meaningless.
    """
    a = a.set_index("image_id").sort_index()
    b = b.set_index("image_id").sort_index()
    shared = a.index.intersection(b.index)
    if len(shared) == 0:
        raise ValueError("the two scorers share no images; comparison is impossible")
    if len(shared) < min(len(a), len(b)):
        log.warning("comparing on %d shared images (a=%d, b=%d)", len(shared), len(a), len(b))
    a, b = a.loc[shared], b.loc[shared]

    rows, pvals = [], []
    for group in sorted(a[group_col].unique()) + ["POOLED"]:
        m = slice(None) if group == "POOLED" else (a[group_col] == group)
        res = delong_test(a.loc[m, "label"].to_numpy(),
                          a.loc[m, "anomaly_score"].to_numpy(dtype=float),
                          b.loc[m, "anomaly_score"].to_numpy(dtype=float))
        rows.append({"group": group, "scorer_a": name_a, "scorer_b": name_b, **res})
        pvals.append(res["p_value"])
    df = pd.DataFrame(rows)
    finite = df["p_value"].fillna(1.0).tolist()
    df["significant_holm"] = holm_bonferroni(finite)
    return df


def estimator_declarations(records: pd.DataFrame) -> pd.DataFrame:
    """What each uncertainty signal in this run *claims* to measure, and how it fails.

    Written into every report so a reader can check a signal's declared failure
    mode against its measured error-prediction AUROC. A signal that under-performs
    in the way its own docstring predicted is evidence; one that under-performs for
    an unanticipated reason is a bug to chase.
    """
    from ..uncertainty.registry import ESTIMATORS

    present = {c.removeprefix(UNCERTAINTY_PREFIX) for c in uncertainty_columns(records)}
    rows = []
    for name in sorted(present):
        cls = ESTIMATORS.get(name)
        rows.append({
            "signal": f"{UNCERTAINTY_PREFIX}{name}",
            "kind": cls.info.kind if cls else "derived",
            "inputs": cls.info.inputs if cls else "conformal p-values",
            "failure_modes": cls.info.failure_modes if cls else
                             "peaks at p=0.5, not at the decision boundary p=delta",
            "is_baseline": bool(cls.info.is_baseline) if cls else False,
        })
    return pd.DataFrame(rows)


def build_report(records: pd.DataFrame, out_dir: str | Path, correct_col: str = "correct",
                 n_boot: int = 1000, seed: int = 0) -> dict[str, Path]:
    """Write the standard CSV tables for one experiment and return their paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    tables = {
        "detection": detection_table(records, n_boot, seed),
        "calibration": calibration_table(records),
        "abstention": abstention_summary(records),
        "estimators": estimator_declarations(records),
    }
    if correct_col in records and uncertainty_columns(records):
        tables["error_prediction"] = error_prediction_table(records, correct_col, n_boot, seed)
        tables["selective"] = selective_table(records, correct_col, uncertainty_columns(records))
    for name, table in tables.items():
        p = out_dir / f"{name}.csv"
        table.to_csv(p, index=False)
        paths[name] = p
        log.info("wrote %s (%d rows)", p, len(table))
    return paths
