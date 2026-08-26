"""Selective prediction: risk-coverage curves, AURC, coverage@risk.

Abstentions (``parse_ok=False``) are treated as the highest-uncertainty items, so
the prototype's silently-dropped Unknowns become a measurable result rather than a
hole in the denominator (defect #6).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class RiskCoverage:
    """A risk-coverage curve and its summary statistics."""

    coverage: np.ndarray
    risk: np.ndarray
    aurc: float
    eaurc: float          # AURC minus the AURC of an oracle ordering
    n: int

    def coverage_at_risk(self, target_risk: float) -> float:
        """Largest coverage whose selective risk stays at or below ``target_risk``."""
        ok = self.risk <= target_risk
        return float(self.coverage[ok].max()) if ok.any() else 0.0

    def risk_at_coverage(self, target_coverage: float) -> float:
        """Selective risk at the smallest coverage that is at least ``target_coverage``."""
        ok = self.coverage >= target_coverage
        return float(self.risk[ok][0]) if ok.any() else float("nan")


def risk_coverage_curve(correct: np.ndarray, uncertainty: np.ndarray,
                        abstain: np.ndarray | None = None) -> RiskCoverage:
    """Risk-coverage curve for rejecting the most uncertain items first.

    Parameters
    ----------
    correct:
        1 if the prediction on this item is right.
    uncertainty:
        Higher = reject sooner. NaNs are treated as maximal uncertainty.
    abstain:
        Optional boolean mask of items the system refused to answer. These are
        forced to the front of the rejection order (and count as errors at full
        coverage), which is the honest accounting for a parse failure.

    Notes
    -----
    ``eaurc`` (excess AURC) subtracts the AURC an oracle achieves by rejecting
    exactly the errors first, so it isolates the *ranking quality* of the
    uncertainty signal from the base error rate. Comparing raw AURC across
    methods with different accuracies is misleading; compare eAURC.
    """
    correct = np.asarray(correct, dtype=float)
    u = np.asarray(uncertainty, dtype=float).copy()
    u = np.where(np.isfinite(u), u, np.inf)
    if abstain is not None:
        u = np.where(np.asarray(abstain, dtype=bool), np.inf, u)
    n = correct.size
    if n == 0:
        return RiskCoverage(np.array([]), np.array([]), float("nan"), float("nan"), 0)

    order = np.argsort(u, kind="mergesort")           # keep the most certain first
    errors = 1.0 - correct[order]
    cum_err = np.cumsum(errors)
    k = np.arange(1, n + 1)
    coverage = k / n
    risk = cum_err / k
    aurc = float(np.mean(risk))

    oracle_err = np.sort(1.0 - correct)               # oracle rejects every error first
    oracle_risk = np.cumsum(oracle_err) / k
    eaurc = aurc - float(np.mean(oracle_risk))
    return RiskCoverage(coverage, risk, aurc, eaurc, n)


def accuracy_at_coverage(correct: np.ndarray, uncertainty: np.ndarray, coverage: float,
                         abstain: np.ndarray | None = None) -> float:
    """Selective accuracy when answering only the most certain ``coverage`` fraction."""
    rc = risk_coverage_curve(correct, uncertainty, abstain)
    return 1.0 - rc.risk_at_coverage(coverage)


def selective_table(records: pd.DataFrame, correct_col: str, uncertainty_cols: list[str],
                    coverages: tuple[float, ...] = (0.5, 0.7, 0.8, 0.9, 1.0),
                    risks: tuple[float, ...] = (0.05, 0.10, 0.20)) -> pd.DataFrame:
    """One row per uncertainty signal: AURC, eAURC, accuracy@coverage, coverage@risk."""
    abstain = (~records["parse_ok"].to_numpy(dtype=bool)) if "parse_ok" in records else None
    rows = []
    for col in uncertainty_cols:
        rc = risk_coverage_curve(records[correct_col].to_numpy(), records[col].to_numpy(), abstain)
        row = {"signal": col, "aurc": rc.aurc, "eaurc": rc.eaurc, "n": rc.n}
        for c in coverages:
            row[f"acc@cov{c:g}"] = 1.0 - rc.risk_at_coverage(c)
        for r in risks:
            row[f"cov@risk{r:g}"] = rc.coverage_at_risk(r)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("aurc").reset_index(drop=True)


def abstention_summary(records: pd.DataFrame) -> pd.DataFrame:
    """Abstention rate per category, and the accuracy on the answered subset."""
    rows = []
    for category, g in records.groupby("category", sort=True):
        parsed = g["parse_ok"].to_numpy(dtype=bool)
        rows.append({
            "category": category, "n": len(g),
            "n_abstain": int((~parsed).sum()),
            "abstention_rate": float((~parsed).mean()),
            "n_answered": int(parsed.sum()),
        })
    total = pd.DataFrame(rows)
    total.loc[len(total)] = {
        "category": "POOLED", "n": len(records),
        "n_abstain": int((~records["parse_ok"].to_numpy(dtype=bool)).sum()),
        "abstention_rate": float((~records["parse_ok"].to_numpy(dtype=bool)).mean()),
        "n_answered": int(records["parse_ok"].sum()),
    }
    return total
