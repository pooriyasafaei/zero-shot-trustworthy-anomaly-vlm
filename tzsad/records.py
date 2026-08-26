"""The per-image record: the single interface between GPU work and everything else.

Core design rule from the brief: a scoring run writes per-image records to disk,
and every uncertainty, calibration and evaluation step consumes those records
offline. No evaluation change may require a GPU forward pass.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

#: Columns every scorer must emit. Uncertainty columns are added later with an
#: ``u_`` prefix; scorer-specific extras are free-form but should stay stable.
CORE_COLUMNS: tuple[str, ...] = (
    "run_id",            # results subdirectory this record came from
    "scorer_name",       # e.g. clip:ViT-B-16/openai, qwen:logprob
    "model_id",          # exact backbone identifier
    "category",
    "defect_type",
    "path",
    "image_id",
    "label",             # 1 = anomalous, 0 = normal
    "split",             # train (calibration pool) / test
    "subset_tag",        # which --subset produced this row
    "corruption",        # 'none' or an ImageNet-C name
    "severity",          # 0 for clean
    "anomaly_score",     # calibrated to [0, 1]; comparable within a category
    "raw_score",         # native scale of the scorer (cosine diff, logit, vote fraction)
    "parse_ok",          # False for unparseable VLM generations -> abstention, never dropped
    "n_valid_votes",     # multi-sample scorers only; 1 for single-pass
)

#: Scorer-specific text fields, kept in the same table so the qualitative viewer
#: and the hallucination module need only one file.
VLM_COLUMNS: tuple[str, ...] = (
    "observation", "predicted_defect", "predicted_location", "predicted_confidence",
)

UNCERTAINTY_PREFIX = "u_"


def empty_records() -> pd.DataFrame:
    """An empty frame with the core schema, for concatenation."""
    return pd.DataFrame({c: pd.Series(dtype=_dtype_for(c)) for c in CORE_COLUMNS})


def _dtype_for(column: str) -> str:
    if column in ("label", "severity", "n_valid_votes"):
        return "int64"
    if column in ("anomaly_score", "raw_score"):
        return "float64"
    if column == "parse_ok":
        return "bool"
    return "object"


def validate_records(df: pd.DataFrame) -> pd.DataFrame:
    """Check the core schema is present and types are sane. Raises on violation."""
    missing = [c for c in CORE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"records missing core columns: {missing}")
    if not df["label"].isin((0, 1)).all():
        raise ValueError("label must be 0/1")
    scores = df.loc[df["parse_ok"], "anomaly_score"]
    if len(scores) and not scores.between(-1e-9, 1 + 1e-9).all():
        raise ValueError(
            "anomaly_score must lie in [0, 1] for parse_ok rows; raw scorer output "
            "belongs in raw_score. (Defect #4: raw cosine differences are not probabilities.)"
        )
    if df.duplicated(["scorer_name", "image_id", "corruption", "severity"]).any():
        raise ValueError("duplicate (scorer_name, image_id, corruption, severity) rows")
    return df


def uncertainty_columns(df: pd.DataFrame) -> list[str]:
    """Names of the uncertainty signal columns present in ``df``."""
    return sorted(c for c in df.columns if c.startswith(UNCERTAINTY_PREFIX))


def write_records(df: pd.DataFrame, path: str | Path) -> Path:
    """Write records to parquet (preferred) or csv, inferred from the suffix."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    validate_records(df)
    if path.suffix == ".parquet":
        try:
            df.to_parquet(path, index=False)
            return path
        except (ImportError, ValueError):  # no pyarrow -> degrade to csv, loudly
            path = path.with_suffix(".csv")
    df.to_csv(path, index=False)
    return path


def read_records(paths: str | Path | Iterable[str | Path]) -> pd.DataFrame:
    """Read one or many record files and concatenate them."""
    if isinstance(paths, (str, Path)):
        paths = [paths]
    frames = []
    for p in paths:
        p = Path(p)
        if not p.exists():
            raise FileNotFoundError(f"records not found: {p}")
        frames.append(pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p))
    df = pd.concat(frames, ignore_index=True)
    if "parse_ok" in df:
        df["parse_ok"] = df["parse_ok"].astype(bool)
    return df
