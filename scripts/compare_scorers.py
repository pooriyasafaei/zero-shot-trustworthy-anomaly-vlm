#!/usr/bin/env python3
"""DeLong comparison of two scorers on the identical image subset.

Refuses to compare frames that do not share images, which is the mistake behind
the prototype's meaningless 0.8228-vs-0.8095 claim.

Usage::

    python scripts/compare_scorers.py --a results/clip_full/records_test_scored.parquet \
        --b results/qwen_full/records_test_scored.parquet --name-a clip --name-b qwen \
        --out results/report/clip_vs_qwen.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from tzsad.eval.report import scorer_comparison_table
from tzsad.records import read_records
from tzsad.utils.logging import setup_logging


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--name-a", default="A")
    ap.add_argument("--name-b", default="B")
    ap.add_argument("--out", default="results/report/scorer_comparison.csv")
    args = ap.parse_args(argv)

    log = setup_logging()
    a, b = read_records(args.a), read_records(args.b)
    table = scorer_comparison_table(a, b, args.name_a, args.name_b)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)

    pooled = table[table.group == "POOLED"].iloc[0]
    log.info("%s AUROC %.4f vs %s AUROC %.4f | delta %+.4f | DeLong p=%.4g | n=%d",
             args.name_a, pooled.auc_a, args.name_b, pooled.auc_b, pooled.delta,
             pooled.p_value, int(pooled.n))
    if pooled.p_value > 0.05:
        log.info("not significant at 0.05: these two scorers are not distinguishable on this subset")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
