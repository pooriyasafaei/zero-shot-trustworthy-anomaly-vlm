#!/usr/bin/env python3
"""Build the synthetic-anomaly calibration set used by the logistic trust combiner.

Corrupts ``train/good`` images with CutPaste/NSA, scores them through the same
CLIP pipeline, and writes records with a ``synthetic_anomaly`` label. No real
MVTec defect is ever seen, so the fused trust score stays zero-shot.

Usage::

    python scripts/make_synthetic.py --config configs/clip_full.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from PIL import Image

from tzsad import pipeline as P
from tzsad.data.synthetic import make_synthetic
from tzsad.utils.config import load_config, to_dict
from tzsad.utils.logging import setup_logging
from tzsad.utils.repro import start_run


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args(argv)

    cfg = load_config(args.config, args.overrides)
    run_dir = P.run_dir_for(cfg)
    log = setup_logging(run_dir)
    start_run(run_dir, to_dict(cfg), int(cfg.seed))

    index = P.stage_index(cfg)
    train = index[(index.split == "train") & (index.label == 0)]
    out_root = run_dir / "synthetic_images"
    n_per = int(cfg.fusion.synthetic.n_per_category)
    method = str(cfg.fusion.synthetic.method)

    rows = []
    for category, g in train.groupby("category", sort=True):
        g = g.sort_values("image_id").head(n_per * 2)
        for i, (_, r) in enumerate(g.iterrows()):
            is_anom = i % 2 == 1
            dest_dir = out_root / category / ("synthetic" if is_anom else "good")
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / f"{Path(r['path']).stem}.png"
            img = Image.open(r["path"]).convert("RGB")
            if is_anom:
                img, _ = make_synthetic(img, seed=int(cfg.seed) + i, method=method)
            img.save(dest)
            rows.append({"category": category, "split": "synthetic",
                         "defect_type": "synthetic" if is_anom else "good",
                         "path": str(dest), "label": 0, "mask_path": "",
                         "image_id": f"{category}/synthetic/{dest.stem}_{i}",
                         "synthetic_anomaly": int(is_anom)})
    synth_index = pd.DataFrame(rows)
    log.info("built %d synthetic images (%d anomalous)", len(synth_index),
             int(synth_index.synthetic_anomaly.sum()))
    synth_index.to_csv(run_dir / "synthetic_index.csv", index=False)

    embedders = P.stage_embed(cfg, synth_index.drop(columns=["synthetic_anomaly"]))
    per_backbone = P.stage_score_clip(cfg, synth_index.drop(columns=["synthetic_anomaly"]), embedders)
    records = per_backbone[P.backbones_of(cfg)[0].cache_tag]
    records = records.merge(synth_index[["image_id", "synthetic_anomaly"]], on="image_id")
    records.to_parquet(run_dir / "records_synthetic.parquet", index=False)
    log.info("wrote %s", run_dir / "records_synthetic.parquet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
