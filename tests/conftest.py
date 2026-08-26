"""Shared fixtures: a tiny synthetic MVTec-shaped dataset built on the fly."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture(scope="session")
def fake_mvtec(tmp_path_factory) -> Path:
    """A two-category dataset with MVTec's directory layout and real PNG files."""
    root = tmp_path_factory.mktemp("mvtec")
    rng = np.random.default_rng(0)
    for category, base in (("bottle", 40), ("carpet", 120)):
        for split, folder, n in (("train", "good", 12), ("test", "good", 6),
                                 ("test", "scratch", 5)):
            d = root / category / split / folder
            d.mkdir(parents=True, exist_ok=True)
            for i in range(n):
                arr = np.clip(base + rng.normal(0, 12, (64, 64, 3)), 0, 255).astype(np.uint8)
                if folder != "good":
                    arr[20:34, 30:44] = 250          # a bright square "defect"
                Image.fromarray(arr).save(d / f"{i:03d}.png")
            if folder != "good":
                gt = root / category / "ground_truth" / folder
                gt.mkdir(parents=True, exist_ok=True)
                for i in range(n):
                    m = np.zeros((64, 64), dtype=np.uint8)
                    m[20:34, 30:44] = 255
                    Image.fromarray(m).save(gt / f"{i:03d}_mask.png")
    return root
