"""MVTec-AD (and VisA/BTAD-compatible) indexing.

The whole pipeline is driven by a single dataframe index built here, so that
every scorer sees byte-identical image lists. That is the fix for defect #5 in
the brief: the two prototype branches were evaluated on different image sets.
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

from ..utils.logging import get_logger

log = get_logger("data.mvtec")

MVTEC_CATEGORIES: tuple[str, ...] = (
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather",
    "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper",
)

_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".JPG", ".PNG", ".bmp")

#: Index columns produced by :func:`build_index`. Kept stable on purpose.
INDEX_COLUMNS = ["category", "split", "defect_type", "path", "label", "mask_path", "image_id"]


@dataclass(frozen=True)
class SubsetSpec:
    """How to sub-sample the test set. Applied identically to every scorer.

    Attributes
    ----------
    kind:
        ``"full"`` keeps every test image; ``"n_per_folder"`` samples
        ``n`` images from each ``test/<defect_type>`` folder.
    n:
        Sample size for ``n_per_folder``.
    seed:
        Sampling seed, logged into the run manifest.
    """

    kind: str = "full"
    n: int = 20
    seed: int = 42

    @classmethod
    def parse(cls, spec: str | None, seed: int = 42) -> "SubsetSpec":
        """Parse ``"full"`` or ``"n_per_folder=20"`` into a :class:`SubsetSpec`."""
        if spec is None or spec == "full":
            return cls("full", seed=seed)
        m = re.fullmatch(r"n_per_folder\s*=\s*(\d+)", spec.strip())
        if not m:
            raise ValueError(f"unparseable subset spec {spec!r}; use 'full' or 'n_per_folder=20'")
        return cls("n_per_folder", int(m.group(1)), seed=seed)

    @property
    def tag(self) -> str:
        """Short string identifying this subset in filenames and records."""
        return "full" if self.kind == "full" else f"n{self.n}"


def resolve_data_root(data_root: str | Path) -> Path:
    """Return the directory that directly contains the category folders.

    MVTec archives unpack inconsistently (``mvtec_ad/``, ``mvtec_anomaly_detection/``
    or a bare directory), so we search one level down for a folder that looks like
    a category. Failing loudly here beats an empty index later.
    """
    root = Path(data_root)
    if not root.exists():
        raise FileNotFoundError(f"data_root does not exist: {root}")
    if _looks_like_dataset(root):
        return root
    for child in sorted(p for p in root.iterdir() if p.is_dir()):
        if _looks_like_dataset(child):
            log.info("resolved data_root %s -> %s", root, child)
            return child
    raise FileNotFoundError(
        f"no category folders with a test/ subdirectory found under {root}. "
        "Expected <root>/<category>/{train/good,test/<defect>}."
    )


def _looks_like_dataset(path: Path) -> bool:
    for child in path.iterdir() if path.is_dir() else []:
        if child.is_dir() and (child / "test").is_dir():
            return True
    return False


def available_categories(data_root: str | Path) -> list[str]:
    """List category folders actually present on disk, sorted."""
    root = resolve_data_root(data_root)
    return sorted(p.name for p in root.iterdir() if p.is_dir() and (p / "test").is_dir())


def mask_path_for(image_path: str | Path, root: Path | None = None) -> Path | None:
    """Ground-truth mask for a test image, or ``None`` for ``good`` images.

    Handles both MVTec (``<stem>_mask.png``) and VisA-style (``<stem>.png``) naming.
    """
    p = Path(image_path)
    if p.parent.name == "good":
        return None
    gt_dir = p.parents[2] / "ground_truth" / p.parent.name
    for candidate in (gt_dir / f"{p.stem}_mask.png", gt_dir / f"{p.stem}.png", gt_dir / f"{p.stem}_mask{p.suffix}"):
        if candidate.exists():
            return candidate
    return None


def _list_images(folder: Path) -> list[Path]:
    files: list[Path] = []
    for suffix in _IMAGE_SUFFIXES:
        files.extend(folder.glob(f"*{suffix}"))
    return sorted(set(files))


def build_index(
    data_root: str | Path,
    categories: Sequence[str] | None = None,
    splits: Iterable[str] = ("train", "test"),
    subset: SubsetSpec | None = None,
    cal_subset: SubsetSpec | None = None,
) -> pd.DataFrame:
    """Build the canonical image index.

    Parameters
    ----------
    data_root:
        Dataset root (auto-resolved one level down if needed).
    categories:
        Categories to include; ``None`` means every category present on disk.
    splits:
        ``"train"`` yields the normal-only calibration pool, ``"test"`` the eval set.
    subset:
        Sub-sampling applied to the **test** split.
    cal_subset:
        Sub-sampling applied to the **train** (calibration) split. Defaults to
        keeping every normal image, which is what the CLIP branch should do since
        embedding is cheap. The VLM branch cannot afford 3,600 forward passes just
        to calibrate, so it caps this - at the cost of a wider conformal threshold,
        which `n_cal_sensitivity` quantifies rather than hides.

    Returns
    -------
    DataFrame with :data:`INDEX_COLUMNS`. ``label`` is 1 for anomalous, 0 for normal.
    """
    root = resolve_data_root(data_root)
    subset = subset or SubsetSpec("full")
    cal_subset = cal_subset or SubsetSpec("full")
    present = available_categories(root)
    if categories is None:
        categories = present
    missing = [c for c in categories if c not in present]
    if missing:
        raise FileNotFoundError(f"categories missing under {root}: {missing}. Present: {present}")

    rows: list[dict] = []
    for category in categories:
        for split in splits:
            split_dir = root / category / split
            if not split_dir.is_dir():
                continue
            for folder in sorted(p for p in split_dir.iterdir() if p.is_dir()):
                images = _list_images(folder)
                active = subset if split == "test" else cal_subset
                if active.kind == "n_per_folder" and len(images) > active.n:
                    rng = random.Random(f"{active.seed}:{category}:{split}:{folder.name}")
                    images = sorted(rng.sample(images, active.n))
                label = 0 if folder.name == "good" else 1
                for img in images:
                    mask = mask_path_for(img)
                    rows.append(
                        {
                            "category": category,
                            "split": split,
                            "defect_type": folder.name,
                            "path": str(img),
                            "label": label,
                            "mask_path": str(mask) if mask else "",
                            "image_id": f"{category}/{split}/{folder.name}/{img.stem}",
                        }
                    )
    if not rows:
        raise RuntimeError(f"index is empty for {root} categories={list(categories)}")
    df = pd.DataFrame(rows, columns=INDEX_COLUMNS)
    log.info(
        "indexed %d images | %d categories | test=%d (anom %d) train=%d | subset=%s",
        len(df), df["category"].nunique(),
        (df.split == "test").sum(), ((df.split == "test") & (df.label == 1)).sum(),
        (df.split == "train").sum(), subset.tag,
    )
    return df
