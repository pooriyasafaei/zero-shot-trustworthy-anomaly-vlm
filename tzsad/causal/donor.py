"""Aligned normal donor selection.

The counterfactual "this image without its defect" needs normal content to put in
the defect's place. Tonight's constraint is that the fill must be deterministic,
fast, dependency-light and - critically - **identical for the real intervention and
for every placebo**, so that donor identity cannot confound the comparison.

So: pick one donor image per target from that category's ``train/good`` pool, by
similarity computed **outside a dilated version of the defect mask**. Excluding the
defect from the matching is what stops the donor choice from being informed by the
thing we are about to remove.

This is deliberately not a generative inpainter. Donor filling is not the
contribution - the causal verification is - and a diffusion model would add a
dependency, a seed-sensitivity and a runtime cost for no scientific gain tonight.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from PIL import Image

from ..utils.logging import get_logger

log = get_logger("causal.donor")

#: Side length at which donor similarity is computed. Small on purpose: we want
#: gross layout/illumination agreement, not texture-level matching that would start
#: to select donors for reasons correlated with the defect.
MATCH_SIZE = 64


@dataclass(frozen=True)
class DonorMatch:
    """The donor chosen for one target image, with its matching diagnostics."""

    donor_image_id: str
    donor_path: str
    donor_distance: float
    n_candidates: int
    valid_fraction: float       # fraction of pixels used for matching (outside the mask)

    def as_dict(self) -> dict:
        """Flat dict for the interventions table."""
        return {
            "donor_image_id": self.donor_image_id,
            "donor_path": self.donor_path,
            "donor_distance": self.donor_distance,
            "donor_n_candidates": self.n_candidates,
            "donor_valid_fraction": self.valid_fraction,
        }


class DonorBank:
    """Per-category bank of ``train/good`` images used as counterfactual content.

    Only the normal training split may enter the bank. Using anomalous test images
    as donors would put defect pixels into a "defect removed" counterfactual, which
    would silently destroy the experiment.
    """

    def __init__(self, index: pd.DataFrame, match_size: int = MATCH_SIZE) -> None:
        good = index[(index["split"] == "train") & (index["label"] == 0)]
        if len(good) == 0:
            raise ValueError(
                "donor bank is empty: it needs train/good rows. Build the index with "
                "splits=('train', 'test') - anomalous images must never be donors."
            )
        if (index["split"] == "train").sum() != len(good):
            raise ValueError("the train split contains non-normal rows; refusing to build a donor bank")
        self.index = good.reset_index(drop=True)
        self.match_size = int(match_size)
        self._thumbs: dict[str, np.ndarray] = {}
        self._by_category: dict[str, pd.DataFrame] = {
            str(cat): g.reset_index(drop=True) for cat, g in self.index.groupby("category", sort=True)
        }

    @property
    def categories(self) -> list[str]:
        """Categories with at least one donor."""
        return sorted(self._by_category)

    def candidates(self, category: str) -> pd.DataFrame:
        """Donor rows available for a category."""
        if category not in self._by_category:
            raise KeyError(f"no train/good donors for category {category!r}; "
                           f"available: {self.categories}")
        return self._by_category[category]

    def _thumb(self, path: str) -> np.ndarray:
        """Cached small RGB thumbnail used for matching."""
        if path not in self._thumbs:
            with Image.open(path) as im:
                small = im.convert("RGB").resize((self.match_size, self.match_size), Image.BILINEAR)
            self._thumbs[path] = np.asarray(small, dtype=np.float32) / 255.0
        return self._thumbs[path]

    def select(self, target_path: str, category: str, exclude_mask: np.ndarray,
               max_candidates: int | None = None) -> DonorMatch:
        """Choose the donor closest to the target *outside* ``exclude_mask``.

        Parameters
        ----------
        target_path:
            The anomalous image being intervened on.
        category:
            Restricts donors to the same MVTec category.
        exclude_mask:
            Boolean array at the target's full resolution. Pixels inside it are
            ignored when matching - this is the dilated defect region.
        max_candidates:
            Cap the pool for speed; the first N rows in index order are used, which
            keeps the choice deterministic.
        """
        pool = self.candidates(category)
        if max_candidates is not None and len(pool) > max_candidates:
            pool = pool.iloc[:max_candidates]

        with Image.open(target_path) as im:
            target = np.asarray(
                im.convert("RGB").resize((self.match_size, self.match_size), Image.BILINEAR),
                dtype=np.float32) / 255.0

        small_mask = np.asarray(
            Image.fromarray((exclude_mask.astype(np.uint8) * 255)).resize(
                (self.match_size, self.match_size), Image.NEAREST)) > 127
        valid = ~small_mask
        if valid.sum() < 16:        # a mask covering almost everything leaves nothing to match on
            log.warning("exclusion mask covers %.1f%% of %s; matching on the full image instead",
                        100 * small_mask.mean(), Path(target_path).name)
            valid = np.ones_like(small_mask, dtype=bool)

        weights = valid[..., None].astype(np.float32)
        denom = float(weights.sum()) or 1.0
        best_id, best_path, best_dist = "", "", float("inf")
        for row in pool.itertuples(index=False):
            dist = float((np.abs(self._thumb(row.path) - target) * weights).sum() / denom)
            if dist < best_dist:
                best_id, best_path, best_dist = row.image_id, row.path, dist
        return DonorMatch(best_id, best_path, best_dist, len(pool), float(valid.mean()))


def load_rgb(path: str | Path) -> np.ndarray:
    """Load an image as uint8 RGB."""
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"), dtype=np.uint8)


def load_mask(path: str | Path, shape: tuple[int, int] | None = None) -> np.ndarray:
    """Load a ground-truth mask as a boolean array, optionally resized to ``shape``."""
    with Image.open(path) as im:
        mask = im.convert("L")
        if shape is not None and mask.size != (shape[1], shape[0]):
            mask = mask.resize((shape[1], shape[0]), Image.NEAREST)
        return np.asarray(mask) > 127


def aligned_donor_pixels(donor_path: str | Path, shape: tuple[int, int]) -> np.ndarray:
    """Donor image resized to the target's resolution, as uint8 RGB."""
    with Image.open(donor_path) as im:
        return np.asarray(im.convert("RGB").resize((shape[1], shape[0]), Image.BICUBIC),
                          dtype=np.uint8)
