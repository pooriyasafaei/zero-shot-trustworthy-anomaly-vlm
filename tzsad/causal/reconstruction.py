"""Applying an intervention: replace a region with aligned normal donor content.

The contract that makes the comparison interpretable:

* **Outside the intervention mask the image is bit-identical to the original.** Any
  change out there would mean the model could be reacting to something other than
  the edited region, and the whole effect would be uninterpretable. Enforced by
  construction and asserted.
* **The same donor is used for the real edit and for every placebo on that image**,
  so donor identity cannot explain a difference between them.
* Feathering happens **only inside** the mask, via a distance transform, so the seam
  is softened without touching a single outside pixel.

Every intervention also reports artifact diagnostics. A reviewer will ask whether the
model reacted to the defect or to a bad splice, and that question needs numbers, not
reassurance.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

from ..utils.logging import get_logger

log = get_logger("causal.reconstruction")

#: Outside-mask pixels must match the original exactly. Kept at 0 rather than a
#: tolerance: the code path writes originals through untouched, so any drift is a bug.
OUTSIDE_TOLERANCE = 0


@dataclass
class Intervention:
    """One reconstructed image plus everything needed to audit it."""

    kind: str                    # original | defect_removed | placebo
    image: np.ndarray            # uint8 RGB
    mask: np.ndarray             # boolean region that was replaced
    edit_energy: float           # mean |change| inside the mask, 0-255 scale
    outside_change_max: float    # must be 0
    seam_score: float            # gradient jump across the mask boundary
    mask_area: int
    mask_fraction: float
    feather_px: int

    def as_dict(self) -> dict:
        """Flat dict for the interventions table."""
        return {"intervention_type": self.kind, "mask_area": self.mask_area,
                "mask_fraction": self.mask_fraction, "edit_energy": self.edit_energy,
                "outside_change_max": self.outside_change_max, "seam_score": self.seam_score,
                "feather_px": self.feather_px}


def _feather_weights(mask: np.ndarray, feather_px: int) -> np.ndarray:
    """Blend weights that are 1 deep inside the mask and ramp to 0 at its edge.

    Computed from the distance transform *of the mask itself*, so the ramp lives
    entirely inside the region. A feather that extended outwards would modify
    original pixels and break the core guarantee.
    """
    if feather_px <= 0:
        return mask.astype(np.float32)
    dist = ndimage.distance_transform_edt(mask)
    return np.clip(dist / float(feather_px), 0.0, 1.0).astype(np.float32) * mask


def seam_score(original: np.ndarray, edited: np.ndarray, mask: np.ndarray,
               ring_px: int = 3) -> float:
    """How visible the splice boundary is.

    Mean absolute gradient-magnitude difference in a thin band straddling the mask
    edge. A clean fill leaves the local gradient statistics roughly unchanged; a bad
    one introduces a step the model might react to.
    """
    band = ndimage.binary_dilation(mask, iterations=ring_px) & \
        ~ndimage.binary_erosion(mask, iterations=ring_px)
    if band.sum() == 0:
        return 0.0
    def grad(img: np.ndarray) -> np.ndarray:
        grey = img.astype(np.float32).mean(axis=2)
        gy, gx = np.gradient(grey)
        return np.hypot(gx, gy)
    return float(np.abs(grad(edited)[band] - grad(original)[band]).mean())


def apply_intervention(original: np.ndarray, donor: np.ndarray, mask: np.ndarray,
                       kind: str, feather_px: int = 3) -> Intervention:
    """Replace ``mask`` in ``original`` with content from ``donor``.

    Parameters
    ----------
    original, donor:
        uint8 RGB arrays of identical shape. The donor must already be resized to
        the target's resolution (see :func:`~tzsad.causal.donor.aligned_donor_pixels`).
    mask:
        Boolean region to replace.
    kind:
        ``defect_removed`` or ``placebo`` - recorded, not used in the computation, so
        the two are guaranteed to go through exactly the same code path.
    feather_px:
        Width of the inward-only blend ramp.
    """
    if original.shape != donor.shape:
        raise ValueError(f"donor shape {donor.shape} != target shape {original.shape}")
    if mask.shape != original.shape[:2]:
        raise ValueError(f"mask shape {mask.shape} != image shape {original.shape[:2]}")

    weights = _feather_weights(mask, feather_px)[..., None]
    edited = (original.astype(np.float32) * (1.0 - weights)
              + donor.astype(np.float32) * weights)
    out = np.clip(np.rint(edited), 0, 255).astype(np.uint8)
    # Belt and braces: write the untouched region back verbatim so rounding can
    # never introduce an outside-mask difference.
    out[~mask] = original[~mask]

    outside = float(np.abs(out[~mask].astype(np.int32)
                           - original[~mask].astype(np.int32)).max()) if (~mask).any() else 0.0
    if outside > OUTSIDE_TOLERANCE:
        raise AssertionError(
            f"intervention modified {outside} levels outside its mask; the counterfactual "
            "would not be interpretable")
    inside = float(np.abs(out[mask].astype(np.int32)
                          - original[mask].astype(np.int32)).mean()) if mask.any() else 0.0
    return Intervention(
        kind=kind, image=out, mask=mask, edit_energy=inside, outside_change_max=outside,
        seam_score=seam_score(original, out, mask), mask_area=int(mask.sum()),
        mask_fraction=float(mask.mean()), feather_px=int(feather_px),
    )


def identity_intervention(original: np.ndarray) -> Intervention:
    """The untouched image, recorded in the same schema as a real intervention."""
    empty = np.zeros(original.shape[:2], dtype=bool)
    return Intervention("original", original, empty, 0.0, 0.0, 0.0, 0, 0.0, 0)


def reconstruction_diagnostics(interventions: list[Intervention]) -> dict:
    """Summary comparing real-defect against placebo reconstructions.

    The number that matters is the ratio: if the true-defect fill is systematically
    seamier than the placebo fills, a positive effect may be an artifact response
    rather than causal dependence on the defect.
    """
    defect = [i for i in interventions if i.kind == "defect_removed"]
    placebo = [i for i in interventions if i.kind == "placebo"]
    d_seam = float(np.mean([i.seam_score for i in defect])) if defect else float("nan")
    p_seam = float(np.mean([i.seam_score for i in placebo])) if placebo else float("nan")
    d_edit = float(np.mean([i.edit_energy for i in defect])) if defect else float("nan")
    p_edit = float(np.mean([i.edit_energy for i in placebo])) if placebo else float("nan")
    return {
        "defect_seam_score": d_seam, "placebo_seam_score_mean": p_seam,
        "seam_ratio": (d_seam / p_seam) if p_seam else float("nan"),
        "defect_edit_energy": d_edit, "placebo_edit_energy_mean": p_edit,
        "edit_energy_ratio": (d_edit / p_edit) if p_edit else float("nan"),
        "max_outside_change": float(max((i.outside_change_max for i in interventions), default=0.0)),
    }


def save_image(image: np.ndarray, path: str | Path) -> Path:
    """Write an intervention image as PNG (lossless: JPEG would add its own artifacts)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path)
    return path
