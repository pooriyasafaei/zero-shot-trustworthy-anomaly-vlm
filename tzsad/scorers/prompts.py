"""Prompt ensembles for CLIP-family zero-shot anomaly detection.

The prototype's five normal / five anomaly templates are preserved verbatim as
``BASE``, because they work; the point of §4.3 defect #1 is that their *spread*
is not an uncertainty estimate, not that the ensemble itself is bad.

``DIVERSE`` deliberately widens the wording so that ``NormalizedPromptEnsemble``
has something other than near-paraphrases to disagree about.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PromptSet:
    """A named pair of normal/anomaly templates, each containing one ``{}``."""

    name: str
    normal: tuple[str, ...]
    anomaly: tuple[str, ...]

    def render(self, class_name: str) -> tuple[list[str], list[str]]:
        """Fill the class name into both template lists."""
        return ([t.format(class_name) for t in self.normal],
                [t.format(class_name) for t in self.anomaly])


BASE = PromptSet(
    name="base",
    normal=(
        "a photo of a normal {}.",
        "a photo of a flawless {}.",
        "a photo of a {} without any defect.",
        "a cropped photo of a normal {}.",
        "an industrial photo of an undamaged {}.",
    ),
    anomaly=(
        "a photo of a damaged {}.",
        "a photo of a {} with a defect.",
        "a photo of a flawed {}.",
        "a cropped photo of an anomalous {}.",
        "an industrial photo of a broken {}.",
    ),
)

DIVERSE = PromptSet(
    name="diverse",
    normal=BASE.normal + (
        "{}.",
        "a close-up inspection image of a {} that passed quality control.",
        "a blurry photo of a perfect {}.",
        "a dark photo of a {} in good condition.",
        "this {} has no scratches, cracks, holes or contamination.",
    ),
    anomaly=BASE.anomaly + (
        "a defective {}.",
        "a close-up inspection image of a {} that failed quality control.",
        "a blurry photo of a {} with a manufacturing fault.",
        "a dark photo of a {} in bad condition.",
        "this {} has a scratch, crack, hole or contamination.",
    ),
)

PROMPT_SETS: dict[str, PromptSet] = {p.name: p for p in (BASE, DIVERSE)}


def get_prompt_set(name: str) -> PromptSet:
    """Look up a prompt set by name, with a helpful error."""
    if name not in PROMPT_SETS:
        raise KeyError(f"unknown prompt set {name!r}; available: {sorted(PROMPT_SETS)}")
    return PROMPT_SETS[name]


def class_name_for(category: str) -> str:
    """MVTec folder name -> natural-language class name ('metal_nut' -> 'metal nut')."""
    return category.replace("_", " ")
