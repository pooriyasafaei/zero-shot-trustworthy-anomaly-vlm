"""Parsing of the structured VLM response. Unparseable output is a result, not a loss.

The prototype dropped generations whose ``ANSWER:`` field it could not find, which
silently removed an unmodelled abstention population from every metric (defect #6).
Here a failed parse yields ``parse_ok=False`` and is carried through to the
selective-prediction evaluation as an abstention.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

ANSWER_RE = re.compile(r"ANSWER:\s*\**\s*(YES|NO)", re.IGNORECASE)
OBSERVATION_RE = re.compile(r"OBSERVATION:\s*(.+?)(?=DEFECT_TYPE:|ANSWER:|$)", re.IGNORECASE | re.DOTALL)
DEFECT_TYPE_RE = re.compile(r"DEFECT_TYPE:\s*(.+?)(?=DEFECT_LOCATION:|ANSWER:|$)", re.IGNORECASE | re.DOTALL)
LOCATION_RE = re.compile(r"DEFECT_LOCATION:\s*(.+?)(?=CONFIDENCE:|ANSWER:|$)", re.IGNORECASE | re.DOTALL)
CONFIDENCE_RE = re.compile(r"CONFIDENCE:\s*\**\s*(HIGH|MEDIUM|LOW)", re.IGNORECASE)

CONFIDENCE_MAP = {"HIGH": 1.0, "MEDIUM": 0.5, "LOW": 0.0}


@dataclass
class VlmSample:
    """One parsed generation."""

    vote: int | None = None            # 1 = YES (anomalous), 0 = NO, None = unparseable
    observation: str = ""
    defect_type: str = ""
    location: str = ""
    confidence: str = ""
    text: str = ""

    @property
    def parse_ok(self) -> bool:
        """Whether a YES/NO verdict was recoverable from this generation."""
        return self.vote is not None


def parse_generation(text: str) -> VlmSample:
    """Parse one raw generation into a :class:`VlmSample`.

    Robust to the two failure modes seen in practice: markdown bolding around the
    field values, and the model answering with a bare ``YES``/``NO`` on the last
    line without the ``ANSWER:`` header.
    """
    sample = VlmSample(text=text)
    m = ANSWER_RE.search(text)
    if m:
        sample.vote = 1 if m.group(1).upper() == "YES" else 0
    else:
        tail = [ln.strip(" *.\t") for ln in text.strip().splitlines() if ln.strip()]
        if tail and tail[-1].upper() in ("YES", "NO"):
            sample.vote = 1 if tail[-1].upper() == "YES" else 0

    for attr, regex in (("observation", OBSERVATION_RE), ("defect_type", DEFECT_TYPE_RE),
                        ("location", LOCATION_RE)):
        hit = regex.search(text)
        if hit:
            setattr(sample, attr, _clean(hit.group(1)))
    conf = CONFIDENCE_RE.search(text)
    if conf:
        sample.confidence = conf.group(1).upper()
    return sample


def _clean(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().strip("*").strip()).strip()


@dataclass
class PromptVariant:
    """A semantically equivalent rewrite of the inspection prompt.

    Used by ``PromptPerturbationConsistency`` (VL-Uncertainty, arXiv:2411.11919):
    a trustworthy model should give the same verdict under paraphrase.
    """

    name: str
    header: str
    blur_radius: float = 0.0


DEFECT_MENU = (
    "crack", "scratch", "hole", "dent", "contamination",
    "broken part", "deformation", "discoloration",
)

RESPONSE_FORMAT = """Respond EXACTLY in this format:

OBSERVATION:
<what you visually observe>

DEFECT_TYPE:
<one word or NONE>

DEFECT_LOCATION:
<location or NONE>

CONFIDENCE:
<HIGH or MEDIUM or LOW>

ANSWER:
<YES or NO>"""


def build_prompt(class_name: str, header: str | None = None) -> str:
    """The prototype's structured inspection prompt, preserved.

    ``header`` swaps only the framing sentences, so paraphrase perturbations never
    change the response schema the parser depends on.
    """
    head = header or (
        f"You are an expert industrial quality inspector evaluating a manufactured "
        f"{class_name} from the MVTec AD benchmark.\n\n"
        "Inspect the object carefully.\n\n"
        "Only report defects that are clearly visible.\n\nDo NOT guess.\n\n"
        "If you cannot confidently determine whether the object is normal or defective, "
        "report the uncertainty explicitly and avoid inventing defects."
    )
    menu = "\n".join(f"- {d}" for d in DEFECT_MENU)
    return f"{head}\n\nPossible defects include:\n{menu}\n\n{RESPONSE_FORMAT}"


PROMPT_VARIANTS: tuple[PromptVariant, ...] = (
    PromptVariant("base", ""),
    PromptVariant(
        "qc_operator",
        "You work in quality control on a production line. The image shows a {class_name}. "
        "Decide whether this unit should be rejected. Report only defects you can actually see; "
        "do not speculate.",
    ),
    PromptVariant(
        "terse",
        "Examine this image of a {class_name} and determine whether it is defective. "
        "Base your judgement only on visible evidence.",
    ),
    PromptVariant(
        "blurred",
        "You are an expert industrial quality inspector evaluating a manufactured {class_name}. "
        "Inspect the object carefully and report only clearly visible defects.",
        blur_radius=1.5,
    ),
)


def render_variant(variant: PromptVariant, class_name: str) -> str:
    """Build the full prompt for one perturbation variant."""
    if not variant.header:
        return build_prompt(class_name)
    return build_prompt(class_name, variant.header.format(class_name=class_name))
