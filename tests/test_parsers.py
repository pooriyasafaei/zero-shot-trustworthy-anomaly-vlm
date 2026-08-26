"""Parser tests: the VLM response format and the natural-language location claims."""
from __future__ import annotations

import numpy as np
import pytest

from tzsad.hallucination.location import parse_location, region_iou
from tzsad.scorers.qwen_parse import build_prompt, parse_generation, render_variant, PROMPT_VARIANTS

WELL_FORMED = """OBSERVATION:
The bottle neck shows a chipped edge.

DEFECT_TYPE:
broken part

DEFECT_LOCATION:
upper left of the bottle

CONFIDENCE:
HIGH

ANSWER:
YES"""


def test_parses_well_formed_response():
    s = parse_generation(WELL_FORMED)
    assert s.vote == 1 and s.parse_ok
    assert s.defect_type == "broken part"
    assert "upper left" in s.location
    assert s.confidence == "HIGH"
    assert s.observation.startswith("The bottle neck")


def test_parses_markdown_bolded_fields():
    text = WELL_FORMED.replace("ANSWER:\nYES", "ANSWER: **NO**").replace("CONFIDENCE:\nHIGH",
                                                                        "CONFIDENCE: **LOW**")
    s = parse_generation(text)
    assert s.vote == 0
    assert s.confidence == "LOW"


def test_parses_bare_trailing_verdict():
    s = parse_generation("The object looks fine to me.\n\nNO")
    assert s.vote == 0


def test_unparseable_generation_is_recorded_not_dropped():
    """Defect #6: a failed parse must be an abstention, never a silent removal."""
    s = parse_generation("I am unable to assess this image.")
    assert s.vote is None
    assert s.parse_ok is False
    assert s.text


def test_prompt_variants_keep_the_response_schema():
    """Paraphrases may change the framing but never the fields the parser needs."""
    for variant in PROMPT_VARIANTS:
        prompt = render_variant(variant, "metal nut")
        for field in ("OBSERVATION:", "DEFECT_TYPE:", "DEFECT_LOCATION:", "CONFIDENCE:", "ANSWER:"):
            assert field in prompt
    assert "metal nut" in build_prompt("metal nut")


@pytest.mark.parametrize("text,kind", [
    ("upper left corner", "corner"),
    ("bottom right", "quadrant"),
    ("center of the object", "quadrant"),
    ("along the outer edge", "edge"),
    ("across the entire surface", "surface"),
    ("NONE", "none"),
    ("", "none"),
    ("somewhere near the thing", "unparsed"),
])
def test_location_kinds(text, kind):
    assert parse_location(text).kind == kind


def test_quadrant_masks_land_in_the_right_place():
    top_left = parse_location("upper left").mask(64)
    bottom_right = parse_location("bottom right").mask(64)
    assert top_left[:20, :20].all()
    assert not top_left[50:, 50:].any()
    assert region_iou(top_left, bottom_right) == 0.0


def test_surface_claims_are_not_localized():
    """A claim covering everything cannot be contradicted, so it must not count as grounded."""
    assert parse_location("across the surface").is_localized is False
    assert parse_location("upper left").is_localized is True


def test_edge_mask_is_a_frame():
    m = parse_location("along the border").mask(64, edge_width=0.2)
    assert m[0].all() and m[-1].all()
    assert not m[32, 32]


def test_dtype_is_fp16_on_pre_ampere_gpus(monkeypatch):
    """bf16 on a T4 is emulated and slow; `auto` must not pick it there."""
    import torch

    from tzsad.scorers.qwen_scorer import QwenScorer, QwenScorerConfig

    scorer = QwenScorer(QwenScorerConfig(dtype="auto"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: (7, 5))   # Turing
    assert scorer._resolve_dtype() is torch.float16
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: (8, 0))   # Ampere
    assert scorer._resolve_dtype() is torch.bfloat16
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert scorer._resolve_dtype() is torch.float32


def test_verdict_token_ids_cover_spaced_and_cased_forms():
    """P(YES) must be summed over every surface form of the verdict token."""
    from tzsad.scorers.qwen_scorer import _variant_token_ids

    class FakeTok:
        table = {"YES": [1], " YES": [2], "Yes": [3], " Yes": [3], "yes": [4, 9], " yes": [5]}

        def encode(self, s, add_special_tokens=False):
            return self.table.get(s, [7, 8])

    ids = _variant_token_ids(FakeTok(), ["YES", "Yes", "yes"])
    assert ids == [1, 2, 3, 5]            # deduped, multi-token "yes" dropped


class _FakeTok:
    """Tokeniser stand-in: one id per word, matching Qwen's real collision."""

    VOCAB = {"OBSERVATION": 1, ":": 2, "The": 3, "bottle": 4, "is": 5, "intact": 6,
             "with": 7, " no": 8, "visible": 9, "cracks": 10, "DEFECT_TYPE": 11,
             "NONE": 12, "ANSWER": 13, "YES": 14, "NO": 15, "\n": 16}
    INV = {v: k for k, v in VOCAB.items()}

    def decode(self, ids):
        return " ".join(self.INV[i] for i in ids)


def test_verdict_is_read_after_the_answer_marker_not_before():
    """The bug that corrupted 50% of the VLM scores, pinned.

    'intact with no visible cracks' in OBSERVATION contains a ' no' token with the
    same id as the verdict 'No'. Scanning from the start reads P(YES) at a position
    unrelated to the verdict.
    """
    from tzsad.scorers.qwen_scorer import QwenScorer

    tok = _FakeTok()
    v = tok.VOCAB
    ids = [v["OBSERVATION"], v[":"], v["The"], v["bottle"], v["is"], v["intact"],
           v["with"], v[" no"], v["visible"], v["cracks"],
           v["DEFECT_TYPE"], v[":"], v["NONE"],
           v["ANSWER"], v[":"], v["YES"]]
    yes_no = {v["YES"], v["NO"], v[" no"]}

    pos = QwenScorer._verdict_position(tok, ids, yes_no)
    assert pos == 15, "must land on the verdict token, not the ' no' inside OBSERVATION"
    assert ids[pos] == v["YES"]
    # The naive scan is what we are guarding against.
    assert next(k for k, t in enumerate(ids) if t in yes_no) == 7


def test_verdict_position_falls_back_to_the_last_candidate():
    """With no ANSWER marker the verdict is still last in the response format."""
    from tzsad.scorers.qwen_scorer import QwenScorer

    tok = _FakeTok()
    v = tok.VOCAB
    ids = [v["The"], v["bottle"], v["is"], v["intact"], v["with"], v[" no"],
           v["cracks"], v["NO"]]
    pos = QwenScorer._verdict_position(tok, ids, {v["YES"], v["NO"], v[" no"]})
    assert pos == 7 and ids[pos] == v["NO"]


def test_verdict_position_is_none_without_any_candidate():
    from tzsad.scorers.qwen_scorer import QwenScorer

    tok = _FakeTok()
    v = tok.VOCAB
    assert QwenScorer._verdict_position(tok, [v["The"], v["bottle"]], {v["YES"], v["NO"]}) is None
