"""Qwen2.5-VL anomaly scorer.

Three modes, all writing the same record schema:

``logprob`` (default)
    One greedy forward pass. The score is P(YES) recovered from the token
    logits at the position where the verdict is emitted - a continuous score
    from a single pass instead of an 11-valued vote fraction.
``vote``
    The prototype's multi-sample generation, kept because semantic entropy and
    prompt-perturbation consistency need multiple samples.
``both``
    A greedy pass for the score plus samples for the sampling-based signals.

Unparseable generations are recorded with ``parse_ok=False`` (never dropped).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFilter

from ..data.corruptions import apply_corruption
from ..utils.budget import TimeBudget, UnitTimer
from ..utils.checkpoint import RecordCheckpoint
from ..utils.logging import get_logger
from .base import Scorer, ScoringContext
from .prompts import class_name_for
from .qwen_parse import (PROMPT_VARIANTS, VlmSample, build_prompt, parse_generation,
                         render_variant)

log = get_logger("scorers.qwen")


@dataclass
class QwenScorerConfig:
    """Configuration of a Qwen scoring pass."""

    model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    mode: str = "logprob"              # logprob | vote | both
    n_samples: int = 10
    temperature: float = 0.6
    top_p: float = 0.9
    max_new_tokens: int = 160
    dtype: str = "auto"                # auto -> bf16 where supported, else fp16
    max_pixels: int = 512 * 28 * 28    # caps visual tokens; the T4-friendly setting
    prompt_variants: tuple[str, ...] = ()   # names from qwen_parse.PROMPT_VARIANTS
    generations_dir: str = ""          # if set, every raw generation is written here
    checkpoint_path: str = ""          # if set, per-image records are appended here and resumed


class QwenScorer(Scorer):
    """Wraps Qwen2.5-VL-Instruct behind the common :class:`Scorer` interface."""

    def __init__(self, config: QwenScorerConfig | None = None, device_map: str = "auto") -> None:
        self.config = config or QwenScorerConfig()
        self.device_map = device_map
        self.model_id = self.config.model_id
        self.scorer_name = f"qwen:{self.config.mode}"
        self._model = None
        self._processor = None
        self._yes_ids: list[int] = []
        self._no_ids: list[int] = []

    # -- model -------------------------------------------------------------
    def load(self) -> None:
        """Load the VLM. Fails loudly instead of downloading mid-experiment when offline."""
        if self._model is not None:
            return
        from transformers import AutoProcessor

        try:
            from transformers import Qwen2_5_VLForConditionalGeneration as _Model
        except ImportError as exc:
            raise RuntimeError(
                "transformers is too old for Qwen2.5-VL; need >= 4.49"
            ) from exc

        dtype = self._resolve_dtype()
        try:
            self._model = _Model.from_pretrained(
                self.config.model_id, torch_dtype=dtype, device_map=self.device_map,
                low_cpu_mem_usage=True,
            ).eval()
            self._processor = AutoProcessor.from_pretrained(
                self.config.model_id, max_pixels=self.config.max_pixels,
            )
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"could not load {self.config.model_id}. Pre-cache the weights into "
                "HF_HOME if this environment has no internet. Underlying error: "
                f"{exc}"
            ) from exc
        tok = self._processor.tokenizer
        self._yes_ids = _variant_token_ids(tok, ["YES", "Yes", "yes"])
        self._no_ids = _variant_token_ids(tok, ["NO", "No", "no"])
        log.info("loaded %s (%s), %d YES / %d NO token ids",
                 self.config.model_id, dtype, len(self._yes_ids), len(self._no_ids))

    def _resolve_dtype(self) -> torch.dtype:
        """Pick a dtype. ``auto`` uses bf16 only on hardware that runs it natively.

        ``torch.cuda.is_bf16_supported()`` returns True on Turing (T4, sm_75)
        because PyTorch counts *emulated* bf16, which is several times slower than
        fp16. Checking the compute capability directly is what keeps a T4 run from
        silently taking hours longer than it should.
        """
        if self.config.dtype == "float16":
            return torch.float16
        if self.config.dtype == "bfloat16":
            return torch.bfloat16
        if self.config.dtype == "float32":
            return torch.float32
        if not torch.cuda.is_available():
            return torch.float32
        major, _ = torch.cuda.get_device_capability()
        return torch.bfloat16 if major >= 8 else torch.float16

    # -- inference ---------------------------------------------------------
    def _inputs(self, image: Image.Image, prompt_text: str):
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image}, {"type": "text", "text": prompt_text}]}]
        chat = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return self._processor(text=[chat], images=[image], padding=True,
                               return_tensors="pt").to(self._model.device)

    @staticmethod
    def _verdict_position(tokenizer, token_ids: list[int], yes_no: set[int]) -> int | None:
        """Index of the token carrying the verdict, or ``None``.

        **The search must start after the ``ANSWER:`` marker.** The free-text
        ``OBSERVATION`` field routinely contains a lowercase "no" - "intact with no
        visible cracks" - which tokenises to the same id as the verdict "No".
        Scanning from the start of the generation therefore reads P(YES) at a
        position that has nothing to do with the verdict; measured on MVTec that
        hit 50.3% of generations and silently corrupted every logprob score.

        Falls back to the *last* candidate when no marker is found, since the
        response format puts the verdict last.
        """
        start = None
        for k in range(1, len(token_ids) + 1):
            if "ANSWER" in tokenizer.decode(token_ids[:k]):
                start = k
                break
        if start is not None:
            for k in range(start, len(token_ids)):
                if token_ids[k] in yes_no:
                    return k
        hits = [k for k, t in enumerate(token_ids) if t in yes_no]
        return hits[-1] if hits else None

    @torch.no_grad()
    def _greedy_with_logprob(self, image: Image.Image, prompt_text: str) -> tuple[VlmSample, float | None]:
        """Greedy decode, then read P(YES) at the verdict token position."""
        inputs = self._inputs(image, prompt_text)
        out = self._model.generate(
            **inputs, max_new_tokens=self.config.max_new_tokens, do_sample=False,
            temperature=None, top_p=None, top_k=None,
            output_scores=True, return_dict_in_generate=True,
        )
        new_tokens = out.sequences[0, inputs.input_ids.shape[1]:]
        text = self._processor.tokenizer.decode(new_tokens, skip_special_tokens=True)
        sample = parse_generation(text)

        ids = new_tokens.tolist()
        step = self._verdict_position(self._processor.tokenizer, ids,
                                      set(self._yes_ids) | set(self._no_ids))
        p_yes = None
        if step is not None and step < len(out.scores):
            lp = torch.log_softmax(out.scores[step][0].float(), dim=-1)
            yes = torch.logsumexp(lp[self._yes_ids], dim=0)
            no = torch.logsumexp(lp[self._no_ids], dim=0)
            p_yes = float(torch.sigmoid(yes - no))
        return sample, p_yes

    # -- direct causal-audit probe ----------------------------------------
    #: Short prompt used by :meth:`score_direct_verdict`. Semantically the same task
    #: as the structured prompt, but it asks only for the verdict, so the answer
    #: token is the *next* token and one forward pass suffices. Bumping this string
    #: must bump PROBE_VERSION, because cached scores are keyed on it.
    DIRECT_VERDICT_PROMPT = (
        "You are an expert industrial quality inspector.\n"
        "Inspect this manufactured {class_name}.\n"
        "Based only on clearly visible evidence, is the item defective?\n"
        "Respond with exactly one answer: YES or NO."
    )
    PROBE_VERSION = "direct-verdict-v1"

    @torch.inference_mode()
    def score_direct_verdict(self, image: Image.Image, class_name: str) -> dict:
        """One forward pass; read P(YES) from the next-token logits.

        The structured scorer generates a full OBSERVATION/DEFECT_TYPE/.../ANSWER
        block before its verdict is available. A causal audit needs the verdict for
        every intervention - 21 of them per image at K=19 - and does not need the
        prose, so generating 120 tokens each time would dominate the runtime for no
        scientific gain.

        This is a *different measurement* from the structured scorer, not a faster
        version of it: a model asked to answer immediately may answer differently
        from one that reasons first. The audit therefore reports the correlation
        between the two rather than assuming they agree.

        Returns ``audit_p_yes``, ``audit_logit`` (= ``log P(YES)/P(NO)``) and probe
        metadata. Leaves the scorer's own state and outputs untouched.
        """
        self.load()
        prompt = self.DIRECT_VERDICT_PROMPT.format(class_name=class_name)
        inputs = self._inputs(image, prompt)
        logits = self._model(**inputs).logits[0, -1].float()
        log_probs = torch.log_softmax(logits, dim=-1)
        log_yes = torch.logsumexp(log_probs[self._yes_ids], dim=0)
        log_no = torch.logsumexp(log_probs[self._no_ids], dim=0)
        audit_logit = float(log_yes - log_no)
        top_id = int(torch.argmax(logits))
        return {
            "audit_p_yes": float(torch.sigmoid(log_yes - log_no)),
            "audit_logit": audit_logit,
            "audit_log_yes": float(log_yes),
            "audit_log_no": float(log_no),
            "audit_top_token": self._processor.tokenizer.decode([top_id]),
            "audit_top_is_verdict": bool(top_id in set(self._yes_ids) | set(self._no_ids)),
            "probe_version": self.PROBE_VERSION,
            "parse_ok": True,        # a logit read cannot fail to parse
        }

    @torch.inference_mode()
    def score_generated_verdict(self, image: Image.Image, class_name: str) -> dict:
        """Confirmation arm: the existing structured path, reported in probe schema.

        Slower, but it exercises the same generation the rest of the repository
        scores with, so a pilot conclusion can be checked against it.
        """
        self.load()
        sample, p_yes = self._greedy_with_logprob(image, build_prompt(class_name))
        parse_ok = p_yes is not None
        return {
            "audit_p_yes": float(p_yes) if parse_ok else float("nan"),
            "audit_logit": (float(np.log(max(p_yes, 1e-9) / max(1 - p_yes, 1e-9)))
                            if parse_ok else float("nan")),
            "audit_top_token": "", "audit_top_is_verdict": parse_ok,
            "probe_version": "generated-verdict-v1",
            "parse_ok": parse_ok,
            "observation": sample.observation,
            "predicted_defect": sample.defect_type,
            "predicted_location": sample.location,
            "predicted_confidence": sample.confidence,
            "generated_vote": sample.vote,
            "generation_text": sample.text,
        }

    @torch.inference_mode()
    def score_direct_verdict_batch(self, images: Sequence[Image.Image],
                                   class_name: str) -> list[dict]:
        """:meth:`score_direct_verdict` for several images in one forward pass.

        Same prompt, same token read-out, same output schema. Padding is on the
        left so the last position is the real next-token position for every row;
        in the causal audit every image in a batch comes from one target and has
        the same resolution, so in practice the rows are equal length and there is
        no padding at all. fp16 batched kernels can differ from the unbatched path
        in the last bits, which the v2 smoke run measures and logs.
        """
        if len(images) == 1:
            return [self.score_direct_verdict(images[0], class_name)]
        self.load()
        prompt = self.DIRECT_VERDICT_PROMPT.format(class_name=class_name)
        messages = [{"role": "user", "content": [
            {"type": "image", "image": images[0]}, {"type": "text", "text": prompt}]}]
        chat = self._processor.apply_chat_template(messages, tokenize=False,
                                                   add_generation_prompt=True)
        tok = self._processor.tokenizer
        side = tok.padding_side
        tok.padding_side = "left"
        try:
            inputs = self._processor(text=[chat] * len(images), images=list(images),
                                     padding=True, return_tensors="pt").to(self._model.device)
        finally:
            tok.padding_side = side
        logits = self._model(**inputs).logits[:, -1].float()
        log_probs = torch.log_softmax(logits, dim=-1)
        log_yes = torch.logsumexp(log_probs[:, self._yes_ids], dim=1)
        log_no = torch.logsumexp(log_probs[:, self._no_ids], dim=1)
        verdict = set(self._yes_ids) | set(self._no_ids)
        out = []
        for i in range(len(images)):
            top_id = int(torch.argmax(logits[i]))
            out.append({
                "audit_p_yes": float(torch.sigmoid(log_yes[i] - log_no[i])),
                "audit_logit": float(log_yes[i] - log_no[i]),
                "audit_log_yes": float(log_yes[i]),
                "audit_log_no": float(log_no[i]),
                "audit_top_token": tok.decode([top_id]),
                "audit_top_is_verdict": bool(top_id in verdict),
                "probe_version": self.PROBE_VERSION,
                "parse_ok": True,
            })
        return out

    @torch.no_grad()
    def _sample(self, image: Image.Image, prompt_text: str, n: int) -> list[VlmSample]:
        inputs = self._inputs(image, prompt_text)
        gen = self._model.generate(
            **inputs, max_new_tokens=self.config.max_new_tokens, do_sample=True,
            temperature=self.config.temperature, top_p=self.config.top_p,
            num_return_sequences=n,
        )
        trimmed = gen[:, inputs.input_ids.shape[1]:]
        texts = self._processor.batch_decode(trimmed, skip_special_tokens=True,
                                             clean_up_tokenization_spaces=False)
        return [parse_generation(t) for t in texts]

    # -- Scorer API --------------------------------------------------------
    def score_index(self, index: pd.DataFrame, ctx: ScoringContext,
                    budget: TimeBudget | None = None) -> pd.DataFrame:
        """Run the VLM over every row of ``index`` and return per-image records.

        With ``checkpoint_path`` set, each image's record is appended as it is
        produced and already-scored images are skipped on a later run. A VLM pass
        over MVTec takes hours; losing it to an expiring session is not acceptable,
        and the generations dumped alongside are useless without their scores.
        """
        gen_dir = Path(self.config.generations_dir) if self.config.generations_dir else None
        if gen_dir:
            gen_dir.mkdir(parents=True, exist_ok=True)
        variants = [v for v in PROMPT_VARIANTS if v.name in self.config.prompt_variants]

        ckpt = RecordCheckpoint(self.config.checkpoint_path) if self.config.checkpoint_path else None
        todo = index
        if ckpt is not None:
            done = ckpt.done_ids()
            todo = index[~index["image_id"].astype(str).isin(done)]
            if len(done):
                log.info("resuming: %d/%d images already scored, %d to go",
                         len(done), len(index), len(todo))
        if len(todo) == 0 and ckpt is not None:
            return self.finalize(_ordered(ckpt.to_frame(), index), ctx)

        self.load()
        rows: list[dict] = []
        stopped_early = False
        for n_done, (_, r) in enumerate(todo.iterrows(), 1):
            # Stop before starting an image that cannot finish: a half-scored image
            # is not a resumable state, and the checkpoint must stay consistent.
            if budget is not None and not budget.has_room():
                log.warning("time budget reached after %d/%d images this pass; %s",
                            n_done - 1, len(todo), budget.report())
                stopped_early = True
                break
            timer = UnitTimer(budget)
            timer.__enter__()
            image = Image.open(r["path"]).convert("RGB")
            if ctx.corruption not in ("none", ""):
                image = apply_corruption(image, ctx.corruption, int(ctx.severity))
            cname = class_name_for(r["category"])
            prompt_text = build_prompt(cname)

            rec: dict = {
                "category": r["category"], "defect_type": r["defect_type"], "path": r["path"],
                "image_id": r["image_id"], "label": int(r["label"]), "split": r["split"],
            }
            samples: list[VlmSample] = []
            p_yes = None
            greedy: VlmSample | None = None

            if self.config.mode in ("logprob", "both"):
                greedy, p_yes = self._greedy_with_logprob(image, prompt_text)
                samples.append(greedy)
            if self.config.mode in ("vote", "both"):
                samples.extend(self._sample(image, prompt_text, self.config.n_samples))

            votes = [s.vote for s in samples if s.vote is not None]
            vote_fraction = float(np.mean(votes)) if votes else float("nan")

            if self.config.mode == "vote":
                score, raw = vote_fraction, vote_fraction
                parse_ok = bool(votes)
            else:
                parse_ok = p_yes is not None
                score = float(p_yes) if parse_ok else float("nan")
                raw = float(np.log(max(p_yes, 1e-9) / max(1 - p_yes, 1e-9))) if parse_ok else float("nan")

            head = greedy or (samples[0] if samples else VlmSample())
            rec.update({
                "anomaly_score": score, "raw_score": raw, "parse_ok": parse_ok,
                "n_valid_votes": len(votes),
                "vote_fraction": vote_fraction,
                "observation": head.observation, "predicted_defect": head.defect_type,
                "predicted_location": head.location, "predicted_confidence": head.confidence,
                "n_generations": len(samples),
            })

            # Prompt-perturbation consistency needs verdicts under paraphrase.
            for variant in variants:
                v_img = image.filter(ImageFilter.GaussianBlur(variant.blur_radius)) \
                    if variant.blur_radius else image
                v_sample, v_p = self._greedy_with_logprob(v_img, render_variant(variant, cname))
                rec[f"variant_vote__{variant.name}"] = (
                    float(v_sample.vote) if v_sample.vote is not None else float("nan"))
                rec[f"variant_pyes__{variant.name}"] = float(v_p) if v_p is not None else float("nan")

            if gen_dir:
                _dump_generations(gen_dir, r["image_id"], samples)
            rows.append(rec)
            if ckpt is not None:
                ckpt.append(rec)
            timer.__exit__()
            if n_done % 25 == 0:
                log.info("qwen scored %d/%d images this pass", n_done, len(todo))

        if ckpt is not None:
            ckpt.flush()
            out = self.finalize(_ordered(ckpt.to_frame(), index), ctx)
            out.attrs["complete"] = not stopped_early
            return out
        records = self.finalize(pd.DataFrame(rows), ctx)
        records.attrs["complete"] = not stopped_early
        return records


def _ordered(records: pd.DataFrame, index: pd.DataFrame) -> pd.DataFrame:
    """Put checkpointed records back into index order and drop anything stale.

    A checkpoint may hold images from a previous, wider index; restricting to the
    current one keeps the record file honest about what this run covers.
    """
    if records.empty:
        return records
    keep = records[records["image_id"].astype(str).isin(index["image_id"].astype(str))]
    order = {v: i for i, v in enumerate(index["image_id"].astype(str))}
    return keep.sort_values("image_id", key=lambda c: c.map(order)).reset_index(drop=True)


def _dump_generations(gen_dir: Path, image_id: str, samples: Sequence[VlmSample]) -> None:
    """Persist raw generations so semantic entropy can be recomputed with no GPU."""
    safe = image_id.replace("/", "__")
    payload = [{"vote": s.vote, "observation": s.observation, "defect_type": s.defect_type,
                "location": s.location, "confidence": s.confidence, "text": s.text}
               for s in samples]
    (gen_dir / f"{safe}.json").write_text(json.dumps(payload, indent=1))


def _variant_token_ids(tokenizer, words: Sequence[str]) -> list[int]:
    """Token ids for a word, with and without a leading space, deduplicated.

    We need every id whose surface form is the verdict, because whether the
    verdict token carries a leading space depends on what preceded it.
    """
    ids: list[int] = []
    for word in words:
        for surface in (word, f" {word}"):
            toks = tokenizer.encode(surface, add_special_tokens=False)
            if len(toks) == 1:
                ids.append(toks[0])
    if not ids:
        raise RuntimeError(f"no single-token id found for any of {list(words)}")
    return sorted(set(ids))


def load_generations(gen_dir: str | Path, image_id: str) -> list[VlmSample]:
    """Read back generations dumped by a scoring run."""
    path = Path(gen_dir) / f"{image_id.replace('/', '__')}.json"
    if not path.exists():
        return []
    return [VlmSample(**d) for d in json.loads(path.read_text())]
