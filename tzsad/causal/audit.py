"""Stage-I orchestration: build interventions, score them, persist everything.

Two properties drive the design.

**A Kaggle session can die at any moment.** Every intervention score is appended to a
crash-safe cache the instant it is produced, and every completed target image is
checkpointed. A restart reloads both and scores only what is missing.

**Work is embarrassingly parallel across target images.** Each worker takes a
deterministic shard of the targets and owns its own cache file, so N GPUs run N
workers with no coordination and no interleaved writes. Shard assignment is by a
stable hash of ``image_id``, so it does not change when the subset changes.
"""
from __future__ import annotations

import hashlib
import io
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol, Sequence

import numpy as np
import pandas as pd
from PIL import Image

from ..scorers.prompts import class_name_for
from ..utils.budget import TimeBudget, UnitTimer
from ..utils.checkpoint import RecordCheckpoint
from ..utils.logging import get_logger
from .donor import DonorBank, DonorMatch, aligned_donor_pixels, load_mask, load_rgb
from .effects import compute_effects
from .placebo import PlaceboPlan, generate_placebos, placebo_mask_array
from .reconstruction import (Intervention, apply_intervention, identity_intervention,
                             reconstruction_diagnostics, save_image)

log = get_logger("causal.audit")


class VerdictProbe(Protocol):
    """Anything that maps (image, class name) to ``audit_p_yes`` / ``audit_logit``.

    A protocol rather than a concrete type so the tests can drive the whole pipeline
    with a stub and never need Qwen weights.
    """

    def __call__(self, image: Image.Image, class_name: str) -> dict: ...


@dataclass
class CausalConfig:
    """Everything that changes what a causal run computes."""

    k_placebos: int = 19
    seed: int = 1234
    mask_source: str = "gt"          # gt | bank_map (see docs/CERM_AD.md)
    score_mode: str = "direct_verdict"   # direct_verdict | generated_verdict
    feather_px: int = 3
    dilate_for_donor: int = 16
    max_iou: float = 0.01
    ring_width: int = 12
    min_separation: int = 8
    candidate_stride: int = 8
    match_quantile: float = 0.4
    min_pool: int = 0
    use_gradient: bool = True
    max_donor_candidates: int = 80
    max_candidates: int = 600         # cap on placebo translations evaluated per target
    max_targets_per_folder: int = 2
    save_images: bool = True
    save_placebo_images: int = 2     # how many placebo images to keep per target
    model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct"

    def signature(self) -> str:
        """Hash of every field that can change a score, for the cache key.

        Anything that alters the pixels fed to the model, or the model/probe used,
        belongs here; anything purely cosmetic (how many images we save) does not.
        """
        payload = json.dumps({
            "k": self.k_placebos, "seed": self.seed, "mask_source": self.mask_source,
            "score_mode": self.score_mode, "feather": self.feather_px,
            "dilate": self.dilate_for_donor, "max_iou": self.max_iou,
            "ring": self.ring_width, "sep": self.min_separation,
            "stride": self.candidate_stride, "quantile": self.match_quantile,
            "grad": self.use_gradient, "donors": self.max_donor_candidates,
            "cands": self.max_candidates,
            "model": self.model_id,
        }, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


class ScoreCache:
    """Append-only cache of intervention scores, keyed by image content + signature.

    The key is a SHA-256 of the exact PNG bytes scored, combined with the run
    signature. That makes reuse safe across restarts and impossible across a change
    of model, prompt version or intervention geometry - a cached score can only be
    reused for byte-identical input under identical settings.

    Each worker writes its own shard; readers take the union, so parallel workers
    never interleave writes into one file.
    """

    def __init__(self, path: str | Path, signature: str, probe_version: str = "") -> None:
        self.path = Path(path)
        self.signature = signature
        self.probe_version = probe_version
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._memory: dict[str, dict] = {}
        self.hits = 0
        self.misses = 0
        self._load_all_shards()

    def _load_all_shards(self) -> None:
        pattern = self.path.name.replace(".jsonl", "*.jsonl")
        for shard in sorted(self.path.parent.glob(pattern)):
            try:
                with shard.open() as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            row = json.loads(line)
                        except json.JSONDecodeError:
                            continue      # torn final line from a killed process
                        if row.get("key"):
                            self._memory[row["key"]] = row
            except OSError:
                continue
        if self._memory:
            log.info("score cache: %d entries loaded from %s*",
                     len(self._memory), self.path.stem)

    @staticmethod
    def image_digest(image: np.ndarray) -> str:
        """SHA-256 of the image's canonical PNG encoding."""
        buf = io.BytesIO()
        Image.fromarray(image).save(buf, format="PNG", optimize=False)
        return hashlib.sha256(buf.getvalue()).hexdigest()

    def key(self, image_digest: str, class_name: str) -> str:
        return f"{self.signature}:{self.probe_version}:{class_name}:{image_digest}"

    def get(self, key: str) -> dict | None:
        row = self._memory.get(key)
        if row is not None:
            self.hits += 1
            return row["score"]
        self.misses += 1
        return None

    def put(self, key: str, score: dict) -> None:
        row = {"key": key, "score": score}
        self._memory[key] = row
        with self.path.open("a") as fh:
            fh.write(json.dumps(row, default=float) + "\n")
            fh.flush()


def assign_shards(image_ids: Sequence[str], num_shards: int) -> np.ndarray:
    """Balanced, deterministic shard assignment over a sorted target list.

    Round-robin over the deterministically ordered targets, so every worker gets
    within one image of the same load. A hash of the id would be stable under a
    change of subset but badly unbalanced on small runs - a 40-image pilot split
    25/15 leaves one GPU idle for a third of the run.

    Reassignment across runs is harmless because a worker skips any image already
    completed by *any* worker (see :func:`completed_ids`).
    """
    n = max(int(num_shards), 1)
    return np.arange(len(image_ids)) % n


def completed_ids(run_dir: Path) -> set[str]:
    """Targets already audited by any worker, so a reshard never redoes work."""
    done: set[str] = set()
    for path in Path(run_dir).glob("causal_records*.jsonl"):
        done |= RecordCheckpoint(path).done_ids()
    return done


def select_targets(index: pd.DataFrame, config: CausalConfig) -> pd.DataFrame:
    """Anomalous test images with a usable mask, capped per defect folder.

    Only anomalous images take part: the counterfactual "remove the defect" is
    undefined for a normal image, and a placebo-only control has nothing to contrast.
    """
    test = index[(index["split"] == "test") & (index["label"] == 1)]
    usable = test[test["mask_path"].astype(str).str.len() > 0]
    dropped = len(test) - len(usable)
    if dropped:
        log.warning("%d anomalous test images have no mask_path and cannot be audited", dropped)
    if config.max_targets_per_folder:
        usable = (usable.sort_values("image_id")
                  .groupby(["category", "defect_type"], sort=True)
                  .head(config.max_targets_per_folder))
    return usable.sort_values("image_id").reset_index(drop=True)


@dataclass
class TargetResult:
    """Everything produced for one target image."""

    record: dict
    interventions: list[dict] = field(default_factory=list)


def audit_target(
    row: pd.Series,
    bank: DonorBank,
    probe: VerdictProbe,
    config: CausalConfig,
    cache: ScoreCache | None = None,
    images_dir: Path | None = None,
) -> TargetResult:
    """Run the full intervention set for one anomalous image.

    The order is deliberate: pick the donor once, then build every intervention from
    that same donor, so the real edit and the placebos differ *only* in where the
    mask sits.
    """
    from scipy import ndimage

    image_id = str(row["image_id"])
    category = str(row["category"])
    class_name = class_name_for(category)
    original = load_rgb(row["path"])
    mask = load_mask(row["mask_path"], original.shape[:2])

    base: dict = {
        "image_id": image_id, "category": category,
        "defect_type": str(row["defect_type"]), "path": str(row["path"]),
        "mask_path": str(row["mask_path"]), "mask_area": int(mask.sum()),
        "mask_fraction": float(mask.mean()),
    }
    if mask.sum() == 0:
        base.update({"status": "empty_mask", "n_valid_placebos": 0})
        return TargetResult(base)

    # Donor matching ignores a dilated defect region, so the choice of donor cannot
    # be informed by the very thing the intervention removes.
    exclude = ndimage.binary_dilation(mask, iterations=max(config.dilate_for_donor, 1))
    donor: DonorMatch = bank.select(str(row["path"]), category, exclude,
                                    config.max_donor_candidates)
    donor_px = aligned_donor_pixels(donor.donor_path, original.shape[:2])

    plan: PlaceboPlan = generate_placebos(
        original, mask, config.k_placebos, config.seed, image_id,
        max_iou=config.max_iou, ring_width=config.ring_width,
        min_separation=config.min_separation, candidate_stride=config.candidate_stride,
        match_quantile=config.match_quantile, min_pool=config.min_pool,
        use_gradient=config.use_gradient, max_candidates=config.max_candidates)

    interventions: list[tuple[Intervention, dict]] = [
        (identity_intervention(original), {"placebo_index": -1})]
    interventions.append((apply_intervention(original, donor_px, mask, "defect_removed",
                                             config.feather_px), {"placebo_index": -1}))
    for placebo in plan.placebos:
        pmask = placebo_mask_array(mask, placebo)
        interventions.append((apply_intervention(original, donor_px, pmask, "placebo",
                                                 config.feather_px), placebo.as_dict()))

    rows: list[dict] = []
    scores: dict[str, float] = {}
    placebo_deltas: list[float] = []
    for pos, (iv, extra) in enumerate(interventions):
        digest = ScoreCache.image_digest(iv.image) if cache else ""
        key = cache.key(digest, class_name) if cache else ""
        result = cache.get(key) if cache else None
        cache_hit = result is not None
        if result is None:
            result = probe(Image.fromarray(iv.image), class_name)
            if cache:
                cache.put(key, result)

        out_path = ""
        if images_dir is not None and config.save_images:
            keep = (iv.kind != "placebo"
                    or int(extra.get("placebo_index", 0)) < config.save_placebo_images)
            if keep:
                name = f"{image_id.replace('/', '__')}__{iv.kind}"
                if iv.kind == "placebo":
                    name += f"{extra.get('placebo_index')}"
                out_path = str(save_image(iv.image, images_dir / f"{name}.png"))

        entry = {**base, **iv.as_dict(), **donor.as_dict(), **extra,
                 "audit_p_yes": result.get("audit_p_yes"),
                 "audit_logit": result.get("audit_logit"),
                 "probe_version": result.get("probe_version", ""),
                 "probe_parse_ok": bool(result.get("parse_ok", True)),
                 "cache_hit": cache_hit, "image_output_path": out_path,
                 "image_digest": digest}
        rows.append(entry)

        z = float(result.get("audit_logit", float("nan")))
        if iv.kind == "original":
            scores["original"] = z
            scores["original_p"] = float(result.get("audit_p_yes", float("nan")))
        elif iv.kind == "defect_removed":
            scores["defect_removed"] = z
            scores["defect_removed_p"] = float(result.get("audit_p_yes", float("nan")))
        else:
            placebo_deltas.append(scores["original"] - z)

    delta_defect = scores["original"] - scores["defect_removed"]
    effect = compute_effects(delta_defect, placebo_deltas)
    diagnostics = reconstruction_diagnostics([iv for iv, _ in interventions])

    record = {
        **base, "status": "ok",
        "audit_p_yes_original": scores.get("original_p"),
        "audit_logit_original": scores.get("original"),
        "audit_p_yes_defect_removed": scores.get("defect_removed_p"),
        "audit_logit_defect_removed": scores.get("defect_removed"),
        "original_direct_prediction": "YES" if scores.get("original_p", 0) >= 0.5 else "NO",
        # Every target here is anomalous, so predicting YES is the correct call.
        "original_is_correct_for_anomaly": bool(scores.get("original_p", 0) >= 0.5),
        **effect.as_dict(), **plan.as_dict(), **donor.as_dict(), **diagnostics,
        "probe_version": rows[0].get("probe_version", ""),
        "score_mode": config.score_mode, "mask_source": config.mask_source,
    }
    return TargetResult(record, rows)


def run_audit(
    index: pd.DataFrame,
    probe: VerdictProbe,
    config: CausalConfig,
    run_dir: Path,
    shard: int = 0,
    num_shards: int = 1,
    budget: TimeBudget | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Audit every target in this worker's shard, resuming from checkpoints.

    Returns ``(causal_records, interventions)`` for this shard. The aggregation step
    merges shards later, so a worker never needs to see another worker's output.
    """
    targets = select_targets(index, config)
    if num_shards > 1:
        mine = assign_shards(targets["image_id"].tolist(), num_shards) == shard
        targets = targets[mine].reset_index(drop=True)
    log.info("shard %d/%d: %d target images", shard, num_shards, len(targets))

    suffix = "" if num_shards <= 1 else f".shard{shard}"
    bank = DonorBank(index)
    cache = ScoreCache(run_dir / f"score_cache{suffix}.jsonl", config.signature(),
                       config.score_mode)
    records_ckpt = RecordCheckpoint(run_dir / f"causal_records{suffix}.jsonl")
    iv_ckpt = RecordCheckpoint(run_dir / f"interventions{suffix}.jsonl", key="_row_key")
    images_dir = run_dir / "intervention_images" if config.save_images else None

    done = completed_ids(run_dir)      # any worker's work counts, not just this one
    todo = targets[~targets["image_id"].astype(str).isin(done)]
    if len(done):
        log.info("resuming: %d already audited, %d to go", len(done), len(todo))

    for n, (_, row) in enumerate(todo.iterrows(), 1):
        if budget is not None and not budget.has_room():
            log.warning("time budget reached after %d/%d targets this pass", n - 1, len(todo))
            break
        with UnitTimer(budget):
            try:
                result = audit_target(row, bank, probe, config, cache, images_dir)
            except Exception as exc:                      # noqa: BLE001
                # A failed intervention is recorded, never silently dropped: a
                # systematically failing category would otherwise vanish from the
                # denominator and inflate the headline.
                log.exception("target %s failed", row["image_id"])
                result = TargetResult({"image_id": str(row["image_id"]),
                                       "category": str(row["category"]),
                                       "defect_type": str(row["defect_type"]),
                                       "status": f"failed: {type(exc).__name__}: {exc}"})
        records_ckpt.append(result.record)
        for i, entry in enumerate(result.interventions):
            iv_ckpt.append({**entry, "_row_key": f"{entry['image_id']}#{i}"})
        records_ckpt.flush()
        iv_ckpt.flush()
        if n % 5 == 0 or n == len(todo):
            log.info("audited %d/%d targets this pass (cache hits %d, misses %d)",
                     n, len(todo), cache.hits, cache.misses)

    records_ckpt.flush()
    iv_ckpt.flush()
    log.info("shard %d done: cache hits %d, newly scored %d", shard, cache.hits, cache.misses)
    return records_ckpt.to_frame(), iv_ckpt.to_frame()


def merge_shards(run_dir: Path, stem: str, key: str = "image_id") -> pd.DataFrame:
    """Union of every worker's checkpoint for one table."""
    frames = []
    for path in sorted(Path(run_dir).glob(f"{stem}*.jsonl")):
        frame = RecordCheckpoint(path, key=key).to_frame()
        if len(frame):
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    return out.drop_duplicates(subset=[key], keep="last").reset_index(drop=True)
