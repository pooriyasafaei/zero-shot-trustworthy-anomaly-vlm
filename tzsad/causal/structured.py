"""Structured-generation confirmation arm (CERM-AD v2, P0/P1).

The causal arms use the fast direct-verdict probe. This module runs the
repository's *existing* structured scorer (OBSERVATION / DEFECT_TYPE /
DEFECT_LOCATION / CONFIDENCE / ANSWER) on a list of images, so that

* P0 - every original target image - measures how well the direct probe tracks the
  structured scorer on exactly the audited images, and
* P1 - a handful of casebook targets - checks whether the *direction* of the removal
  and insertion effects agrees between the two scorers, and saves the exact
  generations for qualitative inspection.

Whatever changes in the generated text when the visual evidence changes is a
qualitative observation only. It is not rationale mediation: Stage II
(teacher-forced mediation) is not implemented.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from PIL import Image

from ..scorers.prompts import class_name_for
from ..utils.checkpoint import RecordCheckpoint
from ..utils.logging import get_logger
from .audit import ScoreCache, assign_shards
from .scoring import array_digest

log = get_logger("causal.structured")

#: Roles a P1 casebook target is scored under, in display order.
P1_ROLES = ("original", "defect_removed", "artifact_placebo_best",
            "recipient", "defect_inserted", "sham_best")


def run_structured(items: pd.DataFrame, generate: Callable[[Image.Image, str], dict],
                   cache: ScoreCache, ckpt_path: Path, shard: int = 0,
                   num_shards: int = 1) -> pd.DataFrame:
    """Generate for every row of ``items`` that is not yet checkpointed.

    ``items`` needs ``item_key, image_id, category, role, image_path``. Generation is
    keyed by pixel digest, so an image that appears under two roles (or in P0 and
    P1) is generated once.
    """
    items = items.sort_values("item_key").reset_index(drop=True)
    if num_shards > 1:
        items = items[assign_shards(items["item_key"].tolist(), num_shards) == shard]
    ckpt = RecordCheckpoint(ckpt_path, key="item_key")
    done = completed_structured(Path(ckpt_path).parent)
    todo = items[~items["item_key"].isin(done)]
    log.info("structured shard %d/%d: %d items, %d to generate", shard, num_shards,
             len(items), len(todo))
    for n, row in enumerate(todo.itertuples(index=False), 1):
        with Image.open(row.image_path) as im:
            image = np.asarray(im.convert("RGB"), dtype=np.uint8)
        class_name = class_name_for(row.category)
        digest = array_digest(image)
        key = cache.key(digest, class_name)
        result = cache.get(key)
        hit = result is not None
        if result is None:
            try:
                result = generate(Image.fromarray(image), class_name)
            except Exception as exc:                     # noqa: BLE001
                log.exception("structured generation failed for %s", row.item_key)
                result = {"parse_ok": False, "status": f"failed: {type(exc).__name__}: {exc}"}
            else:
                cache.put(key, result)
        ckpt.append({"item_key": row.item_key, "image_id": row.image_id,
                     "category": row.category, "role": row.role,
                     "image_path": row.image_path, "image_digest": digest,
                     "cache_hit": hit, **{f"structured_{k}": v for k, v in result.items()}})
        ckpt.flush()
        if n % 5 == 0 or n == len(todo):
            log.info("structured: %d/%d generated this pass", n, len(todo))
    return ckpt.to_frame()


def completed_structured(run_dir: Path) -> set[str]:
    """Items generated successfully by any worker. Failed items are retried on resume
    (their failure row stays in the checkpoint until a success supersedes it)."""
    done: set[str] = set()
    for path in Path(run_dir).glob("structured_records*.jsonl"):
        frame = RecordCheckpoint(path, key="item_key").to_frame()
        if not len(frame):
            continue
        if "structured_status" in frame:
            frame = frame[frame["structured_status"].isna()]
        done |= set(frame["item_key"].astype(str))
    return done


def p0_items(targets: pd.DataFrame) -> pd.DataFrame:
    """One structured generation per original target image."""
    return pd.DataFrame({
        "item_key": targets["image_id"].astype(str) + "#original",
        "image_id": targets["image_id"].astype(str),
        "category": targets["category"].astype(str),
        "role": "original",
        "image_path": targets["path"].astype(str),
    })


def structured_signature(model_id: str, max_pixels: int, dtype: str,
                         max_new_tokens: int) -> str:
    """Cache signature for structured generations (greedy, so fully determined)."""
    from .scoring import scoring_signature

    return scoring_signature(model_id, "generated_verdict",
                             f"generated-verdict-v1/mnt{int(max_new_tokens)}",
                             max_pixels, dtype)
