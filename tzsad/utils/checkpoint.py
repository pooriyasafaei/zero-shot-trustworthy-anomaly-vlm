"""Append-only per-item checkpoints, so an interrupted run resumes instead of restarting.

Long GPU runs die for reasons that have nothing to do with the code: a session
expires, a notebook is closed, a machine is preempted. A scorer that only writes
its records at the end throws away everything it computed. This module gives every
stage a cheap way to record "item X is done" as it goes.

JSON Lines rather than Parquet: appending one row must be crash-safe and must not
rewrite the file. A truncated final line (the process died mid-write) is detected
and dropped on read rather than poisoning the run.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

import pandas as pd

from .logging import get_logger

log = get_logger("utils.checkpoint")


class RecordCheckpoint:
    """An append-only JSONL store keyed by a unique id column.

    Usage::

        ckpt = RecordCheckpoint(run_dir / "records.jsonl", key="image_id")
        todo = ckpt.pending(index["image_id"])
        for image_id in todo:
            ckpt.append(score_one(image_id))
        records = ckpt.to_frame()
    """

    def __init__(self, path: str | Path, key: str = "image_id", flush_every: int = 1) -> None:
        self.path = Path(path)
        self.key = key
        self.flush_every = max(int(flush_every), 1)
        self._buffer: list[dict[str, Any]] = []
        self.path.parent.mkdir(parents=True, exist_ok=True)

    # -- reading -----------------------------------------------------------
    def _iter_rows(self) -> Iterator[dict[str, Any]]:
        if not self.path.exists():
            return
        with self.path.open("r") as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    # Only the final line can be torn, and only if the process died
                    # mid-write. Anything earlier means real corruption.
                    log.warning("dropping unreadable checkpoint line %d in %s", lineno, self.path)

    def done_ids(self) -> set[str]:
        """Ids already recorded."""
        return {str(r[self.key]) for r in self._iter_rows() if self.key in r}

    def pending(self, wanted: Iterable[str]) -> list[str]:
        """Requested ids that are not yet recorded, order preserved."""
        done = self.done_ids()
        return [str(w) for w in wanted if str(w) not in done]

    def to_frame(self) -> pd.DataFrame:
        """All recorded rows as a DataFrame, de-duplicated on the key (last wins)."""
        rows = list(self._iter_rows())
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        if self.key in df.columns:
            df = df.drop_duplicates(subset=[self.key], keep="last").reset_index(drop=True)
        return df

    # -- writing -----------------------------------------------------------
    def append(self, row: Mapping[str, Any]) -> None:
        """Record one item. Flushed to disk according to ``flush_every``."""
        self._buffer.append(dict(row))
        if len(self._buffer) >= self.flush_every:
            self.flush()

    def flush(self) -> None:
        """Write buffered rows and fsync, so a hard kill cannot lose them."""
        if not self._buffer:
            return
        with self.path.open("a") as fh:
            for row in self._buffer:
                fh.write(json.dumps(row, default=_jsonable) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        self._buffer.clear()

    def __enter__(self) -> "RecordCheckpoint":
        return self

    def __exit__(self, *exc: object) -> None:
        self.flush()

    @property
    def n_done(self) -> int:
        """How many items are already recorded."""
        return len(self.done_ids())


def _jsonable(obj: Any) -> Any:
    if hasattr(obj, "item"):          # numpy scalar
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


class StageLedger:
    """Which coarse stages/cells of a multi-part run have completed.

    Used by the corruption sweep, whose unit of work is a (corruption, severity)
    cell rather than an image. Small enough to rewrite whole on each update.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._done: dict[str, Any] = {}
        if self.path.exists():
            try:
                self._done = json.loads(self.path.read_text())
            except json.JSONDecodeError:
                log.warning("unreadable ledger %s; starting fresh", self.path)

    def is_done(self, key: str) -> bool:
        """Whether this stage/cell already completed."""
        return key in self._done

    def mark(self, key: str, payload: Mapping[str, Any] | None = None) -> None:
        """Record a stage/cell as complete, with optional summary payload."""
        self._done[key] = dict(payload or {})
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._done, indent=1, default=_jsonable))
        tmp.replace(self.path)        # atomic: a crash never leaves a half-written ledger

    def payload(self, key: str) -> dict[str, Any]:
        """Stored summary for a completed stage/cell."""
        return dict(self._done.get(key, {}))

    @property
    def done_keys(self) -> list[str]:
        """Completed keys, in insertion order."""
        return list(self._done)
