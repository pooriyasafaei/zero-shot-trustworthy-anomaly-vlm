"""Resumability: an interrupted run must continue, not restart."""
from __future__ import annotations

import json

import pandas as pd
import pytest

from tzsad.utils.checkpoint import RecordCheckpoint, StageLedger


def test_checkpoint_reports_pending_items(tmp_path):
    ck = RecordCheckpoint(tmp_path / "r.jsonl")
    wanted = [f"img{i}" for i in range(5)]
    assert ck.pending(wanted) == wanted

    for i in range(2):
        ck.append({"image_id": f"img{i}", "anomaly_score": 0.1 * i})
    ck.flush()
    assert ck.pending(wanted) == ["img2", "img3", "img4"]
    assert ck.n_done == 2


def test_checkpoint_survives_a_torn_final_line(tmp_path):
    """A process killed mid-write leaves a partial line; it must not poison the run."""
    path = tmp_path / "r.jsonl"
    ck = RecordCheckpoint(path)
    for i in range(3):
        ck.append({"image_id": f"img{i}", "v": i})
    ck.flush()
    with path.open("a") as fh:
        fh.write('{"image_id": "img3", "v":')      # torn write

    reopened = RecordCheckpoint(path)
    assert reopened.done_ids() == {"img0", "img1", "img2"}
    assert len(reopened.to_frame()) == 3


def test_checkpoint_is_append_only_across_instances(tmp_path):
    path = tmp_path / "r.jsonl"
    with RecordCheckpoint(path) as a:
        a.append({"image_id": "x", "v": 1})
    with RecordCheckpoint(path) as b:
        b.append({"image_id": "y", "v": 2})
    assert RecordCheckpoint(path).done_ids() == {"x", "y"}


def test_checkpoint_last_write_wins_on_duplicate_ids(tmp_path):
    path = tmp_path / "r.jsonl"
    with RecordCheckpoint(path) as ck:
        ck.append({"image_id": "x", "v": 1})
        ck.append({"image_id": "x", "v": 2})
    df = RecordCheckpoint(path).to_frame()
    assert len(df) == 1 and df.iloc[0]["v"] == 2


def test_checkpoint_flush_every_batches_writes(tmp_path):
    path = tmp_path / "r.jsonl"
    ck = RecordCheckpoint(path, flush_every=3)
    ck.append({"image_id": "a"})
    ck.append({"image_id": "b"})
    assert not path.exists(), "should still be buffered"
    ck.append({"image_id": "c"})
    assert RecordCheckpoint(path).n_done == 3


def test_ledger_marks_and_resumes_cells(tmp_path):
    led = StageLedger(tmp_path / "cells.json")
    assert not led.is_done("gaussian_noise@3")
    led.mark("gaussian_noise@3", {"auroc": 0.61})
    assert StageLedger(tmp_path / "cells.json").is_done("gaussian_noise@3")
    assert StageLedger(tmp_path / "cells.json").payload("gaussian_noise@3")["auroc"] == 0.61


def test_ledger_write_is_atomic(tmp_path):
    """A crash must never leave a half-written ledger that reads as corrupt."""
    path = tmp_path / "cells.json"
    led = StageLedger(path)
    for i in range(5):
        led.mark(f"c{i}", {"i": i})
    assert json.loads(path.read_text())
    assert not path.with_suffix(".tmp").exists()
    assert len(StageLedger(path).done_keys) == 5


def test_ledger_tolerates_an_unreadable_file(tmp_path):
    path = tmp_path / "cells.json"
    path.write_text("{not json")
    assert StageLedger(path).done_keys == []
