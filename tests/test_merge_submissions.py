"""Unit tests for merge_submissions -- upserts by video_id across each
sample's own submission.json into one aggregate file, re-run on every
run_all.py invocation (unlike every other artifact here, which is written
once and never re-read). Covers the corrupted-aggregate recovery path and
the atomic-write behavior added alongside it."""
from __future__ import annotations

import json
from pathlib import Path

from aero_eyes.utils.io import merge_submissions


def _write_submission(work_dir: Path, sample_id: str, video_id: str, filename: str = "submission.json") -> None:
    sdir = work_dir / sample_id
    sdir.mkdir(parents=True, exist_ok=True)
    (sdir / filename).write_text(json.dumps([
        {"video_id": video_id, "annotations": [{"bboxes": [{"frame": 0, "x1": 0, "y1": 0, "x2": 10, "y2": 10}]}]},
    ]))


def test_merge_submissions_creates_new_aggregate(tmp_path):
    _write_submission(tmp_path, "A_0", "A_0")
    _write_submission(tmp_path, "B_0", "B_0")
    out_path = tmp_path / "submission_all.json"

    merge_submissions(tmp_path, ["A_0", "B_0"], "submission.json", out_path)

    merged = json.loads(out_path.read_text())
    assert {e["video_id"] for e in merged} == {"A_0", "B_0"}
    assert not (tmp_path / "submission_all.json.tmp").exists()


def test_merge_submissions_upserts_without_dropping_other_samples(tmp_path):
    _write_submission(tmp_path, "A_0", "A_0")
    _write_submission(tmp_path, "B_0", "B_0")
    out_path = tmp_path / "submission_all.json"
    merge_submissions(tmp_path, ["A_0", "B_0"], "submission.json", out_path)

    # Second run only touches A_0 (e.g. --sample A_0) -- B_0's entry from
    # the first run must still be present in the aggregate afterward.
    merge_submissions(tmp_path, ["A_0"], "submission.json", out_path)

    merged = json.loads(out_path.read_text())
    assert {e["video_id"] for e in merged} == {"A_0", "B_0"}


def test_merge_submissions_recovers_from_corrupted_aggregate(tmp_path, caplog):
    """A pre-existing corrupted out_path (e.g. from a process killed
    mid-write in a previous run) must not crash the whole pipeline -- log a
    warning and rebuild from this run's own sample_ids instead."""
    _write_submission(tmp_path, "A_0", "A_0")
    out_path = tmp_path / "submission_all.json"
    out_path.write_text('[{"video_id": "A_0", "annota')  # truncated/corrupted

    import logging
    with caplog.at_level(logging.WARNING):
        merge_submissions(tmp_path, ["A_0"], "submission.json", out_path)

    assert any("corrupted" in r.message for r in caplog.records)
    merged = json.loads(out_path.read_text())
    assert {e["video_id"] for e in merged} == {"A_0"}


def test_merge_submissions_skips_missing_samples(tmp_path, caplog):
    _write_submission(tmp_path, "A_0", "A_0")
    out_path = tmp_path / "submission_all.json"

    import logging
    with caplog.at_level(logging.WARNING):
        merge_submissions(tmp_path, ["A_0", "B_0_never_ran"], "submission.json", out_path)

    merged = json.loads(out_path.read_text())
    assert {e["video_id"] for e in merged} == {"A_0"}
    assert any("B_0_never_ran" in r.message for r in caplog.records)
