"""Artifact (de)serialization + GT/submission format conversion.

CRITICAL: all internal boxes are absolute xyxy floats with 0-based frame
indices. Conversion to/from the user-CONFIRMED GT & submission schema
happens ONLY here (the single I/O boundary).

Confirmed GT/submission schema:
  [{"video_id": str,
    "annotations": [{"bboxes": [{"frame": int, "x1": int, "y1": int,
                                  "x2": int, "y2": int}]}]}]
  - box format: xyxy, absolute pixels, 0-based frame indices
  - absent frames are omitted (not present in bboxes list)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from aero_eyes.types import Box, Detection, SpatioTemporalTube

log = logging.getLogger(__name__)

SCHEMA_VERSION = "1.0"


# ---------------------------------------------------------------------------
# GT loading
# ---------------------------------------------------------------------------

def load_gt(annotations_path: str | Path, video_id: str, cfg=None) -> dict[int, Box]:
    """Load ground truth for a single video_id from the global annotations file.

    Returns dict[frame_idx -> Box] (absent frames simply not present).
    """
    with open(annotations_path) as f:
        data = json.load(f)

    for entry in data:
        if entry["video_id"] == video_id:
            gt: dict[int, Box] = {}
            # annotations is a list; we use the first (one object per video)
            for ann in entry.get("annotations", []):
                for bbox in ann.get("bboxes", []):
                    frame = bbox["frame"]
                    box = Box(
                        x1=float(bbox["x1"]),
                        y1=float(bbox["y1"]),
                        x2=float(bbox["x2"]),
                        y2=float(bbox["y2"]),
                    )
                    gt[frame] = box
            return gt

    raise KeyError(f"video_id '{video_id}' not found in {annotations_path}")


def list_video_ids(annotations_path: str | Path) -> list[str]:
    """Return all video_ids present in the global annotations file."""
    with open(annotations_path) as f:
        data = json.load(f)
    return [e["video_id"] for e in data]


# ---------------------------------------------------------------------------
# Submission writing
# ---------------------------------------------------------------------------

def write_submission(
    tube: dict[int, Box],
    video_id: str,
    path: str | Path,
    cfg=None,
) -> None:
    """Write a single video's tube to submission JSON (same schema as GT annotations)."""
    bboxes = []
    for frame_idx in sorted(tube.keys()):
        box = tube[frame_idx]
        bboxes.append({
            "frame": frame_idx,
            "x1": int(round(box.x1)),
            "y1": int(round(box.y1)),
            "x2": int(round(box.x2)),
            "y2": int(round(box.y2)),
        })

    payload = [{
        "video_id": video_id,
        "annotations": [{"bboxes": bboxes}],
    }]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    log.info("Wrote submission: %s (%d frames)", path, len(bboxes))


def append_submission(
    tube: dict[int, Box],
    video_id: str,
    path: str | Path,
) -> None:
    """Append a video's tube to an existing (or new) submission file."""
    path = Path(path)
    if path.exists():
        with open(path) as f:
            data = json.load(f)
        # Remove any existing entry for this video_id
        data = [e for e in data if e["video_id"] != video_id]
    else:
        data = []
        path.parent.mkdir(parents=True, exist_ok=True)

    bboxes = [
        {"frame": fi, "x1": int(round(b.x1)), "y1": int(round(b.y1)),
         "x2": int(round(b.x2)), "y2": int(round(b.y2))}
        for fi, b in sorted(tube.items())
    ]
    data.append({"video_id": video_id, "annotations": [{"bboxes": bboxes}]})
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def merge_submissions(
    work_dir: str | Path,
    sample_ids: list[str] | None,
    submission_filename: str,
    out_path: str | Path,
) -> Path:
    """Merge every sample's own <work_dir>/<sample_id>/<submission_filename>
    (as written by Stage 5's write_submission) into one combined file with
    the same schema, upserting by video_id -- re-running a subset of
    samples only touches their own entries, matching append_submission's
    semantics. Samples with no submission file yet are skipped (logged),
    not treated as an error, since Stage 5 may not have run for all of them.
    """
    work_dir = Path(work_dir)
    if sample_ids is None:
        sample_ids = sorted(d.name for d in work_dir.iterdir() if d.is_dir())

    out_path = Path(out_path)
    if out_path.exists():
        with open(out_path) as f:
            merged = {e["video_id"]: e for e in json.load(f)}
    else:
        merged = {}
        out_path.parent.mkdir(parents=True, exist_ok=True)

    missing: list[str] = []
    for sid in sample_ids:
        sub_path = work_dir / sid / submission_filename
        if not sub_path.exists():
            missing.append(sid)
            continue
        with open(sub_path) as f:
            for entry in json.load(f):
                merged[entry["video_id"]] = entry

    with open(out_path, "w") as f:
        json.dump(list(merged.values()), f, indent=2)

    if missing:
        log.warning("merge_submissions: %d sample(s) had no submission file yet: %s",
                     len(missing), missing)
    log.info("merge_submissions: wrote %d video(s) -> %s", len(merged), out_path)
    return out_path


# ---------------------------------------------------------------------------
# prototype.npz
# ---------------------------------------------------------------------------

def write_prototype(
    prototype: np.ndarray,
    meta: dict,
    per_ref_features: list[np.ndarray] | None,
    path: str | Path,
) -> None:
    """Save prototype vector + optional per-ref feature matrix + metadata."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    save_kwargs: dict = {
        "prototype": prototype,
        "meta_json": np.bytes_(json.dumps({**meta, "schema_version": SCHEMA_VERSION})),
    }
    if per_ref_features:
        for i, feat in enumerate(per_ref_features):
            save_kwargs[f"ref_{i}"] = feat
    np.savez_compressed(str(path), **save_kwargs)
    log.debug("Wrote prototype: %s shape=%s", path, prototype.shape)


def read_prototype(path: str | Path) -> tuple[np.ndarray, dict, list[np.ndarray]]:
    """Load prototype. Returns (prototype, meta_dict, per_ref_features)."""
    data = np.load(str(path), allow_pickle=False)
    prototype = data["prototype"]
    meta = json.loads(data["meta_json"].item().decode())
    per_ref: list[np.ndarray] = []
    i = 0
    while f"ref_{i}" in data:
        per_ref.append(data[f"ref_{i}"])
        i += 1
    return prototype, meta, per_ref


# ---------------------------------------------------------------------------
# candidates.json
# ---------------------------------------------------------------------------

def write_candidates(candidates: dict[int, list[Detection]], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "frames": {
            str(fi): [d.to_dict() for d in dets]
            for fi, dets in candidates.items()
        },
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    log.debug("Wrote candidates: %s (%d keyframes)", path, len(candidates))


def read_candidates(path: str | Path) -> dict[int, list[Detection]]:
    with open(path) as f:
        payload = json.load(f)
    return {
        int(fi): [Detection.from_dict(d) for d in dets]
        for fi, dets in payload["frames"].items()
    }


# ---------------------------------------------------------------------------
# detections.json
# ---------------------------------------------------------------------------

def write_detections(
    detections: dict[int, list[Detection]],
    path: str | Path,
    threshold: float | None = None,
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "frames": {
            str(fi): [d.to_dict() for d in dets]
            for fi, dets in detections.items()
        },
        # The effective match threshold Stage 3 actually used (adaptive
        # z-score threshold when enabled, else the fixed match_threshold).
        # Stage 4 re-detection must reuse this exact value -- falling back
        # to the fixed config default there would silently ignore adaptive
        # tuning and reject nearly every re-detect attempt.
        "threshold": threshold,
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    log.debug("Wrote detections: %s (%d frames)", path, len(detections))


def read_detections(path: str | Path) -> dict[int, list[Detection]]:
    with open(path) as f:
        payload = json.load(f)
    return {
        int(fi): [Detection.from_dict(d) for d in dets]
        for fi, dets in payload["frames"].items()
    }


def read_detections_threshold(path: str | Path) -> float | None:
    """Read the effective match threshold Stage 3 recorded, if any."""
    with open(path) as f:
        payload = json.load(f)
    return payload.get("threshold")


# ---------------------------------------------------------------------------
# tracks.json
# ---------------------------------------------------------------------------

def write_tracks(tracks: dict[int, Box | None], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "frames": {
            str(fi): (box.to_dict() if box is not None else None)
            for fi, box in tracks.items()
        },
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    log.debug("Wrote tracks: %s (%d frames)", path, len(tracks))


def read_tracks(path: str | Path) -> dict[int, Box | None]:
    with open(path) as f:
        payload = json.load(f)
    return {
        int(fi): (Box.from_dict(v) if v is not None else None)
        for fi, v in payload["frames"].items()
    }


# ---------------------------------------------------------------------------
# tube.json
# ---------------------------------------------------------------------------

def write_tube(tube: dict[int, Box], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "frames": {str(fi): box.to_dict() for fi, box in sorted(tube.items())},
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    log.debug("Wrote tube: %s (%d frames)", path, len(tube))


def read_tube(path: str | Path) -> dict[int, Box]:
    with open(path) as f:
        payload = json.load(f)
    return {int(fi): Box.from_dict(v) for fi, v in payload["frames"].items()}
