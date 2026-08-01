"""Convert CVAT XML 1.1 (single-object track, rectangle shapes) exports into
the aero_eyes ground-truth schema and upsert them into the global
annotations file.

aero_eyes GT schema (see aero_eyes/utils/io.py, docs/HUONG_DAN_SU_DUNG.md):
    [{"video_id": str,
      "annotations": [{"bboxes": [{"frame": int, "x1": int, "y1": int,
                                    "x2": int, "y2": int}, ...]}]}]
    - xyxy, absolute pixels, 0-based frames, absent frames simply omitted.

CVAT box semantics handled here:
  - Every <box frame=".." xtl=".." ytl=".." xbr=".." ybr=".." outside="0|1">
    inside a <track> is one frame's box.
  - outside="1" marks the frame where the object leaves the scene -- CVAT
    still records a (stale) box there, but the object is NOT present, so
    that frame is dropped (matches aero_eyes' "absent = omit" rule).
  - Coordinates are rounded to the nearest int and clipped to the video's
    <original_size> (defensive -- CVAT interpolation can produce
    coordinates a fraction of a pixel outside the frame at the edges).

Usage:
    python -m scripts.convert_cvat_xml_to_annotations \\
        --xml BlackBox_0.xml --out "annotations (1).json"

    # video_id defaults to the XML filename stem (BlackBox_0.xml -> "BlackBox_0");
    # override explicitly, or convert several files in one call:
    python -m scripts.convert_cvat_xml_to_annotations \\
        --xml BlackBox_0.xml Person_1.xml --out "annotations (1).json"
"""
from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path


def _select_track(root: ET.Element, xml_path: Path, track_id: str | None) -> ET.Element:
    tracks = root.findall("track")
    if not tracks:
        raise ValueError(f"{xml_path}: no <track> elements found.")
    if track_id is not None:
        for t in tracks:
            if t.get("id") == track_id:
                return t
        raise ValueError(
            f"{xml_path}: no <track id=\"{track_id}\"> found. "
            f"Available ids: {[t.get('id') for t in tracks]}"
        )
    if len(tracks) > 1:
        raise ValueError(
            f"{xml_path}: found {len(tracks)} tracks "
            f"(ids={[t.get('id') for t in tracks]}), but aero_eyes expects "
            "exactly one object per video. Pass --track-id to pick one."
        )
    return tracks[0]


def convert_track_to_bboxes(track: ET.Element, width: float | None, height: float | None) -> list[dict]:
    """Extract one aero_eyes 'bboxes' list from a single CVAT <track>."""
    by_frame: dict[int, dict] = {}
    for box in track.findall("box"):
        if box.get("outside") == "1":
            continue  # object not present at this frame
        frame = int(box.get("frame"))
        x1 = float(box.get("xtl"))
        y1 = float(box.get("ytl"))
        x2 = float(box.get("xbr"))
        y2 = float(box.get("ybr"))
        if width is not None:
            x1 = min(max(x1, 0.0), width)
            x2 = min(max(x2, 0.0), width)
        if height is not None:
            y1 = min(max(y1, 0.0), height)
            y2 = min(max(y2, 0.0), height)
        by_frame[frame] = {
            "frame": frame,
            "x1": int(round(x1)), "y1": int(round(y1)),
            "x2": int(round(x2)), "y2": int(round(y2)),
        }
    return [by_frame[f] for f in sorted(by_frame)]


def convert_file(xml_path: str | Path, video_id: str | None = None, track_id: str | None = None) -> dict:
    """Parse one CVAT XML export into a single aero_eyes annotation entry."""
    xml_path = Path(xml_path)
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size_el = root.find("./meta/original_size")
    width = float(size_el.findtext("width")) if size_el is not None else None
    height = float(size_el.findtext("height")) if size_el is not None else None

    track = _select_track(root, xml_path, track_id)
    bboxes = convert_track_to_bboxes(track, width, height)
    if not bboxes:
        raise ValueError(f"{xml_path}: track has 0 non-'outside' boxes -- nothing to convert.")

    return {
        "video_id": video_id or xml_path.stem,
        "annotations": [{"bboxes": bboxes}],
    }


def upsert_entries(out_path: str | Path, entries: list[dict]) -> None:
    """Merge entries into the global annotations file, replacing any
    existing entry with the same video_id (same upsert semantics as
    aero_eyes.utils.io.append_submission).
    """
    out_path = Path(out_path)
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = []
        out_path.parent.mkdir(parents=True, exist_ok=True)

    new_ids = {e["video_id"] for e in entries}
    data = [e for e in data if e["video_id"] not in new_ids]
    data.extend(entries)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def main():
    p = argparse.ArgumentParser(description="Convert CVAT XML 1.1 track export(s) to aero_eyes GT schema")
    p.add_argument("--xml", nargs="+", required=True, help="One or more CVAT XML export files")
    p.add_argument("--out", default="annotations (1).json",
                    help="Global annotations file to upsert into (created if missing)")
    p.add_argument("--video-id", default=None,
                    help="Override video_id (only valid with a single --xml file; "
                         "default = XML filename stem)")
    p.add_argument("--track-id", default=None,
                    help="CVAT <track id=...> to use if the XML has more than one track")
    args = p.parse_args()

    if args.video_id and len(args.xml) > 1:
        raise SystemExit("--video-id can only be used with a single --xml file")

    entries = []
    for xml_file in args.xml:
        entry = convert_file(xml_file, video_id=args.video_id, track_id=args.track_id)
        n = len(entry["annotations"][0]["bboxes"])
        print(f"{xml_file} -> video_id={entry['video_id']!r} ({n} annotated frames)")
        entries.append(entry)

    upsert_entries(args.out, entries)
    print(f"Wrote {len(entries)} video(s) into {args.out}")


if __name__ == "__main__":
    main()
