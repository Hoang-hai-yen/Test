"""Estimate stage123_geco2.scale_calibration.expected_object_px per sample
directly from the GT annotations file, instead of eyeballing a frame.

expected_object_px is supposed to be "how big the object looks in the RAW
video frame" -- and the GT bboxes in the global annotations file already
ARE absolute pixel boxes on the raw video frame (box_format: xyxy,
normalized: false -- see configs/config.yaml::data.gt), so we don't need
the actual video files at all: the object's width/height across its GT
boxes for a given video_id IS the ground truth for "how big it looks",
frame by frame.

Reports median (the recommended value -- robust to the near/far outliers
every video has) plus the full spread (p10-p90) so you can judge whether a
single expected_object_px is even a reasonable summary for that video, or
whether the object's apparent size varies too much across the clip for one
canvas-scale to help everywhere.

Usage:
    python -m scripts.estimate_expected_object_px --annotations "annotations (1).json"
    python -m scripts.estimate_expected_object_px --annotations "annotations (1).json" --video BlackBox_0
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def compute_stats(annotations_path: str | Path) -> dict[str, dict]:
    with open(annotations_path, encoding="utf-8") as f:
        data = json.load(f)

    results: dict[str, dict] = {}
    for entry in data:
        video_id = entry["video_id"]
        widths, heights = [], []
        for ann in entry.get("annotations", []):
            for bbox in ann.get("bboxes", []):
                widths.append(bbox["x2"] - bbox["x1"])
                heights.append(bbox["y2"] - bbox["y1"])
        if not widths:
            results[video_id] = {"n_frames": 0}
            continue
        w, h = np.array(widths, dtype=np.float64), np.array(heights, dtype=np.float64)
        results[video_id] = {
            "n_frames": len(widths),
            "median_w": float(np.median(w)), "median_h": float(np.median(h)),
            "p10_w": float(np.percentile(w, 10)), "p90_w": float(np.percentile(w, 90)),
            "p10_h": float(np.percentile(h, 10)), "p90_h": float(np.percentile(h, 90)),
            "min_w": float(w.min()), "max_w": float(w.max()),
            "min_h": float(h.min()), "max_h": float(h.max()),
        }
    return results


def _fmt(v: float) -> str:
    return f"{v:.0f}"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--annotations", required=True, help='e.g. "annotations (1).json"')
    p.add_argument("--video", default=None, help="only report this video_id (omit for all)")
    args = p.parse_args()

    stats = compute_stats(args.annotations)
    video_ids = [args.video] if args.video else sorted(stats.keys())

    for vid in video_ids:
        s = stats.get(vid)
        if s is None:
            print(f"{vid}: not found in {args.annotations}")
            continue
        if s["n_frames"] == 0:
            print(f"{vid}: 0 GT frames -- cannot estimate.")
            continue
        spread_w = s["p90_w"] / max(s["p10_w"], 1e-6)
        spread_h = s["p90_h"] / max(s["p10_h"], 1e-6)
        warn = ""
        if max(spread_w, spread_h) > 3:
            warn = ("  <-- CANH BAO: kich thuoc bien dong >3x giua p10 va p90 "
                    "(vat the luc gan luc xa) -- 1 gia tri scale_calibration duy nhat "
                    "chi dung tot cho 1 phan doan video, khong dung deu ca clip.")
        print(f"{vid}: n_frames_gt={s['n_frames']}")
        print(f"  median (x,y) = [{_fmt(s['median_w'])}, {_fmt(s['median_h'])}]  <- goi y expected_object_px")
        print(f"  p10-p90 w = [{_fmt(s['p10_w'])}, {_fmt(s['p90_w'])}]   "
              f"p10-p90 h = [{_fmt(s['p10_h'])}, {_fmt(s['p90_h'])}]")
        print(f"  min-max  w = [{_fmt(s['min_w'])}, {_fmt(s['max_w'])}]   "
              f"min-max  h = [{_fmt(s['min_h'])}, {_fmt(s['max_h'])}]{warn}")
        print()


if __name__ == "__main__":
    main()
