"""Generate a deterministic tiny synthetic test fixture.

Creates tests/fixtures/synth001/:
  refs/ref_0.jpg, ref_1.jpg, ref_2.jpg  (224x224 coloured rectangle on noise)
  video.mp4                              (~30 frames, object enters/moves/exits)
  gt.json                                (annotations schema; absent for some frames)

Usage:
    python -m scripts.make_synthetic_fixture --out tests/fixtures
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def _make_ref_image(seed: int, size: int = 224) -> np.ndarray:
    """Coloured rectangle on Gaussian noise background."""
    rng = np.random.default_rng(seed)
    bg = rng.integers(40, 80, (size, size, 3), dtype=np.uint8)
    color = [int(c) for c in rng.integers(100, 255, 3)]
    x1 = size // 4
    y1 = size // 4
    x2 = 3 * size // 4
    y2 = 3 * size // 4
    cv2.rectangle(bg, (x1, y1), (x2, y2), color, -1)
    cv2.rectangle(bg, (x1, y1), (x2, y2), (255, 255, 255), 2)
    return bg


def _make_synthetic_video(
    out_path: Path,
    width: int = 640,
    height: int = 480,
    total_frames: int = 30,
    obj_enter: int = 5,
    obj_exit: int = 26,
    seed: int = 42,
) -> list[dict]:
    """Write a synthetic video with a moving rectangle; return GT bboxes."""
    rng = np.random.default_rng(seed)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, 10.0, (width, height))

    obj_w, obj_h = 22, 22
    color = (80, 180, 255)

    x_start = 60
    x_end = width - 80
    y_base = height // 2

    bboxes: list[dict] = []
    present_range = range(obj_enter, obj_exit)

    for fi in range(total_frames):
        frame = rng.integers(20, 60, (height, width, 3), dtype=np.uint8)

        if fi in present_range:
            t = (fi - obj_enter) / max(1, obj_exit - obj_enter - 1)
            cx = int(x_start + t * (x_end - x_start))
            cy = y_base + int(8 * np.sin(t * np.pi * 2))
            x1 = cx - obj_w // 2
            y1 = cy - obj_h // 2
            x2 = x1 + obj_w
            y2 = y1 + obj_h
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
            bboxes.append({"frame": fi, "x1": x1, "y1": y1, "x2": x2, "y2": y2})

        writer.write(frame)

    writer.release()
    return bboxes


def make_fixture(out_dir: str | Path, fixture_id: str = "synth001") -> Path:
    """Create the synthetic fixture directory. Returns its path."""
    out_dir = Path(out_dir)
    fixture_dir = out_dir / fixture_id
    fixture_dir.mkdir(parents=True, exist_ok=True)

    refs_dir = fixture_dir / "refs"
    refs_dir.mkdir(exist_ok=True)
    for i in range(3):
        img = _make_ref_image(seed=42 + i)
        cv2.imwrite(str(refs_dir / f"ref_{i}.jpg"), img)

    video_path = fixture_dir / "video.mp4"
    bboxes = _make_synthetic_video(video_path, seed=99)

    gt_data = [
        {
            "video_id": fixture_id,
            "annotations": [{"bboxes": bboxes}],
        }
    ]
    gt_path = fixture_dir / "gt.json"
    with open(gt_path, "w") as f:
        json.dump(gt_data, f, indent=2)

    print(f"Synthetic fixture created at {fixture_dir}")
    print(f"  3 reference images in {refs_dir}")
    print(f"  video: {video_path} (30 frames, object in frames 5-25)")
    print(f"  GT: {gt_path} ({len(bboxes)} annotated frames)")
    return fixture_dir


def main():
    p = argparse.ArgumentParser(description="Generate synthetic test fixture")
    p.add_argument("--out", default="tests/fixtures", help="Output directory")
    p.add_argument("--id", default="synth001", help="Fixture sample ID")
    args = p.parse_args()
    make_fixture(args.out, args.id)


if __name__ == "__main__":
    main()
