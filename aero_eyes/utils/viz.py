"""Per-stage visualization overlays.

Gated by config runtime.save_visualizations.
Written under <work_dir>/<sample>/viz/<stage>/.
"""
from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from aero_eyes.types import Box

log = logging.getLogger(__name__)

_COLORS = {
    "detect": (0, 255, 0),   # green
    "track": (0, 165, 255),  # orange
    "gt": (0, 0, 255),       # red
    "tile": (200, 200, 0),   # cyan-ish
}


def draw_box(img: np.ndarray, box: Box, label: str = "", color=(0, 255, 0), thickness: int = 2) -> None:
    cv2.rectangle(img, (int(box.x1), int(box.y1)), (int(box.x2), int(box.y2)), color, thickness)
    if label:
        cv2.putText(img, label, (int(box.x1), max(0, int(box.y1) - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)


def save_stage1_refs(ref_imgs: list[np.ndarray], masks: list[np.ndarray],
                     out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, (img, mask) in enumerate(zip(ref_imgs, masks)):
        overlay = img.copy()
        overlay[~mask] = overlay[~mask] // 2
        cv2.imwrite(str(out_dir / f"ref_{i}_masked.jpg"), overlay)


def save_stage2_keyframe(frame: np.ndarray, boxes: list[Box],
                         tiles: list | None, frame_idx: int, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    vis = frame.copy()
    if tiles:
        for tile in tiles:
            cv2.rectangle(vis, (tile.x1, tile.y1), (tile.x2, tile.y2), _COLORS["tile"], 1)
    for b in boxes:
        draw_box(vis, b, f"{b.score:.2f}", _COLORS["detect"])
    cv2.imwrite(str(out_dir / f"frame_{frame_idx:06d}.jpg"), vis)


def save_stage3_detections(frame: np.ndarray, boxes: list[Box], sims: list[float],
                            frame_idx: int, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    vis = frame.copy()
    for b, s in zip(boxes, sims):
        draw_box(vis, b, f"{s:.2f}", _COLORS["detect"])
    cv2.imwrite(str(out_dir / f"frame_{frame_idx:06d}_det.jpg"), vis)


def draw_frame_annotation(frame: np.ndarray, box: Box | None, source: str,
                           frame_idx: int) -> np.ndarray:
    vis = frame.copy()
    if box is not None:
        color = _COLORS.get(source, (255, 255, 255))
        draw_box(vis, box, f"{source} #{frame_idx}", color)
    return vis


def save_stage5_timeline(tube: dict[int, Box], total_frames: int, out_path: Path) -> None:
    """Draw a horizontal timeline strip showing present/absent frames."""
    strip_w = min(total_frames, 2000)
    strip_h = 32
    strip = np.zeros((strip_h, strip_w, 3), dtype=np.uint8)
    for fi, _ in tube.items():
        x = int(fi * strip_w / max(total_frames, 1))
        cv2.line(strip, (x, 0), (x, strip_h), (0, 255, 0), 1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), strip)
