"""Video IO + keyframe sampling helpers."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

log = logging.getLogger(__name__)


def frame_iterator(video_path: str | Path) -> Iterator[tuple[int, np.ndarray]]:
    """Yield (frame_idx, bgr_frame) for every frame in the video."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")
    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            yield frame_idx, frame
            frame_idx += 1
    finally:
        cap.release()


def video_info(video_path: str | Path) -> dict:
    """Return basic metadata: total_frames, fps, width, height."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")
    info = {
        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    cap.release()
    return info


def read_frame(video_path: str | Path, frame_idx: int) -> np.ndarray:
    """Read a single frame by index."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise ValueError(f"Could not read frame {frame_idx} from {video_path}")
    return frame


def keyframe_indices(total_frames: int, interval: int) -> list[int]:
    """Return 0-based keyframe indices spaced `interval` frames apart."""
    if total_frames <= 0 or interval <= 0:
        return []
    return list(range(0, total_frames, interval))


class AnnotatedVideoWriter:
    """Context manager for writing an annotated debug video."""

    def __init__(self, path: str | Path, fps: float, width: int, height: int):
        self.path = str(path)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(self.path, fourcc, fps, (width, height))
        if not self._writer.isOpened():
            log.warning("VideoWriter could not open %s — visualizations disabled", self.path)
            self._writer = None

    def write(self, frame_bgr: np.ndarray) -> None:
        if self._writer is not None:
            self._writer.write(frame_bgr)

    def release(self) -> None:
        if self._writer is not None:
            self._writer.release()

    def __enter__(self) -> "AnnotatedVideoWriter":
        return self

    def __exit__(self, *_) -> None:
        self.release()
