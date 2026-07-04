"""Tracker backends: builtin (OpenCV) | litetrack (ONNX) | none.

Selected by config stage4.tracker.
  builtin   : OpenCV CSRT/KCF/MOSSE — no extra weights (DEFAULT)
  litetrack : LiteTrack-B4 via ONNX — REQUIRES stage4.litetrack.onnx_path
  none      : sentinel; Stage 4 will run detection on every frame
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import cv2
import numpy as np

from aero_eyes.types import Box

log = logging.getLogger(__name__)


class Tracker(ABC):
    @abstractmethod
    def init(self, frame_bgr: np.ndarray, box: Box) -> None:
        """Initialize the tracker on the given frame and bounding box."""

    @abstractmethod
    def update(self, frame_bgr: np.ndarray) -> tuple[Box | None, float]:
        """Advance tracker by one frame. Returns (box, confidence) or (None, 0)."""


def _make_cv2_tracker(name: str):
    """Create an OpenCV tracker, trying multiple API paths across versions."""
    candidates = {
        "csrt":  ["cv2.legacy.TrackerCSRT_create",  "cv2.TrackerCSRT_create"],
        "kcf":   ["cv2.legacy.TrackerKCF_create",   "cv2.TrackerKCF_create"],
        "mosse": ["cv2.legacy.TrackerMOSSE_create", "cv2.TrackerMOSSE_create"],
    }
    for dotpath in candidates[name]:
        parts = dotpath.split(".")
        obj = cv2
        try:
            for p in parts[1:]:
                obj = getattr(obj, p)
            return obj()
        except AttributeError:
            continue
    raise RuntimeError(
        f"OpenCV tracker '{name}' not found. "
        "Install opencv-contrib-python:  pip install opencv-contrib-python-headless  "
        "or switch to  stage4.tracker: none  in config."
    )


class BuiltinTracker(Tracker):
    """OpenCV tracking algorithms — no extra weights required."""

    _FACTORY = {
        "csrt":  lambda: _make_cv2_tracker("csrt"),
        "kcf":   lambda: _make_cv2_tracker("kcf"),
        "mosse": lambda: _make_cv2_tracker("mosse"),
    }

    def __init__(self, algorithm: str = "csrt"):
        if algorithm not in self._FACTORY:
            raise ValueError(
                f"Unknown tracker algorithm '{algorithm}'. Choose from {list(self._FACTORY)}."
            )
        self.algorithm = algorithm
        self._tracker = None
        self._initialized = False

    def init(self, frame_bgr: np.ndarray, box: Box) -> None:
        self._tracker = self._FACTORY[self.algorithm]()
        x1, y1 = int(box.x1), int(box.y1)
        w, h = int(box.x2 - box.x1), int(box.y2 - box.y1)
        self._tracker.init(frame_bgr, (x1, y1, w, h))
        self._initialized = True

    def update(self, frame_bgr: np.ndarray) -> tuple[Box | None, float]:
        if not self._initialized or self._tracker is None:
            return None, 0.0
        success, rect = self._tracker.update(frame_bgr)
        if not success:
            return None, 0.0
        x, y, w, h = rect
        box = Box(float(x), float(y), float(x + w), float(y + h))
        # OpenCV trackers don't expose a confidence — use a fixed high value
        # when tracking succeeds; Stage 4 will re-detect if it fails.
        return box, 0.9


class NoneTracker(Tracker):
    """Sentinel tracker — Stage 4 runs full detection on every frame."""

    def init(self, frame_bgr: np.ndarray, box: Box) -> None:
        pass

    def update(self, frame_bgr: np.ndarray) -> tuple[Box | None, float]:
        return None, 0.0


class LiteTrackTracker(Tracker):
    """LiteTrack-B4 ONNX tracker."""

    def __init__(self, onnx_path: str, input_size: int = 256):
        import os
        if not os.path.isfile(onnx_path):
            raise FileNotFoundError(
                f"LiteTrack ONNX weights not found at '{onnx_path}'. "
                "Download LiteTrack-B4 from https://github.com/LitingLin/LiteTrack "
                "and set stage4.litetrack.onnx_path in your config."
            )
        import onnxruntime as ort  # type: ignore
        self.input_size = input_size
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self._sess = ort.InferenceSession(onnx_path, providers=providers)
        self._template: np.ndarray | None = None
        log.info("LiteTrack loaded from %s", onnx_path)

    def init(self, frame_bgr: np.ndarray, box: Box) -> None:
        self._init_box = box
        self._template = self._crop_and_preprocess(frame_bgr, box)

    def update(self, frame_bgr: np.ndarray) -> tuple[Box | None, float]:
        if self._template is None:
            return None, 0.0
        h, w = frame_bgr.shape[:2]
        search = self._preprocess_search(frame_bgr)
        inputs = {
            self._sess.get_inputs()[0].name: self._template,
            self._sess.get_inputs()[1].name: search,
        }
        outputs = self._sess.run(None, inputs)
        # Typical LiteTrack output: [pred_box, score] in normalized coords
        pred_box_norm = outputs[0][0]
        score = float(outputs[1][0]) if len(outputs) > 1 else 0.8
        cx, cy, bw, bh = pred_box_norm[:4]
        x1 = (cx - bw / 2) * w
        y1 = (cy - bh / 2) * h
        x2 = (cx + bw / 2) * w
        y2 = (cy + bh / 2) * h
        return Box(x1, y1, x2, y2), score

    def _crop_and_preprocess(self, frame_bgr: np.ndarray, box: Box) -> np.ndarray:
        s = self.input_size
        x1, y1, x2, y2 = int(box.x1), int(box.y1), int(box.x2), int(box.y2)
        crop = frame_bgr[max(0, y1):y2, max(0, x1):x2]
        if crop.size == 0:
            crop = np.zeros((s, s, 3), dtype=np.uint8)
        crop = cv2.resize(crop, (s, s))
        return self._to_tensor(crop)

    def _preprocess_search(self, frame_bgr: np.ndarray) -> np.ndarray:
        s = self.input_size * 2
        resized = cv2.resize(frame_bgr, (s, s))
        return self._to_tensor(resized)

    @staticmethod
    def _to_tensor(img_bgr: np.ndarray) -> np.ndarray:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_rgb = (img_rgb - mean) / std
        return img_rgb.transpose(2, 0, 1)[None]  # [1, 3, H, W]


def build_tracker(cfg) -> Tracker:
    """Factory: construct the configured tracker."""
    name = cfg.stage4.tracker
    if name == "builtin":
        return BuiltinTracker(algorithm=cfg.stage4.builtin.algorithm)
    if name == "none":
        return NoneTracker()
    if name == "litetrack":
        onnx_path = cfg.stage4.litetrack.onnx_path
        if not onnx_path:
            raise ValueError(
                "stage4.tracker is 'litetrack' but stage4.litetrack.onnx_path is not set. "
                "Download LiteTrack-B4 ONNX weights and set the path in your config, "
                "or switch to tracker: builtin."
            )
        return LiteTrackTracker(
            onnx_path=onnx_path,
            input_size=cfg.stage4.litetrack.input_size,
        )
    raise ValueError(f"Unknown tracker '{name}'. Choose from: builtin, litetrack, none.")
