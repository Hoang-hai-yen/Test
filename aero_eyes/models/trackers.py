"""Tracker backends: builtin (OpenCV) | litetrack (ONNX) | none.

Selected by config stage4.tracker.
  builtin   : OpenCV CSRT/KCF/MOSSE/MIL — no extra weights (DEFAULT)
  litetrack : LiteTrack ONNX — REQUIRES stage4.litetrack.onnx_path_z/onnx_path_x
  none      : sentinel; Stage 4 will run detection on every frame
"""
from __future__ import annotations

import logging
import math
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


def _resolve_cv2_tracker_ctor(name: str):
    """Return the constructor for an OpenCV tracker, trying multiple API paths."""
    candidates = {
        "csrt":  ["cv2.legacy.TrackerCSRT_create",  "cv2.TrackerCSRT_create"],
        "kcf":   ["cv2.legacy.TrackerKCF_create",   "cv2.TrackerKCF_create"],
        "mosse": ["cv2.legacy.TrackerMOSSE_create", "cv2.TrackerMOSSE_create"],
        "mil":   ["cv2.legacy.TrackerMIL_create",   "cv2.TrackerMIL_create"],
    }
    for dotpath in candidates[name]:
        parts = dotpath.split(".")
        obj = cv2
        try:
            for p in parts[1:]:
                obj = getattr(obj, p)
            return obj
        except AttributeError:
            continue
    return None


def _make_cv2_tracker(name: str):
    """Create an OpenCV tracker, falling back to MIL if unavailable."""
    ctor = _resolve_cv2_tracker_ctor(name)
    if ctor is not None:
        return ctor()

    if name != "mil":
        fallback_ctor = _resolve_cv2_tracker_ctor("mil")
        if fallback_ctor is not None:
            log.warning(
                "OpenCV tracker '%s' not available in this OpenCV build "
                "-- falling back to 'mil'.", name,
            )
            return fallback_ctor()

    raise RuntimeError(
        f"OpenCV tracker '{name}' not found. "
        "Install opencv-contrib-python: pip install opencv-contrib-python-headless "
        "or switch to stage4.tracker: none in config."
    )


class BuiltinTracker(Tracker):
    """OpenCV tracking algorithms — no extra weights required."""

    _FACTORY = {
        "csrt":  lambda: _make_cv2_tracker("csrt"),
        "kcf":   lambda: _make_cv2_tracker("kcf"),
        "mosse": lambda: _make_cv2_tracker("mosse"),
        "mil":   lambda: _make_cv2_tracker("mil"),
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
        if w <= 0 or h <= 0:
            log.warning(
                "BuiltinTracker.init: degenerate box (w=%d, h=%d) -- refusing to initialize.", w, h
            )
            self._initialized = False
            return
        try:
            self._tracker.init(frame_bgr, (x1, y1, w, h))
            self._initialized = True
        except cv2.error as e:
            log.warning("BuiltinTracker.init failed (%s) -- treating as tracker-not-active.", e)
            self._initialized = False

    def update(self, frame_bgr: np.ndarray) -> tuple[Box | None, float]:
        if not self._initialized or self._tracker is None:
            return None, 0.0
        try:
            success, rect = self._tracker.update(frame_bgr)
        except cv2.error as e:
            log.warning("BuiltinTracker.update failed (%s) -- treating as tracking lost.", e)
            self._initialized = False
            return None, 0.0
        if not success:
            return None, 0.0
        x, y, w, h = rect
        box = Box(float(x), float(y), float(x + w), float(y + h))
        return box, 0.9


class NoneTracker(Tracker):
    """Sentinel tracker — Stage 4 runs full detection on every frame."""

    def init(self, frame_bgr: np.ndarray, box: Box) -> None:
        pass

    def update(self, frame_bgr: np.ndarray) -> tuple[Box | None, float]:
        return None, 0.0


# ============================================================================
# Thuật toán phụ trợ xử lý khung tìm kiếm và tọa độ cho LiteTrack (ONNX)
# ============================================================================

def _lt_sample_target(
    im: np.ndarray, target_xywh: tuple[float, float, float, float],
    search_area_factor: float, output_sz: int,
) -> tuple[np.ndarray, float]:
    """Cắt vùng ảnh vuông bao quanh mục tiêu với độ phóng search_area_factor."""
    x, y, w, h = target_xywh
    crop_sz = math.ceil(math.sqrt(max(w, 1e-3) * max(h, 1e-3)) * search_area_factor)
    if crop_sz < 1:
        crop_sz = 1

    x1 = round(x + 0.5 * w - crop_sz * 0.5)
    y1 = round(y + 0.5 * h - crop_sz * 0.5)
    x2, y2 = x1 + crop_sz, y1 + crop_sz

    im_h, im_w = im.shape[:2]
    x1_pad, y1_pad = max(0, -x1), max(0, -y1)
    x2_pad, y2_pad = max(x2 - im_w, 0), max(y2 - im_h, 0)

    im_crop = im[y1 + y1_pad:y2 - y2_pad, x1 + x1_pad:x2 - x2_pad]
    if im_crop.size == 0:
        im_crop = np.zeros((1, 1, 3), dtype=im.dtype)
    im_crop_padded = cv2.copyMakeBorder(
        im_crop, y1_pad, y2_pad, x1_pad, x2_pad, cv2.BORDER_CONSTANT, value=0,
    )
    resize_factor = output_sz / crop_sz
    patch = cv2.resize(im_crop_padded, (output_sz, output_sz))
    return patch, resize_factor


def _lt_template_bb_xyxy_norm(
    w: float, h: float, resize_factor: float, output_sz: int,
) -> tuple[float, float, float, float]:
    """Tọa độ normalized xyxy của hộp bao mục tiêu nằm trong patch template."""
    cx = cy = (output_sz - 1) / 2.0
    out_w, out_h = w * resize_factor, h * resize_factor
    x1 = (cx - 0.5 * out_w) / output_sz
    y1 = (cy - 0.5 * out_h) / output_sz
    return x1, y1, x1 + out_w / output_sz, y1 + out_h / output_sz


def _lt_cal_bbox(
    response: np.ndarray, size_map: np.ndarray, offset_map: np.ndarray, feat_size: int,
) -> tuple[tuple[float, float, float, float], float]:
    """Tìm tọa độ argmax score và kích thước từ feature map của LiteTrack."""
    flat_response = response.reshape(-1)
    idx = int(np.argmax(flat_response))
    max_score = float(flat_response[idx])
    idx_y, idx_x = divmod(idx, feat_size)

    size_flat = size_map.reshape(2, -1)
    offset_flat = offset_map.reshape(2, -1)
    w, h = size_flat[:, idx]
    off_x, off_y = offset_flat[:, idx]

    cx = (idx_x + off_x) / feat_size
    cy = (idx_y + off_y) / feat_size
    return (float(cx), float(cy), float(w), float(h)), max_score


def _lt_map_box_back(
    pred_cxcywh_norm: tuple[float, float, float, float], resize_factor: float,
    search_size: int, prev_state_xywh: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    """Chiếu ngược tọa độ từ không gian search patch về tọa độ pixel gốc của frame."""
    cx_n, cy_n, w_n, h_n = pred_cxcywh_norm
    scale = search_size / resize_factor
    cx, cy, w, h = cx_n * scale, cy_n * scale, w_n * scale, h_n * scale
    prev_x, prev_y, prev_w, prev_h = prev_state_xywh
    cx_prev, cy_prev = prev_x + 0.5 * prev_w, prev_y + 0.5 * prev_h
    half_side = 0.5 * scale
    cx_real, cy_real = cx + (cx_prev - half_side), cy + (cy_prev - half_side)
    return cx_real - 0.5 * w, cy_real - 0.5 * h, w, h


def _lt_clip_box(
    box_xywh: tuple[float, float, float, float], img_h: int, img_w: int, margin: float = 10.0,
) -> tuple[float, float, float, float]:
    """Giới hạn bounding box không vượt ra ngoài biên ảnh."""
    x1, y1, w, h = box_xywh
    x2, y2 = x1 + w, y1 + h
    x1 = min(max(0.0, x1), img_w - margin)
    x2 = min(max(margin, x2), img_w)
    y1 = min(max(0.0, y1), img_h - margin)
    y2 = min(max(margin, y2), img_h)
    return x1, y1, max(margin, x2 - x1), max(margin, y2 - y1)


class LiteTrackTracker(Tracker):
    """LiteTrack ONNX tracker gồm 2 đồ thị: Template (z) và Search (x)."""

    def __init__(
        self,
        onnx_path_z: str,
        onnx_path_x: str,
        template_size: int = 128,
        search_size: int = 256,
        template_factor: float = 2.0,
        search_factor: float = 4.0,
        stride: int = 16,
    ):
        import os
        for path in (onnx_path_z, onnx_path_x):
            if not os.path.isfile(path):
                raise FileNotFoundError(
                    f"Không tìm thấy file LiteTrack ONNX tại '{path}'. "
                    "Hãy kiểm tra lại đường dẫn stage4.litetrack.onnx_path_z / onnx_path_x trong config."
                )
        import onnxruntime as ort  # type: ignore
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self._sess_z = ort.InferenceSession(onnx_path_z, providers=providers)
        self._sess_x = ort.InferenceSession(onnx_path_x, providers=providers)
        self.template_size = template_size
        self.search_size = search_size
        self.template_factor = template_factor
        self.search_factor = search_factor
        self.feat_size = search_size // stride
        self._template_feats: np.ndarray | None = None
        self._state_xywh: tuple[float, float, float, float] | None = None
        log.info("LiteTrack loaded (z=%s, x=%s)", onnx_path_z, onnx_path_x)

    def init(self, frame_bgr: np.ndarray, box: Box) -> None:
        w, h = box.x2 - box.x1, box.y2 - box.y1
        if w <= 0 or h <= 0:
            log.warning(
                "LiteTrackTracker.init: Bounding box lỗi (w=%.1f, h=%.1f) -- từ chối khởi tạo.", w, h
            )
            self._template_feats = None
            self._state_xywh = None
            return
        state = (box.x1, box.y1, w, h)
        patch, resize_factor = _lt_sample_target(frame_bgr, state, self.template_factor, self.template_size)
        template = self._to_tensor(patch)
        template_bb = np.array(
            [_lt_template_bb_xyxy_norm(w, h, resize_factor, self.template_size)],
            dtype=np.float32,
        )  # Tensor shape (1, 4)
        outputs = self._sess_z.run(None, {
            self._sess_z.get_inputs()[0].name: template,
            self._sess_z.get_inputs()[1].name: template_bb,
        })
        self._template_feats = outputs[0]
        self._state_xywh = state

    def update(self, frame_bgr: np.ndarray) -> tuple[Box | None, float]:
        if self._template_feats is None or self._state_xywh is None:
            return None, 0.0
        img_h, img_w = frame_bgr.shape[:2]
        patch, resize_factor = _lt_sample_target(frame_bgr, self._state_xywh, self.search_factor, self.search_size)
        search = self._to_tensor(patch)
        response, size_map, offset_map = self._sess_x.run(None, {
            self._sess_x.get_inputs()[0].name: self._template_feats,
            self._sess_x.get_inputs()[1].name: search,
        })
        pred_cxcywh_norm, score = _lt_cal_bbox(response, size_map, offset_map, self.feat_size)
        new_state = _lt_map_box_back(pred_cxcywh_norm, resize_factor, self.search_size, self._state_xywh)
        new_state = _lt_clip_box(new_state, img_h, img_w)
        self._state_xywh = new_state
        x, y, w, h = new_state
        if w <= 0 or h <= 0:
            return None, 0.0
        return Box(x, y, x + w, y + h), score

    @staticmethod
    def _to_tensor(patch_bgr: np.ndarray) -> np.ndarray:
        img_rgb = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_rgb = (img_rgb - mean) / std
        return img_rgb.transpose(2, 0, 1)[None].astype(np.float32)  # [1, 3, H, W]


def build_tracker(cfg) -> Tracker:
    """Factory: khởi tạo tracker tương ứng theo config."""
    name = cfg.stage4.tracker
    if name == "builtin":
        return BuiltinTracker(algorithm=cfg.stage4.builtin.algorithm)
    if name == "none":
        return NoneTracker()
    if name == "litetrack":
        lt_cfg = cfg.stage4.litetrack
        # Đọc 2 đường dẫn z và x
        path_z = getattr(lt_cfg, "onnx_path_z", None)
        path_x = getattr(lt_cfg, "onnx_path_x", None)

        if not path_z or not path_x:
            raise ValueError(
                "stage4.tracker là 'litetrack' nhưng thiếu 'onnx_path_z' hoặc 'onnx_path_x'. "
                "Vui lòng kiểm tra lại cấu hình stage4.litetrack trong config.yaml."
            )
        return LiteTrackTracker(
            onnx_path_z=path_z,
            onnx_path_x=path_x,
            template_size=getattr(lt_cfg, "template_size", 128),
            search_size=getattr(lt_cfg, "search_size", 256),
            template_factor=getattr(lt_cfg, "template_factor", 2.0),
            search_factor=getattr(lt_cfg, "search_factor", 4.0),
            stride=getattr(lt_cfg, "stride", 16),
        )
    raise ValueError(f"Unknown tracker '{name}'. Choose from: builtin, litetrack, none.")