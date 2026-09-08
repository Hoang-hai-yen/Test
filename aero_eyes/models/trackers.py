"""Tracker backends: builtin (OpenCV) | litetrack (ONNX) | none.

Selected by config stage4.tracker.
  builtin   : OpenCV CSRT/KCF/MOSSE — no extra weights (DEFAULT)
  litetrack : LiteTrack ONNX — REQUIRES stage4.litetrack.onnx_path_z/onnx_path_x
  none      : sentinel; Stage 4 will run detection on every frame

LiteTrack needs 2 ONNX graphs, not 1 -- its real forward pass is split into
`forward_z` (encode the template crop ONCE per track init) and `forward`
(template features + search crop -> box, every tracked frame); see
LiteTrack/lib/models/litetrack/litetrack.py. Both are produced from a real
trained checkpoint by LiteTrack/tracking/export_litetrack_onnx.py.

The preprocessing/decoding below (_lt_sample_target, _lt_template_bb_xyxy_norm,
_lt_cal_bbox, _lt_map_box_back, _lt_clip_box) is a from-scratch numpy port of
LiteTrack's own lib/test/tracker/litetrack.py (LiteTrack.initialize/
track_center) -- that file can't actually run as published: it imports
`sample_target` from lib.train.data.processing_utils, a module missing from
the LiteTrack repo (TsingWei/LiteTrack) as cloned. This reconstructs the
same STARK/OSTrack-family crop-and-decode algorithm the whole LiteTrack
model family is trained against.
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
        # Guard against a degenerate (near-zero-area) detection box -- CSRT's
        # init() itself may not raise on this, but its internal search-window
        # bookkeeping can later divide/scale down to an empty region and
        # crash inside a SUBSEQUENT update() call with a raw C++ assertion
        # (cv2.error, not a Python exception any caller here was catching).
        # That crash is unrecoverable and propagates all the way up through
        # run_all(), aborting every remaining sample in the batch -- not just
        # this one frame. Refusing to initialize on an already-degenerate box
        # (and catching cv2.error below as a second line of defense for
        # degeneracy that only manifests after a few tracked frames) turns
        # this into an ordinary "tracking lost" signal, which Stage 4's
        # existing re-detect-on-low-confidence path already handles.
        if w <= 0 or h <= 0:
            log.warning(
                "BuiltinTracker.init: degenerate box (w=%d, h=%d) -- refusing to "
                "initialize, treating as tracker-not-active.", w, h,
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
        # OpenCV trackers don't expose a confidence — use a fixed high value
        # when tracking succeeds; Stage 4 will re-detect if it fails.
        return box, 0.9


class NoneTracker(Tracker):
    """Sentinel tracker — Stage 4 runs full detection on every frame."""

    def init(self, frame_bgr: np.ndarray, box: Box) -> None:
        pass

    def update(self, frame_bgr: np.ndarray) -> tuple[Box | None, float]:
        return None, 0.0


def _lt_sample_target(
    im: np.ndarray, target_xywh: tuple[float, float, float, float],
    search_area_factor: float, output_sz: int,
) -> tuple[np.ndarray, float]:
    """Crop a square region centered on `target_xywh`, padded to
    `search_area_factor` x the box's own size, then resize to `output_sz`.
    Port of LiteTrack/STARK/OSTrack's own sample_target (see module
    docstring for why this can't just be imported). Returns
    (patch_bgr [output_sz, output_sz, 3], resize_factor)."""
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
    """Where the target box sits inside its OWN template crop, normalized
    xyxy -- since _lt_sample_target always centers the crop on the box
    itself, this is a closed form of LiteTrack's
    transform_bbox_to_crop/transform_image_to_crop for that special case
    (box_in == box_extract). LiteTrack.forward_z needs this as
    `template_bb` (MODEL.USE_CLS_TOKEN=True feeds it through a target-token
    embedding)."""
    cx = cy = (output_sz - 1) / 2.0
    out_w, out_h = w * resize_factor, h * resize_factor
    x1 = (cx - 0.5 * out_w) / output_sz
    y1 = (cy - 0.5 * out_h) / output_sz
    return x1, y1, x1 + out_w / output_sz, y1 + out_h / output_sz


def _lt_cal_bbox(
    response: np.ndarray, size_map: np.ndarray, offset_map: np.ndarray, feat_size: int,
) -> tuple[tuple[float, float, float, float], float]:
    """Port of LiteTrack's CenterPredictor.cal_bbox (lib/models/layers/head.py):
    argmax the response map (already hann-windowed inside the exported ONNX
    graph -- see export_litetrack_onnx.py), then read size/offset at that
    same spatial cell. Returns ((cx, cy, w, h) normalized to the search
    canvas, confidence in ~(0,1))."""
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
    """Undo the search-crop resize/recenter to get the predicted box back in
    full-frame pixel coords. Port of LiteTrack.map_box_back."""
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
    """Port of LiteTrack's clip_box (lib/utils/box_ops.py)."""
    x1, y1, w, h = box_xywh
    x2, y2 = x1 + w, y1 + h
    x1 = min(max(0.0, x1), img_w - margin)
    x2 = min(max(margin, x2), img_w)
    y1 = min(max(0.0, y1), img_h - margin)
    y2 = min(max(margin, y2), img_h)
    return x1, y1, max(margin, x2 - x1), max(margin, y2 - y1)


class LiteTrackTracker(Tracker):
    """LiteTrack ONNX tracker -- see module docstring for why this needs 2
    ONNX graphs and a from-scratch numpy pre/postprocessing port instead of
    a single onnxruntime call."""

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
                    f"LiteTrack ONNX weights not found at '{path}'. Export both graphs from a "
                    "trained checkpoint with LiteTrack/tracking/export_litetrack_onnx.py and set "
                    "stage4.litetrack.onnx_path_z / onnx_path_x in your config."
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
                "LiteTrackTracker.init: degenerate box (w=%.1f, h=%.1f) -- "
                "refusing to initialize, treating as tracker-not-active.", w, h,
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
        )  # (1, 4) -- see export_litetrack_onnx.py's dummy_template_bb comment for why not (1,1,4)
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
        return img_rgb.transpose(2, 0, 1)[None].astype(np.float32)  # (1,3,H,W)


def build_tracker(cfg) -> Tracker:
    """Factory: construct the configured tracker."""
    name = cfg.stage4.tracker
    if name == "builtin":
        return BuiltinTracker(algorithm=cfg.stage4.builtin.algorithm)
    if name == "none":
        return NoneTracker()
    if name == "litetrack":
        lt_cfg = cfg.stage4.litetrack
        missing = [f for f in ("onnx_path_z", "onnx_path_x") if not getattr(lt_cfg, f)]
        if missing:
            raise ValueError(
                f"stage4.tracker is 'litetrack' but stage4.litetrack.{missing[0]} is not set. "
                "Export both ONNX graphs from a trained checkpoint with "
                "LiteTrack/tracking/export_litetrack_onnx.py and set the paths in your config, "
                "or switch to tracker: builtin."
            )
        return LiteTrackTracker(
            onnx_path_z=lt_cfg.onnx_path_z,
            onnx_path_x=lt_cfg.onnx_path_x,
            template_size=lt_cfg.template_size,
            search_size=lt_cfg.search_size,
            template_factor=lt_cfg.template_factor,
            search_factor=lt_cfg.search_factor,
            stride=lt_cfg.stride,
        )
    raise ValueError(f"Unknown tracker '{name}'. Choose from: builtin, litetrack, none.")
