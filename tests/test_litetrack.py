"""LiteTrack tracker tests -- no onnxruntime, no weights, no GPU needed.

Covers: config validation (both ONNX graphs required), the from-scratch numpy
port of LiteTrack's crop/decode math (`_lt_*` in aero_eyes/models/trackers.py),
and `LiteTrackTracker` init/update driven by fake ONNX sessions.
"""
from __future__ import annotations

import math
from pathlib import Path

import pytest

np = pytest.importorskip("numpy", reason="numpy not importable", exc_type=ImportError)
pytest.importorskip("cv2", reason="opencv-python not importable", exc_type=ImportError)

CONFIG_YAML = Path(__file__).resolve().parents[1] / "configs" / "config.yaml"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _load(overrides):
    from aero_eyes.config import load_config
    return load_config(str(CONFIG_YAML), overrides)


def test_default_config_does_not_require_litetrack_weights():
    cfg = _load([])
    assert cfg.stage4.tracker == "builtin"
    lt = cfg.stage4.litetrack
    assert lt.onnx_path_z is None and lt.onnx_path_x is None
    assert (lt.template_size, lt.search_size, lt.stride) == (128, 256, 16)
    assert (lt.template_factor, lt.search_factor) == (2.0, 4.0)


@pytest.mark.parametrize("z, x, missing", [
    (None, None, "onnx_path_z"),
    ("a_z.onnx", None, "onnx_path_x"),
    (None, "a_x.onnx", "onnx_path_z"),
])
def test_litetrack_requires_both_onnx_paths(z, x, missing):
    overrides = ["stage4.tracker=litetrack"]
    if z:
        overrides.append(f"stage4.litetrack.onnx_path_z={z}")
    if x:
        overrides.append(f"stage4.litetrack.onnx_path_x={x}")
    with pytest.raises(Exception) as exc_info:
        _load(overrides)
    assert missing in str(exc_info.value)


def test_litetrack_config_valid_with_both_paths():
    cfg = _load([
        "stage4.tracker=litetrack",
        "stage4.litetrack.onnx_path_z=/w/lt_z.onnx",
        "stage4.litetrack.onnx_path_x=/w/lt_x.onnx",
    ])
    assert cfg.stage4.litetrack.onnx_path_z == "/w/lt_z.onnx"
    assert cfg.stage4.litetrack.onnx_path_x == "/w/lt_x.onnx"


def test_build_tracker_litetrack_missing_file_raises_actionable_error(tmp_path):
    from aero_eyes.models.trackers import build_tracker
    cfg = _load([
        "stage4.tracker=litetrack",
        f"stage4.litetrack.onnx_path_z={tmp_path / 'nope_z.onnx'}",
        f"stage4.litetrack.onnx_path_x={tmp_path / 'nope_x.onnx'}",
    ])
    with pytest.raises(FileNotFoundError, match="export_litetrack_onnx"):
        build_tracker(cfg)


# ---------------------------------------------------------------------------
# Crop / decode math
# ---------------------------------------------------------------------------

def test_sample_target_shape_and_resize_factor():
    from aero_eyes.models.trackers import _lt_sample_target
    im = np.full((480, 640, 3), 127, dtype=np.uint8)
    patch, rf = _lt_sample_target(im, (300.0, 200.0, 40.0, 30.0), 4.0, 256)
    assert patch.shape == (256, 256, 3)
    crop_sz = math.ceil(math.sqrt(40 * 30) * 4.0)
    assert rf == pytest.approx(256 / crop_sz)


def test_sample_target_pads_outside_frame_with_zeros():
    from aero_eyes.models.trackers import _lt_sample_target
    im = np.full((100, 100, 3), 200, dtype=np.uint8)
    patch, _ = _lt_sample_target(im, (0.0, 0.0, 10.0, 10.0), 4.0, 64)  # box in a corner
    assert patch.shape == (64, 64, 3)
    assert patch[0, 0].sum() == 0          # out-of-frame corner is zero padding
    assert patch[40, 40].sum() > 0         # in-frame region kept


def test_sample_target_survives_box_entirely_outside_frame():
    from aero_eyes.models.trackers import _lt_sample_target
    im = np.full((100, 100, 3), 200, dtype=np.uint8)
    patch, _ = _lt_sample_target(im, (5000.0, 5000.0, 10.0, 10.0), 2.0, 32)
    assert patch.shape == (32, 32, 3)


def test_template_bb_is_centered_and_normalized():
    from aero_eyes.models.trackers import _lt_template_bb_xyxy_norm
    x1, y1, x2, y2 = _lt_template_bb_xyxy_norm(w=40.0, h=30.0, resize_factor=1.0, output_sz=128)
    assert (x1 + x2) / 2 == pytest.approx((y1 + y2) / 2)     # centered
    assert (x2 - x1) == pytest.approx(40 / 128)
    assert (y2 - y1) == pytest.approx(30 / 128)
    assert 0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1


def test_cal_bbox_reads_size_and_offset_at_response_peak():
    from aero_eyes.models.trackers import _lt_cal_bbox
    fs = 16
    response = np.zeros((1, 1, fs, fs), dtype=np.float32)
    response[0, 0, 5, 9] = 0.83                                # peak at row 5, col 9
    size_map = np.zeros((1, 2, fs, fs), dtype=np.float32)
    size_map[0, 0, 5, 9], size_map[0, 1, 5, 9] = 0.30, 0.20
    offset_map = np.zeros((1, 2, fs, fs), dtype=np.float32)
    offset_map[0, 0, 5, 9], offset_map[0, 1, 5, 9] = 0.25, 0.50
    (cx, cy, w, h), score = _lt_cal_bbox(response, size_map, offset_map, fs)
    assert cx == pytest.approx((9 + 0.25) / fs)
    assert cy == pytest.approx((5 + 0.50) / fs)
    assert (w, h) == (pytest.approx(0.30), pytest.approx(0.20))
    assert score == pytest.approx(0.83)


def test_map_box_back_center_prediction_keeps_previous_center():
    from aero_eyes.models.trackers import _lt_map_box_back
    prev = (100.0, 50.0, 40.0, 30.0)
    crop_sz = 139.0
    rf = 256 / crop_sz
    x, y, w, h = _lt_map_box_back((0.5, 0.5, 40 / crop_sz, 30 / crop_sz), rf, 256, prev)
    assert (w, h) == (pytest.approx(40.0), pytest.approx(30.0))
    assert x + w / 2 == pytest.approx(100 + 20)
    assert y + h / 2 == pytest.approx(50 + 15)


def test_clip_box_keeps_box_inside_frame_with_min_size():
    from aero_eyes.models.trackers import _lt_clip_box
    x, y, w, h = _lt_clip_box((-30.0, -20.0, 500.0, 400.0), img_h=100, img_w=200)
    assert x >= 0 and y >= 0
    assert x + w <= 200 and y + h <= 100
    x, y, w, h = _lt_clip_box((10.0, 10.0, 0.0, 0.0), img_h=100, img_w=200)
    assert w >= 10.0 and h >= 10.0


# ---------------------------------------------------------------------------
# LiteTrackTracker with fake ONNX sessions
# ---------------------------------------------------------------------------

class _FakeInput:
    def __init__(self, name):
        self.name = name


class _FakeSessionZ:
    def __init__(self):
        self.calls = []

    def get_inputs(self):
        return [_FakeInput("template"), _FakeInput("template_bb")]

    def run(self, _outputs, feed):
        self.calls.append(feed)
        return [np.ones((1, 8, 4), dtype=np.float32)]        # fake template_feats


class _FakeSessionX:
    """Always predicts 'target stays exactly where it was' with a fixed score."""

    def __init__(self, box_w, box_h, search_factor, feat_size, score):
        crop_sz = math.ceil(math.sqrt(box_w * box_h) * search_factor)
        self.w_n, self.h_n = box_w / crop_sz, box_h / crop_sz
        self.feat_size, self.score = feat_size, score
        self.calls = []

    def get_inputs(self):
        return [_FakeInput("template_feats"), _FakeInput("search")]

    def run(self, _outputs, feed):
        self.calls.append(feed)
        fs, c = self.feat_size, self.feat_size // 2
        response = np.zeros((1, 1, fs, fs), dtype=np.float32)
        response[0, 0, c, c] = self.score
        size_map = np.zeros((1, 2, fs, fs), dtype=np.float32)
        size_map[0, 0, c, c], size_map[0, 1, c, c] = self.w_n, self.h_n
        offset_map = np.zeros((1, 2, fs, fs), dtype=np.float32)
        return [response, size_map, offset_map]


def _make_tracker(sess_z, sess_x, template_size=128, search_size=256, stride=16):
    from aero_eyes.models.trackers import LiteTrackTracker
    t = object.__new__(LiteTrackTracker)      # skip __init__: no onnxruntime / files
    t._sess_z, t._sess_x = sess_z, sess_x
    t.template_size, t.search_size = template_size, search_size
    t.template_factor, t.search_factor = 2.0, 4.0
    t.feat_size = search_size // stride
    t._template_feats = None
    t._state_xywh = None
    return t


def test_tracker_update_before_init_returns_none():
    tracker = _make_tracker(_FakeSessionZ(), _FakeSessionX(40, 30, 4.0, 16, 0.9))
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    assert tracker.update(frame) == (None, 0.0)


def test_tracker_init_feeds_template_and_normalized_template_bb():
    from aero_eyes.types import Box
    sz = _FakeSessionZ()
    tracker = _make_tracker(sz, _FakeSessionX(40, 30, 4.0, 16, 0.9))
    frame = np.full((480, 640, 3), 90, dtype=np.uint8)
    tracker.init(frame, Box(300.0, 200.0, 340.0, 230.0))
    feed = sz.calls[0]
    template, template_bb = feed["template"], feed["template_bb"]
    assert template.shape == (1, 3, 128, 128) and template.dtype == np.float32
    assert template_bb.shape == (1, 4) and template_bb.dtype == np.float32
    assert 0 <= template_bb.min() and template_bb.max() <= 1


def test_tracker_update_follows_static_target_and_reports_score():
    from aero_eyes.types import Box
    sx = _FakeSessionX(40, 30, 4.0, 16, 0.87)
    tracker = _make_tracker(_FakeSessionZ(), sx)
    frame = np.full((480, 640, 3), 90, dtype=np.uint8)
    tracker.init(frame, Box(300.0, 200.0, 340.0, 230.0))
    box, score = tracker.update(frame)
    assert score == pytest.approx(0.87)
    assert (box.x1, box.y1) == (pytest.approx(300.0, abs=1.0), pytest.approx(200.0, abs=1.0))
    assert (box.x2 - box.x1, box.y2 - box.y1) == (pytest.approx(40.0, abs=1.0), pytest.approx(30.0, abs=1.0))
    feed = sx.calls[0]
    assert feed["search"].shape == (1, 3, 256, 256) and feed["search"].dtype == np.float32
    # template features from z are passed through untouched
    assert feed["template_feats"].shape == (1, 8, 4)


def test_tracker_state_carries_over_between_frames():
    from aero_eyes.types import Box
    tracker = _make_tracker(_FakeSessionZ(), _FakeSessionX(40, 30, 4.0, 16, 0.9))
    frame = np.full((480, 640, 3), 90, dtype=np.uint8)
    tracker.init(frame, Box(300.0, 200.0, 340.0, 230.0))
    for _ in range(3):
        box, score = tracker.update(frame)
        assert box is not None and score == pytest.approx(0.9)
    assert tracker._state_xywh is not None


def test_tracker_degenerate_init_box_leaves_tracker_inactive():
    from aero_eyes.types import Box
    sz = _FakeSessionZ()
    tracker = _make_tracker(sz, _FakeSessionX(40, 30, 4.0, 16, 0.9))
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    tracker.init(frame, Box(100.0, 100.0, 100.0, 120.0))     # zero width
    assert sz.calls == []
    assert tracker.update(frame) == (None, 0.0)


def test_tracker_reinit_replaces_previous_state():
    from aero_eyes.types import Box
    tracker = _make_tracker(_FakeSessionZ(), _FakeSessionX(40, 30, 4.0, 16, 0.9))
    frame = np.full((480, 640, 3), 90, dtype=np.uint8)
    tracker.init(frame, Box(300.0, 200.0, 340.0, 230.0))
    tracker.init(frame, Box(50.0, 60.0, 90.0, 90.0))
    assert tracker._state_xywh == (50.0, 60.0, 40.0, 30.0)
