"""Unit tests for stage123_geco2.py's _run_geco2_default_pass and
_run_geco2_candidate_pass -- the two helpers factored out so
dynamic_prototype.second_pass can re-sweep the whole video a second time
with a frozen prototype without duplicating (and risking drift from) pass
1's per-frame body. No torch import needed: these helpers only call
detector.detect_frame()/extractor.extract_crops(), both faked here."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from aero_eyes.stages import stage123_geco2
from aero_eyes.types import Box


def _frame(idx: int) -> np.ndarray:
    """A tiny frame that encodes its own frame_idx so fakes can look
    themselves up without needing the real frame_iterator's frame_idx arg
    (detect_frame only receives frame_bgr)."""
    arr = np.zeros((2, 2, 3), dtype=np.uint8)
    arr[0, 0, 0] = idx
    return arr


def _frame_idx_of(frame_bgr: np.ndarray) -> int:
    return int(frame_bgr[0, 0, 0])


def _patch_frame_iterator(monkeypatch, frames: dict[int, np.ndarray]):
    def _fake_iterator(video_path):
        for idx in sorted(frames):
            yield idx, frames[idx]

    monkeypatch.setattr(stage123_geco2, "frame_iterator", _fake_iterator, raising=False)
    import aero_eyes.utils.video as video_mod
    monkeypatch.setattr(video_mod, "frame_iterator", _fake_iterator)


class _FakeDetector:
    """boxes_by_frame: {frame_idx: [Box, ...]}. Records (frame_idx,
    prototype) for every detect_frame() call so tests can assert which
    prototype value was in effect at each frame."""

    def __init__(self, boxes_by_frame: dict[int, list[Box]]):
        self.boxes_by_frame = boxes_by_frame
        self.calls: list[tuple[int, object]] = []

    def detect_frame(self, frame_bgr, prototype):
        idx = _frame_idx_of(frame_bgr)
        self.calls.append((idx, prototype))
        return list(self.boxes_by_frame.get(idx, []))


class _FakeExtractor:
    def __init__(self, dim: int = 2):
        self.dim = dim
        self.calls: list[tuple[int, int]] = []

    def _feature_dim(self) -> int:
        return self.dim

    def extract_crops(self, frame_bgr, boxes, pad_ratio, batch_size):
        idx = _frame_idx_of(frame_bgr)
        self.calls.append((idx, len(boxes)))
        return np.ones((len(boxes), self.dim), dtype=np.float32)


def test_default_pass_detects_every_keyframe_and_calls_on_result(monkeypatch):
    frames = {10: _frame(10), 20: _frame(20), 30: _frame(30)}
    _patch_frame_iterator(monkeypatch, frames)
    boxes_by_frame = {10: [Box(0, 0, 5, 5, score=0.9)], 20: [], 30: [Box(1, 1, 6, 6, score=0.8)]}
    detector = _FakeDetector(boxes_by_frame)
    seen: list[tuple[int, int]] = []

    detections = stage123_geco2._run_geco2_default_pass(
        detector, Path("/nonexistent.mp4"), {10, 20, 30}, lambda: "PROTO",
        color_sig=None, cpf_cfg=None, color_stats=None, viz_dir=Path("/nonexistent"), save_viz=False,
        on_result=lambda frame_idx, frame_bgr, result_dets: seen.append((frame_idx, len(result_dets))),
    )

    assert set(detections.keys()) == {10, 20, 30}
    assert len(detections[10]) == 1 and len(detections[20]) == 0 and len(detections[30]) == 1
    assert seen == [(10, 1), (20, 0), (30, 1)]
    assert [c[1] for c in detector.calls] == ["PROTO", "PROTO", "PROTO"]


def test_default_pass_skips_non_keyframes(monkeypatch):
    frames = {1: _frame(1), 2: _frame(2), 3: _frame(3)}
    _patch_frame_iterator(monkeypatch, frames)
    detector = _FakeDetector({1: [Box(0, 0, 5, 5, score=0.9)], 3: [Box(0, 0, 5, 5, score=0.9)]})

    detections = stage123_geco2._run_geco2_default_pass(
        detector, Path("/nonexistent.mp4"), {1, 3}, lambda: "PROTO",
        color_sig=None, cpf_cfg=None, color_stats=None, viz_dir=Path("/nonexistent"), save_viz=False,
    )

    assert set(detections.keys()) == {1, 3}
    assert [c[0] for c in detector.calls] == [1, 3]


def test_default_pass_reads_get_prototype_fresh_each_frame(monkeypatch):
    """Simulates second_pass's use case: get_prototype must be RE-CALLED
    per frame (not captured once), so pass 1's growing dynamic_prototype
    (or pass 2's frozen one) is always the CURRENT value, never stale."""
    frames = {10: _frame(10), 20: _frame(20)}
    _patch_frame_iterator(monkeypatch, frames)
    detector = _FakeDetector({10: [], 20: []})
    state = {"value": "v1"}

    stage123_geco2._run_geco2_default_pass(
        detector, Path("/nonexistent.mp4"), {10, 20}, lambda: state["value"],
        color_sig=None, cpf_cfg=None, color_stats=None, viz_dir=Path("/nonexistent"), save_viz=False,
        on_result=lambda *a: state.__setitem__("value", "v2"),
    )

    assert [c[1] for c in detector.calls] == ["v1", "v2"], (
        "frame 20 must see the prototype AFTER frame 10's on_result mutated it -- "
        "exactly what lets dynamic_prototype's online update affect later keyframes"
    )


def test_candidate_pass_builds_features_and_calls_on_result(monkeypatch):
    frames = {10: _frame(10), 20: _frame(20)}
    _patch_frame_iterator(monkeypatch, frames)
    boxes_by_frame = {10: [Box(0, 0, 5, 5, score=0.9), Box(1, 1, 6, 6, score=0.7)], 20: []}
    detector = _FakeDetector(boxes_by_frame)
    extractor = _FakeExtractor(dim=3)
    cfg = SimpleNamespace(
        stage2=SimpleNamespace(candidate=SimpleNamespace(feature_crop_pad=0.1)),
        runtime=SimpleNamespace(batch_size=8),
    )
    seen: list[tuple[int, int]] = []

    candidates = stage123_geco2._run_geco2_candidate_pass(
        detector, extractor, Path("/nonexistent.mp4"), {10, 20}, lambda: "PROTO",
        color_sig=None, cpf_cfg=None, cfg=cfg,
        on_result=lambda frame_idx, frame_bgr, boxes, feats: seen.append((frame_idx, len(boxes))),
    )

    assert len(candidates[10]) == 2
    assert candidates[10][0]._feature.shape == (3,)
    assert len(candidates[20]) == 0
    assert extractor.calls == [(10, 2)], "extract_crops must be skipped (not called with 0 boxes) for frame 20"
    assert seen == [(10, 2), (20, 0)]


def test_candidate_pass_reads_get_prototype_fresh_each_frame(monkeypatch):
    frames = {10: _frame(10), 20: _frame(20)}
    _patch_frame_iterator(monkeypatch, frames)
    detector = _FakeDetector({10: [], 20: []})
    extractor = _FakeExtractor()
    cfg = SimpleNamespace(
        stage2=SimpleNamespace(candidate=SimpleNamespace(feature_crop_pad=0.1)),
        runtime=SimpleNamespace(batch_size=8),
    )
    state = {"value": "v1"}

    stage123_geco2._run_geco2_candidate_pass(
        detector, extractor, Path("/nonexistent.mp4"), {10, 20}, lambda: state["value"],
        color_sig=None, cpf_cfg=None, cfg=cfg,
        on_result=lambda *a: state.__setitem__("value", "v2"),
    )

    assert [c[1] for c in detector.calls] == ["v1", "v2"]


def test_candidate_pass_saves_raw_candidate_frames_only_when_viz_dir_given(monkeypatch, tmp_path):
    frames = {10: _frame(10), 20: _frame(20)}
    _patch_frame_iterator(monkeypatch, frames)
    boxes_by_frame = {10: [Box(0, 0, 5, 5, score=0.9)], 20: []}
    cfg = SimpleNamespace(
        stage2=SimpleNamespace(candidate=SimpleNamespace(feature_crop_pad=0.1)),
        runtime=SimpleNamespace(batch_size=8),
    )

    def run(viz_dir):
        return stage123_geco2._run_geco2_candidate_pass(
            _FakeDetector(boxes_by_frame), _FakeExtractor(dim=3), Path("/nonexistent.mp4"), {10, 20},
            lambda: "PROTO", color_sig=None, cpf_cfg=None, cfg=cfg, viz_dir=viz_dir,
        )

    run(None)
    assert not list(tmp_path.glob("**/*.jpg"))                       # off by default: nothing written

    out = tmp_path / "candidates"
    run(out)
    assert sorted(p.name for p in out.glob("*.jpg")) == ["frame_000010.jpg"]   # frame 20 had no candidates
