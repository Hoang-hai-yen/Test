"""Unit tests for aero_eyes.models.geco2_auto_scale (Track A: per-sample
automatic replacement for hand-tuning ref_downscale_factor/
crop_context_margin -- see docs/GECO2_scale_domain_gap_plan.md).

Uses fakes for the GeCo2 detector and video I/O so this runs without the
real GECO2 repo/weights/GPU/video files.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="torch not importable", exc_type=ImportError)

from aero_eyes.models import geco2_auto_scale as asc
from aero_eyes.types import Box


# ---------------------------------------------------------------------------
# select_weights / build_candidate_grid / sample_uniform_frame_indices
# ---------------------------------------------------------------------------

def test_select_weights_hard_picks_argmax():
    weights = asc.select_weights([0.1, 0.9, 0.5], "hard", temperature=0.5)
    assert weights == [0.0, 1.0, 0.0]


def test_select_weights_soft_sums_to_one_and_favors_best():
    weights = asc.select_weights([0.1, 0.9, 0.5], "soft", temperature=0.5)
    assert weights[1] == max(weights)
    assert abs(sum(weights) - 1.0) < 1e-9


def test_select_weights_soft_uniform_when_qualities_tied():
    weights = asc.select_weights([0.5, 0.5, 0.5], "soft", temperature=0.5)
    assert all(abs(w - 1 / 3) < 1e-9 for w in weights)


def test_build_candidate_grid_full_when_under_cap():
    grid = asc.build_candidate_grid([0.5, 1.0], [1.0, 0.5, 0.25], max_candidates=100)
    assert grid == [(0.5, 1.0), (0.5, 0.5), (0.5, 0.25), (1.0, 1.0), (1.0, 0.5), (1.0, 0.25)]


def test_build_candidate_grid_truncates_evenly_when_over_cap():
    grid = asc.build_candidate_grid([0.5, 1.0, 2.0, 4.0], [1.0, 0.5, 0.25, 0.125, 0.0625, 0.03], max_candidates=12)
    assert len(grid) == 12


def test_sample_uniform_frame_indices_deterministic():
    idx1 = asc.sample_uniform_frame_indices(1000, 12)
    idx2 = asc.sample_uniform_frame_indices(1000, 12)
    assert idx1 == idx2
    assert idx1[0] == 0
    assert idx1[-1] == 999


def test_sample_uniform_frame_indices_caps_at_total_frames():
    idx = asc.sample_uniform_frame_indices(3, 12)
    assert idx == [0, 1, 2]


# ---------------------------------------------------------------------------
# score_candidate_self_supervised / score_candidate_gt_iou / top1_box_and_score
# ---------------------------------------------------------------------------

class _RawScoresDetector:
    """Fake detector for testing the scoring functions in isolation."""

    def __init__(self, scores_by_frame: dict[int, np.ndarray], boxes_by_frame: dict[int, Box] | None = None):
        self._scores_by_frame = scores_by_frame
        self._boxes_by_frame = boxes_by_frame or {}

    def raw_scores(self, frame_bgr, prototype):
        return self._scores_by_frame[frame_bgr]

    def forward_scores(self, frame_bgr, prototype):
        box = self._boxes_by_frame.get(frame_bgr)
        if box is None:
            return torch.zeros(0, 4), torch.zeros(0), 1.0
        return torch.tensor([[box.x1, box.y1, box.x2, box.y2]]), torch.tensor([box.score]), 1.0

    def filter_boxes_by_threshold(self, pred_boxes, box_v, scale, frame_bgr, threshold):
        box = self._boxes_by_frame.get(frame_bgr)
        return [box] if box is not None else []


def test_score_candidate_self_supervised_is_peakiness_proxy(monkeypatch):
    monkeypatch.setattr(asc, "read_frame", lambda video_path, idx: idx)  # frame_bgr = its own index
    peaky = np.array([10.0, 0.0, 0.0, 0.0])       # high max, low mean -> high margin
    flat = np.array([1.0, 1.0, 1.0, 1.0])          # zero std -> low margin
    detector = _RawScoresDetector({1: peaky, 2: flat})
    q_peaky = asc.score_candidate_self_supervised(detector, {}, "video.mp4", [1], eps=1e-6)
    q_flat = asc.score_candidate_self_supervised(detector, {}, "video.mp4", [2], eps=1e-6)
    assert q_peaky > q_flat


def test_score_candidate_gt_iou_perfect_match(monkeypatch):
    monkeypatch.setattr(asc, "read_frame", lambda video_path, idx: idx)
    box = Box(0, 0, 10, 10, score=0.9)
    detector = _RawScoresDetector({}, boxes_by_frame={1: box})
    gt = {1: Box(0, 0, 10, 10)}
    q = asc.score_candidate_gt_iou(detector, {}, "video.mp4", gt, present_frame_idxs=[1])
    assert q == pytest.approx(1.0)


def test_score_candidate_gt_iou_no_detection_scores_zero(monkeypatch):
    monkeypatch.setattr(asc, "read_frame", lambda video_path, idx: idx)
    detector = _RawScoresDetector({}, boxes_by_frame={})  # no box for frame 1
    gt = {1: Box(0, 0, 10, 10)}
    q = asc.score_candidate_gt_iou(detector, {}, "video.mp4", gt, present_frame_idxs=[1])
    assert q == 0.0


# ---------------------------------------------------------------------------
# build_auto_scaled_prototype orchestration
# ---------------------------------------------------------------------------

class _FakeEncodeDetector:
    """Records encode_exemplars calls; each call returns a token dict
    filled with a distinct constant (encode-call-index + 1) so a test can
    verify which candidate's tokens ended up in a blend."""

    def __init__(self, use_shape_token: bool = False):
        self.use_shape_token = use_shape_token
        self.encode_calls: list[tuple] = []

    def encode_exemplars(self, ref_images_bgr, ref_boxes=None):
        idx = len(self.encode_calls)
        self.encode_calls.append((ref_images_bgr, ref_boxes))
        n_tokens = len(ref_images_bgr) * (2 if self.use_shape_token else 1)
        val = float(idx + 1)
        return {
            "main": torch.full((1, n_tokens, 4), val),
            "l1": torch.full((1, n_tokens, 4), val),
            "l2": torch.full((1, n_tokens, 4), val),
        }


def _make_cfg(tmp_path: Path, quality_metric="self_supervised_margin", selection_mode="soft",
              crop_margins=(1.0, 2.0), downscale_factors=(1.0, 0.5), max_candidates=100):
    return SimpleNamespace(
        data=SimpleNamespace(gt=SimpleNamespace(global_file=str(tmp_path / "no_such_annotations.json"))),
        stage123_geco2=SimpleNamespace(
            auto_scale_calibration=SimpleNamespace(
                candidate_crop_margins=list(crop_margins),
                candidate_downscale_factors=list(downscale_factors),
                num_probe_frames=4,
                selection_mode=selection_mode,
                quality_metric=quality_metric,
                temperature=0.5,
                max_candidates=max_candidates,
                eps=1e-6,
            ),
        ),
    )


def _make_ref_imgs_and_boxes():
    imgs = [np.zeros((40, 40, 3), dtype=np.uint8) for _ in range(3)]
    boxes = [(5.0, 5.0, 35.0, 35.0) for _ in range(3)]
    return imgs, boxes


def test_build_auto_scaled_prototype_calls_encode_exemplars_once_per_candidate(tmp_path, monkeypatch):
    cfg = _make_cfg(tmp_path, crop_margins=(1.0, 2.0), downscale_factors=(1.0, 0.5, 0.25))
    detector = _FakeEncodeDetector()
    ref_imgs, raw_boxes = _make_ref_imgs_and_boxes()

    monkeypatch.setattr(asc, "video_info", lambda video_path: {"total_frames": 100})
    monkeypatch.setattr(asc, "read_frame", lambda video_path, idx: idx)  # frame_bgr = its own index
    # Deterministic, distinguishable quality per candidate (call order).
    monkeypatch.setattr(asc, "score_candidate_self_supervised",
                         lambda detector, prototype, video_path, idxs, eps: float(len(detector.encode_calls)))

    prototype, debug_info = asc.build_auto_scaled_prototype(cfg, "sample_0", detector, ref_imgs, raw_boxes, "video.mp4")

    assert len(detector.encode_calls) == 2 * 3  # 2 margins x 3 factors
    assert len(debug_info["candidates"]) == 6
    assert debug_info["metric_used"] == "self_supervised_margin"
    assert len(debug_info["weights"]) == 6
    assert prototype["main"].shape == (1, 3, 4)  # 3 refs, no shape token


def test_build_auto_scaled_prototype_blend_matches_manual_weighted_sum(tmp_path, monkeypatch):
    cfg = _make_cfg(tmp_path, crop_margins=(1.0, 2.0), downscale_factors=(1.0,), selection_mode="soft")
    detector = _FakeEncodeDetector()
    ref_imgs, raw_boxes = _make_ref_imgs_and_boxes()

    monkeypatch.setattr(asc, "video_info", lambda video_path: {"total_frames": 100})
    monkeypatch.setattr(asc, "read_frame", lambda video_path, idx: idx)
    scripted_qualities = [0.2, 0.8]
    monkeypatch.setattr(asc, "score_candidate_self_supervised",
                         lambda detector, prototype, video_path, idxs, eps: scripted_qualities[len(detector.encode_calls) - 1])

    prototype, debug_info = asc.build_auto_scaled_prototype(cfg, "sample_0", detector, ref_imgs, raw_boxes, "video.mp4")

    expected_weights = asc.select_weights(scripted_qualities, "soft", cfg.stage123_geco2.auto_scale_calibration.temperature)
    # Candidate i's tokens are all filled with (i+1); blended value must equal the weighted sum.
    expected_value = sum(w * (i + 1) for i, w in enumerate(expected_weights))
    assert prototype["main"][0, 0, 0].item() == pytest.approx(expected_value)
    assert debug_info["weights"] == pytest.approx(expected_weights)


def test_build_auto_scaled_prototype_shape_token_uses_best_candidate_unblended(tmp_path, monkeypatch):
    cfg = _make_cfg(tmp_path, crop_margins=(1.0, 2.0), downscale_factors=(1.0,), selection_mode="soft")
    detector = _FakeEncodeDetector(use_shape_token=True)
    ref_imgs, raw_boxes = _make_ref_imgs_and_boxes()

    monkeypatch.setattr(asc, "video_info", lambda video_path: {"total_frames": 100})
    monkeypatch.setattr(asc, "read_frame", lambda video_path, idx: idx)
    scripted_qualities = [0.9, 0.1]  # candidate 0 (value=1.0) is best
    monkeypatch.setattr(asc, "score_candidate_self_supervised",
                         lambda detector, prototype, video_path, idxs, eps: scripted_qualities[len(detector.encode_calls) - 1])

    prototype, debug_info = asc.build_auto_scaled_prototype(cfg, "sample_0", detector, ref_imgs, raw_boxes, "video.mp4")

    # Layout is [app_1, shape_1, app_2, shape_2, app_3, shape_3] -- shape
    # indices 1, 3, 5 must equal the BEST candidate's raw value (1.0),
    # unblended, even though candidate 0 doesn't get weight 1.0 in "soft" mode.
    assert prototype["main"].shape == (1, 6, 4)
    for shape_idx in (1, 3, 5):
        assert prototype["main"][0, shape_idx, 0].item() == pytest.approx(1.0)
    # Appearance tokens (even indices) ARE the soft blend, so they should
    # NOT all equal a single candidate's raw value (best has weight < 1).
    expected_weights = asc.select_weights(scripted_qualities, "soft", cfg.stage123_geco2.auto_scale_calibration.temperature)
    assert expected_weights[0] < 1.0
    expected_app_value = sum(w * (i + 1) for i, w in enumerate(expected_weights))
    assert prototype["main"][0, 0, 0].item() == pytest.approx(expected_app_value)


def test_build_auto_scaled_prototype_hard_mode_picks_single_candidate(tmp_path, monkeypatch):
    cfg = _make_cfg(tmp_path, crop_margins=(1.0, 2.0), downscale_factors=(1.0,), selection_mode="hard")
    detector = _FakeEncodeDetector()
    ref_imgs, raw_boxes = _make_ref_imgs_and_boxes()

    monkeypatch.setattr(asc, "video_info", lambda video_path: {"total_frames": 100})
    monkeypatch.setattr(asc, "read_frame", lambda video_path, idx: idx)
    scripted_qualities = [0.2, 0.8]
    monkeypatch.setattr(asc, "score_candidate_self_supervised",
                         lambda detector, prototype, video_path, idxs, eps: scripted_qualities[len(detector.encode_calls) - 1])

    prototype, debug_info = asc.build_auto_scaled_prototype(cfg, "sample_0", detector, ref_imgs, raw_boxes, "video.mp4")

    assert debug_info["weights"] == [0.0, 1.0]
    assert prototype["main"][0, 0, 0].item() == pytest.approx(2.0)  # candidate index 1 -> value 2.0
