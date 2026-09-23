"""Unit tests for stage123_geco2.dynamic_prototype.topk_fusion.
peakiness_weight -- the optional 3rd fusion axis (alongside cosine/geco2
score) built from Box.peak_contrast (stage123_geco2.peak_contrast_filter).
See Geco2DynamicPrototypeTopKFusionConfig's own docstring (aero_eyes/
config.py) for the full fused_score formula.

Reuses tests/test_geco2_dynamic_prototype.py's fake-config/fake-detector
scaffolding, same as test_geco2_dynamic_prototype_margin_verification.py,
so this runs without the real GECO2 repo, weights, or GPU.
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="torch not importable", exc_type=ImportError)

from aero_eyes.types import Box
from tests.test_geco2_dynamic_prototype import _FakeDetector, _make_cfg, _make_tracker


def test_peakiness_weight_zero_ignores_peak_contrast_even_when_present():
    """Default (peakiness_weight=0.0) -- unchanged 2-way formula, even when
    every Box happens to carry a peak_contrast. Cosine AND geco2 score are
    tied between the two candidates, so the plain 2-way fused_score is also
    tied -- argmax picks boxes[0] (the first max), same as pre-peakiness
    behavior."""
    cfg = _make_cfg(
        min_consecutive_hits=1,
        topk_fusion_overrides={"enabled": True, "min_window_for_zscore": 5, "peakiness_weight": 0.0},
    )
    detector = _FakeDetector()
    tracker = _make_tracker(cfg, detector=detector)
    tracker._cross_prototype = np.array([1.0, 0.0])
    tracker._topk_cosine_history.extend([0.08, 0.10, 0.12, 0.09, 0.11])
    tracker._topk_geco2_history.extend([0.9, 1.0, 1.1, 0.95, 1.05])

    boxes = [
        Box(0, 0, 10, 10, score=1.0, peak_contrast=0.0),
        Box(20, 20, 30, 30, score=1.0, peak_contrast=5.0),  # decisively higher, but weight is 0
    ]
    feats = np.array([[0.10, 0.0], [0.10, 0.0]])
    tracker.offer_topk(np.zeros((10, 10, 3), dtype=np.uint8), boxes, feats)

    assert tracker._n_topk_fused_selected_non_top1 == 0
    assert detector.calls[0][1] == [(boxes[0].x1, boxes[0].y1, boxes[0].x2, boxes[0].y2)]
    assert len(tracker._topk_peakiness_history) == 0, "history must not accumulate when peakiness_weight is 0"


def test_peakiness_weight_can_flip_selection_when_cosine_and_geco2_tie():
    """cosine AND geco2 score tied between the 2 candidates (2-way formula
    alone can't distinguish them, picks boxes[0] by tie-break) -- but
    boxes[1]'s peak_contrast is decisively higher than anything in the
    warmed-up peakiness baseline. With peakiness_weight > 0, boxes[1] must
    win selection instead."""
    cfg = _make_cfg(
        min_consecutive_hits=1,
        topk_fusion_overrides={
            "enabled": True, "min_window_for_zscore": 5,
            "cosine_weight": 0.3, "peakiness_weight": 0.5,  # geco2 implicitly gets 0.2
        },
    )
    detector = _FakeDetector()
    tracker = _make_tracker(cfg, detector=detector)
    tracker._cross_prototype = np.array([1.0, 0.0])
    tracker._topk_cosine_history.extend([0.08, 0.10, 0.12, 0.09, 0.11])
    tracker._topk_geco2_history.extend([0.9, 1.0, 1.1, 0.95, 1.05])
    tracker._topk_peakiness_history.extend([0.0, 0.5, -0.5, 0.2, -0.2])  # mean ~0, modest spread

    boxes = [
        Box(0, 0, 10, 10, score=1.0, peak_contrast=0.0),   # average peakiness
        Box(20, 20, 30, 30, score=1.0, peak_contrast=5.0),  # far outside the baseline's spread
    ]
    feats = np.array([[0.10, 0.0], [0.10, 0.0]])
    tracker.offer_topk(np.zeros((10, 10, 3), dtype=np.uint8), boxes, feats)

    assert tracker._n_topk_fused_selected_non_top1 == 1
    eff = tracker.effective_prototype()
    assert eff["main"].shape[1] == 2, "boxes[1] must have been confirmed+accepted"
    assert detector.calls[0][1] == [(boxes[1].x1, boxes[1].y1, boxes[1].x2, boxes[1].y2)]
    assert len(tracker._topk_peakiness_history) == 6, "the chosen candidate's peak_contrast must be recorded too"


def test_peakiness_weight_falls_back_with_warning_when_peak_contrast_missing():
    """peakiness_weight > 0 but candidates carry no peak_contrast (default
    None -- stage123_geco2.peak_contrast_filter.enabled is false) -- must
    warn once and silently behave like the 2-way formula, not crash."""
    cfg = _make_cfg(
        min_consecutive_hits=1,
        topk_fusion_overrides={"enabled": True, "min_window_for_zscore": 100, "peakiness_weight": 0.5},
    )
    tracker = _make_tracker(cfg)
    tracker._cross_prototype = np.array([1.0, 0.0])

    boxes = [Box(0, 0, 10, 10, score=0.9), Box(20, 20, 30, 30, score=0.8)]  # peak_contrast defaults to None
    feats = np.array([[0.8, 0.0], [0.95, 0.0]])
    tracker.offer_topk(np.zeros((10, 10, 3), dtype=np.uint8), boxes, feats)

    assert tracker._warned_peakiness_unavailable is True
    assert len(tracker._topk_peakiness_history) == 0

    # A second call must not raise / must not warn again (one-time only) --
    # exercised implicitly by just calling it again without error.
    tracker.offer_topk(np.zeros((10, 10, 3), dtype=np.uint8), boxes, feats)
