"""cosine_rescore.candidate_score_threshold_abs: per-box absolute floor applied
before the frame-relative ratio; both 0 = no score filtering."""
import numpy as np
import pytest
import torch

from tests.test_geco2_peak_contrast_filter import _make_detector

_BOXES = torch.tensor([
    [0.05, 0.05, 0.25, 0.25],
    [0.30, 0.05, 0.50, 0.25],
    [0.55, 0.05, 0.75, 0.25],
    [0.05, 0.50, 0.25, 0.70],
])
_SCORES = torch.tensor([0.9, 0.5, 0.09, 0.05])


def _detect(box_abs, ratio, generic_abs=0.0):
    det = _make_detector()
    det.score_threshold_ratio = ratio
    det.score_threshold_abs = generic_abs
    det.score_threshold_box_abs = box_abs
    det._forward_scores = lambda frame, proto: (_BOXES, _SCORES, 1.0, None, None, None)
    boxes = det.detect_frame(np.zeros((20, 20, 3), np.uint8), None)
    return sorted(round(b.score, 2) for b in boxes)


def test_legacy_behaviour_unchanged_when_box_abs_is_none():
    assert _detect(None, 0.15) == [0.5, 0.9]                       # ratio only: > 0.9 * 0.15 = 0.135
    assert _detect(None, 0.15, generic_abs=0.95) == []             # generic abs = floor on the frame's best score


def test_abs_applied_before_ratio():
    assert _detect(0.4, 0.15) == [0.5, 0.9]      # abs 0.4 dominates the ratio cut (0.135)
    assert _detect(0.6, 0.15) == [0.9]
    assert _detect(0.0, 0.15) == [0.5, 0.9]      # abs off: ratio alone
    assert _detect(0.05, 0.0) == [0.09, 0.5, 0.9]  # ratio off: abs alone (strictly above 0.05)


def test_ratio_still_cuts_when_it_is_stricter_than_abs():
    assert _detect(0.05, 0.6) == [0.9]           # 0.9 * 0.6 = 0.54 > abs


def test_frame_whose_best_box_is_below_abs_yields_nothing():
    assert _detect(0.95, 0.15) == []


def test_both_zero_means_no_score_filtering():
    assert _detect(0.0, 0.0) == [0.05, 0.09, 0.5, 0.9]
