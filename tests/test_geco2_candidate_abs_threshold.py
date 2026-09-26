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


# ---------------------------------------------------------------------------
# cosine_rescore.skip_candidate_encoding
# ---------------------------------------------------------------------------

def _skip_cfg(skip=True, recompute=True, dp_enabled=False, source="feature_extractor"):
    from types import SimpleNamespace as NS

    return NS(
        stage3=NS(recompute_candidate_features=recompute),
        stage123_geco2=NS(
            cosine_rescore=NS(skip_candidate_encoding=skip),
            dynamic_prototype=NS(enabled=dp_enabled, cross_check_source=source),
        ),
    )


def test_skip_encoding_needs_recompute_and_a_compatible_dynamic_prototype():
    from aero_eyes.stages.stage123_geco2 import validate_skip_candidate_encoding as v

    v(_skip_cfg(skip=False, recompute=False))                        # off: nothing to check
    v(_skip_cfg(recompute=True))
    v(_skip_cfg(recompute=True, dp_enabled=True, source="hiera"))
    with pytest.raises(ValueError, match="recompute_candidate_features"):
        v(_skip_cfg(recompute=False))
    with pytest.raises(ValueError, match="cross_check_source"):
        v(_skip_cfg(recompute=True, dp_enabled=True, source="feature_extractor"))


def test_candidate_pass_without_encoding_never_touches_the_extractor(monkeypatch):
    from pathlib import Path
    from types import SimpleNamespace as NS

    from aero_eyes.stages import stage123_geco2
    from aero_eyes.types import Box
    from tests.test_geco2_second_pass import _FakeDetector, _frame, _patch_frame_iterator

    _patch_frame_iterator(monkeypatch, {10: _frame(10), 20: _frame(20)})
    cfg = NS(stage2=NS(candidate=NS(feature_crop_pad=0.1)), runtime=NS(batch_size=8))
    cands = stage123_geco2._run_geco2_candidate_pass(
        _FakeDetector({10: [Box(0, 0, 5, 5, score=0.9), Box(1, 1, 6, 6, score=0.7)], 20: []}),
        None, Path("/nonexistent.mp4"), {10, 20}, lambda: "P",
        color_sig=None, cpf_cfg=None, cfg=cfg, encode=False,          # extractor=None would crash if used
    )
    assert len(cands[10]) == 2 and cands[10][0]._feature.shape == (1,) and not cands[10][0]._feature.any()
    assert cands[20] == []


def test_placeholder_marker_roundtrip_and_stage3_guard(tmp_path):
    from types import SimpleNamespace as NS

    from aero_eyes.stages.stage2 import _write_candidates_with_features, candidates_have_placeholder_features
    from aero_eyes.stages.stage3 import check_placeholder_features
    from aero_eyes.types import Box, Detection

    det = Detection(frame_idx=1, box=Box(0, 0, 5, 5), similarity=0.0, source="candidate")
    det._feature = np.zeros(1, np.float32)
    path = tmp_path / "candidates.json"
    _write_candidates_with_features({1: [det]}, path)                        # real features (default)
    assert not candidates_have_placeholder_features(path)
    check_placeholder_features(path, NS(recompute_candidate_features=False), None)

    _write_candidates_with_features({1: [det]}, path, placeholder_features=True)
    assert candidates_have_placeholder_features(path)
    check_placeholder_features(path, NS(recompute_candidate_features=True), tmp_path / "v.mp4")   # recompute will replace them
    with pytest.raises(ValueError, match="placeholders"):
        check_placeholder_features(path, NS(recompute_candidate_features=False), tmp_path / "v.mp4")
    with pytest.raises(ValueError, match="placeholders"):
        check_placeholder_features(path, NS(recompute_candidate_features=True), None)      # no video -> cannot recompute

    _write_candidates_with_features({1: [det]}, path)                        # what recompute does afterwards
    assert not candidates_have_placeholder_features(path)


def test_skip_candidate_encoding_defaults_off():
    from aero_eyes.config import Geco2CosineRescoreConfig

    assert Geco2CosineRescoreConfig().skip_candidate_encoding is False
