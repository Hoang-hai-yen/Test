"""Unit tests for aero_eyes.stages.stage4._detect_on_frame_gdino --
stage4.gdino_redetect_cosine_filter's own logic (Grounding DINO proposes,
an instance-level cosine check on stage1.feature_extractor's prototype
optionally confirms/rejects), mirroring the untested-but-analogous
_detect_on_frame_geco2 -- exercised here with fakes, no real model needed."""
from __future__ import annotations

import numpy as np
import pytest

from aero_eyes.stages import stage4 as stage4_mod
from aero_eyes.types import Box


class _FakeGDinoDetector:
    def __init__(self, boxes: list[Box]):
        self._boxes = boxes

    def detect_frame(self, frame_bgr, text_prompt):
        return self._boxes


class _FakeExtractor:
    """extract_crops returns a fixed feature per call index, so the test
    controls each box's cosine similarity directly via `prototype`."""
    def __init__(self, feats: np.ndarray):
        self._feats = feats

    def extract_crops(self, frame_bgr, boxes, pad_ratio, batch_size):
        return self._feats


class _FakeCfg:
    class stage2:
        class candidate:
            feature_crop_pad = 0.1
    class runtime:
        batch_size = 16
    class accuracy:
        mode = "baseline"
        class cheap_boosters:
            multi_reference_embedding = False


def test_detect_on_frame_gdino_returns_none_when_no_detector():
    box, source = stage4_mod._detect_on_frame_gdino(np.zeros((10, 10, 3)), None, "an object")
    assert (box, source) == (None, "none")


def test_detect_on_frame_gdino_returns_none_when_no_text_prompt():
    det = _FakeGDinoDetector([Box(0, 0, 10, 10, score=0.9)])
    box, source = stage4_mod._detect_on_frame_gdino(np.zeros((10, 10, 3)), det, "")
    assert (box, source) == (None, "none")


def test_detect_on_frame_gdino_without_cosine_filter_picks_best_gdino_score():
    boxes = [Box(0, 0, 10, 10, score=0.5), Box(20, 20, 30, 30, score=0.9)]
    det = _FakeGDinoDetector(boxes)
    box, source = stage4_mod._detect_on_frame_gdino(np.zeros((40, 40, 3)), det, "an object")
    assert source == "detect"
    assert box.score == pytest.approx(0.9)


def test_detect_on_frame_gdino_cosine_filter_drops_low_similarity_boxes():
    # Two GDINO candidates: box 0 has the HIGHER gdino score but a LOW
    # cosine similarity to the prototype (wrong instance); box 1 has a
    # lower gdino score but clears match_threshold -- cosine filter must
    # pick box 1, not just the highest gdino score.
    boxes = [Box(0, 0, 10, 10, score=0.9), Box(20, 20, 30, 30, score=0.6)]
    det = _FakeGDinoDetector(boxes)
    feats = np.array([[1.0, 0.0], [0.0, 1.0]])  # box0 || prototype axis0, box1 || axis1
    extractor = _FakeExtractor(feats)
    prototype = np.array([0.0, 1.0])  # matches box1 (cos=1.0), not box0 (cos=0.0)

    box, source = stage4_mod._detect_on_frame_gdino(
        np.zeros((40, 40, 3)), det, "an object",
        cosine_extractor=extractor, cosine_prototype=prototype,
        per_ref_features=[], cfg=_FakeCfg(), match_threshold=0.5,
    )
    assert source == "detect"
    assert box.score == pytest.approx(0.6)  # box1, despite its lower gdino score


def test_detect_on_frame_gdino_cosine_filter_returns_none_when_all_dropped():
    boxes = [Box(0, 0, 10, 10, score=0.9)]
    det = _FakeGDinoDetector(boxes)
    extractor = _FakeExtractor(np.array([[1.0, 0.0]]))
    prototype = np.array([0.0, 1.0])  # orthogonal -- cos=0.0, well below threshold

    box, source = stage4_mod._detect_on_frame_gdino(
        np.zeros((40, 40, 3)), det, "an object",
        cosine_extractor=extractor, cosine_prototype=prototype,
        per_ref_features=[], cfg=_FakeCfg(), match_threshold=0.5,
    )
    assert (box, source) == (None, "none")
