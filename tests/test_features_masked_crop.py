"""Unit tests for MaskedCropFeatureExtractor -- stage1.feature_extractor.
candidate_background_masking's opt-in wrapper that masks each VIDEO
CANDIDATE crop's background before extraction (see its own docstring in
aero_eyes/models/features.py and FeatureExtractorConfig.
candidate_background_masking's docstring in aero_eyes/config.py for the
asymmetry this addresses vs. reference-photo masking). Uses a fake
segmenter/base extractor so this runs without any real segmentation model.
"""
from __future__ import annotations

import numpy as np
import pytest

from aero_eyes.models import features as features_mod
from aero_eyes.models.features import MaskedCropFeatureExtractor
from aero_eyes.types import Box


class _FakeSegmenter:
    def __init__(self, mask=None, raise_on_segment: bool = False):
        self._mask = mask
        self._raise = raise_on_segment
        self.calls: list[np.ndarray] = []

    def segment(self, image_bgr: np.ndarray) -> np.ndarray:
        self.calls.append(image_bgr)
        if self._raise:
            raise RuntimeError("segmentation backend unavailable")
        if self._mask is not None:
            return self._mask
        mask = np.zeros(image_bgr.shape[:2], dtype=bool)
        mask[2:-2, 2:-2] = True  # a small foreground box, background around it
        return mask


class _FakeBase:
    def __init__(self, dim: int = 4):
        self._d = dim
        self.received: list[np.ndarray] | None = None

    def extract(self, images: list[np.ndarray], batch_size: int = 16) -> np.ndarray:
        self.received = list(images)
        return np.zeros((len(images), self._d), dtype=np.float32)

    def extract_crops(self, frame_bgr, boxes, pad_ratio=0.10, batch_size=16):
        raise AssertionError("MaskedCropFeatureExtractor must not delegate extract_crops to the base when boxes exist")

    def _dim(self) -> int:
        return self._d

    def _feature_dim(self) -> int:
        return self._d


def _frame() -> np.ndarray:
    """Box(10,10,30,30)/pad_ratio=0 crops frame[10:30, 10:30] (20x20) --
    background (10,20,30) everywhere, with a foreground square (200,150,100)
    at [12:28, 12:28] matching _FakeSegmenter's default mask interior
    (rows/cols 2:-2 of the crop) exactly, so mean_fill's background replacement
    is visibly different from both colors (a genuine mix of the two)."""
    frame = np.zeros((50, 50, 3), dtype=np.uint8)
    frame[:, :] = (10, 20, 30)
    frame[12:28, 12:28] = (200, 150, 100)
    return frame


def test_extract_crops_masks_background_before_extraction():
    seg = _FakeSegmenter()
    base = _FakeBase()
    ext = MaskedCropFeatureExtractor(base, seg, background_mode="mean_fill", blur_sigma=25.0)

    boxes = [Box(10, 10, 30, 30)]
    ext.extract_crops(_frame(), boxes, pad_ratio=0.0)

    assert len(seg.calls) == 1, "the segmenter must be called once per candidate crop"
    assert base.received is not None and len(base.received) == 1
    raw_crop = seg.calls[0]
    masked_crop = base.received[0]
    assert not np.array_equal(raw_crop, masked_crop), "background pixels must have been replaced"
    # Foreground region (rows/cols 2:-2 of the crop) must be untouched.
    assert np.array_equal(raw_crop[2:-2, 2:-2], masked_crop[2:-2, 2:-2])


def test_extract_passthrough_does_not_touch_base_extract_crops():
    seg = _FakeSegmenter()
    base = _FakeBase()
    ext = MaskedCropFeatureExtractor(base, seg, background_mode="mean_fill", blur_sigma=25.0)

    images = [np.zeros((8, 8, 3), dtype=np.uint8)]
    ext.extract(images)

    assert seg.calls == [], "extract() (already-prepared images, no box/frame) must never invoke the segmenter"
    assert base.received == images


def test_segmentation_failure_falls_back_to_unmasked_crop():
    seg = _FakeSegmenter(raise_on_segment=True)
    base = _FakeBase()
    ext = MaskedCropFeatureExtractor(base, seg, background_mode="mean_fill", blur_sigma=25.0)

    boxes = [Box(10, 10, 30, 30)]
    ext.extract_crops(_frame(), boxes, pad_ratio=0.0)

    assert base.received is not None and len(base.received) == 1
    assert np.array_equal(base.received[0], seg.calls[0]), "a segmentation failure must fall back to the RAW crop, unmasked"


def test_empty_boxes_returns_early_without_segmenting():
    seg = _FakeSegmenter()
    base = _FakeBase()
    ext = MaskedCropFeatureExtractor(base, seg, background_mode="mean_fill", blur_sigma=25.0)

    out = ext.extract_crops(_frame(), [])
    assert out.shape == (0, 4)
    assert seg.calls == []


def test_dim_delegates_to_base():
    ext = MaskedCropFeatureExtractor(_FakeBase(dim=7), _FakeSegmenter(), "mean_fill", 25.0)
    assert ext._dim() == 7
    assert ext._feature_dim() == 7


def test_keep_real_mode_leaves_crop_unchanged():
    seg = _FakeSegmenter()
    base = _FakeBase()
    ext = MaskedCropFeatureExtractor(base, seg, background_mode="keep_real", blur_sigma=25.0)

    boxes = [Box(10, 10, 30, 30)]
    ext.extract_crops(_frame(), boxes, pad_ratio=0.0)

    assert np.array_equal(base.received[0], seg.calls[0]), "keep_real must leave the crop untouched entirely"


def test_build_feature_extractor_wraps_with_masking_when_enabled(monkeypatch):
    from aero_eyes.config import load_config

    monkeypatch.setattr(features_mod, "DINOv2FeatureExtractor", lambda *a, **k: _FakeBase())
    monkeypatch.setattr(
        "aero_eyes.models.segmentation.build_segmenter",
        lambda seg_cfg, cfg: _FakeSegmenter(),
    )

    cfg = load_config("configs/config.yaml", overrides=[
        "stage1.feature_extractor.model=dinov2",
        "stage1.feature_extractor.candidate_background_masking.enabled=true",
    ])
    ext = features_mod.build_feature_extractor(cfg)
    assert isinstance(ext, MaskedCropFeatureExtractor)


def test_build_feature_extractor_no_wrapping_when_disabled(monkeypatch):
    from aero_eyes.config import load_config

    fake = _FakeBase()
    monkeypatch.setattr(features_mod, "DINOv2FeatureExtractor", lambda *a, **k: fake)

    cfg = load_config("configs/config.yaml", overrides=["stage1.feature_extractor.model=dinov2"])
    ext = features_mod.build_feature_extractor(cfg)
    assert ext is fake
