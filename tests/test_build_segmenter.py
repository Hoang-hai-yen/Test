"""Unit tests for build_segmenter() -- the SegmentationConfig.model
(mobilesam | fastsam | sam2) dispatch factory used by stage1.py and
stage123_geco2.py. Monkeypatches the 3 segmenter classes so this runs
without any real model weights/GPU."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

cv2 = pytest.importorskip("cv2", reason="opencv-python not importable", exc_type=ImportError)

import aero_eyes.models.segmentation as segmentation_module
from aero_eyes.models.segmentation import build_segmenter


def _seg_cfg(model: str) -> SimpleNamespace:
    return SimpleNamespace(
        model=model, weights="/mobilesam.pt", fallback_if_missing="passthrough",
        min_area_frac=0.05, max_area_frac=0.95, score_ratio_floor=0.85,
        max_border_touch_frac=0.02, use_point_prompt=True, reject_implausible_mask=True,
    )


def _full_cfg() -> SimpleNamespace:
    return SimpleNamespace(
        stage2=SimpleNamespace(fastsam_s=SimpleNamespace(weights="/fastsam.pt", conf=0.2, iou=0.7, imgsz=640)),
        stage123_geco2=SimpleNamespace(repo_path="./GECO2"),
    )


def test_build_segmenter_defaults_to_mobilesam(monkeypatch):
    captured = {}

    class _FakeMobileSAM:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(segmentation_module, "MobileSAMSegmenter", _FakeMobileSAM)
    result = build_segmenter(_seg_cfg("mobilesam"), _full_cfg())

    assert isinstance(result, _FakeMobileSAM)
    assert captured["weights_path"] == "/mobilesam.pt"
    assert captured["fallback_if_missing"] == "passthrough"
    assert captured["min_area_frac"] == 0.05
    assert captured["use_point_prompt"] is True


def test_build_segmenter_fastsam_reuses_stage2_fastsam_s_config(monkeypatch):
    captured = {}

    class _FakeFastSAM:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(segmentation_module, "FastSAMSegmenter", _FakeFastSAM)
    result = build_segmenter(_seg_cfg("fastsam"), _full_cfg())

    assert isinstance(result, _FakeFastSAM)
    # Must come from cfg.stage2.fastsam_s, NOT seg_cfg.weights (mobilesam path).
    assert captured["weights"] == "/fastsam.pt"
    assert captured["conf"] == 0.2
    assert captured["iou"] == 0.7
    assert captured["imgsz"] == 640
    assert captured["min_area_frac"] == 0.05
    assert captured["max_border_touch_frac"] == 0.02


def test_build_segmenter_sam2_uses_geco2_repo_path(monkeypatch):
    captured_args = []
    captured_kwargs = {}

    class _FakeSAM2:
        def __init__(self, *args, **kwargs):
            captured_args.extend(args)
            captured_kwargs.update(kwargs)

    monkeypatch.setattr(segmentation_module, "SAM2Segmenter", _FakeSAM2)
    result = build_segmenter(_seg_cfg("sam2"), _full_cfg())

    assert isinstance(result, _FakeSAM2)
    assert captured_args == ["./GECO2"]
    assert captured_kwargs["score_ratio_floor"] == 0.85
    assert captured_kwargs["use_point_prompt"] is True
