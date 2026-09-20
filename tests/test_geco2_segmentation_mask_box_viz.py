"""Tests for the MobileSAM tight-box debug viz (stage123_geco2.py::
_save_mask_box_viz), added so segmentation quality can be sanity-checked
visually BEFORE trusting a downstream crop/scale sweep built from it
(auto_scale_calibration, learned_scale_fusion, and the plain
segmentation.enabled path all pool from exactly this box).

Uses a fake segmenter (avoids a real MobileSAM weights download) across
all three stage123_geco2.py::build_exemplar_prototype branches that
compute `raw_boxes = [mask_bbox(m) for m in masks]`.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="torch not importable", exc_type=ImportError)
pytest.importorskip("cv2", reason="opencv-python not importable", exc_type=ImportError)

from aero_eyes.stages.stage123_geco2 import _save_mask_box_viz, build_exemplar_prototype  # noqa: E402
from aero_eyes.types import Box  # noqa: E402

FIXTURE_ID = "synth001"


class _FakeSegmenter:
    def segment(self, image_bgr):
        h, w = image_bgr.shape[:2]
        mask = np.zeros((h, w), dtype=bool)
        mask[h // 4: 3 * h // 4, w // 4: 3 * w // 4] = True
        return mask


class _FakeEncodeDetector:
    """Minimal fake for the plain segmentation.enabled path, which ends in
    a real detector.encode_exemplars(...) call."""

    use_shape_token = False

    def encode_exemplars(self, ref_images_bgr, ref_boxes=None):
        n = len(ref_images_bgr)
        return {"main": torch.zeros(1, n, 4), "l1": torch.zeros(1, n, 4), "l2": torch.zeros(1, n, 4)}


@pytest.fixture(scope="module")
def synth_fixture(tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("geco2_mask_box_viz_fixtures")
    from scripts.make_synthetic_fixture import make_fixture
    make_fixture(out_dir, FIXTURE_ID)
    return out_dir


def _assert_mask_box_viz_written(work_dir: Path, num_refs: int = 3) -> None:
    out_dir = work_dir / "viz" / "stage123_geco2" / "refs_mask_box"
    files = sorted(out_dir.glob("ref_*_mask_box.jpg"))
    assert len(files) == num_refs
    for f in files:
        img = cv2.imread(str(f))
        assert img is not None and img.shape[0] > 0 and img.shape[1] > 0


# ---------------------------------------------------------------------------
# _save_mask_box_viz -- unit level
# ---------------------------------------------------------------------------

def test_save_mask_box_viz_draws_box(tmp_path):
    imgs = [np.zeros((50, 50, 3), dtype=np.uint8)]
    boxes = [(10.0, 10.0, 40.0, 40.0)]
    out_dir = tmp_path / "refs_mask_box"
    _save_mask_box_viz(imgs, boxes, out_dir)

    saved = cv2.imread(str(out_dir / "ref_0_mask_box.jpg"))
    assert saved is not None

    def _is_greenish(px) -> bool:
        b, g, r = (int(c) for c in px)  # BGR -- cv2.rectangle drew (0,255,0)
        return g > 150 and g > b + 50 and g > r + 50

    # The box's green border must actually be drawn on the (all-black)
    # source image -- allow for JPEG compression, so check "greenish"
    # rather than an exact [0,255,0] match.
    assert _is_greenish(saved[10, 25]) or _is_greenish(saved[39, 25])


def test_save_mask_box_viz_handles_none_box(tmp_path):
    imgs = [np.zeros((50, 50, 3), dtype=np.uint8)]
    out_dir = tmp_path / "refs_mask_box"
    _save_mask_box_viz(imgs, [None], out_dir)  # must not raise
    assert (out_dir / "ref_0_mask_box.jpg").exists()


# ---------------------------------------------------------------------------
# Wired into build_exemplar_prototype's 3 branches
# ---------------------------------------------------------------------------

def _make_cfg(synth_fixture, tmp_path, **stage_overrides):
    from aero_eyes.config import (
        AeroEyesConfig, DataConfig, GTConfig, ProjectConfig, RuntimeConfig,
        SegmentationConfig, Stage123Geco2Config,
    )

    return AeroEyesConfig(
        project=ProjectConfig(work_dir=str(tmp_path / "runs"), use_cache=False, seed=42),
        data=DataConfig(
            data_root=str(synth_fixture), refs_subdir="refs", video_glob="*.mp4", num_references=3,
            gt=GTConfig(global_file=str(synth_fixture / FIXTURE_ID / "gt.json")),
        ),
        runtime=RuntimeConfig(save_visualizations=True),
        stage123_geco2=Stage123Geco2Config(
            segmentation=SegmentationConfig(enabled=True),
            **stage_overrides,
        ),
    )


def test_plain_segmentation_path_writes_mask_box_viz(synth_fixture, tmp_path, monkeypatch):
    import aero_eyes.models.segmentation as seg_mod
    monkeypatch.setattr(seg_mod, "build_segmenter", lambda seg_cfg, cfg: _FakeSegmenter())

    cfg = _make_cfg(synth_fixture, tmp_path)
    detector = _FakeEncodeDetector()
    work_dir = Path(cfg.project.work_dir) / FIXTURE_ID
    build_exemplar_prototype(cfg, FIXTURE_ID, detector, work_dir)

    _assert_mask_box_viz_written(work_dir)


def test_auto_scale_calibration_path_writes_mask_box_viz(synth_fixture, tmp_path, monkeypatch):
    import aero_eyes.models.segmentation as seg_mod
    import aero_eyes.models.geco2_auto_scale as asc_mod
    monkeypatch.setattr(seg_mod, "build_segmenter", lambda seg_cfg, cfg: _FakeSegmenter())
    monkeypatch.setattr(
        asc_mod, "build_auto_scaled_prototype",
        lambda cfg, sample_id, detector, ref_imgs, raw_boxes, video_path: (
            {"main": torch.zeros(1, 3, 4), "l1": torch.zeros(1, 3, 4), "l2": torch.zeros(1, 3, 4)},
            {"candidates": [], "qualities": [], "weights": [], "metric_used": "self_supervised_margin"},
        ),
    )

    cfg = _make_cfg(synth_fixture, tmp_path)
    cfg.stage123_geco2.auto_scale_calibration.enabled = True
    detector = _FakeEncodeDetector()
    work_dir = Path(cfg.project.work_dir) / FIXTURE_ID
    build_exemplar_prototype(cfg, FIXTURE_ID, detector, work_dir)

    _assert_mask_box_viz_written(work_dir)


def test_learned_scale_fusion_path_writes_mask_box_viz(synth_fixture, tmp_path, monkeypatch):
    import aero_eyes.models.segmentation as seg_mod
    monkeypatch.setattr(seg_mod, "build_segmenter", lambda seg_cfg, cfg: _FakeSegmenter())

    cfg = _make_cfg(synth_fixture, tmp_path, use_shape_token=False)
    cfg.stage123_geco2.learned_scale_fusion.enabled = True

    class _FakeFusedDetector:
        def encode_exemplars_fused(self, ref_images_bgr, ref_boxes, ref_group_ids, context_frames_bgr):
            n = len(set(ref_group_ids))
            return {"main": torch.zeros(1, n, 4), "l1": torch.zeros(1, n, 4), "l2": torch.zeros(1, n, 4)}

    work_dir = Path(cfg.project.work_dir) / FIXTURE_ID
    build_exemplar_prototype(cfg, FIXTURE_ID, _FakeFusedDetector(), work_dir)

    _assert_mask_box_viz_written(work_dir)
