"""Wiring tests for Track B's inference-side learned_scale_fusion branch in
aero_eyes/stages/stage123_geco2.py::build_exemplar_prototype (see
docs/GECO2_scale_domain_gap_plan.md).

Uses a fake segmenter (avoids a real MobileSAM weights download) and a fake
GeCo2Detector.encode_exemplars_fused -- this checks the PIPELINE WIRING
(candidate building, ref_group_ids, context-frame sampling, config
validation), not GeCo2Detector.encode_exemplars_fused's own tensor math,
which needs a GPU + a real (trained) Track B checkpoint to exercise.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="torch not importable", exc_type=ImportError)
pytest.importorskip("cv2", reason="opencv-python not importable", exc_type=ImportError)

from aero_eyes.stages.stage123_geco2 import build_exemplar_prototype  # noqa: E402

FIXTURE_ID = "synth001"


class _FakeSegmenter:
    def segment(self, image_bgr):
        h, w = image_bgr.shape[:2]
        mask = np.zeros((h, w), dtype=bool)
        mask[h // 4: 3 * h // 4, w // 4: 3 * w // 4] = True
        return mask


class _FakeFusedDetector:
    def __init__(self):
        self.calls: list[tuple] = []

    def encode_exemplars_fused(self, ref_images_bgr, ref_boxes, ref_group_ids, context_frames_bgr):
        self.calls.append((ref_images_bgr, ref_boxes, ref_group_ids, context_frames_bgr))
        n_groups = len(set(ref_group_ids))
        return {
            "main": torch.zeros(1, n_groups, 4),
            "l1": torch.zeros(1, n_groups, 4),
            "l2": torch.zeros(1, n_groups, 4),
        }


@pytest.fixture(scope="module")
def synth_fixture(tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("geco2_lsf_fixtures")
    from scripts.make_synthetic_fixture import make_fixture
    make_fixture(out_dir, FIXTURE_ID)
    return out_dir


def _make_cfg(synth_fixture, tmp_path, candidate_factors=(1.0, 0.5), num_context_frames=3,
              use_shape_token=False, seg_enabled=True):
    from aero_eyes.config import (
        AeroEyesConfig, DataConfig, GTConfig, ProjectConfig, RuntimeConfig,
        SegmentationConfig, Stage123Geco2Config, Geco2LearnedScaleFusionConfig,
    )

    lsf_cfg = Geco2LearnedScaleFusionConfig(
        enabled=True, candidate_factors=list(candidate_factors), num_context_frames=num_context_frames,
    )
    return AeroEyesConfig(
        project=ProjectConfig(work_dir=str(tmp_path / "runs"), use_cache=False, seed=42),
        data=DataConfig(
            data_root=str(synth_fixture), refs_subdir="refs", video_glob="*.mp4", num_references=3,
            gt=GTConfig(global_file=str(synth_fixture / FIXTURE_ID / "gt.json")),
        ),
        runtime=RuntimeConfig(save_visualizations=False),
        stage123_geco2=Stage123Geco2Config(
            segmentation=SegmentationConfig(enabled=seg_enabled),
            use_shape_token=use_shape_token,
            learned_scale_fusion=lsf_cfg,
        ),
    )


def test_requires_segmentation_enabled(synth_fixture, tmp_path, monkeypatch):
    cfg = _make_cfg(synth_fixture, tmp_path, seg_enabled=False)
    detector = _FakeFusedDetector()
    with pytest.raises(ValueError, match="segmentation.enabled"):
        build_exemplar_prototype(cfg, FIXTURE_ID, detector, Path(cfg.project.work_dir) / FIXTURE_ID)


def test_requires_shape_token_disabled(synth_fixture, tmp_path):
    cfg = _make_cfg(synth_fixture, tmp_path, use_shape_token=True)
    detector = _FakeFusedDetector()
    with pytest.raises(ValueError, match="use_shape_token"):
        build_exemplar_prototype(cfg, FIXTURE_ID, detector, Path(cfg.project.work_dir) / FIXTURE_ID)


def test_builds_correct_candidate_grid_and_group_ids(synth_fixture, tmp_path, monkeypatch):
    import aero_eyes.models.segmentation as seg_mod
    monkeypatch.setattr(seg_mod, "build_segmenter", lambda seg_cfg, cfg: _FakeSegmenter())

    cfg = _make_cfg(synth_fixture, tmp_path, candidate_factors=(1.0, 0.5, 0.25), num_context_frames=3)
    detector = _FakeFusedDetector()
    prototype = build_exemplar_prototype(cfg, FIXTURE_ID, detector, Path(cfg.project.work_dir) / FIXTURE_ID)

    assert len(detector.calls) == 1
    ref_images_bgr, ref_boxes, ref_group_ids, context_frames_bgr = detector.calls[0]
    assert len(ref_images_bgr) == 3 * 3   # 3 refs x 3 factors
    assert len(ref_boxes) == 9
    assert ref_group_ids == [0, 0, 0, 1, 1, 1, 2, 2, 2]
    assert len(context_frames_bgr) == 3   # num_context_frames
    # Returned prototype is exactly what the fake detector produced.
    assert prototype["main"].shape == (1, 3, 4)  # 3 distinct groups


def test_context_frames_are_deterministic_across_runs(synth_fixture, tmp_path, monkeypatch):
    """Re-running with an unchanged config must sample the SAME context
    frames -- required for project.use_cache to stay safe (see
    Geco2LearnedScaleFusionConfig's docstring / AutoScaleCalibrationConfig's
    equivalent note)."""
    import aero_eyes.models.segmentation as seg_mod
    monkeypatch.setattr(seg_mod, "build_segmenter", lambda seg_cfg, cfg: _FakeSegmenter())

    cfg1 = _make_cfg(synth_fixture, tmp_path, num_context_frames=4)
    d1 = _FakeFusedDetector()
    build_exemplar_prototype(cfg1, FIXTURE_ID, d1, Path(cfg1.project.work_dir) / (FIXTURE_ID + "_a"))

    cfg2 = _make_cfg(synth_fixture, tmp_path, num_context_frames=4)
    d2 = _FakeFusedDetector()
    build_exemplar_prototype(cfg2, FIXTURE_ID, d2, Path(cfg2.project.work_dir) / (FIXTURE_ID + "_b"))

    frames1 = d1.calls[0][3]
    frames2 = d2.calls[0][3]
    assert len(frames1) == len(frames2) == 4
    for f1, f2 in zip(frames1, frames2):
        assert np.array_equal(f1, f2)


def test_warns_when_overriding_manual_knobs(synth_fixture, tmp_path, monkeypatch, caplog):
    import aero_eyes.models.segmentation as seg_mod
    monkeypatch.setattr(seg_mod, "build_segmenter", lambda seg_cfg, cfg: _FakeSegmenter())

    cfg = _make_cfg(synth_fixture, tmp_path)
    cfg.stage123_geco2.ref_downscale_factor = 0.5  # non-default -> should trigger a warning
    detector = _FakeFusedDetector()
    with caplog.at_level("WARNING"):
        build_exemplar_prototype(cfg, FIXTURE_ID, detector, Path(cfg.project.work_dir) / FIXTURE_ID)
    assert any("learned_scale_fusion.enabled overrides" in r.message for r in caplog.records)
