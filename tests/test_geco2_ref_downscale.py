"""Unit test for build_exemplar_prototype's ref_downscale_factor/
ref_downscale_levels: previously a no-op whenever stage123_geco2.
segmentation.enabled was false (the code that applied it lived entirely
inside the segmentation-enabled branch) -- now applies to the RAW
(unmasked) reference images in that case too. Uses a fake GeCo2Detector so
this runs without the real GECO2 repo/weights/GPU."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="torch not importable", exc_type=ImportError)

from aero_eyes.stages.stage123_geco2 import build_exemplar_prototype


class _FakeDetector:
    """Records encode_exemplars' arguments; returns a minimal torch dict
    save_prototype can serialize."""

    def __init__(self):
        self.calls: list[tuple] = []

    def encode_exemplars(self, ref_images_bgr, ref_boxes=None):
        self.calls.append((ref_images_bgr, ref_boxes))
        return {"main": torch.zeros(1, 1, 4), "l1": torch.zeros(1, 1, 4), "l2": torch.zeros(1, 1, 4)}


def _make_cfg(tmp_path: Path, sample_id: str, ref_downscale_factor: float, seg_enabled: bool,
              ref_downscale_levels=None) -> SimpleNamespace:
    data_root = tmp_path / "data"
    refs_dir = data_root / sample_id / "refs"
    refs_dir.mkdir(parents=True, exist_ok=True)
    for i in range(3):
        img = np.full((80, 80, 3), 255 - i * 20, dtype=np.uint8)  # distinct per ref
        cv2.imwrite(str(refs_dir / f"ref_{i}.jpg"), img)

    return SimpleNamespace(
        project=SimpleNamespace(use_cache=False, work_dir=str(tmp_path / f"work_{ref_downscale_factor}_{ref_downscale_levels}")),
        data=SimpleNamespace(data_root=str(data_root), refs_subdir="refs", num_references=3),
        runtime=SimpleNamespace(save_visualizations=False),
        stage123_geco2=SimpleNamespace(
            prototype_cache_name="geco2_prototype.pt",
            segmentation=SimpleNamespace(enabled=seg_enabled),
            scale_calibration=SimpleNamespace(enabled=False),
            ref_downscale_factor=ref_downscale_factor,
            ref_downscale_levels=ref_downscale_levels,
            crop_to_object=False,
            domain_calibration=SimpleNamespace(enabled=False),
            use_shape_token=False,
        ),
    )


def test_ref_downscale_factor_applies_when_segmentation_disabled(tmp_path):
    """The bug: with segmentation.enabled=false, changing ref_downscale_
    factor used to have ZERO effect -- the ref images passed to
    encode_exemplars must now actually differ in size between two
    different factors."""
    sample_id = "IDCard_0"
    detector_noop = _FakeDetector()
    cfg_noop = _make_cfg(tmp_path, sample_id, ref_downscale_factor=1.0, seg_enabled=False)
    build_exemplar_prototype(cfg_noop, sample_id, detector_noop, Path(cfg_noop.project.work_dir) / sample_id)

    detector_shrunk = _FakeDetector()
    cfg_shrunk = _make_cfg(tmp_path, sample_id, ref_downscale_factor=0.1, seg_enabled=False)
    build_exemplar_prototype(cfg_shrunk, sample_id, detector_shrunk, Path(cfg_shrunk.project.work_dir) / sample_id)

    imgs_noop = detector_noop.calls[0][0]
    imgs_shrunk = detector_shrunk.calls[0][0]
    assert len(imgs_noop) == len(imgs_shrunk) == 3
    # A 0.1 downscale-then-upscale-back-to-original-size round trip must
    # visibly differ from the untouched original -- if factor=0.1 were
    # silently ignored (the bug), the two calls would be pixel-identical.
    assert not np.array_equal(imgs_noop[0], imgs_shrunk[0])


def test_ref_downscale_levels_multiplies_exemplar_count_when_segmentation_disabled(tmp_path):
    sample_id = "IDCard_0"
    detector = _FakeDetector()
    cfg = _make_cfg(
        tmp_path, sample_id, ref_downscale_factor=1.0, seg_enabled=False,
        ref_downscale_levels=[1.0, 0.3],
    )
    build_exemplar_prototype(cfg, sample_id, detector, Path(cfg.project.work_dir) / sample_id)

    imgs = detector.calls[0][0]
    assert len(imgs) == 3 * 2  # 3 ref images x 2 levels


def test_refs_final_viz_saved_when_enabled(tmp_path):
    """The debug viz saves the ACTUAL post-downscale images that
    encode_exemplars receives -- one file per exemplar entry, matching
    the count encode_exemplars was actually called with."""
    sample_id = "IDCard_0"
    detector = _FakeDetector()
    cfg = _make_cfg(
        tmp_path, sample_id, ref_downscale_factor=0.1, seg_enabled=False,
        ref_downscale_levels=[1.0, 0.3],
    )
    cfg.runtime.save_visualizations = True
    work_dir = Path(cfg.project.work_dir) / sample_id
    build_exemplar_prototype(cfg, sample_id, detector, work_dir)

    viz_dir = work_dir / "viz" / "stage123_geco2" / "refs_final"
    files = sorted(f.name for f in viz_dir.glob("ref_final_*.jpg") if "_box" not in f.name)
    assert len(files) == len(detector.calls[0][0]) == 6  # 3 refs x 2 levels

    # Must be the ACTUAL images passed to encode_exemplars, not some other
    # intermediate copy.
    saved = cv2.imread(str(viz_dir / files[0]))
    assert saved.shape == detector.calls[0][0][0].shape


def test_refs_final_viz_not_saved_when_disabled(tmp_path):
    sample_id = "IDCard_0"
    detector = _FakeDetector()
    cfg = _make_cfg(tmp_path, sample_id, ref_downscale_factor=1.0, seg_enabled=False)
    work_dir = Path(cfg.project.work_dir) / sample_id
    build_exemplar_prototype(cfg, sample_id, detector, work_dir)

    assert not (work_dir / "viz" / "stage123_geco2" / "refs_final").exists()
