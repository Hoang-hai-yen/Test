"""Integration tests for stage3.margin_verification -- margin-over-runner-up
(WildFusion, arXiv:2608.02469): a keyframe's top accepted candidate is only
trusted if it has a clear similarity margin over the runner-up. See
MarginVerificationConfig's own docstring in aero_eyes/config.py.
"""
from __future__ import annotations

import numpy as np
import pytest

from aero_eyes.config import load_config
from aero_eyes.stages.stage2 import _write_candidates_with_features
from aero_eyes.stages.stage3 import run_stage3
from aero_eyes.types import Box, Detection
from aero_eyes.utils.io import read_detections, write_prototype


def _make_unit(vecs: np.ndarray) -> np.ndarray:
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


def _at_cosine(ref_unit: np.ndarray, cos_theta: float, orth: np.ndarray) -> np.ndarray:
    """A unit vector with EXACT cosine similarity `cos_theta` to `ref_unit`
    -- scaling a unit vector by a positive scalar does NOT change its
    direction after re-normalization, so mixing in an orthogonal
    component is required to get a controlled, distinct cosine value."""
    sin_theta = (1.0 - cos_theta ** 2) ** 0.5
    v = cos_theta * ref_unit + sin_theta * orth
    return v / np.linalg.norm(v)


def _det(frame_idx: int, x: float, feat: np.ndarray) -> Detection:
    box = Box(x1=x, y1=10.0, x2=x + 20.0, y2=30.0)
    det = Detection(frame_idx=frame_idx, box=box, similarity=0.0, source="detect")
    det._feature = feat
    return det


@pytest.fixture()
def margin_setup(tmp_path):
    d = 16
    view = np.zeros(d)
    view[0] = 1.0
    rng = np.random.default_rng(0)
    ref = _make_unit(view[None, :] + 0.01 * rng.normal(size=(3, d)))
    fused_ref = ref.mean(axis=0)
    fused_ref /= np.linalg.norm(fused_ref)

    # A vector orthogonal to fused_ref (Gram-Schmidt against a basis
    # vector not aligned with it) -- used by _at_cosine to construct
    # candidates with an EXACT, controlled cosine similarity.
    raw_orth = np.zeros(d)
    raw_orth[-1] = 1.0
    raw_orth = raw_orth - float(raw_orth @ fused_ref) * fused_ref
    orth = raw_orth / np.linalg.norm(raw_orth)

    work_dir = tmp_path / "sample1"
    work_dir.mkdir(parents=True)
    write_prototype(fused_ref, {}, list(ref), work_dir / "prototype.npz")
    return work_dir, fused_ref, orth


def _base_overrides(tmp_path) -> list[str]:
    return [
        f"project.work_dir={tmp_path}",
        "project.use_cache=false",
        "runtime.save_visualizations=false",
        "stage3.verification_method=threshold",
        "stage3.match_threshold=0.3",
        "stage3.topk_per_keyframe=10",
        "stage3.margin_verification.enabled=true",
        "stage3.margin_verification.tau_margin=0.1",
    ]


def test_drops_ambiguous_keyframe_below_margin(margin_setup, tmp_path, caplog):
    import logging

    work_dir, fused_ref, orth = margin_setup
    # Two candidates whose cosine to the exemplar differ by less than
    # tau_margin=0.1 -- both clear match_threshold=0.3, but the keyframe is
    # ambiguous: cannot tell which (if either) is the genuine match.
    feat_a = _at_cosine(fused_ref, 0.50, orth)
    feat_b = _at_cosine(fused_ref, 0.45, orth)
    dets = [_det(10, 0.0, feat_a), _det(10, 50.0, feat_b)]
    _write_candidates_with_features({10: dets}, work_dir / "candidates.json")

    cfg = load_config("configs/config.yaml", overrides=_base_overrides(tmp_path))
    with caplog.at_level(logging.INFO):
        det_path = run_stage3(cfg, "sample1")
    detections = read_detections(det_path)

    assert len(detections[10]) == 0, "an ambiguous keyframe (no clear margin) should be dropped entirely"
    assert "margin_verification" in caplog.text


def test_keeps_keyframe_with_clear_margin(margin_setup, tmp_path):
    work_dir, fused_ref, orth = margin_setup
    # Top candidate clearly ahead of the runner-up (margin well above 0.1).
    feat_top = _at_cosine(fused_ref, 0.95, orth)
    feat_runner_up = _at_cosine(fused_ref, 0.40, orth)
    dets = [_det(10, 0.0, feat_top), _det(10, 50.0, feat_runner_up)]
    _write_candidates_with_features({10: dets}, work_dir / "candidates.json")

    cfg = load_config("configs/config.yaml", overrides=_base_overrides(tmp_path))
    det_path = run_stage3(cfg, "sample1")
    detections = read_detections(det_path)

    assert len(detections[10]) == 1
    assert detections[10][0].box.x1 == 0.0


def test_single_survivor_keyframe_unaffected(margin_setup, tmp_path):
    """A keyframe with only 1 threshold-surviving candidate has no
    runner-up to compare against -- margin_verification must never touch
    it, regardless of tau_margin."""
    work_dir, fused_ref, orth = margin_setup
    feat = _at_cosine(fused_ref, 0.50, orth)
    _write_candidates_with_features({10: [_det(10, 0.0, feat)]}, work_dir / "candidates.json")

    cfg = load_config("configs/config.yaml", overrides=_base_overrides(tmp_path))
    det_path = run_stage3(cfg, "sample1")
    detections = read_detections(det_path)

    assert len(detections[10]) == 1


def test_noop_when_disabled(margin_setup, tmp_path):
    work_dir, fused_ref, orth = margin_setup
    feat_a = _at_cosine(fused_ref, 0.50, orth)
    feat_b = _at_cosine(fused_ref, 0.45, orth)
    dets = [_det(10, 0.0, feat_a), _det(10, 50.0, feat_b)]
    _write_candidates_with_features({10: dets}, work_dir / "candidates.json")

    overrides = [o for o in _base_overrides(tmp_path) if not o.startswith("stage3.margin_verification")]
    cfg = load_config("configs/config.yaml", overrides=overrides)
    det_path = run_stage3(cfg, "sample1")
    detections = read_detections(det_path)

    assert len(detections[10]) == 2, "without margin_verification, both ambiguous candidates should survive"
