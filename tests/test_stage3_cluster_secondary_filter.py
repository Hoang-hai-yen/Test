"""Integration tests for stage3.cluster_secondary_filter -- the precision-
focused add-on to verification_method="threshold" (see
ClusterSecondaryFilterConfig's own docstring in aero_eyes/config.py for the
full empirical rationale: standalone verification_method="cluster"
underperformed threshold-based verification on this project's own real
footage, and this add-on borrows threshold's winning ingredient --
multi-keyframe aggregate context, via a rolling window of recently-accepted
candidates -- instead of repeating cluster's per-keyframe-isolation mistake).
"""
from __future__ import annotations

import logging

import numpy as np
import pytest

from aero_eyes.config import load_config
from aero_eyes.stages.stage2 import _write_candidates_with_features
from aero_eyes.stages.stage3 import run_stage3
from aero_eyes.types import Box, Detection
from aero_eyes.utils.io import read_detections, write_prototype


def _make_unit(vecs: np.ndarray) -> np.ndarray:
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


def _det(frame_idx: int, x: float, feat: np.ndarray) -> Detection:
    box = Box(x1=x, y1=10.0, x2=x + 20.0, y2=30.0)
    det = Detection(frame_idx=frame_idx, box=box, similarity=0.0, source="detect")
    det._feature = feat
    return det


@pytest.fixture()
def secondary_filter_setup(tmp_path):
    """3 exemplars near `view`; genuine TP candidate features also near
    `view`; confuser features that clear a flat match_threshold=0.5 on raw
    cosine ALONE (cosine to `view` ~= 0.7, a real but moderate match) yet
    sit structurally far from the tight exemplar/TP cluster -- exactly the
    "clears the threshold bar but doesn't structurally resemble anything
    confirmed real" case this filter targets. Multiple points per group
    (not just 1-2) since HDBSCAN needs enough density to cluster reliably
    -- a single candidate vs. 3 exemplars (4 total points) is too few for
    it to trust any structure at all (confirmed empirically while writing
    these tests), same class of small-N fragility documented in
    tests/test_cluster_verify.py.
    """
    d = 16
    view = np.zeros(d)
    view[0] = 1.0
    confuser_dir = np.zeros(d)
    confuser_dir[0] = 0.7
    confuser_dir[1] = 0.7
    confuser_dir = confuser_dir / np.linalg.norm(confuser_dir)

    rng = np.random.default_rng(0)
    noise = 0.02
    ref = _make_unit(view[None, :] + noise * rng.normal(size=(3, d)))
    fused_ref = ref.mean(axis=0)
    fused_ref /= np.linalg.norm(fused_ref)

    def make_tp(n, seed_offset=0):
        r = np.random.default_rng(10 + seed_offset)
        return _make_unit(view[None, :] + noise * r.normal(size=(n, d)))

    def make_confuser(n, seed_offset=0):
        r = np.random.default_rng(20 + seed_offset)
        return _make_unit(confuser_dir[None, :] + noise * r.normal(size=(n, d)))

    # Sanity-check the premise every test relies on.
    sample_confuser = make_confuser(1)[0]
    sample_tp = make_tp(1)[0]
    assert float(sample_confuser @ fused_ref) > 0.5, "confuser must clear the flat match_threshold used below"
    assert float(sample_confuser @ sample_tp) < 0.8, "confuser must be structurally distinct from genuine TP"

    work_dir = tmp_path / "sample1"
    work_dir.mkdir(parents=True)
    write_prototype(fused_ref, {}, list(ref), work_dir / "prototype.npz")

    return work_dir, ref, fused_ref, make_tp, make_confuser


def _base_overrides(tmp_path) -> list[str]:
    return [
        f"project.work_dir={tmp_path}",
        "project.use_cache=false",
        "runtime.save_visualizations=false",
        "stage3.verification_method=threshold",
        "stage3.match_threshold=0.5",
        "stage3.topk_per_keyframe=10",
        "stage3.cluster_secondary_filter.enabled=true",
        "stage3.cluster_secondary_filter.cluster_verification.min_candidates_for_cluster=2",
    ]


def test_rejects_confuser_that_clears_threshold_but_not_cluster(secondary_filter_setup, tmp_path, caplog):
    work_dir, ref, fused_ref, make_tp, make_confuser = secondary_filter_setup

    tp_feats = make_tp(3)
    confuser_feats = make_confuser(2)
    dets = (
        [_det(10, float(i * 20), f) for i, f in enumerate(tp_feats)]
        + [_det(10, float(100 + i * 20), f) for i, f in enumerate(confuser_feats)]
    )
    _write_candidates_with_features({10: dets}, work_dir / "candidates.json")

    cfg = load_config("configs/config.yaml", overrides=_base_overrides(tmp_path))
    with caplog.at_level(logging.INFO):
        det_path = run_stage3(cfg, "sample1")
    detections = read_detections(det_path)

    assert len(detections[10]) == 3, "only the 3 genuine TPs should survive the secondary filter"
    assert {d.box.x1 for d in detections[10]} == {0.0, 20.0, 40.0}
    assert "cluster_secondary_filter" in caplog.text


def test_noop_when_disabled(secondary_filter_setup, tmp_path):
    """Same scene, filter OFF -- both groups (TP and confuser) must survive
    the flat threshold, proving the rejection above came from the filter,
    not from match_threshold itself."""
    work_dir, ref, fused_ref, make_tp, make_confuser = secondary_filter_setup

    tp_feats = make_tp(3)
    confuser_feats = make_confuser(2)
    dets = (
        [_det(10, float(i * 20), f) for i, f in enumerate(tp_feats)]
        + [_det(10, float(100 + i * 20), f) for i, f in enumerate(confuser_feats)]
    )
    _write_candidates_with_features({10: dets}, work_dir / "candidates.json")

    overrides = [o for o in _base_overrides(tmp_path) if not o.startswith("stage3.cluster_secondary_filter")]
    cfg = load_config("configs/config.yaml", overrides=overrides)
    det_path = run_stage3(cfg, "sample1")
    detections = read_detections(det_path)

    assert len(detections[10]) == 5, "without the filter, both groups should pass the flat threshold"


def test_noop_below_min_candidates_for_cluster(secondary_filter_setup, tmp_path):
    """A keyframe with too few threshold-survivors (< min_candidates_for_
    cluster) -- too little evidence to cluster meaningfully -- must keep
    the threshold's own decision unchanged rather than guessing."""
    work_dir, ref, fused_ref, make_tp, make_confuser = secondary_filter_setup

    confuser_feats = make_confuser(2)  # both clear match_threshold=0.5 alone
    dets = [_det(10, float(100 + i * 20), f) for i, f in enumerate(confuser_feats)]
    _write_candidates_with_features({10: dets}, work_dir / "candidates.json")

    overrides = _base_overrides(tmp_path) + [
        "stage3.cluster_secondary_filter.cluster_verification.min_candidates_for_cluster=4",
    ]
    cfg = load_config("configs/config.yaml", overrides=overrides)
    det_path = run_stage3(cfg, "sample1")
    detections = read_detections(det_path)

    assert len(detections[10]) == 2, "too few survivors to cluster -- filter must be a no-op here"


def test_rolling_window_carries_trusted_candidates_across_keyframes(secondary_filter_setup, tmp_path):
    """Keyframe 10 has ONLY genuine TPs (accepted, added to the trusted
    window). Keyframe 20 has ONLY confusers (clear match_threshold alone,
    and are structurally distinct from the 3 static exemplars too) --
    confirms the filter correctly rejects them using the accumulated
    context (exemplars + trusted window from keyframe 10), and that this
    multi-keyframe pipeline wiring behaves consistently end-to-end."""
    work_dir, ref, fused_ref, make_tp, make_confuser = secondary_filter_setup

    tp_feats = make_tp(3)
    confuser_feats = make_confuser(3, seed_offset=1)
    candidates = {
        10: [_det(10, float(i * 20), f) for i, f in enumerate(tp_feats)],
        20: [_det(20, float(i * 20), f) for i, f in enumerate(confuser_feats)],
    }
    _write_candidates_with_features(candidates, work_dir / "candidates.json")

    cfg = load_config("configs/config.yaml", overrides=_base_overrides(tmp_path))
    det_path = run_stage3(cfg, "sample1")
    detections = read_detections(det_path)

    assert len(detections[10]) == 3, "the genuine TPs (cluster with the exemplars) should survive"
    assert len(detections[20]) == 0, "the confusers (cluster with neither exemplars nor the trusted window) should be rejected"


def test_no_effect_when_verification_method_is_cluster(secondary_filter_setup, tmp_path):
    """cluster_secondary_filter is defined as an add-on to
    verification_method="threshold" ONLY -- must be inert when
    verification_method="cluster" already made the primary decision."""
    work_dir, ref, fused_ref, make_tp, make_confuser = secondary_filter_setup

    tp_feats = make_tp(3)
    dets = [_det(10, float(i * 20), f) for i, f in enumerate(tp_feats)]
    _write_candidates_with_features({10: dets}, work_dir / "candidates.json")

    overrides = [o for o in _base_overrides(tmp_path) if not o.startswith("stage3.verification_method")]
    overrides += ["stage3.verification_method=cluster"]
    cfg = load_config("configs/config.yaml", overrides=overrides)
    # Must not raise even though verification_method="cluster" bypasses the
    # secondary filter entirely (asserting only that this combination is
    # safe, not any particular keep/reject outcome).
    run_stage3(cfg, "sample1")
