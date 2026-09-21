"""Integration tests for stage3.verification_method="cluster" -- confirms
run_stage3 groups candidates by keyframe, feeds each keyframe's own
candidates + per_ref_features through cluster_verify_candidates, and unions
the result back into detections.json correctly (causal per-keyframe
decisions, no dependency on other keyframes' similarity scores). See
tests/test_cluster_verify.py for the underlying primitive's own unit tests,
and ClusterVerificationConfig's docstring (aero_eyes/config.py) for the
full rationale.
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


@pytest.fixture()
def cluster_stage3_setup(tmp_path):
    """3 exemplars near a single reference appearance (`view`), well
    separated from 2 DIFFERENT confuser directions used on 2 different
    keyframes -- exercises run_stage3's per-keyframe grouping/union and
    that each keyframe's own confuser doesn't need to look anything like
    the other keyframe's confuser (independent per-frame decisions), while
    keeping the underlying clustering scenario itself as robust/well-
    separated as test_cluster_verify.py's own basic case (that file already
    covers the harder overlapping-similarity scenario at the primitive
    level -- this integration test's job is the plumbing around it, not
    re-proving the primitive's own separability guarantees)."""
    rng = np.random.default_rng(0)
    d = 32
    noise = 0.02
    view = np.zeros(d)
    view[0] = 1.0
    confuser_10 = np.zeros(d)
    confuser_10[1] = 1.0
    confuser_20 = np.zeros(d)
    confuser_20[2] = 1.0
    ref = _make_unit(view[None, :] + noise * rng.normal(size=(3, d)))

    work_dir = tmp_path / "sample1"
    work_dir.mkdir(parents=True)

    fused_prototype = ref.mean(axis=0)
    fused_prototype /= np.linalg.norm(fused_prototype)
    write_prototype(fused_prototype, {}, list(ref), work_dir / "prototype.npz")

    def _det(frame_idx, x, feat):
        box = Box(x1=x, y1=10.0, x2=x + 20.0, y2=30.0)
        det = Detection(frame_idx=frame_idx, box=box, similarity=0.0, source="detect")
        det._feature = feat
        return det

    tp10_feats = _make_unit(view[None, :] + noise * rng.normal(size=(2, d)))
    fp10_feats = _make_unit(confuser_10[None, :] + noise * rng.normal(size=(2, d)))
    frame10 = [
        _det(10, 0.0, tp10_feats[0]), _det(10, 50.0, tp10_feats[1]),
        _det(10, 100.0, fp10_feats[0]), _det(10, 150.0, fp10_feats[1]),
    ]

    tp20_feats = _make_unit(view[None, :] + noise * rng.normal(size=(2, d)))
    fp20_feats = _make_unit(confuser_20[None, :] + noise * rng.normal(size=(2, d)))
    frame20 = [
        _det(20, 0.0, tp20_feats[0]), _det(20, 50.0, tp20_feats[1]),
        _det(20, 100.0, fp20_feats[0]), _det(20, 150.0, fp20_feats[1]),
    ]

    candidates = {10: frame10, 20: frame20}
    _write_candidates_with_features(candidates, work_dir / "candidates.json")

    cfg = load_config(
        "configs/config.yaml",
        overrides=[
            f"project.work_dir={tmp_path}",
            "project.use_cache=false",
            "runtime.save_visualizations=false",
            "stage3.verification_method=cluster",
            "stage3.cluster_verification.min_candidates_for_cluster=4",
            "stage3.topk_per_keyframe=10",
        ],
    )
    return cfg, work_dir


def test_cluster_verification_keeps_tp_rejects_fp_per_keyframe(cluster_stage3_setup, caplog):
    cfg, work_dir = cluster_stage3_setup
    with caplog.at_level(logging.INFO):
        det_path = run_stage3(cfg, "sample1")
    detections = read_detections(det_path)

    assert len(detections[10]) == 2, "both frame-10 TPs kept, both frame-10 confusers rejected"
    assert len(detections[20]) == 2, "both frame-20 TPs kept, both frame-20 confusers rejected"
    assert {d.box.x1 for d in detections[10]} == {0.0, 50.0}
    assert {d.box.x1 for d in detections[20]} == {0.0, 50.0}
    assert "verification_method=cluster" in caplog.text


def test_cluster_verification_is_causal_per_keyframe(cluster_stage3_setup):
    """Removing keyframe 20 entirely must not change keyframe 10's own
    verified set -- each keyframe's decision only ever depends on its own
    candidates + per_ref_features, never on another keyframe's data."""
    cfg, work_dir = cluster_stage3_setup

    det_path_both = run_stage3(cfg, "sample1")
    detections_both = read_detections(det_path_both)
    frame10_both = sorted(d.box.x1 for d in detections_both[10])

    # Rebuild candidates.json with ONLY keyframe 10, same feature vectors.
    import json
    with open(work_dir / "candidates.json") as f:
        payload = json.load(f)
    payload["frames"] = {"10": payload["frames"]["10"]}
    with open(work_dir / "candidates.json", "w") as f:
        json.dump(payload, f)
    (work_dir / "detections.json").unlink()

    det_path_only10 = run_stage3(cfg, "sample1")
    detections_only10 = read_detections(det_path_only10)
    frame10_only = sorted(d.box.x1 for d in detections_only10[10])

    assert frame10_both == frame10_only


def test_dynamic_prototype_and_cluster_mode_mutual_exclusion_warns(cluster_stage3_setup, caplog):
    cfg, work_dir = cluster_stage3_setup
    cfg.stage3.dynamic_prototype.enabled = True
    with caplog.at_level(logging.WARNING):
        run_stage3(cfg, "sample1")
    assert any(
        "verification_method='cluster'" in rec.message and "dynamic_prototype" in rec.message
        for rec in caplog.records
    )


def test_fallback_keeps_tp_even_when_far_below_match_threshold(tmp_path):
    """Regression test for a real-world failure: on footage with a severe
    domain gap, an ENTIRE video's candidate-to-exemplar cosine similarity
    can stay well below stage3.match_threshold's default (0.55) -- observed
    in practice at max=0.335 across a whole video. The fallback path (below
    min_candidates_for_cluster) must judge candidates RELATIVE to this
    keyframe's own top similarity, never against that absolute number, or
    it silently rejects every fallback-path keyframe regardless of whether
    a genuine match is present."""
    rng = np.random.default_rng(7)
    d = 32
    noise = 0.02
    view = np.zeros(d)
    view[0] = 1.0
    confuser = np.zeros(d)
    confuser[1] = 1.0
    ref = _make_unit(view[None, :] + noise * rng.normal(size=(3, d)))

    work_dir = tmp_path / "sample1"
    work_dir.mkdir(parents=True)
    fused_prototype = ref.mean(axis=0)
    fused_prototype /= np.linalg.norm(fused_prototype)
    write_prototype(fused_prototype, {}, list(ref), work_dir / "prototype.npz")

    # Simulate a severe domain gap: shrink the TP feature's alignment with
    # the exemplar direction so its absolute cosine sits around 0.3 -- well
    # under match_threshold's default (0.55) -- while still being clearly
    # the BETTER of this keyframe's own 2 candidates (below
    # min_candidates_for_cluster=4, so this hits the fallback path).
    tp_feat = _make_unit((0.3 * view + 0.95 * confuser)[None, :])[0]  # cosine to `view` ~= 0.3
    fp_feat = _make_unit((0.05 * view + 0.999 * confuser)[None, :])[0]  # cosine to `view` ~= 0.05

    def _det(frame_idx, x, feat):
        box = Box(x1=x, y1=10.0, x2=x + 20.0, y2=30.0)
        det = Detection(frame_idx=frame_idx, box=box, similarity=0.0, source="detect")
        det._feature = feat
        return det

    candidates = {10: [_det(10, 0.0, tp_feat), _det(10, 50.0, fp_feat)]}
    _write_candidates_with_features(candidates, work_dir / "candidates.json")

    cfg = load_config(
        "configs/config.yaml",
        overrides=[
            f"project.work_dir={tmp_path}",
            "project.use_cache=false",
            "runtime.save_visualizations=false",
            "stage3.verification_method=cluster",
            "stage3.match_threshold=0.55",
            "stage3.cluster_verification.min_candidates_for_cluster=4",
            "stage3.cluster_verification.fallback_relative_ratio=0.9",
        ],
    )
    # Sanity-check the premise: the TP's own absolute cosine must be well
    # under match_threshold, so a fixed-threshold fallback would reject it.
    tp_sim = float(tp_feat @ fused_prototype)
    assert tp_sim < cfg.stage3.match_threshold, "test setup didn't actually simulate a domain gap"

    det_path = run_stage3(cfg, "sample1")
    detections = read_detections(det_path)

    assert len(detections[10]) == 1
    assert detections[10][0].box.x1 == 0.0, "the TP (better of the 2, despite low absolute cosine) must survive"
