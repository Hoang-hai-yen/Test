"""Tests for stage3.identity_chain_filter -- KeepTrack-style (arXiv:2103.16556)
multi-candidate identity tracking across keyframes. See
IdentityChainFilterConfig's own docstring (aero_eyes/config.py) and
apply_identity_chain_filter's own docstring (aero_eyes/stages/stage3.py)
for the full rationale.
"""
from __future__ import annotations

import numpy as np
import pytest

from aero_eyes.config import IdentityChainFilterConfig, load_config
from aero_eyes.stages.stage2 import _write_candidates_with_features
from aero_eyes.stages.stage3 import apply_identity_chain_filter, run_stage3
from aero_eyes.types import Box, Detection
from aero_eyes.utils.io import read_detections, write_prototype


def _make_unit(vecs: np.ndarray) -> np.ndarray:
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


def _det(frame_idx: int, x: float, feat: np.ndarray) -> Detection:
    box = Box(x1=x, y1=10.0, x2=x + 20.0, y2=30.0)
    det = Detection(frame_idx=frame_idx, box=box, similarity=0.0, source="detect")
    det._feature = feat
    return det


def test_long_consistent_chain_is_kept_short_one_off_is_dropped():
    """A candidate reappearing with consistent appearance across several
    keyframes forms a long chain (kept); a one-off high-scoring confuser
    that never recurs forms a length-1 chain (dropped)."""
    rng = np.random.default_rng(0)
    d = 8
    real = np.zeros(d)
    real[0] = 1.0
    confuser = np.zeros(d)
    confuser[1] = 1.0

    all_feats_list, all_frame_idxs, all_dets, all_sims_list = [], [], [], []
    for i, fi in enumerate([10, 20, 30, 40]):
        feat = _make_unit((real[None, :] + 0.02 * rng.normal(size=(1, d))))[0]
        all_feats_list.append(feat)
        all_frame_idxs.append(fi)
        all_dets.append(_det(fi, 0.0, feat))
        all_sims_list.append(0.9)
    # A one-off confuser at frame 50, never recurring, with the HIGHEST raw score.
    confuser_feat = _make_unit((confuser[None, :] + 0.02 * rng.normal(size=(1, d))))[0]
    all_feats_list.append(confuser_feat)
    all_frame_idxs.append(50)
    all_dets.append(_det(50, 100.0, confuser_feat))
    all_sims_list.append(0.99)

    all_feats = np.stack(all_feats_list)
    all_sims = np.array(all_sims_list)
    keep_mask = np.ones(len(all_feats), dtype=bool)

    cfg = IdentityChainFilterConfig(enabled=True, top_k_per_keyframe=5, min_chain_length=3, spatial_weight=0.0)
    new_keep_mask, n_chains_total, n_chains_kept = apply_identity_chain_filter(
        all_feats, all_frame_idxs, all_dets, all_sims, keep_mask, cfg,
    )

    assert new_keep_mask[:4].all(), "the 4-keyframe-long consistent chain should be kept"
    assert not new_keep_mask[4], "the one-off confuser (chain length 1) should be dropped despite its higher score"
    assert n_chains_kept == 1


def test_chain_breaks_on_dissimilar_appearance():
    """A candidate that looks nothing like the previous keyframe's tail
    must NOT extend that chain -- it starts its own new (short) chain."""
    d = 8
    real = np.zeros(d)
    real[0] = 1.0
    other = np.zeros(d)
    other[1] = 1.0

    all_feats = np.stack([real, real, other])  # frame 10, 20: same identity; frame 30: unrelated
    all_frame_idxs = [10, 20, 30]
    all_dets = [_det(10, 0.0, real), _det(20, 0.0, real), _det(30, 0.0, other)]
    all_sims = np.array([0.9, 0.9, 0.9])
    keep_mask = np.ones(3, dtype=bool)

    cfg = IdentityChainFilterConfig(enabled=True, top_k_per_keyframe=5, min_chain_length=2, spatial_weight=0.0)
    new_keep_mask, n_chains_total, n_chains_kept = apply_identity_chain_filter(
        all_feats, all_frame_idxs, all_dets, all_sims, keep_mask, cfg,
    )

    assert n_chains_total == 2, "the dissimilar frame-30 candidate must start its own chain, not extend the first"
    assert new_keep_mask[0] and new_keep_mask[1], "the 2-long real chain should be kept"
    assert not new_keep_mask[2], "the length-1 chain (frame 30 alone) should be dropped"


def test_top_k_per_keyframe_truncation():
    """Only the top_k_per_keyframe highest-similarity candidates per
    keyframe are even considered for chain-building."""
    d = 8
    real = np.zeros(d)
    real[0] = 1.0
    feats = np.stack([real, real, real])  # 3 identical candidates, same keyframe
    all_frame_idxs = [10, 10, 10]
    all_dets = [_det(10, float(i * 20), real) for i in range(3)]
    all_sims = np.array([0.9, 0.5, 0.1])
    keep_mask = np.ones(3, dtype=bool)

    cfg = IdentityChainFilterConfig(enabled=True, top_k_per_keyframe=1, min_chain_length=1, spatial_weight=0.0)
    new_keep_mask, n_chains_total, n_chains_kept = apply_identity_chain_filter(
        feats, all_frame_idxs, all_dets, all_sims, keep_mask, cfg,
    )
    assert n_chains_total == 1, "top_k_per_keyframe=1 should only consider the single highest-similarity candidate"
    assert new_keep_mask.sum() == 1
    assert new_keep_mask[0]  # the highest-similarity one


def test_empty_input_returns_empty():
    cfg = IdentityChainFilterConfig(enabled=True)
    keep_mask = np.zeros(0, dtype=bool)
    new_keep_mask, n_chains_total, n_chains_kept = apply_identity_chain_filter(
        np.zeros((0, 8)), [], [], np.zeros(0), keep_mask, cfg,
    )
    assert new_keep_mask.shape == (0,)
    assert n_chains_total == 0


@pytest.fixture()
def identity_chain_stage3_setup(tmp_path):
    d = 8
    rng = np.random.default_rng(1)
    view = np.zeros(d)
    view[0] = 1.0
    ref = _make_unit(view[None, :] + 0.02 * rng.normal(size=(3, d)))
    fused_ref = ref.mean(axis=0)
    fused_ref /= np.linalg.norm(fused_ref)

    work_dir = tmp_path / "sample1"
    work_dir.mkdir(parents=True)
    write_prototype(fused_ref, {}, list(ref), work_dir / "prototype.npz")
    return work_dir, view, rng


def test_run_stage3_end_to_end_with_identity_chain_filter(identity_chain_stage3_setup, tmp_path, caplog):
    import logging

    work_dir, view, rng = identity_chain_stage3_setup
    d = view.shape[0]

    # A consistent 3-keyframe chain (the real target) + a single-keyframe confuser.
    candidates = {}
    for fi in (10, 20, 30):
        feat = _make_unit((view[None, :] + 0.02 * rng.normal(size=(1, d))))[0]
        candidates[fi] = [_det(fi, 0.0, feat)]
    confuser_dir = np.zeros(d)
    confuser_dir[1] = 1.0
    confuser_feat = _make_unit((confuser_dir[None, :] + 0.02 * rng.normal(size=(1, d))))[0]
    candidates[40] = [_det(40, 100.0, confuser_feat)]

    _write_candidates_with_features(candidates, work_dir / "candidates.json")

    cfg = load_config(
        "configs/config.yaml",
        overrides=[
            f"project.work_dir={tmp_path}",
            "project.use_cache=false",
            "runtime.save_visualizations=false",
            "stage3.verification_method=threshold",
            "stage3.match_threshold=0.3",
            "stage3.topk_per_keyframe=10",
            "stage3.identity_chain_filter.enabled=true",
            "stage3.identity_chain_filter.min_chain_length=2",
        ],
    )
    with caplog.at_level(logging.INFO):
        det_path = run_stage3(cfg, "sample1")
    detections = read_detections(det_path)

    assert len(detections[10]) == 1 and len(detections[20]) == 1 and len(detections[30]) == 1
    assert len(detections[40]) == 0, "the one-off confuser (chain length 1) should be dropped"
    assert "identity_chain_filter" in caplog.text
