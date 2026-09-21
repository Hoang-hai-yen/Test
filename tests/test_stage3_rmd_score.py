"""Tests for stage3.similarity="rmd" (Relative Mahalanobis Distance) --
docs/GECO2_precision_improvements_plan.md Phase 2 item 3. See
_fit_rmd_background / _score_against_ref's own docstrings in
aero_eyes/stages/stage3.py for the full rationale.
"""
from __future__ import annotations

import numpy as np
import pytest

from aero_eyes.config import load_config
from aero_eyes.stages.stage2 import _write_candidates_with_features
from aero_eyes.stages.stage3 import _fit_rmd_background, _mahalanobis_sq, _score_against_ref, run_stage3
from aero_eyes.types import Box, Detection
from aero_eyes.utils.io import read_detections, write_prototype


def _make_unit(vecs: np.ndarray) -> np.ndarray:
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


def test_mahalanobis_sq_zero_at_mean():
    mu = np.array([1.0, 2.0, 3.0])
    precision = np.eye(3)
    d = _mahalanobis_sq(mu[None, :], mu, precision)
    assert d[0] == pytest.approx(0.0, abs=1e-8)


def test_mahalanobis_sq_matches_euclidean_for_identity_precision():
    mu = np.zeros(3)
    precision = np.eye(3)
    x = np.array([[3.0, 4.0, 0.0]])
    d = _mahalanobis_sq(x, mu, precision)
    assert d[0] == pytest.approx(25.0)  # 3^2 + 4^2, identity precision reduces to squared Euclidean


def test_score_against_ref_requires_background_for_rmd():
    feats = np.random.default_rng(0).normal(size=(5, 4))
    ref = np.zeros(4)
    with pytest.raises(ValueError, match="requires background stats"):
        _score_against_ref(feats, ref, "rmd")


def test_rmd_scores_higher_for_candidates_near_exemplar_than_background():
    rng = np.random.default_rng(1)
    d = 8
    background_center = np.zeros(d)
    exemplar = np.zeros(d)
    exemplar[0] = 3.0  # far from background center, in a consistent direction

    # The video's own candidate pool: mostly background-like, plus a few near the exemplar.
    background_like = background_center[None, :] + 0.3 * rng.normal(size=(30, d))
    near_exemplar = exemplar[None, :] + 0.1 * rng.normal(size=(5, d))
    all_feats = np.concatenate([background_like, near_exemplar], axis=0)

    background = _fit_rmd_background(all_feats)
    scores = _score_against_ref(all_feats, exemplar, "rmd", background=background)

    assert scores[-5:].mean() > scores[:30].mean(), "candidates near the exemplar should score higher than background-like ones"


def test_dynamic_prototype_and_cluster_fallback_reuse_same_background():
    """Regression guard: RMD background must be threaded through to
    run_dynamic_prototype_rounds and the cluster-mode fallback, not just
    the main scoring path -- both must not raise when similarity='rmd'."""
    rng = np.random.default_rng(2)
    d = 8
    exemplar = np.zeros(d)
    exemplar[0] = 3.0
    all_feats = exemplar[None, :] + 0.5 * rng.normal(size=(20, d))
    background = _fit_rmd_background(all_feats)

    from aero_eyes.config import DynamicPrototypeConfig
    from aero_eyes.stages.stage3 import run_dynamic_prototype_rounds

    dp = DynamicPrototypeConfig(enabled=True, rounds=1, min_support=1, high_conf_abs_floor=-1e9)
    all_sims = _score_against_ref(all_feats, exemplar, "rmd", background=background)
    # Should not raise (background correctly threaded through the re-scoring round).
    run_dynamic_prototype_rounds(
        "sample1", all_feats, all_sims, exemplar, [exemplar], False, "mean", "rmd", dp,
        background=background,
    )


@pytest.fixture()
def rmd_stage3_setup(tmp_path):
    d = 8
    rng = np.random.default_rng(3)
    exemplar = np.zeros(d)
    exemplar[0] = 3.0
    ref = exemplar[None, :] + 0.05 * rng.normal(size=(3, d))

    work_dir = tmp_path / "sample1"
    work_dir.mkdir(parents=True)
    fused = ref.mean(axis=0)
    write_prototype(fused, {}, list(ref), work_dir / "prototype.npz")
    return work_dir, exemplar, rng


def test_run_stage3_end_to_end_with_rmd_similarity(rmd_stage3_setup, tmp_path):
    work_dir, exemplar, rng = rmd_stage3_setup
    d = exemplar.shape[0]

    def _det(frame_idx, x, feat):
        box = Box(x1=x, y1=10.0, x2=x + 20.0, y2=30.0)
        det = Detection(frame_idx=frame_idx, box=box, similarity=0.0, source="detect")
        det._feature = feat
        return det

    candidates = {}
    # Many background-like candidates across several keyframes (feeds the
    # RMD background fit) plus one clearly-near-exemplar candidate.
    for fi in range(0, 100, 10):
        feat = 0.3 * rng.normal(size=d)
        candidates[fi] = [_det(fi, 0.0, feat)]
    tp_feat = exemplar + 0.05 * rng.normal(size=d)
    candidates[100] = [_det(100, 0.0, tp_feat)]

    _write_candidates_with_features(candidates, work_dir / "candidates.json")

    cfg = load_config(
        "configs/config.yaml",
        overrides=[
            f"project.work_dir={tmp_path}",
            "project.use_cache=false",
            "runtime.save_visualizations=false",
            "stage3.similarity=rmd",
            "stage3.verification_method=threshold",
            "stage3.match_threshold=-1e9",  # RMD's scale differs from cosine -- keep everything through threshold for this smoke test
            "stage3.topk_per_keyframe=10",
        ],
    )
    det_path = run_stage3(cfg, "sample1")
    detections = read_detections(det_path)
    # Must not raise, and the near-exemplar candidate must score higher
    # than the background-like ones.
    assert detections[100][0].similarity > detections[0][0].similarity
