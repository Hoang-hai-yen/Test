"""Unit tests for stage123_geco2.dynamic_prototype.cluster_verification --
GeCo2DynamicPrototypeTracker.offer_topk's DAVE-style alternative to
topk_fusion's Z-score fusion (aero_eyes/models/geco2_detector.py's
_offer_topk_cluster). See ClusterVerificationConfig's own docstring
(aero_eyes/config.py) for the shared mechanism, and tests/test_cluster_
verify.py for the underlying primitive's own unit tests.

Reuses tests/test_geco2_dynamic_prototype.py's fake-config/fake-detector
scaffolding so this runs without the real GECO2 repo, weights, or GPU.
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="torch not importable", exc_type=ImportError)

from aero_eyes.types import Box
from tests.test_geco2_dynamic_prototype import _FakeDetector, _make_cfg, _make_tracker, _token_set


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def _scene(rng, d=16, noise=0.02, tp_scores=(0.3, 0.4), fp_scores=(0.9, 0.95)):
    """2 genuine (TP) candidates near the exemplar direction + 2 confuser
    (FP) candidates near an unrelated direction, plus 3 exemplars -- same
    well-separated, HDBSCAN-reliable construction as test_cluster_verify.py's
    own basic scene (a handful of points per cluster; HDBSCAN with
    min_cluster_size=2 is unreliable on a 2-3 point total dataset). FP
    candidates get the HIGHER GeCo2 scores by default, so any test asserting
    a TP wins is actually proving verification (not just raw score) decided
    it."""
    exemplar_center = np.zeros(d)
    exemplar_center[0] = 1.0
    confuser_center = np.zeros(d)
    confuser_center[1] = 1.0

    ref = list(_unit_rows(exemplar_center[None, :] + (noise / 2) * rng.normal(size=(3, d))))
    tp_feats = _unit_rows(exemplar_center[None, :] + noise * rng.normal(size=(2, d)))
    fp_feats = _unit_rows(confuser_center[None, :] + noise * rng.normal(size=(2, d)))

    boxes = [
        Box(0, 0, 10, 10, score=tp_scores[0]), Box(20, 20, 30, 30, score=tp_scores[1]),
        Box(40, 40, 50, 50, score=fp_scores[0]), Box(60, 60, 70, 70, score=fp_scores[1]),
    ]
    feats = np.concatenate([tp_feats, fp_feats], axis=0)
    return boxes, feats, ref


def _unit_rows(vecs: np.ndarray) -> np.ndarray:
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


def test_first_keyframe_verifies_and_accepts_with_no_warmup():
    """The key online/cold-start fix: unlike topk_fusion (which needs
    min_window_for_zscore keyframes before trusting anything but boxes[0]),
    cluster_verification's very FIRST offer_topk() call can verify and
    accept a candidate -- the decision only ever depends on this keyframe's
    own candidates + the current exemplar set, never on accumulated
    history. Also proves GeCo2's raw score alone does NOT decide TP/FP: the
    confusers (boxes[2], boxes[3]) have the highest scores yet must be
    rejected."""
    rng = np.random.default_rng(0)
    cfg = _make_cfg(
        min_consecutive_hits=1,
        cluster_verification_overrides={"enabled": True, "min_cluster_size": 2, "min_candidates_for_cluster": 2},
    )
    detector = _FakeDetector()
    tracker = _make_tracker(cfg, detector=detector)
    boxes, feats, ref = _scene(rng)
    tracker._cross_prototype = None
    tracker._cross_per_ref_features = ref

    tracker.offer_topk(np.zeros((10, 10, 3), dtype=np.uint8), boxes, feats)

    assert tracker._n_cluster_offers == 1
    assert tracker._n_cluster_unverified == 0
    eff = tracker.effective_prototype()
    assert eff["main"].shape[1] == 2, "the higher-scoring VERIFIED candidate (boxes[1]) must have been accepted"
    assert detector.calls[0][1] == [(boxes[1].x1, boxes[1].y1, boxes[1].x2, boxes[1].y2)]


def test_unverified_confuser_never_offered_even_with_highest_geco2_score():
    """When NOTHING verifies (every candidate is a confuser), no candidate
    should even reach the confirmer -- reports absent rather than falling
    back to GeCo2's own unverified top pick."""
    rng = np.random.default_rng(1)
    cfg = _make_cfg(
        min_consecutive_hits=1,
        cluster_verification_overrides={"enabled": True, "min_cluster_size": 2, "min_candidates_for_cluster": 2},
    )
    detector = _FakeDetector()
    tracker = _make_tracker(cfg, detector=detector)
    _, _, ref = _scene(rng)
    tracker._cross_prototype = None
    tracker._cross_per_ref_features = ref

    # 4 candidates, ALL confusers (near the same off-exemplar direction) --
    # nothing here should cluster with an exemplar.
    d = 16
    confuser_center = np.zeros(d)
    confuser_center[1] = 1.0
    fp_feats = _unit_rows(confuser_center[None, :] + 0.02 * rng.normal(size=(4, d)))
    boxes = [Box(i * 20, i * 20, i * 20 + 10, i * 20 + 10, score=0.5 + 0.1 * i) for i in range(4)]

    tracker.offer_topk(np.zeros((10, 10, 3), dtype=np.uint8), boxes, fp_feats)

    assert tracker._n_cluster_unverified == 1
    assert tracker.effective_prototype() is tracker.base_prototype
    assert detector.calls == [], "no candidate should even reach the confirmer, let alone encode_exemplars"


def test_geco2_score_breaks_ties_within_verified_set():
    """When MULTIPLE candidates verify, the one with the highest raw GeCo2
    score among THEM wins -- not necessarily boxes[0], and not a confuser
    even if IT has a higher raw score still."""
    rng = np.random.default_rng(2)
    cfg = _make_cfg(
        min_consecutive_hits=1,
        cluster_verification_overrides={"enabled": True, "min_cluster_size": 2, "min_candidates_for_cluster": 2},
    )
    detector = _FakeDetector()
    tracker = _make_tracker(cfg, detector=detector)
    # tp_scores reversed from the default so boxes[0] (not boxes[1]) is the
    # higher-scoring TP this time -- proves the winner is picked by score,
    # not by array position.
    boxes, feats, ref = _scene(rng, tp_scores=(0.6, 0.4), fp_scores=(0.9, 0.95))
    tracker._cross_prototype = None
    tracker._cross_per_ref_features = ref

    tracker.offer_topk(np.zeros((10, 10, 3), dtype=np.uint8), boxes, feats)

    assert detector.calls[0][1] == [(boxes[0].x1, boxes[0].y1, boxes[0].x2, boxes[0].y2)]


def test_falls_back_to_relative_cosine_gate_below_min_candidates_for_cluster():
    """Fallback keeps candidates RELATIVE to this keyframe's own top cosine
    (fallback_relative_ratio), never a hand-set absolute number -- confirmed
    in practice that an absolute cutoff (e.g. cross_check_threshold-style)
    can be unreachable on footage with a severe domain gap. Uses 2
    candidates (still < min_candidates_for_cluster) so the assertion
    actually exercises the RATIO, not just "the only candidate always
    passes trivially"."""
    cfg = _make_cfg(
        min_consecutive_hits=1,
        cluster_verification_overrides={
            "enabled": True, "min_candidates_for_cluster": 5, "fallback_relative_ratio": 0.9,
        },
    )
    detector = _FakeDetector()
    tracker = _make_tracker(cfg, detector=detector)
    tracker._cross_prototype = _unit(np.array([1.0, 0.0]))
    tracker._cross_per_ref_features = None

    # 2 candidates < min_candidates_for_cluster=5. Both would fail a
    # hand-set absolute cross_check_threshold=0.5 in a domain-gap scenario
    # (simulated here by keeping raw cosines low), but boxes[0]'s cosine is
    # within 90% of the frame's own top (boxes[1]'s) -- should be kept;
    # boxes[1] itself IS the top, always kept.
    boxes = [Box(0, 0, 10, 10, score=0.5), Box(20, 20, 30, 30, score=0.4)]
    feats = np.array([
        _unit(np.array([0.30, 0.10])),   # cosine ~0.949 -- within 90% of boxes[1]'s
        _unit(np.array([0.33, 0.05])),   # cosine ~0.989 -- this frame's own top
    ])

    tracker.offer_topk(np.zeros((10, 10, 3), dtype=np.uint8), boxes, feats)

    assert tracker._n_cluster_fallback == 1
    # Highest GeCo2 score among the (both-verified) fallback set wins -- boxes[0].
    assert tracker.effective_prototype()["main"].shape[1] == 2, "fallback gate should have accepted boxes[0]"
    assert detector.calls[0][1] == [(boxes[0].x1, boxes[0].y1, boxes[0].x2, boxes[0].y2)]


def test_fallback_relative_ratio_rejects_candidate_far_below_frame_top():
    cfg = _make_cfg(
        min_consecutive_hits=1,
        cluster_verification_overrides={
            "enabled": True, "min_candidates_for_cluster": 5, "fallback_relative_ratio": 0.95,
        },
    )
    detector = _FakeDetector()
    tracker = _make_tracker(cfg, detector=detector)
    tracker._cross_prototype = _unit(np.array([1.0, 0.0]))
    tracker._cross_per_ref_features = None

    # boxes[0]'s cosine is far below 95% of boxes[1]'s (this frame's top) --
    # must be rejected even though it has the HIGHER GeCo2 score.
    boxes = [Box(0, 0, 10, 10, score=0.9), Box(20, 20, 30, 30, score=0.1)]
    feats = np.array([
        _unit(np.array([0.10, 0.30])),   # cosine ~0.316 -- well below 95% of boxes[1]'s
        _unit(np.array([0.33, 0.05])),   # cosine ~0.989 -- this frame's own top
    ])

    tracker.offer_topk(np.zeros((10, 10, 3), dtype=np.uint8), boxes, feats)

    assert tracker._n_cluster_fallback == 1
    assert detector.calls[0][1] == [(boxes[1].x1, boxes[1].y1, boxes[1].x2, boxes[1].y2)], (
        "only boxes[1] (this frame's own top) should have been accepted, despite boxes[0]'s higher GeCo2 score"
    )


def test_mutually_exclusive_with_topk_fusion_cluster_wins(caplog):
    cfg = _make_cfg(
        min_consecutive_hits=1,
        topk_fusion_overrides={"enabled": True, "min_window_for_zscore": 100},
        cluster_verification_overrides={"enabled": True, "min_cluster_size": 2, "min_candidates_for_cluster": 2},
    )
    detector = _FakeDetector()
    tracker = _make_tracker(cfg, detector=detector)
    tracker._cross_prototype = _unit(np.array([1.0, 0.0, 0.0, 0.0]))
    tracker._cross_per_ref_features = None

    boxes = [Box(0, 0, 10, 10, score=0.5), Box(20, 20, 30, 30, score=0.9)]
    feats = np.array([
        _unit(np.array([0.99, 0.01, 0.0, 0.0])),
        _unit(np.array([0.0, 0.0, 1.0, 0.0])),
    ])

    tracker.offer_topk(np.zeros((10, 10, 3), dtype=np.uint8), boxes, feats)

    assert tracker._n_cluster_offers == 1, "cluster_verification path must have run, not topk_fusion's"
    assert tracker._n_topk_offers == 0
    assert tracker._warned_cluster_topk_conflict is True


def test_falls_back_when_cross_check_source_is_hiera():
    cfg = _make_cfg(
        cross_check_source="hiera",
        cluster_verification_overrides={"enabled": True},
    )
    tracker = _make_tracker(cfg)
    assert tracker._warned_cluster_verification_unsupported is True

    calls = []
    tracker.offer = lambda frame_bgr, box, precomputed_feature=None, frame_idx=None: calls.append(box)
    boxes = [Box(0, 0, 10, 10, score=0.9), Box(20, 20, 30, 30, score=0.8)]
    feats = np.array([[0.1, 0.0], [0.9, 0.0]])
    tracker.offer_topk(np.zeros((10, 10, 3), dtype=np.uint8), boxes, feats)

    assert calls == [boxes[0]], "must fall back to plain offer() on boxes[0], same as topk_fusion's own fallback"
    assert tracker._n_cluster_offers == 0


def test_no_exemplars_available_falls_back(caplog):
    """No prototype.npz / _cross_prototype available -- must fall back to
    topk_fusion/plain selection with a one-time warning, same convention as
    every other cross_check_source="feature_extractor" requirement."""
    cfg = _make_cfg(cluster_verification_overrides={"enabled": True})
    tracker = _make_tracker(cfg)
    assert tracker._cross_prototype is None

    calls = []
    tracker.offer = lambda frame_bgr, box, precomputed_feature=None, frame_idx=None: calls.append(box)
    boxes = [Box(0, 0, 10, 10, score=0.9)]
    feats = np.array([[0.1, 0.0]])
    tracker.offer_topk(np.zeros((10, 10, 3), dtype=np.uint8), boxes, feats)

    assert calls == [boxes[0]]
    assert tracker._warned_cross_unavailable is True


def test_log_summary_reports_cluster_verification_stats(caplog):
    import logging

    cfg = _make_cfg(
        min_consecutive_hits=1,
        cluster_verification_overrides={"enabled": True, "min_cluster_size": 2, "min_candidates_for_cluster": 2},
    )
    tracker = _make_tracker(cfg)
    tracker._cross_prototype = _unit(np.array([1.0, 0.0, 0.0, 0.0]))
    tracker._cross_per_ref_features = None

    boxes = [Box(0, 0, 10, 10, score=0.5), Box(20, 20, 30, 30, score=0.9)]
    feats = np.array([
        _unit(np.array([0.0, 1.0, 0.0, 0.0])),
        _unit(np.array([0.0, 0.0, 1.0, 0.0])),
    ])
    tracker.offer_topk(np.zeros((10, 10, 3), dtype=np.uint8), boxes, feats)  # unverified

    with caplog.at_level(logging.INFO):
        tracker.log_summary()
    assert "cluster_verification summary" in caplog.text
