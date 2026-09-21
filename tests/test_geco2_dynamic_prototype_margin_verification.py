"""Unit tests for stage123_geco2.dynamic_prototype.margin_verification --
GeCo2DynamicPrototypeTracker.offer_topk's (both the topk_fusion and
cluster_verification code paths) margin-over-runner-up ambiguity guard.
See MarginVerificationConfig's own docstring (aero_eyes/config.py) for the
shared mechanism (also used by stage3.margin_verification).

Reuses tests/test_geco2_dynamic_prototype.py's fake-config/fake-detector
scaffolding so this runs without the real GECO2 repo, weights, or GPU.
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="torch not importable", exc_type=ImportError)

from aero_eyes.types import Box
from tests.test_geco2_dynamic_prototype import _FakeDetector, _make_cfg, _make_tracker


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def _unit_rows(vecs: np.ndarray) -> np.ndarray:
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# topk_fusion path (offer_topk's own cold-start cosine selection)
# ---------------------------------------------------------------------------

def test_topk_fusion_cold_start_ambiguous_keyframe_rejected():
    """Cold start (no Z-score baseline yet) selects purely on raw cosine.
    Two candidates with near-identical cosine similarity are ambiguous --
    margin_verification should reject the whole keyframe rather than
    guessing which is real."""
    cfg = _make_cfg(
        min_consecutive_hits=1,
        topk_fusion_overrides={"enabled": True, "min_window_for_zscore": 100},
        margin_verification_overrides={"enabled": True, "tau_margin": 0.05},
    )
    detector = _FakeDetector()
    tracker = _make_tracker(cfg, detector=detector)
    tracker._cross_prototype = _unit(np.array([1.0, 0.0]))
    tracker._cross_per_ref_features = None

    boxes = [Box(0, 0, 10, 10, score=0.9), Box(20, 20, 30, 30, score=0.8)]
    feats = np.array([
        _unit(np.array([0.99, 0.01])),
        _unit(np.array([0.985, 0.02])),  # near-identical cosine to boxes[0]
    ])

    tracker.offer_topk(np.zeros((10, 10, 3), dtype=np.uint8), boxes, feats)

    assert detector.calls == [], "ambiguous margin should reject the keyframe before ever confirming/encoding"
    assert tracker._n_margin_rejected == 1


def test_topk_fusion_clear_margin_keyframe_accepted():
    """Same shape as the ambiguous case above, but the runner-up is far
    enough below the top candidate that the margin clears -- must proceed
    normally."""
    cfg = _make_cfg(
        min_consecutive_hits=1,
        topk_fusion_overrides={"enabled": True, "min_window_for_zscore": 100},
        margin_verification_overrides={"enabled": True, "tau_margin": 0.05},
    )
    detector = _FakeDetector()
    tracker = _make_tracker(cfg, detector=detector)
    tracker._cross_prototype = _unit(np.array([1.0, 0.0]))
    tracker._cross_per_ref_features = None

    boxes = [Box(0, 0, 10, 10, score=0.9), Box(20, 20, 30, 30, score=0.8)]
    feats = np.array([
        _unit(np.array([0.99, 0.01])),
        _unit(np.array([0.10, 0.99])),  # far lower cosine -- clear margin
    ])

    tracker.offer_topk(np.zeros((10, 10, 3), dtype=np.uint8), boxes, feats)

    assert detector.calls[0][1] == [(boxes[0].x1, boxes[0].y1, boxes[0].x2, boxes[0].y2)]
    assert tracker._n_margin_rejected == 0


def test_topk_fusion_single_candidate_unaffected():
    """A single-candidate keyframe has no runner-up to compare against and
    must never be rejected by margin_verification."""
    cfg = _make_cfg(
        min_consecutive_hits=1,
        topk_fusion_overrides={"enabled": True, "min_window_for_zscore": 100},
        margin_verification_overrides={"enabled": True, "tau_margin": 0.5},
    )
    detector = _FakeDetector()
    tracker = _make_tracker(cfg, detector=detector)
    tracker._cross_prototype = _unit(np.array([1.0, 0.0]))
    tracker._cross_per_ref_features = None

    boxes = [Box(0, 0, 10, 10, score=0.9)]
    feats = np.array([_unit(np.array([0.99, 0.01]))])

    tracker.offer_topk(np.zeros((10, 10, 3), dtype=np.uint8), boxes, feats)

    assert detector.calls[0][1] == [(boxes[0].x1, boxes[0].y1, boxes[0].x2, boxes[0].y2)]
    assert tracker._n_margin_rejected == 0


def test_margin_verification_disabled_by_default_no_behavior_change():
    """margin_verification defaults to disabled -- an ambiguous keyframe
    that WOULD be rejected if enabled must pass through unaffected."""
    cfg = _make_cfg(
        min_consecutive_hits=1,
        topk_fusion_overrides={"enabled": True, "min_window_for_zscore": 100},
    )
    detector = _FakeDetector()
    tracker = _make_tracker(cfg, detector=detector)
    tracker._cross_prototype = _unit(np.array([1.0, 0.0]))
    tracker._cross_per_ref_features = None

    boxes = [Box(0, 0, 10, 10, score=0.9), Box(20, 20, 30, 30, score=0.8)]
    feats = np.array([
        _unit(np.array([0.99, 0.01])),
        _unit(np.array([0.985, 0.02])),
    ])

    tracker.offer_topk(np.zeros((10, 10, 3), dtype=np.uint8), boxes, feats)

    assert detector.calls[0][1] == [(boxes[0].x1, boxes[0].y1, boxes[0].x2, boxes[0].y2)]
    assert tracker._n_margin_rejected == 0


# ---------------------------------------------------------------------------
# cluster_verification path (_offer_topk_cluster)
# ---------------------------------------------------------------------------

def _cluster_scene(rng, d=16, noise=0.02):
    """2 candidates that BOTH cluster with the exemplar (near-identical
    cosine to it -- ambiguous), plus 3 exemplars."""
    exemplar_center = np.zeros(d)
    exemplar_center[0] = 1.0
    ref = list(_unit_rows(exemplar_center[None, :] + (noise / 2) * rng.normal(size=(3, d))))
    tp_feats = _unit_rows(exemplar_center[None, :] + noise * rng.normal(size=(2, d)))
    boxes = [Box(0, 0, 10, 10, score=0.5), Box(20, 20, 30, 30, score=0.9)]
    return boxes, tp_feats, ref


def test_cluster_verification_ambiguous_verified_pair_rejected():
    rng = np.random.default_rng(0)
    cfg = _make_cfg(
        min_consecutive_hits=1,
        cluster_verification_overrides={"enabled": True, "min_cluster_size": 2, "min_candidates_for_cluster": 2},
        margin_verification_overrides={"enabled": True, "tau_margin": 0.9},  # very strict -- forces ambiguity
    )
    detector = _FakeDetector()
    tracker = _make_tracker(cfg, detector=detector)
    boxes, feats, ref = _cluster_scene(rng)
    tracker._cross_prototype = None
    tracker._cross_per_ref_features = ref

    tracker.offer_topk(np.zeros((10, 10, 3), dtype=np.uint8), boxes, feats)

    assert detector.calls == [], "both verified candidates are near-identical in cosine -- should be ambiguous"
    assert tracker._n_margin_rejected == 1


def test_cluster_verification_disabled_margin_default_no_behavior_change():
    rng = np.random.default_rng(0)
    cfg = _make_cfg(
        min_consecutive_hits=1,
        cluster_verification_overrides={"enabled": True, "min_cluster_size": 2, "min_candidates_for_cluster": 2},
    )
    detector = _FakeDetector()
    tracker = _make_tracker(cfg, detector=detector)
    boxes, feats, ref = _cluster_scene(rng)
    tracker._cross_prototype = None
    tracker._cross_per_ref_features = ref

    tracker.offer_topk(np.zeros((10, 10, 3), dtype=np.uint8), boxes, feats)

    assert tracker._n_margin_rejected == 0
    assert len(detector.calls) == 1, "margin_verification defaults to disabled -- verification should proceed"


def test_log_summary_reports_margin_verification_stats(caplog):
    import logging

    cfg = _make_cfg(
        min_consecutive_hits=1,
        topk_fusion_overrides={"enabled": True, "min_window_for_zscore": 100},
        margin_verification_overrides={"enabled": True, "tau_margin": 0.05},
    )
    tracker = _make_tracker(cfg)
    tracker._cross_prototype = _unit(np.array([1.0, 0.0]))
    tracker._cross_per_ref_features = None
    tracker.offer = lambda *a, **k: None

    boxes = [Box(0, 0, 10, 10, score=0.9), Box(20, 20, 30, 30, score=0.8)]
    feats = np.array([
        _unit(np.array([0.99, 0.01])),
        _unit(np.array([0.985, 0.02])),
    ])
    tracker.offer_topk(np.zeros((10, 10, 3), dtype=np.uint8), boxes, feats)

    with caplog.at_level(logging.INFO):
        tracker.log_summary()
    assert "margin_verification summary" in caplog.text
