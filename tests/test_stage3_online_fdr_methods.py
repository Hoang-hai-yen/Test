"""Tests for stage3.adaptive_threshold_online_method's 3 alternatives to
the default window z_score/otsu/gmm dispatch -- ACIOnlineThreshold,
SaffronInspiredOnlineFDR and CorruptionCompensatedThreshold. All three are
FAITHFUL ports of their cited papers' own algorithms; see each class's own
docstring in aero_eyes/stages/stage3.py (and OnlineFDRConfig/
CorruptionCompensatedThresholdConfig in aero_eyes/config.py) for the exact
mapping and each one's own project-specific adaptation.
"""
from __future__ import annotations

import numpy as np
import pytest

from aero_eyes.config import (
    OnlineFDRConfig,
    Stage3Config,
    load_config,
)
from aero_eyes.stages.stage2 import _write_candidates_with_features
from aero_eyes.stages.stage3 import (
    ACIOnlineThreshold,
    CorruptionCompensatedThreshold,
    SaffronInspiredOnlineFDR,
    run_stage3,
)
from aero_eyes.types import Box, Detection
from aero_eyes.utils.io import read_detections, write_prototype


def _make_unit(vecs: np.ndarray) -> np.ndarray:
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


def _det(frame_idx: int, x: float, feat: np.ndarray) -> Detection:
    box = Box(x1=x, y1=10.0, x2=x + 20.0, y2=30.0)
    det = Detection(frame_idx=frame_idx, box=box, similarity=0.0, source="detect")
    det._feature = feat
    return det


# ---------------------------------------------------------------------------
# ACIOnlineThreshold
# ---------------------------------------------------------------------------

def test_aci_cold_start_uses_floor():
    s3 = Stage3Config(adaptive_min_floor=0.05, adaptive_threshold_min_samples=20)
    aci = ACIOnlineThreshold(s3)
    threshold, label = aci.threshold_for_next_frame()
    assert threshold == pytest.approx(0.05)
    assert label == "aci_cold_start"


def test_aci_percentile_rises_when_too_permissive():
    """If a frame's own accept rate exceeds the target error rate, ACI's
    update should make the threshold STRICTER for the next frame (higher
    percentile -> higher threshold value in similarity-score units)."""
    s3 = Stage3Config(aci_target_error_rate=0.1, aci_step_size=0.1, adaptive_threshold_min_samples=5)
    aci = ACIOnlineThreshold(s3)
    percentile_before = aci.percentile
    # Warm up past cold start with SOME history first.
    aci.observe(np.array([0.1, 0.2, 0.3, 0.4, 0.5]), np.array([False] * 5))
    percentile_after_low_accept = aci.percentile

    # Now a frame where ALL candidates were accepted (accept rate 1.0 >> target 0.1).
    aci.observe(np.array([0.6, 0.7]), np.array([True, True]))
    percentile_after_high_accept = aci.percentile

    assert percentile_after_high_accept > percentile_after_low_accept, (
        "an over-permissive frame's accept rate should push the percentile (and threshold) up"
    )


def test_aci_percentile_falls_when_too_strict():
    s3 = Stage3Config(aci_target_error_rate=0.5, aci_step_size=0.1, adaptive_threshold_min_samples=5)
    aci = ACIOnlineThreshold(s3)
    aci.observe(np.array([0.1, 0.2, 0.3, 0.4, 0.5]), np.array([False] * 5))
    percentile_before = aci.percentile
    # Nothing accepted this frame (accept rate 0.0 < target 0.5) -> should loosen (percentile falls).
    aci.observe(np.array([0.6, 0.7]), np.array([False, False]))
    assert aci.percentile < percentile_before


def test_aci_threshold_reflects_window_percentile_after_warmup():
    s3 = Stage3Config(adaptive_threshold_min_samples=5, aci_target_error_rate=0.5)
    aci = ACIOnlineThreshold(s3)
    aci.observe(np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]), np.zeros(10, dtype=bool))
    threshold, label = aci.threshold_for_next_frame()
    assert label == "aci"
    assert 0.0 <= threshold <= 1.0


# ---------------------------------------------------------------------------
# SaffronInspiredOnlineFDR
# ---------------------------------------------------------------------------

def test_saffron_cold_start_p_value_is_neutral():
    cfg = OnlineFDRConfig(target_fdr=0.1, initial_wealth_fraction=0.5, p_value_window=200)
    saffron = SaffronInspiredOnlineFDR(cfg, cfg.p_value_window)
    assert saffron._p_value(0.9) == pytest.approx(0.5)


def test_saffron_high_scoring_candidate_more_likely_accepted():
    cfg = OnlineFDRConfig(target_fdr=0.2, initial_wealth_fraction=1.0, p_value_window=200)
    saffron = SaffronInspiredOnlineFDR(cfg, cfg.p_value_window)
    # Feed a long history of LOW background scores.
    for s in np.linspace(0.0, 0.3, 50):
        saffron.test(float(s))
    # A candidate far above all recent history should get a very low p-value.
    p = saffron._p_value(0.99)
    assert p < 0.05


def test_saffron_wealth_never_goes_negative():
    cfg = OnlineFDRConfig(target_fdr=0.1, initial_wealth_fraction=0.5, p_value_window=50)
    saffron = SaffronInspiredOnlineFDR(cfg, cfg.p_value_window)
    rng = np.random.default_rng(0)
    for _ in range(200):
        saffron.test(float(rng.normal()))
        assert saffron.wealth >= 0.0


def test_saffron_wealth_increases_on_acceptance():
    cfg = OnlineFDRConfig(target_fdr=0.3, initial_wealth_fraction=0.5, p_value_window=200)
    saffron = SaffronInspiredOnlineFDR(cfg, cfg.p_value_window)
    for s in np.linspace(0.0, 0.3, 20):
        saffron.test(float(s))
    wealth_before = saffron.wealth
    accepted, alpha_t = saffron.test(0.99)  # far above history -> low p-value -> likely accepted
    assert accepted
    assert saffron.wealth == pytest.approx(wealth_before - alpha_t + cfg.target_fdr)


# ---------------------------------------------------------------------------
# CorruptionCompensatedThreshold (F-ROCP, faithful port -- see its own
# docstring in stage3.py: wraps ACIOnlineThreshold, overriding the observed
# accept-rate signal with a CERTAIN one whenever the percentile has hit the
# 0/100 boundary, trusting the observed signal only in-range).
# ---------------------------------------------------------------------------

def test_corruption_compensated_threshold_labels_reflect_boundary_state():
    s3 = Stage3Config(adaptive_threshold_min_samples=1)
    cct = CorruptionCompensatedThreshold(s3)
    cct.aci.history.extend([0.1, 0.2, 0.3])
    cct.aci.percentile = 0.0
    _, label = cct.threshold_for_next_frame()
    assert label == "corruption_compensated_permissive_boundary"
    cct.aci.percentile = 100.0
    _, label = cct.threshold_for_next_frame()
    assert label == "corruption_compensated_strict_boundary"
    cct.aci.percentile = 50.0
    _, label = cct.threshold_for_next_frame()
    assert label == "corruption_compensated_in_range"


def test_corruption_compensated_forces_too_permissive_at_permissive_boundary():
    """At the permissive boundary, the frame's OWN accept rate is
    irrelevant -- F-ROCP's boundary certainty must treat it as "too
    permissive" regardless (percentile increases) even when this frame's
    own sample suggests the opposite (nothing accepted)."""
    s3 = Stage3Config(aci_target_error_rate=0.5, aci_step_size=0.1, adaptive_threshold_min_samples=1)
    cct = CorruptionCompensatedThreshold(s3)
    cct.aci.percentile = 0.0
    percentile_before = cct.aci.percentile
    cct.observe(np.array([0.1, 0.2]), False, np.array([False, False]))
    assert cct.aci.percentile > percentile_before


def test_corruption_compensated_forces_too_strict_at_strict_boundary():
    """Symmetric case at the strict boundary -- percentile must decrease
    even when this frame's own sample suggests everything was accepted."""
    s3 = Stage3Config(aci_target_error_rate=0.5, aci_step_size=0.1, adaptive_threshold_min_samples=1)
    cct = CorruptionCompensatedThreshold(s3)
    cct.aci.percentile = 100.0
    cct.observe(np.array([0.1, 0.2]), False, np.array([True, True]))
    assert cct.aci.percentile < 100.0


def test_corruption_compensated_trusts_observed_signal_in_range():
    s3 = Stage3Config(aci_target_error_rate=0.1, aci_step_size=0.1, adaptive_threshold_min_samples=1)
    cct = CorruptionCompensatedThreshold(s3)
    cct.aci.percentile = 50.0
    percentile_before = cct.aci.percentile
    # accept rate 1.0 >> target 0.1 -- too permissive, and this time it's a
    # genuine (not boundary-forced) observation.
    cct.observe(np.array([0.1, 0.2]), False, np.array([True, True]))
    assert cct.aci.percentile > percentile_before


# ---------------------------------------------------------------------------
# End-to-end run_stage3 smoke tests (must not crash, sane output)
# ---------------------------------------------------------------------------

@pytest.fixture()
def online_method_setup(tmp_path):
    d = 8
    rng = np.random.default_rng(0)
    view = np.zeros(d)
    view[0] = 1.0
    ref = _make_unit(view[None, :] + 0.02 * rng.normal(size=(3, d)))
    fused_ref = ref.mean(axis=0)
    fused_ref /= np.linalg.norm(fused_ref)

    work_dir = tmp_path / "sample1"
    work_dir.mkdir(parents=True)
    write_prototype(fused_ref, {}, list(ref), work_dir / "prototype.npz")

    candidates = {}
    for i, fi in enumerate(range(0, 300, 10)):
        feat = _make_unit((view[None, :] + 0.3 * rng.normal(size=(1, d))))[0]
        candidates[fi] = [_det(fi, 0.0, feat)]
    return work_dir, candidates


@pytest.mark.parametrize("method", ["window_stat", "aci", "saffron", "corruption_compensated"])
def test_run_stage3_end_to_end_each_online_method(online_method_setup, tmp_path, method):
    work_dir, candidates = online_method_setup
    _write_candidates_with_features(candidates, work_dir / "candidates.json")

    cfg = load_config(
        "configs/config.yaml",
        overrides=[
            f"project.work_dir={tmp_path}",
            "project.use_cache=false",
            "runtime.save_visualizations=false",
            "stage3.verification_method=threshold",
            "stage3.adaptive_threshold=true",
            "stage3.adaptive_threshold_online=true",
            f"stage3.adaptive_threshold_online_method={method}",
            "stage3.adaptive_threshold_min_samples=3",
            "stage3.dynamic_prototype.enabled=false",
        ],
    )
    det_path = run_stage3(cfg, "sample1")
    detections = read_detections(det_path)
    assert isinstance(detections, dict)
    assert len(detections) == len(candidates)
