"""Integration tests for stage3.negative_prototype_filter -- the hard-
negative "negative prototype" secondary filter (see
NegativePrototypeFilterConfig's own docstring in aero_eyes/config.py for
the full rationale: a single recurring confuser class, e.g. dry leaves,
accounted for 47.8% of all false positives on this project's own real
footage). Uses exactly-controlled cosine similarities (two orthonormal
basis vectors e0/e1, cos(theta) = alpha) so every assertion below can be
reasoned about algebraically rather than empirically.
"""
from __future__ import annotations

import logging
import math

import numpy as np
import pytest

from aero_eyes.config import load_config
from aero_eyes.stages.stage2 import _write_candidates_with_features
from aero_eyes.stages.stage3 import run_stage3
from aero_eyes.types import Box, Detection
from aero_eyes.utils.io import read_detections, write_prototype


def _det(frame_idx: int, x: float, feat: np.ndarray) -> Detection:
    box = Box(x1=x, y1=10.0, x2=x + 20.0, y2=30.0)
    det = Detection(frame_idx=frame_idx, box=box, similarity=0.0, source="detect")
    det._feature = feat
    return det


def _conf_vec(alpha: float, d: int = 16) -> np.ndarray:
    """Unit vector at exactly cosine=alpha from e0, in the e0/e1 plane --
    cosine between _conf_vec(a1) and _conf_vec(a2) is exactly
    cos(arccos(a1) - arccos(a2))."""
    v = np.zeros(d)
    v[0] = alpha
    v[1] = math.sqrt(max(0.0, 1.0 - alpha * alpha))
    return v


@pytest.fixture()
def negative_prototype_setup(tmp_path):
    d = 16
    e0 = _conf_vec(1.0, d)  # exemplar direction

    work_dir = tmp_path / "sample1"
    work_dir.mkdir(parents=True)
    # No noise on the exemplar -- keeps "own cosine to exemplar" exact.
    ref = [e0.copy(), e0.copy(), e0.copy()]
    write_prototype(e0, {}, ref, work_dir / "prototype.npz")
    return work_dir


def _base_overrides(tmp_path, **extra) -> list[str]:
    overrides = [
        f"project.work_dir={tmp_path}",
        "project.use_cache=false",
        "runtime.save_visualizations=false",
        "stage3.verification_method=threshold",
        "stage3.match_threshold=0.5",
        "stage3.topk_per_keyframe=10",
        "stage3.negative_prototype_filter.enabled=true",
        "stage3.negative_prototype_filter.min_window_for_check=2",
    ]
    for k, v in extra.items():
        overrides.append(f"stage3.negative_prototype_filter.{k}={v}")
    return overrides


def _scene():
    """3 confuser 'seeds' (cosine=0.2 to exemplar -- BELOW match_threshold=
    0.5, so the primary threshold itself rejects them, seeding the negative
    window) at frame 5; 1 genuine TP (cosine=1.0) at frame 10; 1 'sneaky'
    confuser (cosine=0.55 -- clears match_threshold=0.5 by itself, same
    underlying confuser direction as the seeds) at frame 15. cosine(seed,
    sneaky) = cos(arccos(0.2) - arccos(0.55)) ~= 0.928, far above the
    sneaky candidate's own 0.55 cosine to the exemplar."""
    seeds = [_conf_vec(0.2), _conf_vec(0.2), _conf_vec(0.2)]
    tp = _conf_vec(1.0)
    sneaky = _conf_vec(0.55)
    candidates = {
        5: [_det(5, float(i * 20), f) for i, f in enumerate(seeds)],
        10: [_det(10, 0.0, tp)],
        15: [_det(15, 0.0, sneaky)],
    }
    return candidates


def test_rejects_candidate_closer_to_negative_window_than_exemplar(negative_prototype_setup, tmp_path, caplog):
    work_dir = negative_prototype_setup
    _write_candidates_with_features(_scene(), work_dir / "candidates.json")

    cfg = load_config("configs/config.yaml", overrides=_base_overrides(tmp_path))
    with caplog.at_level(logging.INFO):
        det_path = run_stage3(cfg, "sample1")
    detections = read_detections(det_path)

    assert len(detections[5]) == 0, "the 3 confuser seeds fail match_threshold=0.5 on their own (cosine=0.2)"
    assert len(detections[10]) == 1, "the genuine TP (cosine=1.0, far from the negative window) must survive"
    assert len(detections[15]) == 0, (
        "the sneaky confuser clears match_threshold alone (cosine=0.55) but is much closer to the "
        "accumulated negative window (~0.928) than to the exemplar -- must be rejected"
    )
    assert "negative_prototype_filter" in caplog.text


def test_noop_when_disabled(negative_prototype_setup, tmp_path):
    work_dir = negative_prototype_setup
    _write_candidates_with_features(_scene(), work_dir / "candidates.json")

    overrides = [o for o in _base_overrides(tmp_path) if not o.startswith("stage3.negative_prototype_filter")]
    cfg = load_config("configs/config.yaml", overrides=overrides)
    det_path = run_stage3(cfg, "sample1")
    detections = read_detections(det_path)

    assert len(detections[15]) == 1, "without the filter, the sneaky confuser survives match_threshold alone"


def test_noop_below_min_window_for_check(negative_prototype_setup, tmp_path):
    """Only 3 negative-window members accumulate (from the 3 seeds) --
    below a min_window_for_check=5 requirement, so the filter must skip
    the check entirely rather than reject off too little evidence."""
    work_dir = negative_prototype_setup
    _write_candidates_with_features(_scene(), work_dir / "candidates.json")

    cfg = load_config("configs/config.yaml", overrides=_base_overrides(tmp_path, min_window_for_check=5))
    det_path = run_stage3(cfg, "sample1")
    detections = read_detections(det_path)

    assert len(detections[15]) == 1, "too few negative-window members -- filter must be a no-op here"


def test_tau_negative_margin_controls_strictness(negative_prototype_setup, tmp_path):
    """Margin ~= 0.928 - 0.55 = 0.378 for the sneaky confuser -- a
    tau_negative_margin ABOVE that must let it survive."""
    work_dir = negative_prototype_setup
    _write_candidates_with_features(_scene(), work_dir / "candidates.json")

    cfg = load_config("configs/config.yaml", overrides=_base_overrides(tmp_path, tau_negative_margin=0.5))
    det_path = run_stage3(cfg, "sample1")
    detections = read_detections(det_path)

    assert len(detections[15]) == 1, "margin (0.5) exceeds the actual gap (~0.378) -- must not reject"
