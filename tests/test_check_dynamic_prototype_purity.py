"""Unit tests for run_dynamic_prototype_rounds's on_round callback -- the
hook scripts.check_dynamic_prototype_purity relies on to compute purity
without duplicating stage3's selection logic."""
from __future__ import annotations

import numpy as np

from aero_eyes.config import DynamicPrototypeConfig
from aero_eyes.stages.stage3 import run_dynamic_prototype_rounds
from aero_eyes.types import Box
from aero_eyes.utils.geometry import box_iou


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def test_on_round_reports_selected_candidates_and_purity():
    """Builds a small synthetic scenario: 4 candidates, 2 clearly closer to
    the prototype (should be selected as high-confidence) and 2 far from
    it. Checks on_round is invoked with the RIGHT mask, and that purity
    (checked externally, the way the real script does) correctly separates
    a genuine match from a confuser."""
    d = 8
    rng = np.random.default_rng(0)
    prototype = _unit(rng.standard_normal(d))

    # 2 candidates near the prototype (would-be "correct" picks), 2 far.
    near_a = _unit(prototype + 0.01 * rng.standard_normal(d))
    near_b = _unit(prototype + 0.01 * rng.standard_normal(d))
    far_a = _unit(-prototype + 0.01 * rng.standard_normal(d))
    far_b = _unit(rng.standard_normal(d))

    all_feats = np.stack([near_a, near_b, far_a, far_b], axis=0)
    all_frame_idxs = [0, 1, 2, 3]
    # frame 0 -> box overlapping GT (genuine match); frame 1 -> box that
    # does NOT overlap GT (a confuser that happens to score high too).
    gt_box = Box(0, 0, 10, 10)
    confuser_box = Box(200, 200, 210, 210)
    dets = [
        type("D", (), {"box": gt_box})(),        # frame 0: matches GT
        type("D", (), {"box": confuser_box})(),  # frame 1: does NOT match GT
        type("D", (), {"box": gt_box})(),         # frame 2 (far, shouldn't be picked)
        type("D", (), {"box": gt_box})(),         # frame 3 (far, shouldn't be picked)
    ]
    gt = {0: gt_box, 1: gt_box}  # GT present on frames 0 and 1 (not 2, 3)

    all_sims = all_feats @ prototype

    dp = DynamicPrototypeConfig(
        enabled=True, rounds=1, alpha=0.3,
        high_conf_percentile=50.0,  # top ~50% -> should select the 2 "near" candidates
        high_conf_abs_floor=-1.0,   # don't let the floor interfere with this synthetic test
        min_support=1,
    )

    calls = []

    def on_round(round_idx, high_conf_mask, threshold):
        calls.append((round_idx, high_conf_mask.copy(), threshold))

    run_dynamic_prototype_rounds(
        "test_sample", all_feats, all_sims, prototype, [],
        use_multi_ref=False, multi_ref_pooling="mean", similarity_metric="cosine",
        dp=dp, on_round=on_round,
    )

    assert len(calls) == 1
    round_idx, high_conf_mask, threshold = calls[0]
    selected = set(np.where(high_conf_mask)[0].tolist())
    # The 2 near-prototype candidates (0, 1) score higher than the 2 far
    # ones (2, 3) -- percentile-50 threshold should select exactly them.
    assert selected == {0, 1}

    # Now replay the SAME purity computation check_dynamic_prototype_purity
    # does, using the mask on_round handed back.
    n_with_gt = 0
    n_correct = 0
    for idx in selected:
        fi = all_frame_idxs[idx]
        if fi in gt:
            n_with_gt += 1
            if box_iou(gt[fi], dets[idx].box) >= 0.5:
                n_correct += 1
    assert n_with_gt == 2       # both selected candidates' frames have GT
    assert n_correct == 1       # only frame 0's box actually matches GT -- 50% purity
