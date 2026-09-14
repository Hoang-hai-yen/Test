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


def test_require_diverse_picks_skips_narrow_frame_span():
    """4 high-confidence candidates that would easily clear min_support are
    all clustered within a few frames (e.g. 3 consecutive keyframes of the
    same unmoving pose) -- require_diverse_picks must refuse to trust them
    (round skipped, on_round never called) since they don't demonstrate the
    target's appearance actually varies. The SAME candidates spread across
    a wide frame span DO get trusted."""
    d = 8
    rng = np.random.default_rng(1)
    prototype = _unit(rng.standard_normal(d))
    # All 4 candidates close to the prototype -- all comfortably clear a
    # low percentile/floor threshold, isolating the diversity gate itself.
    feats = np.stack([_unit(prototype + 0.01 * rng.standard_normal(d)) for _ in range(4)], axis=0)
    all_sims = feats @ prototype

    dp_narrow = DynamicPrototypeConfig(
        enabled=True, rounds=1, alpha=0.3,
        high_conf_percentile=0.0, high_conf_abs_floor=-1.0, min_support=3,
        require_diverse_picks=True, min_frame_span=30,
    )
    calls = []
    run_dynamic_prototype_rounds(
        "narrow", feats, all_sims.copy(), prototype, [],
        use_multi_ref=False, multi_ref_pooling="mean", similarity_metric="cosine",
        dp=dp_narrow, on_round=lambda *a: calls.append(a),
        all_frame_idxs=[10, 11, 12, 13],  # span=3, well under min_frame_span=30
    )
    assert calls == [], "narrow frame span should have skipped the round entirely"

    calls_wide = []
    run_dynamic_prototype_rounds(
        "wide", feats, all_sims.copy(), prototype, [],
        use_multi_ref=False, multi_ref_pooling="mean", similarity_metric="cosine",
        dp=dp_narrow, on_round=lambda *a: calls_wide.append(a),
        all_frame_idxs=[0, 100, 200, 300],  # span=300, clears min_frame_span=30
    )
    assert len(calls_wide) == 1, "wide frame span should have let the round proceed"

    # require_diverse_picks=False (default) -- narrow span no longer matters.
    dp_off = dp_narrow.model_copy(update={"require_diverse_picks": False})
    calls_off = []
    run_dynamic_prototype_rounds(
        "narrow-unchecked", feats, all_sims.copy(), prototype, [],
        use_multi_ref=False, multi_ref_pooling="mean", similarity_metric="cosine",
        dp=dp_off, on_round=lambda *a: calls_off.append(a),
        all_frame_idxs=[10, 11, 12, 13],
    )
    assert len(calls_off) == 1, "diversity check off should reproduce the old count-only gate"
