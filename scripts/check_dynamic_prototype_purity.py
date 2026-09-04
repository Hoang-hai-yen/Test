"""Diagnostic: is stage3.dynamic_prototype actually picking the real
target, or drifting toward a confuser?

dynamic_prototype is a self-training loop: each round picks the CURRENT
highest-scoring candidates and blends their mean feature back into the
prototype (or appends them as a new reference), then re-scores everything
against the updated prototype for the next round. If a round's "high-
confidence" picks are actually a similarly-shaped WRONG object, the
prototype drifts toward finding that wrong object instead -- and nothing
in the pipeline itself would notice, since it only ever trusts its own
past picks.

This script replays the EXACT SAME selection logic (via
aero_eyes.stages.stage3.run_dynamic_prototype_rounds, imported -- not
reimplemented) against candidates.json + prototype.npz, and reports, per
round, what fraction of that round's selected candidates actually overlap
the REAL ground-truth box (IoU >= --iou-threshold) on their own frame --
a purity score. A dropping purity across rounds is a warning sign of
drift; a round with 0% purity (while non-empty) most likely selected a
confuser.

Read-only: does not touch detections.json/tracks.json/prototype.npz, does
not write anything. Ground truth must be available (this is a dev/eval
diagnostic, not something you'd run on unlabeled inference data).

Usage:
    python -m scripts.check_dynamic_prototype_purity --config configs/config.yaml --sample BlackBox_0
    python -m scripts.check_dynamic_prototype_purity --config configs/config.yaml --iou-threshold 0.3
    python -m scripts.check_dynamic_prototype_purity --config configs/config.yaml   # all samples
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from aero_eyes.stages.stage3 import _pool_sims, _score_against_ref, run_dynamic_prototype_rounds
from aero_eyes.utils.geometry import box_iou
from aero_eyes.utils.io import load_gt, read_prototype

log = logging.getLogger(__name__)


def check_sample(cfg, sample_id: str, iou_threshold: float) -> None:
    from aero_eyes.stages.stage2 import read_candidates_with_features

    work_dir = Path(cfg.project.work_dir) / sample_id
    proto_path = work_dir / cfg.stage1.prototype.cache_name
    cand_path = work_dir / "candidates.json"

    print(f"\n=== {sample_id} (IoU threshold={iou_threshold}) ===")

    if not cfg.stage3.dynamic_prototype.enabled:
        print("  stage3.dynamic_prototype.enabled is false in this config -- nothing to check "
              "(pass --set stage3.dynamic_prototype.enabled=true).")
        return
    if not proto_path.exists():
        print(f"  prototype.npz not found at {proto_path} -- run Stage 1 first.")
        return
    if not cand_path.exists():
        print(f"  candidates.json not found at {cand_path} -- run Stage 2 (legacy) or the "
              "GeCo2 candidate stage first.")
        return

    gt_file = cfg.data.gt.global_file
    try:
        gt = load_gt(gt_file, sample_id)
    except KeyError:
        print(f"  not found in {gt_file}, skipping.")
        return

    prototype, _, per_ref_features = read_prototype(proto_path)
    candidates, _ = read_candidates_with_features(cand_path)

    all_entries = []
    for frame_idx, cand_dets in candidates.items():
        for det in cand_dets:
            feat = getattr(det, "_feature", None)
            if feat is not None:
                all_entries.append((frame_idx, det, feat))

    if not all_entries:
        print("  no candidate features found -- nothing to check.")
        return

    all_frame_idxs = [e[0] for e in all_entries]
    all_dets = [e[1] for e in all_entries]
    all_feats = np.stack([e[2] for e in all_entries], axis=0)

    s3 = cfg.stage3
    use_multi_ref = (
        cfg.accuracy.mode in ("cheap_boosters", "max_accuracy")
        and cfg.accuracy.cheap_boosters.multi_reference_embedding
        and len(per_ref_features) > 0
    )
    multi_ref_pooling = cfg.accuracy.cheap_boosters.multi_ref_pooling

    if use_multi_ref:
        sims_per_ref = [_score_against_ref(all_feats, ref_feat, s3.similarity) for ref_feat in per_ref_features]
        all_sims = _pool_sims(sims_per_ref, multi_ref_pooling)
    else:
        all_sims = _score_against_ref(all_feats, prototype, s3.similarity)

    rounds_seen = 0

    def on_round(round_idx: int, high_conf_mask: np.ndarray, threshold: float) -> None:
        nonlocal rounds_seen
        rounds_seen += 1
        selected = np.where(high_conf_mask)[0]
        n_with_gt = 0
        n_correct = 0
        for idx in selected:
            fi = all_frame_idxs[idx]
            if fi in gt:
                n_with_gt += 1
                if box_iou(gt[fi], all_dets[idx].box) >= iou_threshold:
                    n_correct += 1

        if n_with_gt == 0:
            purity_str = "n/a (none of the selected candidates' frames have GT)"
        else:
            purity_str = f"{n_correct}/{n_with_gt} ({100.0 * n_correct / n_with_gt:.0f}%)"

        print(
            f"  Round {round_idx + 1}: threshold={threshold:.3f}, selected={len(selected)} candidates, "
            f"purity vs GT={purity_str}"
        )
        if n_with_gt > 0 and n_correct < n_with_gt:
            print(
                f"    warning: {n_with_gt - n_correct} of {n_with_gt} GT-checkable picks this round "
                f"did NOT match the real object (IoU<{iou_threshold}) -- possible confuser drift."
            )

    run_dynamic_prototype_rounds(
        sample_id, all_feats, all_sims, prototype, per_ref_features,
        use_multi_ref, multi_ref_pooling, s3.similarity, s3.dynamic_prototype,
        on_round=on_round,
    )

    if rounds_seen == 0:
        print(
            f"  0 rounds ran (stage3.dynamic_prototype.min_support={s3.dynamic_prototype.min_support} "
            "was never met on the very first round) -- nothing to report."
        )
    elif rounds_seen < s3.dynamic_prototype.rounds:
        print(
            f"  note: only {rounds_seen}/{s3.dynamic_prototype.rounds} configured rounds ran "
            "(later round(s) stopped early -- too few candidates met min_support)."
        )


def main():
    logging.basicConfig(level=logging.WARNING)
    p = argparse.ArgumentParser(
        description="Check whether stage3.dynamic_prototype's high-confidence picks each "
        "round actually match ground truth, or are drifting toward a confuser."
    )
    p.add_argument("--config", required=True)
    p.add_argument("--sample", default=None, help="omit to check all samples in data_root")
    p.add_argument("--set", action="append", default=[])
    p.add_argument("--iou-threshold", type=float, default=0.5,
                    help="IoU >= this counts as a correct pick (default: 0.5)")
    args = p.parse_args()

    from aero_eyes.config import load_config
    cfg = load_config(args.config, args.set)

    if args.sample:
        sample_ids = [args.sample]
    else:
        data_root = Path(cfg.data.data_root)
        sample_ids = [d.name for d in sorted(data_root.iterdir()) if d.is_dir()]

    for sid in sample_ids:
        check_sample(cfg, sid, args.iou_threshold)


if __name__ == "__main__":
    main()
