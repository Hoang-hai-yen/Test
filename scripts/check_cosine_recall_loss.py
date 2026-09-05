"""Diagnostic: WHY did a specific GT-present frame lose its correct box
during Stage 3, when the candidate pool (before cosine) already had a
good one? check_cosine_effect.py shows THAT recall drops after cosine;
this script shows WHERE in Stage 3's own pipeline each lost frame's
otherwise-correct candidate actually got removed:

    threshold filter -> global_topk cap -> per-frame NMS -> topk_per_keyframe cap

Replays the EXACT SAME functions Stage 3 itself uses (imported from
aero_eyes.stages.stage3, not reimplemented: _score_against_ref, _pool_sims,
run_dynamic_prototype_rounds, and stage3's own adaptive_threshold formula)
against candidates.json + prototype.npz + ground truth, so a mismatch here
would mean this script's replication itself is wrong, not that Stage 3
behaves differently.

For every GT-present frame where at least one candidate in the RAW pool
(before cosine) already reaches --iou-threshold against GT (i.e. cosine
COULD have kept a good box here), tracks the fate of that specific
best-IoU candidate through every filtering step:

  rejected_threshold      -- its own similarity score never cleared
                              effective_threshold (adaptive or fixed) --
                              the object's appearance didn't match well
                              enough on this frame, even after
                              dynamic_prototype's rounds (if enabled).
  cut_global_topk         -- passed threshold, but stage3.global_topk
                              capped the WHOLE video's candidate count and
                              this one ranked below the cutoff.
                              (only possible when global_topk is set)
  suppressed_by_nms       -- passed threshold+global_topk, but a
                              HIGHER-SCORING, OVERLAPPING box on the same
                              frame survived NMS instead -- i.e. cosine
                              similarity picked the wrong one among two
                              boxes covering roughly the same region.
  cut_by_topk_per_keyframe -- passed threshold+global_topk+NMS, but this
                              frame had more than stage3.topk_per_keyframe
                              surviving (non-overlapping) candidates and
                              this one didn't rank high enough by
                              similarity -- i.e. a DIFFERENT, non-
                              overlapping confuser scored higher on the
                              same frame and pushed it out.
  kept                     -- survived every filter; should appear as a TP
                              in detections.json. If it doesn't (see the
                              cross-check against the real file), something
                              downstream of this script's replication
                              (box_refine, a config drift) changed it.

Read-only: does not touch detections.json/tracks.json/prototype.npz.

Usage:
    python -m scripts.check_cosine_recall_loss --config configs/config.yaml --sample LifeJacket_1
    python -m scripts.check_cosine_recall_loss --config configs/config.yaml --iou-threshold 0.3
    python -m scripts.check_cosine_recall_loss --config configs/config.yaml   # all samples
"""
from __future__ import annotations

import argparse
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np

from aero_eyes.stages.stage3 import _pool_sims, _score_against_ref, run_dynamic_prototype_rounds
from aero_eyes.types import Box
from aero_eyes.utils.geometry import box_iou, nms
from aero_eyes.utils.io import load_gt, read_detections, read_prototype

log = logging.getLogger(__name__)

_FATE_ORDER = [
    "kept", "rejected_threshold", "suppressed_by_nms",
    "cut_by_topk_per_keyframe", "cut_global_topk",
]


def check_sample(cfg, sample_id: str, iou_threshold: float) -> None:
    from aero_eyes.stages.stage2 import read_candidates_with_features

    work_dir = Path(cfg.project.work_dir) / sample_id
    proto_path = work_dir / cfg.stage1.prototype.cache_name
    cand_path = work_dir / "candidates.json"
    det_path = work_dir / "detections.json"

    print(f"\n=== {sample_id} (IoU threshold={iou_threshold}) ===")

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

    all_entries = [
        (fi, det, getattr(det, "_feature", None))
        for fi, dets in candidates.items() for det in dets
    ]
    all_entries = [(fi, det, feat) for fi, det, feat in all_entries if feat is not None]
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
        sims_per_ref = [_score_against_ref(all_feats, ref, s3.similarity) for ref in per_ref_features]
        all_sims = _pool_sims(sims_per_ref, multi_ref_pooling)
    else:
        all_sims = _score_against_ref(all_feats, prototype, s3.similarity)

    _, all_sims, _ = run_dynamic_prototype_rounds(
        sample_id, all_feats, all_sims, prototype, per_ref_features,
        use_multi_ref, multi_ref_pooling, s3.similarity, s3.dynamic_prototype,
    )

    if cfg.accuracy.mode == "max_accuracy" and cfg.accuracy.max_accuracy.domain_prompter.enabled:
        print("  warning: accuracy.max_accuracy.domain_prompter is enabled -- this script "
              "does NOT replicate it, so the scores/threshold below may not exactly match "
              "what Stage 3 actually used.")

    # ---- effective threshold (mirrors run_stage3 exactly) ----
    if s3.adaptive_threshold:
        sim_mean = float(all_sims.mean())
        sim_std = float(all_sims.std())
        raw_threshold = sim_mean + s3.adaptive_z_score * sim_std
        effective_threshold = max(s3.adaptive_min_floor, raw_threshold) if s3.similarity == "cosine" else raw_threshold
    else:
        effective_threshold = s3.match_threshold

    keep_mask = all_sims >= effective_threshold
    fate: dict[int, str] = {i: "rejected_threshold" for i in range(len(all_sims)) if not keep_mask[i]}
    selected_idx = [i for i in range(len(all_sims)) if keep_mask[i]]

    # ---- global_topk cap ----
    global_topk = s3.global_topk
    if global_topk is not None and len(selected_idx) > global_topk:
        selected_idx.sort(key=lambda i: all_sims[i], reverse=True)
        for i in selected_idx[global_topk:]:
            fate[i] = "cut_global_topk"
        selected_idx = selected_idx[:global_topk]

    # ---- per-frame NMS + topk_per_keyframe ----
    frame_groups: dict[int, list[int]] = defaultdict(list)
    for i in selected_idx:
        frame_groups[all_frame_idxs[i]].append(i)

    for fi, idxs in frame_groups.items():
        boxes_for_nms = [
            Box(all_dets[i].box.x1, all_dets[i].box.y1, all_dets[i].box.x2, all_dets[i].box.y2,
                score=float(all_sims[i]))
            for i in idxs
        ]
        keep_local = nms(boxes_for_nms, iou_threshold=s3.nms_iou)
        kept_after_nms_ordered = [idxs[k] for k in keep_local]  # already score-desc, per nms()'s own contract
        kept_after_nms_set = set(kept_after_nms_ordered)
        for i in idxs:
            if i not in kept_after_nms_set:
                fate[i] = "suppressed_by_nms"
        topk_kept = set(kept_after_nms_ordered[: s3.topk_per_keyframe])
        for i in kept_after_nms_ordered:
            fate[i] = "kept" if i in topk_kept else "cut_by_topk_per_keyframe"

    # ---- for each GT frame, find the best-achievable candidate and its fate ----
    cand_by_frame: dict[int, list[int]] = defaultdict(list)
    for i, fi in enumerate(all_frame_idxs):
        cand_by_frame[fi].append(i)

    counts: dict[str, int] = defaultdict(int)
    examples: dict[str, list] = defaultdict(list)
    n_gt_achievable = 0
    for fi, gt_box in gt.items():
        idxs = cand_by_frame.get(fi)
        if not idxs:
            continue  # not a candidate/processed frame at all -- see check_cosine_effect's "unsampled" bucket
        best_i = max(idxs, key=lambda i: box_iou(gt_box, all_dets[i].box))
        best_iou = box_iou(gt_box, all_dets[best_i].box)
        if best_iou < iou_threshold:
            continue  # candidate pool itself has no good box here -- not cosine's doing
        n_gt_achievable += 1
        f = fate.get(best_i, "kept")  # never touched by any filter above -> survived everything
        counts[f] += 1
        if f != "kept" and len(examples[f]) < 3:
            examples[f].append((fi, float(all_sims[best_i]), best_iou))

    if n_gt_achievable == 0:
        print(f"  0 GT frames have a candidate reaching IoU>={iou_threshold} before cosine -- "
              "the deficit is upstream of Stage 3 (candidate generation itself); see "
              "check_stage2_recall.py / check_cosine_effect.py's BEFORE-cosine numbers instead.")
        return

    print(f"  {n_gt_achievable} GT frame(s) have a candidate reaching IoU>={iou_threshold} "
          f"before cosine (recall Stage 3 COULD achieve if it always picked the best box):")
    for label in _FATE_ORDER:
        cnt = counts.get(label, 0)
        if cnt == 0 and label not in counts:
            continue
        pct = 100.0 * cnt / n_gt_achievable
        print(f"    {label}: {cnt} ({pct:.0f}%)")
        for fi, sim, iou in examples.get(label, []):
            print(f"      e.g. frame {fi}: score={sim:.3f} vs threshold={effective_threshold:.3f}, "
                  f"achievable GT-IoU={iou:.3f}")

    n_lost = n_gt_achievable - counts.get("kept", 0)
    if n_lost > 0:
        dominant = max((l for l in _FATE_ORDER if l != "kept"), key=lambda l: counts.get(l, 0))
        print(f"  -> {n_lost}/{n_gt_achievable} achievable frames lost their good box during "
              f"Stage 3, dominant cause: {dominant}.")
        if dominant == "rejected_threshold":
            print("     -> the object's own similarity score is too low on these frames even "
                  "though the box is right -- appearance drift/pose/lighting, not a confuser. "
                  "Lowering match_threshold/adaptive_z_score, or dynamic_prototype tuning, is "
                  "the relevant lever, not multi_ref_pooling.")
        elif dominant in ("suppressed_by_nms", "cut_by_topk_per_keyframe"):
            print("     -> a DIFFERENT, higher-scoring candidate on the SAME frame is beating "
                  "the correct one -- a confuser is out-scoring the real object, not a pure "
                  "threshold problem. Check check_dynamic_prototype_purity.py and consider "
                  "whether multi_ref_pooling=max is inflating a confuser's score via a single "
                  "well-matching reference.")

    # ---- cross-check against the real detections.json, if present ----
    if det_path.exists():
        detections = read_detections(det_path)
        mismatches = 0
        for fi, gt_box in gt.items():
            idxs = cand_by_frame.get(fi)
            if not idxs:
                continue
            best_i = max(idxs, key=lambda i: box_iou(gt_box, all_dets[i].box))
            if box_iou(gt_box, all_dets[best_i].box) < iou_threshold or fate.get(best_i) != "kept":
                continue
            actual_boxes = [d.box for d in detections.get(fi, [])]
            actual_best_iou = max((box_iou(gt_box, b) for b in actual_boxes), default=0.0)
            if actual_best_iou < iou_threshold:
                mismatches += 1
        if mismatches > 0:
            print(f"  note: {mismatches} frame(s) this script marked 'kept' do NOT actually have "
                  "a good box in detections.json -- likely box_refine.enabled (moves the box "
                  "AFTER this script's replication ends) or a config difference from when "
                  "detections.json was last generated. Re-run Stage 3 with the current config "
                  "if in doubt.")


def main():
    logging.basicConfig(level=logging.WARNING)
    p = argparse.ArgumentParser(
        description="For GT frames where the raw candidate pool already had a good box, show "
        "exactly which Stage 3 filtering step (threshold / global_topk / NMS / "
        "topk_per_keyframe) removed it."
    )
    p.add_argument("--config", required=True)
    p.add_argument("--sample", default=None, help="omit to check all samples in data_root")
    p.add_argument("--set", action="append", default=[])
    p.add_argument("--iou-threshold", type=float, default=0.5,
                    help="IoU >= this counts as a correct localization (default: 0.5)")
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
