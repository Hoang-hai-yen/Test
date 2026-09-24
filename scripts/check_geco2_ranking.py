"""Diagnostic: WHY does a GeCo2 checkpoint miss the target, per frame that has one?

scripts/check_geco2_score_separation.py only compares the per-frame MAX score
of present vs absent frames -- it cannot say whether the target is missed
because (a) the frame produced no peaks at all, (b) no peak's box is close to
the GT (box regression), or (c) a good box exists but is out-ranked by clutter
(scoring). With score_threshold_ratio=score_threshold_abs=0 the pipeline just
keeps the top-K (topk_per_keyframe) NMS survivors, so recall is decided by
exactly these three things.

For each sampled PRESENT frame (all peaks, before NMS/top-K):
  empty            no peaks at all (boxes_with_scores found no local max > max/8;
                   happens when the score map's max is <= 0)
  good proposal    some peak's box has IoU >= --iou-thr with the GT
  rank             score rank (1 = highest) of the best-IoU peak
  pipeline hit     a box in filter_boxes_by_threshold(threshold=0.0)'s output
                   (NMS + top-K, as the pipeline emits it) has IoU >= --iou-thr
For each sampled ABSENT frame: how many boxes the pipeline would still emit.

Usage:
    python -m scripts.check_geco2_ranking --config configs/config.yaml --sample IDCard_0 \
        --set stage123_geco2.weights_path=<checkpoint>
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


def iou_matrix_np(boxes: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """IoU of each row of boxes [N,4] (xyxy) against one gt [4]."""
    if len(boxes) == 0:
        return np.zeros(0)
    x1 = np.maximum(boxes[:, 0], gt[0])
    y1 = np.maximum(boxes[:, 1], gt[1])
    x2 = np.minimum(boxes[:, 2], gt[2])
    y2 = np.minimum(boxes[:, 3], gt[3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_b = np.clip(boxes[:, 2] - boxes[:, 0], 0, None) * np.clip(boxes[:, 3] - boxes[:, 1], 0, None)
    area_g = max(0.0, gt[2] - gt[0]) * max(0.0, gt[3] - gt[1])
    union = area_b + area_g - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)


def summarize_present_frame(peak_boxes: np.ndarray, peak_scores: np.ndarray, gt: np.ndarray, iou_thr: float) -> dict:
    """peak_boxes [N,4] pixel xyxy, peak_scores [N], gt [4]."""
    n = len(peak_scores)
    if n == 0:
        return {"empty": True, "n_peaks": 0, "n_positive": 0, "best_iou": 0.0, "good_proposal": False, "rank": None}
    ious = iou_matrix_np(peak_boxes, gt)
    best = int(np.argmax(ious))
    order = np.argsort(-peak_scores, kind="stable")
    rank = int(np.where(order == best)[0][0]) + 1
    return {
        "empty": False, "n_peaks": n, "n_positive": int((peak_scores > 0).sum()),
        "best_iou": float(ious[best]), "good_proposal": bool(ious[best] >= iou_thr),
        "rank": rank if ious[best] >= iou_thr else None,
    }


def aggregate_present(rows: list[dict], top_k: int) -> dict:
    n = len(rows)
    if n == 0:
        return {}
    good = [r for r in rows if r["good_proposal"]]
    ranks = [r["rank"] for r in good]
    return {
        "n_frames": n,
        "empty_frac": sum(r["empty"] for r in rows) / n,
        "good_proposal_frac": len(good) / n,
        "median_best_iou": float(np.median([r["best_iou"] for r in rows])),
        "rank1_frac": sum(1 for x in ranks if x == 1) / n,
        f"rank_le_{top_k}_frac": sum(1 for x in ranks if x <= top_k) / n,
        "median_rank_of_good": float(np.median(ranks)) if ranks else None,
        "median_peaks": float(np.median([r["n_peaks"] for r in rows])),
        "pipeline_hit_frac": sum(r.get("pipeline_hit", False) for r in rows) / n,
    }


def _sample(indices: list[int], n: int) -> list[int]:
    if len(indices) <= n:
        return list(indices)
    step = (len(indices) - 1) / (n - 1)
    return sorted({indices[round(i * step)] for i in range(n)})


def check_sample(cfg, sample_id: str, detector, num_samples: int, iou_thr: float) -> None:
    from aero_eyes.stages.stage123_geco2 import build_exemplar_prototype
    from aero_eyes.utils.io import load_gt
    from aero_eyes.utils.video import frame_iterator, video_info
    from aero_eyes.utils.geometry import box_iou

    gt = load_gt(cfg.data.gt.global_file, sample_id)
    if not gt:
        print(f"{sample_id}: 0 GT frames, skipping.")
        return
    video_files = list((Path(cfg.data.data_root) / sample_id).glob(cfg.data.video_glob))
    if not video_files:
        print(f"{sample_id}: no video found.")
        return
    total = video_info(video_files[0])["total_frames"]
    present = _sample(sorted(gt), num_samples)
    absent = _sample([f for f in range(total) if f not in gt], num_samples)
    wanted = set(present) | set(absent)
    prototype = build_exemplar_prototype(cfg, sample_id, detector, Path(cfg.project.work_dir) / sample_id)

    rows, absent_counts, absent_max = [], [], []
    for fi, frame in frame_iterator(video_files[0]):
        if fi not in wanted:
            continue
        pred_boxes, box_v, scale = detector.forward_scores(frame, prototype)
        scores = box_v.reshape(-1).float().cpu().numpy()
        px = (pred_boxes.reshape(-1, 4).clamp(0, 1) / scale * detector.image_size).cpu().numpy() if scores.size else np.zeros((0, 4))
        final = detector.filter_boxes_by_threshold(pred_boxes, box_v, scale, frame, 0.0) if scores.size else []
        if fi in gt and fi in present:
            g = gt[fi]
            row = summarize_present_frame(px, scores, np.array([g.x1, g.y1, g.x2, g.y2]), iou_thr)
            row["pipeline_hit"] = any(box_iou(b, g) >= iou_thr for b in final)
            rows.append(row)
        elif fi in absent:
            absent_counts.append(len(final))
            absent_max.append(float(scores.max()) if scores.size else float("-inf"))

    top_k = detector.topk_per_keyframe
    agg = aggregate_present(rows, top_k)
    print(f"\n{sample_id}  (IoU threshold {iou_thr}, pipeline emits up to top-{top_k} after NMS)")
    print("  PRESENT frames:")
    for k, v in agg.items():
        print(f"    {k}: {v:.3f}" if isinstance(v, float) else f"    {k}: {v}")
    if absent_counts:
        print("  ABSENT frames:")
        print(f"    frames sampled: {len(absent_counts)}")
        print(f"    with zero emitted boxes: {sum(c == 0 for c in absent_counts) / len(absent_counts):.3f}")
        print(f"    mean emitted boxes/frame: {np.mean(absent_counts):.2f}")
    print(
        "  READ: empty_frac high -> frames output nothing (score map <= 0). good_proposal_frac low -> box "
        "regression is the limit. good_proposal_frac high but rank<=K low -> scoring/ranking is the limit."
    )


def main():
    logging.basicConfig(level=logging.WARNING)
    p = argparse.ArgumentParser(description="Why does GeCo2 miss the target: empty / box / ranking")
    p.add_argument("--config", required=True)
    p.add_argument("--sample", default=None, help="omit to check every sample in data_root")
    p.add_argument("--num-samples", type=int, default=60)
    p.add_argument("--iou-thr", type=float, default=0.5)
    p.add_argument("--set", action="append", default=[])
    args = p.parse_args()

    from aero_eyes.config import load_config
    from aero_eyes.models.geco2_detector import GeCo2Detector

    cfg = load_config(args.config, args.set)
    print(f">>> Loading GeCo2 weights from: {cfg.stage123_geco2.weights_path}")
    detector = GeCo2Detector(cfg)
    ids = [args.sample] if args.sample else [d.name for d in sorted(Path(cfg.data.data_root).iterdir()) if d.is_dir()]
    for sid in ids:
        check_sample(cfg, sid, detector, args.num_samples, args.iou_thr)


if __name__ == "__main__":
    main()
