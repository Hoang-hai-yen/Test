"""ST-IoU evaluation.

    python -m aero_eyes.evaluate --pred submission.json --gt annotations.json --config ...

ST-IoU semantics (implemented exactly):
  - Build the predicted tube and the GT tube (frame_idx -> box).
  - Take the TEMPORAL UNION of frames present in either tube.
  - For each frame in that union:
      * both present -> spatial IoU of the two boxes
      * only one present -> IoU contribution = 0.0
  - Video ST-IoU = mean of per-frame IoU values over the union.
  - Leaderboard score = mean ST-IoU across all evaluation videos.
  - Both tubes empty -> 0.0 (documented assumption).
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from aero_eyes.types import Box
from aero_eyes.utils.geometry import box_iou

log = logging.getLogger(__name__)


def st_iou(pred_tube: dict[int, Box], gt_tube: dict[int, Box]) -> float:
    """Compute Spatio-Temporal IoU between two per-frame-box dicts.

    Both tubes empty -> 0.0.
    """
    union_frames = set(pred_tube.keys()) | set(gt_tube.keys())

    if not union_frames:
        # Both empty -> 0.0 (documented convention)
        return 0.0

    total = 0.0
    for fi in union_frames:
        in_pred = fi in pred_tube
        in_gt = fi in gt_tube
        if in_pred and in_gt:
            total += box_iou(pred_tube[fi], gt_tube[fi])
        else:
            total += 0.0  # only one tube has this frame

    return total / len(union_frames)


def _load_submission(path: str | Path) -> dict[str, dict[int, Box]]:
    """Load a submission/annotation JSON file.

    Returns {video_id -> {frame_idx -> Box}}.
    """
    with open(path) as f:
        data = json.load(f)

    result: dict[str, dict[int, Box]] = {}
    for entry in data:
        vid = entry["video_id"]
        tube: dict[int, Box] = {}
        for ann in entry.get("annotations", []):
            for bbox in ann.get("bboxes", []):
                tube[bbox["frame"]] = Box(
                    x1=float(bbox["x1"]),
                    y1=float(bbox["y1"]),
                    x2=float(bbox["x2"]),
                    y2=float(bbox["y2"]),
                )
        result[vid] = tube
    return result


def evaluate_dataset(
    pred_path: str | Path,
    gt_path: str | Path,
    cfg=None,
) -> dict:
    """Compute per-video and mean ST-IoU.

    Returns {'per_video': {video_id: float}, 'mean_st_iou': float}.
    """
    pred_all = _load_submission(pred_path)
    gt_all = _load_submission(gt_path)

    per_video: dict[str, float] = {}
    for vid in gt_all:
        pred_tube = pred_all.get(vid, {})
        gt_tube = gt_all[vid]
        score = st_iou(pred_tube, gt_tube)
        per_video[vid] = score
        log.info("  %-30s ST-IoU = %.4f", vid, score)

    mean = sum(per_video.values()) / len(per_video) if per_video else 0.0
    return {"per_video": per_video, "mean_st_iou": mean}


def main():
    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser(description="ST-IoU evaluation")
    p.add_argument("--pred", required=True, help="Submission JSON file")
    p.add_argument("--gt", required=True, help="Ground-truth annotations JSON file")
    p.add_argument("--config", default=None, help="Config YAML (optional)")
    args = p.parse_args()

    cfg = None
    if args.config:
        from aero_eyes.config import load_config
        cfg = load_config(args.config)

    result = evaluate_dataset(args.pred, args.gt, cfg=cfg)
    print(f"\nMean ST-IoU: {result['mean_st_iou']:.4f}")
    print("\nPer-video scores:")
    for vid, score in sorted(result["per_video"].items()):
        print(f"  {vid}: {score:.4f}")


if __name__ == "__main__":
    main()
