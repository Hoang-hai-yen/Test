"""Convert competition GT annotations into a YOLO training dataset, for
fine-tuning YOLOv11n as a class-agnostic single-class detector.

No first-party train split exists for this competition -- the only labeled
data is the PublicTest (6 videos) + PrivateTest (10 videos) sets themselves,
already used to validate the pipeline. To fine-tune without leaking (train
and evaluate on the exact same video), pair this script with a 4-fold
category-grouped cross-validation scheme: for each fold, call this script
once with --train-videos = the other 3 folds' 12 videos and --val-videos =
this fold's 4 held-out videos, train a checkpoint on the resulting dataset,
then evaluate the aero_eyes pipeline (--set stage2.yolov11n.weights=<ckpt>)
ONLY on the held-out videos. Repeat for all 4 folds; pooling all 16 videos'
results (each scored by a checkpoint that never saw it during training)
gives one honest, full-coverage Mean ST-IoU.

Usage:
    python -m scripts.prepare_yolo_finetune_data \
        --config configs/config.yaml \
        --gt-files "PublicTest/samples/annotations (1).json" \
                   "annotations_converted (1).json" \
        --train-videos BlackBox_0 BlackBox_1 CardboardBox_0 CardboardBox_1 \
                       LifeJacket_0 LifeJacket_1 Helmet_0 Helmet_1 \
                       Motorbike_0 Motorbike_1 Person2_0 Person2_1 \
        --val-videos IDCard_0 IDCard_1 Wallet_0 Wallet_1 \
        --out /kaggle/working/yolo_ft/fold3
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from aero_eyes.types import Box

log = logging.getLogger(__name__)


def box_to_yolo_label(box: Box, img_w: float, img_h: float) -> tuple[float, float, float, float]:
    """Convert an absolute-pixel xyxy Box (the schema `load_gt` returns) to
    normalized YOLO label format (cx, cy, w, h), all in [0, 1]."""
    cx = (box.x1 + box.x2) / 2.0 / img_w
    cy = (box.y1 + box.y2) / 2.0 / img_h
    w = (box.x2 - box.x1) / img_w
    h = (box.y2 - box.y1) / img_h
    return cx, cy, w, h


def convert_video_to_yolo(
    video_path,
    gt: dict[int, Box],
    video_id: str,
    images_dir: Path,
    labels_dir: Path,
) -> int:
    """Write every GT-labeled frame of one video as a YOLO image+label pair.
    Reuses `frame_iterator` (sequential decode) rather than seeking per
    frame -- much faster for compressed video, same pattern stage2.py uses.
    Returns the number of frames written, to sanity-check against len(gt)."""
    import cv2

    from aero_eyes.utils.video import frame_iterator, video_info

    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    info = video_info(video_path)
    img_w, img_h = info["width"], info["height"]

    written = 0
    for frame_idx, frame_bgr in frame_iterator(video_path):
        box = gt.get(frame_idx)
        if box is None:
            continue
        stem = f"{video_id}_{frame_idx}"
        cv2.imwrite(str(images_dir / f"{stem}.jpg"), frame_bgr)
        cx, cy, w, h = box_to_yolo_label(box, img_w, img_h)
        (labels_dir / f"{stem}.txt").write_text(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
        written += 1
    return written


def _load_gt_any(gt_files: list[Path], video_id: str) -> dict[int, Box]:
    """PublicTest and PrivateTest GT live in two separate files -- try each
    until one has this video_id (mirrors scripts/check_stage2_recall.py's
    try/except KeyError pattern for a single file)."""
    from aero_eyes.utils.io import load_gt

    for gt_file in gt_files:
        try:
            return load_gt(gt_file, video_id)
        except KeyError:
            continue
    raise KeyError(f"'{video_id}' not found in any of {gt_files}")


def _find_video(data_root: Path, video_id: str, video_glob: str) -> Path:
    matches = list((data_root / video_id).glob(video_glob))
    if not matches:
        raise FileNotFoundError(f"No video matching '{video_glob}' in {data_root / video_id}")
    return matches[0]


def build_split(
    cfg,
    gt_files: list[Path],
    video_ids: list[str],
    images_dir: Path,
    labels_dir: Path,
) -> None:
    data_root = Path(cfg.data.data_root)
    for video_id in video_ids:
        gt = _load_gt_any(gt_files, video_id)
        video_path = _find_video(data_root, video_id, cfg.data.video_glob)
        written = convert_video_to_yolo(video_path, gt, video_id, images_dir, labels_dir)
        log.info("[prepare_yolo_finetune_data] %s: wrote %d/%d GT frames",
                 video_id, written, len(gt))


def write_data_yaml(out_dir: Path) -> Path:
    data_yaml = out_dir / "data.yaml"
    data_yaml.write_text(
        f"path: {out_dir.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "nc: 1\n"
        "names: ['object']\n"
    )
    return data_yaml


def main():
    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--gt-files", nargs="+", required=True,
                   help="One or more GT annotation JSON files (PublicTest + PrivateTest).")
    p.add_argument("--train-videos", nargs="+", required=True)
    p.add_argument("--val-videos", nargs="+", required=True)
    p.add_argument("--out", required=True, help="Output dir for the YOLO dataset.")
    p.add_argument("--set", action="append", default=[])
    args = p.parse_args()

    from aero_eyes.config import load_config
    cfg = load_config(args.config, args.set)

    out_dir = Path(args.out)
    gt_files = [Path(f) for f in args.gt_files]

    build_split(cfg, gt_files, args.train_videos,
                out_dir / "images" / "train", out_dir / "labels" / "train")
    build_split(cfg, gt_files, args.val_videos,
                out_dir / "images" / "val", out_dir / "labels" / "val")

    data_yaml = write_data_yaml(out_dir)
    log.info("[prepare_yolo_finetune_data] wrote data.yaml -> %s", data_yaml)


if __name__ == "__main__":
    main()
