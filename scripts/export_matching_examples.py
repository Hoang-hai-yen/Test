"""Export a visual montage: reference images vs. a good/bad Stage B match.

Requested in review: "show images of what matched well and what matched
badly, and illustrate the domain gap visually." This picks, from Stage 3's
actual detections:
  - the frame with the HIGHEST IoU against GT among detected frames (good match)
  - the frame with the LOWEST IoU against GT among detected frames (bad match)
and crops both the predicted box and the true GT box from the video at
that frame, so a viewer can see side by side: reference photo vs. what the
system picked vs. what was actually there.

Reads: refs/*.jpg, detections.json, video, GT annotations.
Writes: <work_dir>/<sample>/viz/matching_examples.jpg

Usage:
    python -m scripts.export_matching_examples --config configs/config.yaml --sample BlackBox_0
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import cv2
import numpy as np

from aero_eyes.types import Box
from aero_eyes.utils.geometry import box_iou, crop_with_pad
from aero_eyes.utils.io import load_gt, read_detections
from aero_eyes.utils.video import frame_iterator

log = logging.getLogger(__name__)

PANEL = 220  # px, square panel size for every crop in the montage
LABEL_H = 30
PAD = 10


def _fit_square(img: np.ndarray, size: int = PANEL) -> np.ndarray:
    """Resize (letterbox, preserve aspect ratio) into a size x size square."""
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        img = np.zeros((size, size, 3), dtype=np.uint8)
        h, w = size, size
    scale = size / max(h, w)
    nh, nw = max(1, int(h * scale)), max(1, int(w * scale))
    resized = cv2.resize(img, (nw, nh))
    canvas = np.full((size, size, 3), 40, dtype=np.uint8)
    y0, x0 = (size - nh) // 2, (size - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas


def _panel_with_label(img: np.ndarray, label: str, size: int = PANEL) -> np.ndarray:
    square = _fit_square(img, size)
    out = np.full((size + LABEL_H, size, 3), 255, dtype=np.uint8)
    out[:size] = square
    cv2.putText(out, label, (4, size + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def _hstack_padded(panels: list[np.ndarray], pad: int = PAD) -> np.ndarray:
    h = max(p.shape[0] for p in panels)
    padded = []
    for p in panels:
        if p.shape[0] < h:
            extra = np.full((h - p.shape[0], p.shape[1], 3), 255, dtype=np.uint8)
            p = np.vstack([p, extra])
        padded.append(p)
        padded.append(np.full((h, pad, 3), 255, dtype=np.uint8))
    return np.hstack(padded[:-1])


def export_for_sample(cfg, sample_id: str) -> Path | None:
    work_dir = Path(cfg.project.work_dir) / sample_id
    det_path = work_dir / "detections.json"
    if not det_path.exists():
        print(f"{sample_id}: detections.json not found — run Stage 3 first.")
        return None

    detections = read_detections(det_path)
    gt_file = cfg.data.gt.global_file
    try:
        gt = load_gt(gt_file, sample_id)
    except KeyError:
        print(f"{sample_id}: not found in {gt_file}, skipping.")
        return None

    # For every detected frame that also has GT, compute IoU(best detection, GT)
    scored: list[tuple[int, float, Box, float]] = []  # (frame_idx, iou, pred_box, sim)
    for frame_idx, dets in detections.items():
        if frame_idx not in gt or not dets:
            continue
        best_det = max(dets, key=lambda d: d.similarity)
        iou = box_iou(best_det.box, gt[frame_idx])
        scored.append((frame_idx, iou, best_det.box, best_det.similarity))

    if not scored:
        print(f"{sample_id}: no detected frame overlaps a GT frame — cannot build examples.")
        return None

    scored.sort(key=lambda t: t[1])
    worst = scored[0]
    best = scored[-1]

    data_root = Path(cfg.data.data_root)
    video_files = list((data_root / sample_id).glob(cfg.data.video_glob))
    if not video_files:
        print(f"{sample_id}: no video found.")
        return None
    video_path = video_files[0]

    wanted_frames = {best[0], worst[0]}
    frames_read: dict[int, np.ndarray] = {}
    for frame_idx, frame_bgr in frame_iterator(video_path):
        if frame_idx in wanted_frames:
            frames_read[frame_idx] = frame_bgr
        if len(frames_read) == len(wanted_frames):
            break

    pad_ratio = cfg.stage2.candidate.feature_crop_pad
    refs_dir = data_root / sample_id / cfg.data.refs_subdir
    ref_paths = sorted(refs_dir.glob("*.jpg")) + sorted(refs_dir.glob("*.png")) + \
        sorted(refs_dir.glob("*.JPG")) + sorted(refs_dir.glob("*.jpeg"))
    ref_panels = [
        _panel_with_label(cv2.imread(str(p)), f"Ref {i}") for i, p in enumerate(ref_paths[:3])
    ]

    def _case_panels(frame_idx: int, iou: float, pred_box: Box, sim: float, tag: str) -> list[np.ndarray]:
        frame_bgr = frames_read[frame_idx]
        pred_crop = crop_with_pad(frame_bgr, pred_box, pad_ratio=pad_ratio)
        gt_crop = crop_with_pad(frame_bgr, gt[frame_idx], pad_ratio=pad_ratio)
        return [
            _panel_with_label(pred_crop, f"{tag}: du doan (sim={sim:.3f})"),
            _panel_with_label(gt_crop, f"{tag}: GT that (IoU={iou:.3f})"),
        ]

    rows = []
    if ref_panels:
        rows.append(_hstack_padded(ref_panels))
    rows.append(_hstack_padded(_case_panels(*best, "TOT")))
    rows.append(_hstack_padded(_case_panels(*worst, "XAU")))

    max_w = max(r.shape[1] for r in rows)
    padded_rows = []
    for r in rows:
        if r.shape[1] < max_w:
            extra = np.full((r.shape[0], max_w - r.shape[1], 3), 255, dtype=np.uint8)
            r = np.hstack([r, extra])
        padded_rows.append(r)
        padded_rows.append(np.full((PAD, max_w, 3), 255, dtype=np.uint8))
    montage = np.vstack(padded_rows[:-1])

    out_dir = work_dir / "viz"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "matching_examples.jpg"
    cv2.imwrite(str(out_path), montage)
    print(
        f"{sample_id}: wrote {out_path}  "
        f"(best: frame={best[0]} IoU={best[1]:.3f} sim={best[3]:.3f} | "
        f"worst: frame={worst[0]} IoU={worst[1]:.3f} sim={worst[3]:.3f})"
    )
    return out_path


def main():
    logging.basicConfig(level=logging.WARNING)
    p = argparse.ArgumentParser(description="Export reference vs. good/bad match montage")
    p.add_argument("--config", required=True)
    p.add_argument("--sample", default=None, help="omit to export for all samples in data_root")
    p.add_argument("--set", action="append", default=[])
    args = p.parse_args()

    from aero_eyes.config import load_config
    cfg = load_config(args.config, args.set)

    if args.sample:
        sample_ids = [args.sample]
    else:
        data_root = Path(cfg.data.data_root)
        sample_ids = [d.name for d in sorted(data_root.iterdir()) if d.is_dir()]

    for sid in sample_ids:
        export_for_sample(cfg, sid)


if __name__ == "__main__":
    main()
