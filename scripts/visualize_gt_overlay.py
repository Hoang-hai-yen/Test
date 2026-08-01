"""Sanity-check tool: draw ground-truth boxes onto a video to visually
confirm the annotations actually line up with the object (right place,
right frames -- catches frame-offset / wrong-video / mis-scaled-box bugs
before they silently corrupt an evaluation run).

Works directly off a video file + the global annotations JSON -- does NOT
require the video to already be staged under data/<video_id>/ (useful for
checking a freshly-converted label before deciding where to place it).

Usage:
    # A handful of sampled frames as JPGs (fast eyeball check)
    python -m scripts.visualize_gt_overlay \\
        --video drone_video.mp4 --gt "annotations (1).json" \\
        --video-id BlackBox_0 --out-dir checks/BlackBox_0

    # Also render a full annotated copy of the video (boxes on GT frames)
    python -m scripts.visualize_gt_overlay \\
        --video drone_video.mp4 --gt "annotations (1).json" \\
        --video-id BlackBox_0 --out-dir checks/BlackBox_0 --full-video
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import cv2

from aero_eyes.types import Box
from aero_eyes.utils.io import load_gt
from aero_eyes.utils.video import frame_iterator, read_frame, video_info
from aero_eyes.utils.viz import draw_box

log = logging.getLogger(__name__)


def _sample_frame_indices(present_frames: list[int], n: int) -> list[int]:
    """Evenly pick n frame indices out of the (possibly gappy) present frames,
    always including the first and last so scene entry/exit are checked too.
    """
    if len(present_frames) <= n:
        return present_frames
    step = (len(present_frames) - 1) / (n - 1)
    return sorted({present_frames[round(i * step)] for i in range(n)})


def save_sample_frames(video_path: Path, gt: dict[int, Box], out_dir: Path, num_samples: int) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    present = sorted(gt.keys())
    chosen = _sample_frame_indices(present, num_samples)

    saved: list[Path] = []
    for frame_idx in chosen:
        frame = read_frame(video_path, frame_idx)
        draw_box(frame, gt[frame_idx], label=f"GT frame {frame_idx}", color=(0, 0, 255))
        out_path = out_dir / f"frame_{frame_idx:06d}.jpg"
        cv2.imwrite(str(out_path), frame)
        saved.append(out_path)
    return saved


def save_annotated_video(video_path: Path, gt: dict[int, Box], out_path: Path) -> None:
    info = video_info(video_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, info["fps"] or 25.0, (info["width"], info["height"]))

    for frame_idx, frame in frame_iterator(video_path):
        if frame_idx in gt:
            draw_box(frame, gt[frame_idx], label=f"{frame_idx}", color=(0, 0, 255))
        writer.write(frame)
    writer.release()


def main():
    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser(description="Overlay GT boxes on a video to visually verify alignment")
    p.add_argument("--video", required=True, help="Path to the video file")
    p.add_argument("--gt", default="annotations (1).json", help="Global annotations JSON path")
    p.add_argument("--video-id", required=True, help="video_id to look up in --gt")
    p.add_argument("--out-dir", required=True, help="Output directory for sample frames / annotated video")
    p.add_argument("--num-samples", type=int, default=12, help="Number of sample frame JPGs to save")
    p.add_argument("--full-video", action="store_true", help="Also render a full GT-annotated copy of the video")
    args = p.parse_args()

    video_path = Path(args.video)
    out_dir = Path(args.out_dir)

    gt = load_gt(args.gt, args.video_id)
    if not gt:
        raise SystemExit(f"video_id '{args.video_id}' has 0 GT boxes in {args.gt}")

    info = video_info(video_path)
    log.info("Video: %s (%d frames, %dx%d, %.1f fps)",
              video_path, info["total_frames"], info["width"], info["height"], info["fps"] or 0.0)
    log.info("GT: %d annotated frames, range [%d, %d]", len(gt), min(gt), max(gt))
    out_of_range = [f for f in gt if f >= info["total_frames"]]
    if out_of_range:
        log.warning("%d GT frame(s) reference frames beyond the video's %d frames "
                     "(e.g. %s) -- video/label likely mismatched.",
                     len(out_of_range), info["total_frames"], sorted(out_of_range)[:5])

    saved = save_sample_frames(video_path, gt, out_dir / "samples", args.num_samples)
    log.info("Saved %d sample frame(s) to %s", len(saved), out_dir / "samples")

    if args.full_video:
        annotated_path = out_dir / "annotated.mp4"
        save_annotated_video(video_path, gt, annotated_path)
        log.info("Saved full annotated video to %s", annotated_path)


if __name__ == "__main__":
    main()
