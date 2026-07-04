"""Stage 5 — Spatio-temporal output.

Flow:  tracks.json
       -> aggregate into spatio-temporal tube
       -> temporal EMA smoothing
       -> fill short gaps by linear interpolation
       -> drop short tubes
       -> tube.json + submission file

Reads:  tracks.json
Writes: tube.json + <work_dir>/<sample_id>/submission.json
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np

from aero_eyes.types import Box

log = logging.getLogger(__name__)


def _ema_smooth(tube: dict[int, Box], alpha: float) -> dict[int, Box]:
    """Exponential moving average smoothing over consecutive present frames."""
    if not tube:
        return tube
    frames = sorted(tube.keys())
    smoothed: dict[int, Box] = {}
    prev: Box | None = None
    for fi in frames:
        b = tube[fi]
        if prev is None:
            smoothed[fi] = b
        else:
            smoothed[fi] = Box(
                x1=alpha * b.x1 + (1 - alpha) * prev.x1,
                y1=alpha * b.y1 + (1 - alpha) * prev.y1,
                x2=alpha * b.x2 + (1 - alpha) * prev.x2,
                y2=alpha * b.y2 + (1 - alpha) * prev.y2,
                score=b.score,
            )
        prev = smoothed[fi]
    return smoothed


def _fill_gaps(tube: dict[int, Box], max_gap: int) -> dict[int, Box]:
    """Linearly interpolate gaps of <= max_gap frames."""
    if not tube:
        return tube
    frames = sorted(tube.keys())
    filled = dict(tube)
    for i in range(len(frames) - 1):
        f0, f1 = frames[i], frames[i + 1]
        gap = f1 - f0 - 1
        if 0 < gap <= max_gap:
            b0, b1 = tube[f0], tube[f1]
            for j in range(1, gap + 1):
                t = j / (gap + 1)
                filled[f0 + j] = Box(
                    x1=b0.x1 + t * (b1.x1 - b0.x1),
                    y1=b0.y1 + t * (b1.y1 - b0.y1),
                    x2=b0.x2 + t * (b1.x2 - b0.x2),
                    y2=b0.y2 + t * (b1.y2 - b0.y2),
                    score=(b0.score + b1.score) / 2,
                )
    return filled


def _split_into_segments(tube: dict[int, Box]) -> list[dict[int, Box]]:
    """Split a tube into contiguous segments (consecutive frames)."""
    if not tube:
        return []
    frames = sorted(tube.keys())
    segments: list[dict[int, Box]] = []
    current: dict[int, Box] = {frames[0]: tube[frames[0]]}
    for i in range(1, len(frames)):
        if frames[i] == frames[i - 1] + 1:
            current[frames[i]] = tube[frames[i]]
        else:
            segments.append(current)
            current = {frames[i]: tube[frames[i]]}
    segments.append(current)
    return segments


def run_stage5(cfg, sample_id: str) -> Path:
    """Run Stage 5. Returns path to submission file."""
    from aero_eyes.utils import viz as vizmod
    from aero_eyes.utils.io import (append_submission, read_tracks, write_submission,
                                     write_tube)
    from aero_eyes.utils.video import video_info

    t0 = time.time()
    work_dir = Path(cfg.project.work_dir) / sample_id
    work_dir.mkdir(parents=True, exist_ok=True)

    submission_path = work_dir / cfg.data.submission.path_name

    # ---- Load tracks ----
    tracks_path = work_dir / "tracks.json"
    if not tracks_path.exists():
        raise FileNotFoundError(
            f"tracks.json not found at {tracks_path}. Run Stage 4 first."
        )
    tracks = read_tracks(tracks_path)

    # Keep only non-None entries
    raw_tube: dict[int, Box] = {fi: b for fi, b in tracks.items() if b is not None}
    log.info("[Stage5] %s: %d frames with boxes before smoothing", sample_id, len(raw_tube))

    s5 = cfg.stage5

    # ---- Temporal smoothing ----
    if s5.temporal_smoothing.enabled and s5.temporal_smoothing.method == "ema":
        tube = _ema_smooth(raw_tube, alpha=s5.temporal_smoothing.ema_alpha)
    else:
        tube = dict(raw_tube)

    # ---- Fill short gaps ----
    tube = _fill_gaps(tube, max_gap=s5.fill_short_gaps)

    # ---- Drop short tube segments ----
    segments = _split_into_segments(tube)
    final_tube: dict[int, Box] = {}
    for seg in segments:
        if len(seg) >= s5.min_tube_length:
            final_tube.update(seg)

    log.info("[Stage5] %s: %d frames in final tube", sample_id, len(final_tube))

    # ---- Write artifacts ----
    tube_path = work_dir / "tube.json"
    write_tube(final_tube, tube_path)
    write_submission(final_tube, sample_id, submission_path, cfg)

    # ---- Visualization ----
    if cfg.runtime.save_visualizations:
        data_root = Path(cfg.data.data_root)
        video_files = list((data_root / sample_id).glob(cfg.data.video_glob))
        if video_files:
            try:
                vinfo = video_info(video_files[0])
                total_frames = vinfo["total_frames"]
                viz_dir = work_dir / "viz" / "stage5"
                viz_dir.mkdir(parents=True, exist_ok=True)
                vizmod.save_stage5_timeline(final_tube, total_frames,
                                            viz_dir / "timeline.jpg")
            except Exception as e:
                log.warning("[Stage5] Visualization failed: %s", e)

    elapsed = time.time() - t0
    log.info("[Stage5] %s done in %.1fs -> %s", sample_id, elapsed, submission_path)
    return submission_path


def main():
    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser(description="Stage 5 — spatio-temporal output")
    p.add_argument("--config", required=True)
    p.add_argument("--sample", required=True)
    p.add_argument("--set", action="append", default=[])
    args = p.parse_args()
    from aero_eyes.config import load_config
    cfg = load_config(args.config, args.set)
    run_stage5(cfg, args.sample)


if __name__ == "__main__":
    main()
