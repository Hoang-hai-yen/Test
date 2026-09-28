"""Per-sample Grounding DINO threshold calibration via synthetic paste --
answers "what box_threshold (or online_adaptive_threshold.z_score/abs_floor)
separates THIS object from THIS video's own clutter", using an actual
labeled anchor instead of guessing from an unlabeled score distribution's
shape alone.

Idea: paste the (masked, degraded) reference photo into several REAL frames
of the SAME query video -- reusing the exact copy-paste building blocks from
scripts/prepare_gdino_finetune_data.py -- then run Grounding DINO on both
the pasted and the original version of each sampled frame:
  - PASTED frame -> whichever detected box overlaps the known paste
    location (IoU >= --min-iou) is a "positive" sample: the score Grounding
    DINO assigns to something that IS (an approximation of) the real
    target, in THIS video's own visual context.
  - ORIGINAL (unpasted) frame -> every detected box is a "background"
    sample: a genuine confuser this video's own clutter produces, under
    this same text prompt, with the real object definitely NOT there.

Reports both score distributions and a threshold sweep table (for each
candidate z_score, the resulting threshold, what fraction of positive
samples would clear it, and what fraction of background samples would
ALSO clear it) -- deliberately does NOT auto-pick one number; use the
table to choose stage123_gdino.online_adaptive_threshold.z_score/abs_floor
(or a flat box_threshold) with an actual sense of the tradeoff, matching
this project's own convention for adaptive_z_score (see config.yaml's own
sweep-table comment for stage3.adaptive_z_score).

CAVEAT (read before trusting the numbers): the pasted crop is a
DEGRADED APPROXIMATION of the real object as it would actually appear from
the drone (see prepare_gdino_finetune_data.py's own docstring on the same
domain-gap concern) -- if Grounding DINO scores the synthetic paste
systematically higher or lower than it would score the real, as-filmed
object, the calibrated threshold inherits that bias. Treat this as a
starting point to sanity-check with --preview-n images, not a certainty.

This is a ONE-TIME, OFFLINE calibration step you run before deployment --
it does not run inside the online detection loop itself (see
GDinoOnlineAdaptiveThresholdConfig's own docstring, aero_eyes/config.py,
for why the ONLINE loop itself can't do a whole-video pass).

Usage:
    python -m scripts.calibrate_gdino_threshold --config configs/config.yaml \\
        --sample Helmet_0 --n-frames 20 --preview-n 10
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import cv2
import numpy as np

from aero_eyes.types import Box
from aero_eyes.utils.geometry import box_iou
from scripts.prepare_gdino_finetune_data import (
    build_copy_paste_sample,
    tight_crop_from_mask,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pure logic
# ---------------------------------------------------------------------------

def match_paste_box(
    boxes_xyxy: np.ndarray, scores: np.ndarray, paste_box: Box, min_iou: float = 0.3,
) -> float | None:
    """Score of the detected box with the highest IoU against paste_box,
    if that IoU clears min_iou -- None if nothing overlaps well enough
    (Grounding DINO didn't propose anything near the pasted object at all,
    a LOCALIZATION miss rather than a threshold/scoring problem -- see
    summarize_calibration's own reporting of this rate)."""
    best_iou, best_score = 0.0, None
    for b, s in zip(boxes_xyxy, scores):
        cand = Box(x1=float(b[0]), y1=float(b[1]), x2=float(b[2]), y2=float(b[3]))
        iou = box_iou(cand, paste_box)
        if iou > best_iou:
            best_iou, best_score = iou, float(s)
    if best_iou >= min_iou:
        return best_score
    return None


def sweep_thresholds(
    positive_scores: np.ndarray, background_scores: np.ndarray, z_scores: list[float],
) -> list[dict]:
    """For each candidate z_score, threshold = mean(background) +
    z*std(background) (matching GDinoOnlineAdaptiveThreshold's own formula),
    and the fraction of positive/background samples that would clear it.
    An empty background_scores array degrades to threshold=0.0 for every
    z_score (nothing to calibrate against) -- caller should treat that
    case as "not enough background samples collected" rather than trust
    the resulting all-pass-through row."""
    if len(background_scores) == 0:
        center, spread = 0.0, 0.0
    else:
        center, spread = float(background_scores.mean()), float(background_scores.std())
    rows = []
    for z in z_scores:
        threshold = center + z * spread
        pos_recall = float((positive_scores >= threshold).mean()) if len(positive_scores) else float("nan")
        bg_pass_rate = float((background_scores >= threshold).mean()) if len(background_scores) else float("nan")
        rows.append({
            "z_score": z, "threshold": threshold,
            "positive_recall": pos_recall, "background_pass_rate": bg_pass_rate,
        })
    return rows


def summarize_calibration(positive_scores: list[float | None], background_scores: list[float]) -> dict:
    """positive_scores: one entry per pasted-frame trial, None = miss (no
    box overlapped the paste location at all -- see match_paste_box).
    Returns a report dict with hit/miss counts, percentiles of both score
    distributions (hits only for positive_scores), and does NOT include the
    threshold sweep (call sweep_thresholds separately with the hits array)."""
    hits = np.array([s for s in positive_scores if s is not None], dtype=np.float64)
    n_total = len(positive_scores)
    n_miss = n_total - len(hits)
    bg = np.array(background_scores, dtype=np.float64)

    def _pctiles(arr: np.ndarray) -> dict:
        if len(arr) == 0:
            return {"min": None, "p50": None, "mean": None, "p95": None, "max": None}
        return {
            "min": float(arr.min()), "p50": float(np.percentile(arr, 50)),
            "mean": float(arr.mean()), "p95": float(np.percentile(arr, 95)), "max": float(arr.max()),
        }

    return {
        "n_paste_trials": n_total,
        "n_localization_misses": n_miss,
        "miss_rate": n_miss / n_total if n_total else float("nan"),
        "positive_score_stats": _pctiles(hits),
        "background_score_stats": _pctiles(bg),
        "n_background_samples": len(bg),
        "_positive_hits_array": hits,   # consumed by sweep_thresholds, not printed directly
    }


# ---------------------------------------------------------------------------
# I/O-heavy orchestration
# ---------------------------------------------------------------------------

def _sample_frame_indices(total_frames: int, n_frames: int, rng: np.random.Generator) -> list[int]:
    """Uniform-stride sampling across the WHOLE video, deliberately NOT
    dependent on GT (a real deployment target has none) -- a real inference
    sample only has 3 reference photos, never in-video ground truth."""
    if total_frames <= n_frames:
        return list(range(total_frames))
    stride = total_frames / n_frames
    return sorted({int(i * stride + rng.uniform(0, stride)) for i in range(n_frames)} & set(range(total_frames)))


def _build_ref_crop_and_mask(cfg, sample_id: str):
    from aero_eyes.models.segmentation import MobileSAMSegmenter
    from aero_eyes.stages.stage1 import apply_ref_degradation

    data_root = Path(cfg.data.data_root)
    refs_dir = data_root / sample_id / cfg.data.refs_subdir
    ref_paths = sorted(
        p for p in (refs_dir.iterdir() if refs_dir.is_dir() else [])
        if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    )[: cfg.data.num_references]
    if not ref_paths:
        raise FileNotFoundError(f"No reference images found under {refs_dir}.")
    ref_img = cv2.imread(str(ref_paths[0]))
    segmenter = MobileSAMSegmenter(weights_path=cfg.stage1.segmentation.weights)
    mask = segmenter.segment(ref_img)
    tightened = tight_crop_from_mask(ref_img, mask)
    if tightened is None:
        raise ValueError(f"MobileSAM mask was empty for {ref_paths[0]} -- cannot build a paste crop.")
    crop, crop_mask = tightened
    return apply_ref_degradation(crop, downscale_factor=0.3, blur_ksize=3, jpeg_quality=60), crop_mask


def run_calibration(
    cfg, sample_id: str, n_frames: int, min_iou: float, seed: int,
    preview_dir: Path | None = None, preview_n: int = 10,
) -> dict:
    from aero_eyes.models.grounding_dino_detector import GroundingDinoDetector
    from aero_eyes.stages.stage123_gdino import _locate_video, resolve_text_prompt
    from aero_eyes.utils.video import read_frame, video_info

    rng = np.random.default_rng(seed)
    text_prompt = resolve_text_prompt(cfg, sample_id)
    log.info("[calibrate-gdino] %s: text prompt = %r", sample_id, text_prompt)

    ref_crop, ref_mask = _build_ref_crop_and_mask(cfg, sample_id)
    detector = GroundingDinoDetector(cfg)

    video_path = _locate_video(cfg, sample_id)
    total_frames = video_info(video_path)["total_frames"]
    frame_idxs = _sample_frame_indices(total_frames, n_frames, rng)
    log.info("[calibrate-gdino] %s: video=%s (%d frames), sampling %d frame(s) for calibration",
             sample_id, video_path.name, total_frames, len(frame_idxs))

    positive_scores: list[float | None] = []
    background_scores: list[float] = []
    n_previewed = 0

    for fi in frame_idxs:
        frame = read_frame(video_path, fi)
        h, w = frame.shape[:2]
        ch, cw = ref_crop.shape[:2]
        target_scale = float(rng.uniform(0.5, 1.0))
        tw = max(4, min(w - 1, round(cw * target_scale)))
        th = max(4, min(h - 1, round(ch * target_scale)))

        # ---- Original (unpasted) frame -> background samples ----
        bg_boxes, bg_scores_arr = detector.raw_boxes_and_scores(frame, text_prompt, box_threshold=0.0, text_threshold=0.0)
        background_scores.extend(float(s) for s in bg_scores_arr)

        # ---- Pasted frame -> one positive sample (or a miss) ----
        result = build_copy_paste_sample(frame, ref_crop, ref_mask, (tw, th), rng)
        if result is None:
            log.debug("[calibrate-gdino] %s: frame %d -- no valid paste placement, skipped.", sample_id, fi)
            continue
        pasted_img, paste_box = result
        pos_boxes, pos_scores_arr = detector.raw_boxes_and_scores(pasted_img, text_prompt, box_threshold=0.0, text_threshold=0.0)
        matched = match_paste_box(pos_boxes, pos_scores_arr, paste_box, min_iou)
        positive_scores.append(matched)

        if preview_dir is not None and n_previewed < preview_n:
            from aero_eyes.utils.viz import draw_box
            preview = pasted_img.copy()
            draw_box(preview, paste_box, "pasted (GT)", (0, 255, 0))
            for b, s in zip(pos_boxes, pos_scores_arr):
                cand = Box(x1=float(b[0]), y1=float(b[1]), x2=float(b[2]), y2=float(b[3]), score=float(s))
                if box_iou(cand, paste_box) < min_iou:
                    draw_box(preview, cand, f"{s:.2f}", (0, 0, 255))
            preview_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(preview_dir / f"{sample_id}_{fi:06d}.jpg"), preview)
            n_previewed += 1

    report = summarize_calibration(positive_scores, background_scores)
    report["sweep"] = sweep_thresholds(
        report.pop("_positive_hits_array"), np.array(background_scores, dtype=np.float64),
        z_scores=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
    )
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_report(sample_id: str, report: dict) -> None:
    print(f"\n=== Grounding DINO threshold calibration -- {sample_id} ===")
    print(f"Paste trials: {report['n_paste_trials']}, localization misses: "
          f"{report['n_localization_misses']} ({report['miss_rate']:.1%})")
    if report["miss_rate"] and report["miss_rate"] > 0.3:
        print("WARNING: >30% localization misses -- Grounding DINO often didn't propose ANY box near "
              "the pasted object at all. No threshold fixes this; check the prompt wording and preview images.")
    pos = report["positive_score_stats"]
    bg = report["background_score_stats"]
    print(f"Positive (pasted, hits only) scores -- min={pos['min']} p50={pos['p50']} mean={pos['mean']} "
          f"p95={pos['p95']} max={pos['max']}")
    print(f"Background (real clutter, n={report['n_background_samples']}) scores -- min={bg['min']} "
          f"p50={bg['p50']} mean={bg['mean']} p95={bg['p95']} max={bg['max']}")
    print("\nz_score sweep (threshold = mean(background) + z*std(background)):")
    print(f"{'z_score':>8} {'threshold':>10} {'positive_recall':>16} {'background_pass_rate':>22}")
    for row in report["sweep"]:
        print(f"{row['z_score']:>8.1f} {row['threshold']:>10.3f} {row['positive_recall']:>16.1%} "
              f"{row['background_pass_rate']:>22.1%}")
    print("\nPick the z_score whose positive_recall/background_pass_rate tradeoff you're comfortable with, "
          "then set stage123_gdino.online_adaptive_threshold.z_score (and abs_floor near the background "
          "mean, not 0) -- this table does not auto-select one for you.")


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser(description="Calibrate Grounding DINO's threshold per sample via synthetic paste")
    p.add_argument("--config", required=True)
    p.add_argument("--sample", required=True)
    p.add_argument("--n-frames", type=int, default=20)
    p.add_argument("--min-iou", type=float, default=0.3, help="IoU to count a detected box as matching the pasted object")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--preview-dir", default=None, help="default: <project.work_dir>/<sample>/viz/calibrate_gdino")
    p.add_argument("--preview-n", type=int, default=10)
    args = p.parse_args()

    from aero_eyes.config import load_config
    cfg = load_config(args.config)

    preview_dir = Path(args.preview_dir) if args.preview_dir else (
        Path(cfg.project.work_dir) / args.sample / "viz" / "calibrate_gdino"
    )
    report = run_calibration(
        cfg, args.sample, args.n_frames, args.min_iou, args.seed,
        preview_dir=preview_dir, preview_n=args.preview_n,
    )
    _print_report(args.sample, report)
    print(f"\nPreview images (pasted box in green, other detections in red): {preview_dir}")


if __name__ == "__main__":
    main()
