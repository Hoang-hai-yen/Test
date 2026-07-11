"""Sweep stage3.adaptive_z_score AND estimate its generalization honestly.

Context: the original z-sweep (scripts/sweep_zscore_loocv.py's predecessor,
run ad hoc) picked z by looking at Mean ST-IoU across the same 6 videos that
were then reported as "the result". That is tuning a hyperparameter on the
test set — the chosen z is optimistic for those exact 6 videos and may not
generalize.

This script fixes that with leave-one-video-out cross-validation (LOOCV):
for each video, pick the best z using only the OTHER 5 videos, then score
that z on the held-out video. Averaging the held-out scores across all 6
folds gives an unbiased estimate of what z=<the naive pick> actually buys
you on unseen data — directly comparable to the naive "best on all 6"
number to show the gap.

Writes a JSON report with the full per-video, per-z score matrix so the
analysis is reproducible/inspectable, not just a printed summary.

Usage:
    python -m scripts.sweep_zscore_loocv --config configs/config.yaml \
        --set data.data_root=... --set data.gt.global_file=... \
        --set project.work_dir=/kaggle/working/runs/exp001 \
        --z-values 1.0,1.5,1.8,2.0,2.2,2.5,3.0
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)


def _clear_stage345_cache(work_dir: Path, sample_ids: list[str]) -> None:
    for sid in sample_ids:
        sdir = work_dir / sid
        for name in ("detections.json", "tracks.json", "submission.json"):
            f = sdir / name
            if f.exists():
                f.unlink()


def run_sweep(cfg, sample_ids: list[str], z_values: list[float]) -> dict[float, dict[str, float]]:
    """Returns {z: {video_id: st_iou}}."""
    from aero_eyes.stages.stage3 import run_stage3
    from aero_eyes.stages.stage4 import run_stage4
    from aero_eyes.stages.stage5 import run_stage5
    from aero_eyes.evaluate import evaluate_dataset

    if not cfg.stage3.adaptive_threshold:
        log.warning(
            "stage3.adaptive_threshold was false -- forcing it to true. "
            "This script sweeps adaptive_z_score, which has no effect unless "
            "adaptive_threshold is enabled (otherwise every z gives the same "
            "fixed-match_threshold result)."
        )
        cfg.stage3.adaptive_threshold = True

    work_dir = Path(cfg.project.work_dir)
    results: dict[float, dict[str, float]] = {}

    for z in z_values:
        cfg.stage3.adaptive_z_score = z
        _clear_stage345_cache(work_dir, sample_ids)

        for sid in sample_ids:
            run_stage3(cfg, sid)
            run_stage4(cfg, sid)
            run_stage5(cfg, sid)

        # Gather submissions into one file for evaluate_dataset
        all_preds = []
        for sid in sample_ids:
            sub_path = work_dir / sid / cfg.data.submission.path_name
            if sub_path.exists():
                all_preds.extend(json.loads(sub_path.read_text()))
        combined_path = work_dir / "all_submissions_sweep.json"
        combined_path.write_text(json.dumps(all_preds))

        report = evaluate_dataset(combined_path, cfg.data.gt.global_file, cfg=cfg)
        results[z] = report["per_video"]
        log.info("z=%.2f -> mean=%.4f  per-video=%s", z, report["mean_st_iou"], report["per_video"])
        print(f"z={z:.2f}  mean ST-IoU={report['mean_st_iou']:.4f}  {report['per_video']}")

    return results


def loocv_estimate(results: dict[float, dict[str, float]], sample_ids: list[str]) -> tuple[float, dict[str, float], dict[str, float]]:
    """Leave-one-video-out: for each held-out video, pick best z from the
    other videos, score that z on the held-out video.
    Returns (mean_held_out_score, per_video_held_out_score, per_video_chosen_z).
    """
    held_out_scores: dict[str, float] = {}
    chosen_z_per_video: dict[str, float] = {}

    for held_out in sample_ids:
        others = [s for s in sample_ids if s != held_out]
        best_z, best_mean = None, -1.0
        for z, per_video in results.items():
            mean_on_others = sum(per_video[s] for s in others) / len(others)
            if mean_on_others > best_mean:
                best_mean = mean_on_others
                best_z = z
        held_out_scores[held_out] = results[best_z][held_out]
        chosen_z_per_video[held_out] = best_z

    mean_held_out = sum(held_out_scores.values()) / len(held_out_scores)
    return mean_held_out, held_out_scores, chosen_z_per_video


def main():
    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser(description="Sweep adaptive z-score + LOOCV honest estimate")
    p.add_argument("--config", required=True)
    p.add_argument("--z-values", default="1.0,1.5,1.8,2.0,2.2,2.5,3.0")
    p.add_argument("--set", action="append", default=[])
    p.add_argument("--out", default=None, help="path to write JSON report (default: <work_dir>/zscore_loocv_report.json)")
    args = p.parse_args()

    from aero_eyes.config import load_config
    cfg = load_config(args.config, args.set)

    data_root = Path(cfg.data.data_root)
    sample_ids = [d.name for d in sorted(data_root.iterdir()) if d.is_dir()]
    z_values = [float(z) for z in args.z_values.split(",")]

    print(f"Samples: {sample_ids}")
    print(f"Sweeping z in {z_values}\n")

    results = run_sweep(cfg, sample_ids, z_values)

    # Naive pick: best z on ALL 6 videos (what was reported before -- optimistic)
    naive_best_z, naive_best_mean = None, -1.0
    for z, per_video in results.items():
        m = sum(per_video.values()) / len(per_video)
        if m > naive_best_mean:
            naive_best_mean = m
            naive_best_z = z

    mean_held_out, held_out_scores, chosen_z_per_video = loocv_estimate(results, sample_ids)

    print("\n" + "=" * 60)
    print(f"NAIVE (tuned on all 6, reported before): best z={naive_best_z} -> Mean ST-IoU={naive_best_mean:.4f}")
    print(f"LOOCV HONEST ESTIMATE (z picked without seeing held-out video): Mean ST-IoU={mean_held_out:.4f}")
    print("Per-video held-out score (z chosen from the OTHER 5 videos):")
    for sid in sample_ids:
        print(f"  {sid}: z={chosen_z_per_video[sid]}  ST-IoU={held_out_scores[sid]:.4f}")
    print("=" * 60)

    out_path = Path(args.out) if args.out else Path(cfg.project.work_dir) / "zscore_loocv_report.json"
    out_path.write_text(json.dumps({
        "z_values": z_values,
        "per_z_per_video": results,
        "naive_best_z": naive_best_z,
        "naive_best_mean": naive_best_mean,
        "loocv_mean_held_out": mean_held_out,
        "loocv_held_out_scores": held_out_scores,
        "loocv_chosen_z_per_video": chosen_z_per_video,
    }, indent=2))
    print(f"\nWrote report: {out_path}")


if __name__ == "__main__":
    main()
