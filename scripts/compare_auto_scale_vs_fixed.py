"""Full-pipeline, LOOCV-honest comparison: stage123_geco2.
auto_scale_calibration (per-sample adaptive) vs. the best FIXED
ref_downscale_factor found by leave-one-object-out cross-validation --
same rationale/structure as scripts/sweep_zscore_loocv.py, applied to
this project's own ref_downscale_factor instead of stage3.adaptive_z_score.

Runs stage123_geco2 -> stage4 -> stage5 -> evaluate_dataset for:
  (i)  each candidate ref_downscale_factor in --factors (auto_scale_
       calibration disabled) -- LOOCV picks, per held-out video, the best
       factor using only the OTHER videos, giving an honest (not tuned-on-
       the-test-set) estimate of "the best single global factor".
  (ii) auto_scale_calibration.enabled=true, ONE run (no LOOCV needed --
       the whole point of Track A is that it adapts PER SAMPLE already,
       so there is no single global hyperparameter to overfit by picking
       from the same videos it's scored on; see docs/
       GECO2_scale_domain_gap_plan.md).

Writes a JSON report with the full per-factor, per-video ST-IoU matrix
plus both final numbers side by side.

Usage:
    python -m scripts.compare_auto_scale_vs_fixed --config configs/config.yaml \
        --set data.data_root=... --set data.gt.global_file=... \
        --set project.work_dir=/kaggle/working/runs/exp001 \
        --factors 1.0,0.5,0.25,0.125,0.0625,0.03
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)


def _clear_geco2_cache(work_dir: Path, sample_ids: list[str], prototype_cache_name: str) -> None:
    for sid in sample_ids:
        sdir = work_dir / sid
        for name in ("detections.json", "tracks.json", "submission.json",
                     prototype_cache_name, "geco2_auto_scale_calibration.json"):
            f = sdir / name
            if f.exists():
                f.unlink()


def _run_full_pipeline(cfg, sample_ids: list[str]) -> dict[str, float]:
    """Returns {video_id: st_iou} for the CURRENT cfg."""
    from aero_eyes.stages.stage123_geco2 import run_stage123_geco2
    from aero_eyes.stages.stage4 import run_stage4
    from aero_eyes.stages.stage5 import run_stage5
    from aero_eyes.evaluate import evaluate_dataset

    work_dir = Path(cfg.project.work_dir)
    for sid in sample_ids:
        run_stage123_geco2(cfg, sid)
        run_stage4(cfg, sid)
        run_stage5(cfg, sid)

    all_preds = []
    for sid in sample_ids:
        sub_path = work_dir / sid / cfg.data.submission.path_name
        if sub_path.exists():
            all_preds.extend(json.loads(sub_path.read_text()))
    combined_path = work_dir / "all_submissions_auto_scale_compare.json"
    combined_path.write_text(json.dumps(all_preds))

    report = evaluate_dataset(combined_path, cfg.data.gt.global_file, cfg=cfg)
    return report["per_video"]


def run_fixed_factor_sweep(cfg, sample_ids: list[str], factors: list[float]) -> dict[float, dict[str, float]]:
    cfg.stage123_geco2.auto_scale_calibration.enabled = False
    work_dir = Path(cfg.project.work_dir)
    results: dict[float, dict[str, float]] = {}
    for factor in factors:
        cfg.stage123_geco2.ref_downscale_factor = factor
        _clear_geco2_cache(work_dir, sample_ids, cfg.stage123_geco2.prototype_cache_name)
        per_video = _run_full_pipeline(cfg, sample_ids)
        results[factor] = per_video
        mean = sum(per_video.values()) / len(per_video)
        log.info("ref_downscale_factor=%.4f -> mean ST-IoU=%.4f per-video=%s", factor, mean, per_video)
        print(f"ref_downscale_factor={factor:<8}  mean ST-IoU={mean:.4f}  {per_video}")
    return results


def loocv_estimate(results: dict[float, dict[str, float]], sample_ids: list[str]) -> tuple[float, dict[str, float], dict[str, float]]:
    """Leave-one-OBJECT-out would be more correct if sample_ids include
    _0/_1 same-object pairs (see sweep_topk_fusion_cosine_weight_loocv.py's
    own docstring for why) -- this helper is intentionally leave-one-
    VIDEO-out to stay usable on an arbitrary sample_ids list; pass an
    object-grouped sample_ids ordering upstream if you want the stricter
    leave-one-object-out split.
    """
    held_out_scores: dict[str, float] = {}
    chosen_factor_per_video: dict[str, float] = {}
    for held_out in sample_ids:
        others = [s for s in sample_ids if s != held_out]
        best_factor, best_mean = None, -1.0
        for factor, per_video in results.items():
            mean_on_others = sum(per_video[s] for s in others) / len(others)
            if mean_on_others > best_mean:
                best_mean = mean_on_others
                best_factor = factor
        held_out_scores[held_out] = results[best_factor][held_out]
        chosen_factor_per_video[held_out] = best_factor
    mean_held_out = sum(held_out_scores.values()) / len(held_out_scores)
    return mean_held_out, held_out_scores, chosen_factor_per_video


def run_auto_scale_calibration(cfg, sample_ids: list[str]) -> dict[str, float]:
    cfg.stage123_geco2.auto_scale_calibration.enabled = True
    work_dir = Path(cfg.project.work_dir)
    _clear_geco2_cache(work_dir, sample_ids, cfg.stage123_geco2.prototype_cache_name)
    per_video = _run_full_pipeline(cfg, sample_ids)
    mean = sum(per_video.values()) / len(per_video)
    log.info("auto_scale_calibration -> mean ST-IoU=%.4f per-video=%s", mean, per_video)
    print(f"auto_scale_calibration            mean ST-IoU={mean:.4f}  {per_video}")
    return per_video


def main():
    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser(
        description="Compare auto_scale_calibration vs. the best fixed ref_downscale_factor (LOOCV-honest)"
    )
    p.add_argument("--config", required=True)
    p.add_argument("--factors", default="1.0,0.5,0.25,0.125,0.0625,0.03")
    p.add_argument("--set", action="append", default=[])
    p.add_argument("--out", default=None,
                    help="path to write JSON report (default: <work_dir>/auto_scale_vs_fixed_report.json)")
    args = p.parse_args()

    from aero_eyes.config import load_config
    cfg = load_config(args.config, args.set)

    if cfg.project.use_cache:
        log.warning("project.use_cache=true in config -- forcing false for this sweep script.")
        cfg.project.use_cache = False

    data_root = Path(cfg.data.data_root)
    sample_ids = [d.name for d in sorted(data_root.iterdir()) if d.is_dir()]
    factors = [float(f) for f in args.factors.split(",")]

    print(f"Samples: {sample_ids}")
    print(f"Sweeping fixed ref_downscale_factor in {factors}\n")

    fixed_results = run_fixed_factor_sweep(cfg, sample_ids, factors)
    mean_held_out, held_out_scores, chosen_factor_per_video = loocv_estimate(fixed_results, sample_ids)

    naive_best_factor, naive_best_mean = None, -1.0
    for factor, per_video in fixed_results.items():
        m = sum(per_video.values()) / len(per_video)
        if m > naive_best_mean:
            naive_best_mean = m
            naive_best_factor = factor

    print()
    auto_results = run_auto_scale_calibration(cfg, sample_ids)
    auto_mean = sum(auto_results.values()) / len(auto_results)

    print("\n" + "=" * 70)
    print(f"NAIVE fixed factor (tuned on all videos, optimistic): factor={naive_best_factor} "
          f"-> Mean ST-IoU={naive_best_mean:.4f}")
    print(f"LOOCV HONEST fixed-factor estimate (factor picked without seeing held-out video): "
          f"Mean ST-IoU={mean_held_out:.4f}")
    print("Per-video held-out score (factor chosen from the OTHER videos):")
    for sid in sample_ids:
        print(f"  {sid}: factor={chosen_factor_per_video[sid]}  ST-IoU={held_out_scores[sid]:.4f}")
    print(f"\nauto_scale_calibration (per-sample adaptive, no LOOCV needed): Mean ST-IoU={auto_mean:.4f}")
    for sid in sample_ids:
        print(f"  {sid}: ST-IoU={auto_results[sid]:.4f}")
    print("=" * 70)

    out_path = Path(args.out) if args.out else Path(cfg.project.work_dir) / "auto_scale_vs_fixed_report.json"
    out_path.write_text(json.dumps({
        "factors": factors,
        "fixed_per_factor_per_video": fixed_results,
        "naive_best_factor": naive_best_factor,
        "naive_best_mean": naive_best_mean,
        "loocv_mean_held_out": mean_held_out,
        "loocv_held_out_scores": held_out_scores,
        "loocv_chosen_factor_per_video": chosen_factor_per_video,
        "auto_scale_calibration_per_video": auto_results,
        "auto_scale_calibration_mean": auto_mean,
    }, indent=2))
    print(f"\nWrote report: {out_path}")


if __name__ == "__main__":
    main()
