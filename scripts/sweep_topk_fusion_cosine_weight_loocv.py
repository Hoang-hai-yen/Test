"""Sweep stage123_geco2.dynamic_prototype.topk_fusion.cosine_weight AND
estimate its generalization honestly -- same rationale as
scripts/sweep_zscore_loocv.py (stage3.adaptive_z_score), adapted for
topk_fusion's fused_score_i = cosine_weight*cosine_z_i + (1-cosine_weight)*
geco2_z_i (see Geco2DynamicPrototypeTopKFusionConfig in aero_eyes/config.py).

Why NOT plain leave-one-video-out here: the training set is 14 videos, but
only 7 DISTINCT physical objects -- '_0'/'_1' pairs (e.g. Backpack_0/
Backpack_1) are two takes of the SAME object (see
aero_eyes.models.geco2_finetune_data.video_category/EXPECTED_TRAIN_VIDEOS,
the same grouping already used to keep GeCo2 finetuning's own train/val
split leakage-free). Leave-ONE-VIDEO-out would let Backpack_1 sit in the
"other videos" fold used to pick cosine_weight for held-out Backpack_0 --
optimistic, since they share the same object appearance/domain gap. This
script instead does leave-one-OBJECT-out: each fold holds out BOTH videos
of one object category at once.

topk_fusion only has an effect on the cosine_rescore path
(run_stage12_geco2_candidates -> stage34 -> stage5, i.e.
stage123_geco2.cosine_rescore.enabled=true) -- see
Geco2DynamicPrototypeTopKFusionConfig's own docstring for why. This script
forces cosine_rescore.enabled, dynamic_prototype.enabled,
dynamic_prototype.topk_fusion.enabled, and cross_check_source=
"feature_extractor" all to true (each with a warning if the config on disk
had them off), since sweeping cosine_weight is a no-op otherwise.

Usage:
    python -m scripts.sweep_topk_fusion_cosine_weight_loocv \
        --config configs/config.yaml \
        --set data.data_root=... --set data.gt.global_file=... \
        --set project.work_dir=/kaggle/working/runs/exp001 \
        --cosine-weights 0.0,0.2,0.3,0.4,0.5,0.6,0.7,0.8,1.0
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)


def _clear_topk_sweep_cache(work_dir: Path, sample_ids: list[str], submission_name: str) -> None:
    """Everything downstream of (and including) candidates.json depends on
    cosine_weight through the online dynamic_prototype loop -- all must be
    regenerated every sweep point, unlike geco2_prototype.pt/prototype.npz
    (the static reference-image prototypes upstream of that loop, unaffected
    by cosine_weight and fine to leave cached)."""
    for sid in sample_ids:
        sdir = work_dir / sid
        for name in ("candidates.json", "candidates.feats.npz", "detections.json", "tracks.json", submission_name):
            f = sdir / name
            if f.exists():
                f.unlink()


def run_sweep(cfg, sample_ids: list[str], cosine_weights: list[float]) -> dict[float, dict[str, float]]:
    """Returns {cosine_weight: {video_id: st_iou}}."""
    from aero_eyes.stages.stage123_geco2 import run_stage12_geco2_candidates
    from aero_eyes.stages.stage34 import run_stage34
    from aero_eyes.stages.stage5 import run_stage5
    from aero_eyes.evaluate import evaluate_dataset

    if cfg.pipeline.detector != "geco2":
        raise ValueError(
            "pipeline.detector must be 'geco2' -- dynamic_prototype.topk_fusion only "
            "exists in the GeCo2 stage (aero_eyes/stages/stage123_geco2.py)."
        )
    dp_cfg = cfg.stage123_geco2.dynamic_prototype
    if not cfg.stage123_geco2.cosine_rescore.enabled:
        log.warning(
            "stage123_geco2.cosine_rescore.enabled was false -- forcing true. "
            "topk_fusion is only wired into run_stage12_geco2_candidates (the "
            "cosine_rescore path) -- see its own docstring for why."
        )
        cfg.stage123_geco2.cosine_rescore.enabled = True
    if not dp_cfg.enabled:
        log.warning("stage123_geco2.dynamic_prototype.enabled was false -- forcing true.")
        dp_cfg.enabled = True
    if not dp_cfg.topk_fusion.enabled:
        log.warning(
            "dynamic_prototype.topk_fusion.enabled was false -- forcing true. This "
            "script sweeps topk_fusion.cosine_weight, which has no effect otherwise."
        )
        dp_cfg.topk_fusion.enabled = True
    if dp_cfg.cross_check_source != "feature_extractor":
        log.warning(
            "dynamic_prototype.cross_check_source was %r -- forcing 'feature_extractor' "
            "(topk_fusion silently falls back to plain boxes[0] selection for 'hiera').",
            dp_cfg.cross_check_source,
        )
        dp_cfg.cross_check_source = "feature_extractor"

    work_dir = Path(cfg.project.work_dir)
    results: dict[float, dict[str, float]] = {}

    for w in cosine_weights:
        dp_cfg.topk_fusion.cosine_weight = w
        _clear_topk_sweep_cache(work_dir, sample_ids, cfg.data.submission.path_name)

        for sid in sample_ids:
            run_stage12_geco2_candidates(cfg, sid)
            run_stage34(cfg, sid)
            run_stage5(cfg, sid)

        # Gather submissions into one file for evaluate_dataset
        all_preds = []
        for sid in sample_ids:
            sub_path = work_dir / sid / cfg.data.submission.path_name
            if sub_path.exists():
                all_preds.extend(json.loads(sub_path.read_text()))
        combined_path = work_dir / "all_submissions_topk_sweep.json"
        combined_path.write_text(json.dumps(all_preds))

        report = evaluate_dataset(combined_path, cfg.data.gt.global_file, cfg=cfg)
        results[w] = report["per_video"]
        log.info("cosine_weight=%.2f -> mean=%.4f  per-video=%s", w, report["mean_st_iou"], report["per_video"])
        print(f"cosine_weight={w:.2f}  mean ST-IoU={report['mean_st_iou']:.4f}  {report['per_video']}")

    return results


def loocv_estimate_by_object(
    results: dict[float, dict[str, float]], sample_ids: list[str],
) -> tuple[float, dict[str, float], dict[str, float]]:
    """Leave-one-OBJECT-out (not video-out -- see module docstring): for
    each held-out object category, pick the best cosine_weight using only
    videos from the OTHER 6 categories, then score that weight on BOTH of
    the held-out category's videos. Averaging the held-out scores across
    all 7 folds gives an unbiased estimate of what the naive "best on all
    14 videos" pick actually buys you on an unseen object.
    Returns (mean_held_out_score, per_video_held_out_score,
    per_video_chosen_weight, per_category_chosen_weight) -- the last one is
    the same information as per_video_chosen_weight but deduplicated to one
    entry per object (both '_0'/'_1' videos of a held-out category always
    get the SAME chosen weight, since they're held out together), meant for
    a stability check across the 7 independent fold picks.
    """
    from aero_eyes.models.geco2_finetune_data import video_category

    categories = sorted({video_category(s) for s in sample_ids})
    held_out_scores: dict[str, float] = {}
    chosen_weight_per_video: dict[str, float] = {}
    chosen_weight_per_category: dict[str, float] = {}

    for held_out_cat in categories:
        held_out_videos = [s for s in sample_ids if video_category(s) == held_out_cat]
        others = [s for s in sample_ids if video_category(s) != held_out_cat]
        best_w, best_mean = None, -1.0
        for w, per_video in results.items():
            mean_on_others = sum(per_video[s] for s in others) / len(others)
            if mean_on_others > best_mean:
                best_mean = mean_on_others
                best_w = w
        chosen_weight_per_category[held_out_cat] = best_w
        for sid in held_out_videos:
            held_out_scores[sid] = results[best_w][sid]
            chosen_weight_per_video[sid] = best_w

    mean_held_out = sum(held_out_scores.values()) / len(held_out_scores)
    return mean_held_out, held_out_scores, chosen_weight_per_video, chosen_weight_per_category


def main():
    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser(
        description="Sweep dynamic_prototype.topk_fusion.cosine_weight + leave-one-object-out honest estimate"
    )
    p.add_argument("--config", required=True)
    p.add_argument("--cosine-weights", default="0.0,0.2,0.3,0.4,0.5,0.6,0.7,0.8,1.0")
    p.add_argument("--set", action="append", default=[])
    p.add_argument(
        "--out", default=None,
        help="path to write JSON report (default: <work_dir>/topk_fusion_cosine_weight_loocv_report.json)",
    )
    p.add_argument(
        "--allow-any-videos", action="store_true",
        help="skip the exact-14-training-videos check (validate_training_video_ids) -- use "
        "whatever sample directories are under data.data_root as-is. Without this flag, a "
        "data_root that doesn't contain exactly the 14 known training videos hard-fails, the "
        "same guard aero_eyes.models.geco2_finetune_data uses to prevent accidentally sweeping "
        "against the 6-video TEST set instead.",
    )
    args = p.parse_args()

    from aero_eyes.config import load_config
    from aero_eyes.models.geco2_finetune_data import validate_training_video_ids, video_category

    cfg = load_config(args.config, args.set)

    data_root = Path(cfg.data.data_root)
    sample_ids = sorted(d.name for d in data_root.iterdir() if d.is_dir() and not d.name.startswith("."))
    if not args.allow_any_videos:
        validate_training_video_ids(sample_ids)

    cosine_weights = [float(w) for w in args.cosine_weights.split(",")]
    categories = sorted({video_category(s) for s in sample_ids})

    print(f"Samples ({len(sample_ids)}): {sample_ids}")
    print(f"Objects ({len(categories)}): {categories}")
    print(f"Sweeping cosine_weight in {cosine_weights}\n")

    results = run_sweep(cfg, sample_ids, cosine_weights)

    # Naive pick: best cosine_weight on ALL videos (optimistic -- tuned on the test set)
    naive_best_w, naive_best_mean = None, -1.0
    for w, per_video in results.items():
        m = sum(per_video.values()) / len(per_video)
        if m > naive_best_mean:
            naive_best_mean = m
            naive_best_w = w

    mean_held_out, held_out_scores, chosen_w_per_video, chosen_w_per_category = loocv_estimate_by_object(
        results, sample_ids,
    )

    print("\n" + "=" * 60)
    print(f"NAIVE (tuned on all {len(sample_ids)} videos): best cosine_weight={naive_best_w} -> Mean ST-IoU={naive_best_mean:.4f}")
    print(f"LEAVE-ONE-OBJECT-OUT HONEST ESTIMATE: Mean ST-IoU={mean_held_out:.4f}")
    print("Per-video held-out score (weight chosen from the OTHER 6 objects' videos):")
    for sid in sample_ids:
        print(f"  {sid} ({video_category(sid)}): cosine_weight={chosen_w_per_video[sid]}  ST-IoU={held_out_scores[sid]:.4f}")

    # Stability check: does the naive all-data pick actually agree with what
    # INDEPENDENT folds would have picked on their own, or did it just fit
    # noise in these particular 14 videos? Exact-value comparison is valid
    # here (not a floating-point tolerance issue) -- every chosen weight
    # comes from the same discrete cosine_weights list swept above.
    from statistics import median
    fold_weights = sorted(chosen_w_per_category.values())
    agreement = sum(1 for w in fold_weights if w == naive_best_w)
    median_fold_weight = median(fold_weights)
    agreement_fraction = agreement / len(categories)

    print(f"\nPer-object fold picks: {chosen_w_per_category}")
    print(
        f"{agreement}/{len(categories)} object-folds independently picked the SAME weight "
        f"as the naive all-data pick ({naive_best_w})."
    )

    STABLE_AGREEMENT_FRACTION = 0.5
    if agreement_fraction >= STABLE_AGREEMENT_FRACTION:
        recommended_w = naive_best_w
        print(
            f"\n>>> RECOMMENDED cosine_weight = {recommended_w} (agrees with "
            f"{agreement}/{len(categories)} object-folds; expected held-out "
            f"ST-IoU ~= {mean_held_out:.4f})."
        )
    else:
        recommended_w = median_fold_weight
        print(
            f"\n>>> WARNING: naive best cosine_weight={naive_best_w} agreed with only "
            f"{agreement}/{len(categories)} object-folds ({fold_weights}) -- likely fit to "
            f"noise in these 14 videos rather than a genuinely better weight."
        )
        print(
            f">>> RECOMMENDED cosine_weight = {recommended_w} (median of the 7 independent "
            f"per-object-fold picks, more robust than the naive all-data pick given the "
            f"disagreement above -- NOTE this exact value was not itself scored by the LOOCV "
            f"loop above, only each fold's own pick was; re-run with it as the sole value in "
            f"--cosine-weights to confirm before deploying)."
        )
    print("=" * 60)

    out_path = Path(args.out) if args.out else Path(cfg.project.work_dir) / "topk_fusion_cosine_weight_loocv_report.json"
    out_path.write_text(json.dumps({
        "cosine_weights": cosine_weights,
        "per_weight_per_video": results,
        "naive_best_cosine_weight": naive_best_w,
        "naive_best_mean": naive_best_mean,
        "loocv_mean_held_out": mean_held_out,
        "loocv_held_out_scores": held_out_scores,
        "loocv_chosen_cosine_weight_per_video": chosen_w_per_video,
        "loocv_chosen_cosine_weight_per_category": chosen_w_per_category,
        "fold_agreement_count": agreement,
        "fold_agreement_fraction": agreement_fraction,
        "median_fold_cosine_weight": median_fold_weight,
        "recommended_cosine_weight": recommended_w,
    }, indent=2))
    print(f"\nWrote report: {out_path}")


if __name__ == "__main__":
    main()
