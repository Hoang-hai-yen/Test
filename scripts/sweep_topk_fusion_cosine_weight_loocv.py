"""Sweep stage123_geco2.dynamic_prototype.topk_fusion.cosine_weight AND
estimate its generalization honestly -- same rationale as
scripts/sweep_zscore_loocv.py (stage3.adaptive_z_score), adapted for
topk_fusion's fused_score_i = cosine_weight*cosine_z_i + (1-cosine_weight)*
geco2_z_i (see Geco2DynamicPrototypeTopKFusionConfig in aero_eyes/config.py).

Stops at candidates.json (Precision/Recall/F1), does NOT run Stage 3+:
topk_fusion only affects WHICH exemplar tokens get appended to the online
dynamic prototype while candidates.json is being built -- its effect is
fully visible there already. Scoring all the way through Stage 3's cosine
matching + Stage 4 tracking + Stage 5 (the original version of this script,
via ST-IoU on submission.json) would work too in principle, but
stage3.adaptive_threshold is CURRENTLY UNSTABLE/experimental (see
Stage3Config.adaptive_threshold_method's own docstring -- a fixed z-score
multiplier misbehaves in opposite directions depending on a video's own
FP/TP mixture, and the otsu/gmm alternatives are unvalidated) -- running
through it here would confound topk_fusion's OWN effect on candidate
quality with whatever noise Stage 3's own threshold selection is
contributing that week. Revisit an end-to-end ST-IoU sweep (like this
script's own git history) once Stage 3's threshold is stabilized.

Reports P/R/F1 at MULTIPLE IoU thresholds (default 0.1 and 0.5), not just
one: GeCo2's own box regression can find roughly the right region but with
a loose/imprecise box -- a cosine_weight change might genuinely hurt F1@0.5
(strict localization) while F1@0.1 (loose "did we find the right area at
all") stays flat, which tells you box TIGHTNESS regressed (something a
downstream box-refine step could plausibly fix) rather than the candidate
actually being lost/wrong (which box-refine can't fix -- there's nothing
there to refine). A drop that shows up at BOTH thresholds together is the
more concerning case. --select-iou-threshold picks which of the reported
thresholds actually drives cosine_weight selection/recommendation below
(default 0.5, the stricter one) -- the other threshold(s) are reported for
this diagnosis only, not used to pick anything.

--save-candidates-dir (opt-in): archives EVERY sweep point's candidates.json
(+ .feats.npz) per video, under <dir>/cosine_weight_<w>/<video_id>/, before
the next weight's cache-clear overwrites them -- lets you go back and
visually inspect box tightness/placement for a specific (weight, video,
frame) after the sweep finishes, instead of only ever seeing the aggregate
P/R/F1 numbers. Off by default (candidates.feats.npz has one DINOv2 vector
per candidate box, per keyframe, per weight -- adds up fast over a wide
sweep).

compute_prf1()/read_candidates() are reused (not reimplemented) from
scripts/check_cosine_effect.py / aero_eyes/utils/io.py -- same TP/FN/FP/TN
semantics and the same survivorship-bias fix already established there
(processed_frames = candidates.json's OWN key set, which has an entry for
EVERY keyframe GeCo2 sampled, even an empty one -- see
_run_geco2_candidate_pass in aero_eyes/stages/stage123_geco2.py; naively
using dets_by_frame.keys() instead would silently drop frames where
topk_fusion's choices happened to leave zero surviving candidates, biasing
recall upward for exactly the runs that are actually doing worse).

Why leave-one-OBJECT-out (not leave-one-video-out): the training set is 14
videos but only 7 DISTINCT physical objects -- '_0'/'_1' pairs (e.g.
Backpack_0/Backpack_1) are two takes of the SAME object (see
aero_eyes.models.geco2_finetune_data.video_category/EXPECTED_TRAIN_VIDEOS,
the same grouping GeCo2 finetuning's own train/val split uses to stay
leakage-free). Leave-one-video-out would let Backpack_1 sit in the "other
videos" fold used to pick cosine_weight for held-out Backpack_0 --
optimistic, since they share the same object appearance/domain gap. This
script instead holds out BOTH videos of one object category per fold.

topk_fusion only has an effect on the cosine_rescore path
(run_stage12_geco2_candidates, i.e. stage123_geco2.cosine_rescore.
enabled=true) -- see Geco2DynamicPrototypeTopKFusionConfig's own docstring
for why. This script forces cosine_rescore.enabled, dynamic_prototype.
enabled, dynamic_prototype.topk_fusion.enabled, and cross_check_source=
"feature_extractor" all to true (each with a warning if the config on disk
had them off), since sweeping cosine_weight is a no-op otherwise.

Usage:
    python -m scripts.sweep_topk_fusion_cosine_weight_loocv \
        --config configs/config.yaml \
        --set data.data_root=... --set data.gt.global_file=... \
        --set project.work_dir=/kaggle/working/runs/exp001 \
        --cosine-weights 0.0,0.2,0.3,0.4,0.5,0.6,0.7,0.8,1.0 \
        --iou-thresholds 0.1,0.5 --select-iou-threshold 0.5 \
        --save-candidates-dir /kaggle/working/runs/exp001/topk_sweep_candidates
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

log = logging.getLogger(__name__)


def _iou_key(iou: float) -> str:
    """Stable string key for an IoU threshold, used both as a dict key
    (JSON needs string keys) and for archive directory names."""
    return f"{iou:.2f}"


def _clear_candidates_cache(work_dir: Path, sample_ids: list[str]) -> None:
    """candidates.json (+ .feats.npz) depends on cosine_weight through the
    online dynamic_prototype loop -- must be regenerated every sweep
    point, unlike geco2_prototype.pt/prototype.npz (the static reference-
    image prototypes upstream of that loop, unaffected by cosine_weight
    and fine to leave cached)."""
    for sid in sample_ids:
        sdir = work_dir / sid
        for name in ("candidates.json", "candidates.feats.npz"):
            f = sdir / name
            if f.exists():
                f.unlink()


def _archive_candidates(work_dir: Path, sample_ids: list[str], cosine_weight: float, save_dir: Path) -> None:
    """Copies this sweep point's candidates.json (+ .feats.npz) for every
    sample into <save_dir>/cosine_weight_<w>/<sample_id>/ before the next
    sweep point's _clear_candidates_cache overwrites them."""
    dest_root = save_dir / f"cosine_weight_{cosine_weight:.2f}"
    for sid in sample_ids:
        src_json = work_dir / sid / "candidates.json"
        if not src_json.exists():
            continue
        dest_dir = dest_root / sid
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_json, dest_dir / "candidates.json")
        src_feats = work_dir / sid / "candidates.feats.npz"
        if src_feats.exists():
            shutil.copy2(src_feats, dest_dir / "candidates.feats.npz")


def run_sweep(
    cfg, sample_ids: list[str], cosine_weights: list[float], iou_thresholds: list[float],
    save_candidates_dir: Path | None = None,
) -> dict[float, dict[str, dict[str, dict[str, float]]]]:
    """Returns {cosine_weight: {video_id: {iou_key: {"recall":.., "precision":.., "f1":..}}}}."""
    from aero_eyes.stages.stage123_geco2 import run_stage12_geco2_candidates
    from aero_eyes.utils.io import load_gt, read_candidates
    from scripts.check_cosine_effect import compute_prf1

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
    gt_file = cfg.data.gt.global_file
    results: dict[float, dict[str, dict[str, dict[str, float]]]] = {}

    for w in cosine_weights:
        dp_cfg.topk_fusion.cosine_weight = w
        _clear_candidates_cache(work_dir, sample_ids)

        per_video: dict[str, dict[str, dict[str, float]]] = {}
        for sid in sample_ids:
            run_stage12_geco2_candidates(cfg, sid)
            cand_path = work_dir / sid / "candidates.json"
            candidates = read_candidates(cand_path)
            gt = load_gt(gt_file, sid)
            processed_frames = set(candidates.keys())
            per_video[sid] = {
                _iou_key(iou): {
                    "recall": (r := compute_prf1(candidates, gt, iou, processed_frames=processed_frames))["recall"],
                    "precision": r["precision"],
                    "f1": r["f1"],
                }
                for iou in iou_thresholds
            }
        results[w] = per_video

        if save_candidates_dir is not None:
            _archive_candidates(work_dir, sample_ids, w, save_candidates_dir)

        summary_parts = []
        for iou in iou_thresholds:
            k = _iou_key(iou)
            mean_f1 = sum(m[k]["f1"] for m in per_video.values()) / len(per_video)
            mean_r = sum(m[k]["recall"] for m in per_video.values()) / len(per_video)
            mean_p = sum(m[k]["precision"] for m in per_video.values()) / len(per_video)
            summary_parts.append(f"iou={iou}: F1={mean_f1:.4f} (R={mean_r:.4f} P={mean_p:.4f})")
        line = f"cosine_weight={w:.2f}  " + "  |  ".join(summary_parts)
        log.info(line)
        print(line)

    return results


def loocv_estimate_by_object(
    results: dict[float, dict[str, dict[str, dict[str, float]]]], sample_ids: list[str], select_iou_key: str,
) -> tuple[float, dict[str, dict[str, dict[str, float]]], dict[str, float], dict[str, float]]:
    """Leave-one-OBJECT-out (not video-out -- see module docstring), scored
    on F1 at select_iou_key: for each held-out object category, pick the
    best cosine_weight using only videos from the OTHER 6 categories' mean
    F1@select_iou_key, then score that weight on BOTH of the held-out
    category's videos. Averaging the held-out F1 across all 7 folds gives
    an unbiased estimate of what the naive "best on all 14 videos" pick
    actually buys you on an unseen object.
    Returns (mean_held_out_f1, per_video_held_out_metrics [ALL iou
    thresholds, for the diagnosis this script's docstring describes],
    per_video_chosen_weight, per_category_chosen_weight) -- the last one is
    per_video_chosen_weight deduplicated to one entry per object (both
    '_0'/'_1' videos of a held-out category always get the SAME chosen
    weight, since they're held out together), meant for a stability check
    across the 7 independent fold picks.
    """
    from aero_eyes.models.geco2_finetune_data import video_category

    categories = sorted({video_category(s) for s in sample_ids})
    held_out_metrics: dict[str, dict[str, dict[str, float]]] = {}
    chosen_weight_per_video: dict[str, float] = {}
    chosen_weight_per_category: dict[str, float] = {}

    for held_out_cat in categories:
        held_out_videos = [s for s in sample_ids if video_category(s) == held_out_cat]
        others = [s for s in sample_ids if video_category(s) != held_out_cat]
        best_w, best_mean_f1 = None, -1.0
        for w, per_video in results.items():
            mean_f1_on_others = sum(per_video[s][select_iou_key]["f1"] for s in others) / len(others)
            if mean_f1_on_others > best_mean_f1:
                best_mean_f1 = mean_f1_on_others
                best_w = w
        chosen_weight_per_category[held_out_cat] = best_w
        for sid in held_out_videos:
            held_out_metrics[sid] = results[best_w][sid]  # ALL iou thresholds, for the box-tightness diagnosis
            chosen_weight_per_video[sid] = best_w

    mean_held_out_f1 = sum(m[select_iou_key]["f1"] for m in held_out_metrics.values()) / len(held_out_metrics)
    return mean_held_out_f1, held_out_metrics, chosen_weight_per_video, chosen_weight_per_category


def main():
    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser(
        description="Sweep dynamic_prototype.topk_fusion.cosine_weight on candidates.json "
        "Precision/Recall/F1 (at multiple IoU thresholds) + leave-one-object-out honest estimate"
    )
    p.add_argument("--config", required=True)
    p.add_argument("--cosine-weights", default="0.0,0.2,0.3,0.4,0.5,0.6,0.7,0.8,1.0")
    p.add_argument(
        "--iou-thresholds", default="0.1,0.5",
        help="comma list of IoU thresholds to report P/R/F1 at (default: 0.1,0.5 -- a big gap "
        "between them for the same cosine_weight signals loose/imprecise boxes rather than "
        "genuinely missing candidates; see module docstring)",
    )
    p.add_argument(
        "--select-iou-threshold", type=float, default=0.5,
        help="which of --iou-thresholds' values actually drives cosine_weight selection/"
        "recommendation below (default: 0.5) -- must be one of --iou-thresholds' values; the "
        "others are reported for diagnosis only.",
    )
    p.add_argument("--set", action="append", default=[])
    p.add_argument(
        "--out", default=None,
        help="path to write JSON report (default: <work_dir>/topk_fusion_cosine_weight_loocv_report.json)",
    )
    p.add_argument(
        "--save-candidates-dir", default=None,
        help="opt-in: archive every sweep point's candidates.json (+ .feats.npz) per video under "
        "this directory (see module docstring) for later manual inspection. Off by default.",
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
    iou_thresholds = [float(t) for t in args.iou_thresholds.split(",")]
    if args.select_iou_threshold not in iou_thresholds:
        raise ValueError(
            f"--select-iou-threshold={args.select_iou_threshold} must be one of "
            f"--iou-thresholds={iou_thresholds}."
        )
    select_key = _iou_key(args.select_iou_threshold)
    save_candidates_dir = Path(args.save_candidates_dir) if args.save_candidates_dir else None
    categories = sorted({video_category(s) for s in sample_ids})

    print(f"Samples ({len(sample_ids)}): {sample_ids}")
    print(f"Objects ({len(categories)}): {categories}")
    print(f"Sweeping cosine_weight in {cosine_weights}")
    print(f"Reporting IoU thresholds: {iou_thresholds} (selection uses {args.select_iou_threshold})")
    if save_candidates_dir is not None:
        print(f"Archiving candidates.json per sweep point to: {save_candidates_dir}")
    print()

    results = run_sweep(cfg, sample_ids, cosine_weights, iou_thresholds, save_candidates_dir)

    # Naive pick: best cosine_weight on ALL videos' mean F1@select_iou_threshold (optimistic --
    # tuned on the test set)
    naive_best_w, naive_best_f1 = None, -1.0
    for w, per_video in results.items():
        m = sum(v[select_key]["f1"] for v in per_video.values()) / len(per_video)
        if m > naive_best_f1:
            naive_best_f1 = m
            naive_best_w = w

    mean_held_out_f1, held_out_metrics, chosen_w_per_video, chosen_w_per_category = loocv_estimate_by_object(
        results, sample_ids, select_key,
    )

    print("\n" + "=" * 60)
    print(f"Selection metric: F1@IoU={args.select_iou_threshold}")
    print(f"NAIVE (tuned on all {len(sample_ids)} videos): best cosine_weight={naive_best_w} -> Mean F1={naive_best_f1:.4f}")
    print(f"LEAVE-ONE-OBJECT-OUT HONEST ESTIMATE: Mean F1={mean_held_out_f1:.4f}")
    print("Per-video held-out metrics (weight chosen from the OTHER 6 objects' videos), all reported IoU thresholds:")
    for sid in sample_ids:
        parts = [f"iou={iou}: F1={held_out_metrics[sid][_iou_key(iou)]['f1']:.4f}" for iou in iou_thresholds]
        print(f"  {sid} ({video_category(sid)}): cosine_weight={chosen_w_per_video[sid]}  " + "  ".join(parts))

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
            f"{agreement}/{len(categories)} object-folds; expected held-out F1@"
            f"{args.select_iou_threshold} ~= {mean_held_out_f1:.4f})."
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
    if len(iou_thresholds) > 1:
        print(
            "\nTIP: compare F1 across the reported IoU thresholds above for the SAME "
            "cosine_weight -- if the LOOSE threshold's F1 stays high while the STRICT one drops, "
            "GeCo2 is finding roughly the right region but with an imprecise box (a box-refine "
            "step downstream could plausibly help); if both drop together, candidates are "
            "actually being lost/misplaced (box-refine can't fix a candidate that isn't there)."
        )
    print(
        "\nNOTE: this measures topk_fusion's effect on CANDIDATE quality (before Stage 3), "
        "not the final ST-IoU submission metric -- see module docstring for why Stage 3+ is "
        "skipped for now. Re-validate the chosen cosine_weight end-to-end once stage3."
        "adaptive_threshold is stabilized."
    )

    out_path = Path(args.out) if args.out else Path(cfg.project.work_dir) / "topk_fusion_cosine_weight_loocv_report.json"
    out_path.write_text(json.dumps({
        "iou_thresholds": iou_thresholds,
        "select_iou_threshold": args.select_iou_threshold,
        "cosine_weights": cosine_weights,
        "per_weight_per_video": results,
        "naive_best_cosine_weight": naive_best_w,
        "naive_best_f1": naive_best_f1,
        "loocv_mean_held_out_f1": mean_held_out_f1,
        "loocv_held_out_metrics": held_out_metrics,
        "loocv_chosen_cosine_weight_per_video": chosen_w_per_video,
        "loocv_chosen_cosine_weight_per_category": chosen_w_per_category,
        "fold_agreement_count": agreement,
        "fold_agreement_fraction": agreement_fraction,
        "median_fold_cosine_weight": median_fold_weight,
        "recommended_cosine_weight": recommended_w,
        "candidates_archive_dir": str(save_candidates_dir) if save_candidates_dir else None,
    }, indent=2))
    print(f"\nWrote report: {out_path}")


if __name__ == "__main__":
    main()
