"""Runner + diagnostic: sweep stage4.kalman_motion_check.max_dist_ratio
across several values and report each one's ST-IoU / MATCH-LOOSE-
MISSING_PRED-MISSING_GT breakdown, to pick the next value to try instead of
hand-running the full pipeline once per candidate value and re-typing the
comparison yourself. Mirrors scripts/sweep_verify_interval.py -- see that
script's docstring for the general approach; this one sweeps a different
stage4 knob.

UNLIKE every other script under scripts/ (except sweep_verify_interval.py),
this one is NOT read-only -- it actually RE-RUNS Stage 4
(aero_eyes.stages.stage4.run_stage4) once per swept value, since
max_dist_ratio changes tracking behavior itself, not just how an
already-produced artifact gets analyzed. Stage 3's own detections.json is
UNAFFECTED by this knob, so it's reused as-is for every value -- Stage 3
itself is never re-run.

Every swept run forces stage4.kalman_motion_check.enabled=true regardless
of what the base config/--set says -- sweeping max_dist_ratio is pointless
with the check disabled, and the whole point of this script is testing it
enabled. Pass --set stage4.kalman_motion_check.enabled=false explicitly and
it will simply be overridden back to true for every swept run.

Reuses sweep_verify_interval.py's isolated-scratch-work_dir machinery
(_prepare_scratch_sample_dir) so your real tracks.json/submission.json are
never touched -- see that script's docstring for exactly what gets copied
where.

For each value, reports classify_frames()'s own ST-IoU + MATCH/LOOSE/
MISSING_PRED/MISSING_GT breakdown (imported from check_st_iou_breakdown.py,
not reimplemented) against tracks.json (raw Stage 4). Stage 5 is NOT run
by default -- pass --include-stage5 to also run+report Stage 5's
submission.json for every value (extra cost: doubles the work per value).

Ends with a ranked summary (best ST-IoU first) and flags whether the best
value found sits at either end of --values, so you know whether to widen
the sweep range next rather than assuming the tested range already
contains the true optimum.

Usage:
    python -m scripts.sweep_kalman_max_dist_ratio --config configs/config.yaml \\
        --sample LifeJacket_1 --values 1.5,2,3,5,8
    python -m scripts.sweep_kalman_max_dist_ratio --config configs/config.yaml \\
        --sample LifeJacket_1 --values 1,1.5,2 --include-stage5
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from aero_eyes.utils.io import load_gt, read_tracks
from scripts.check_st_iou_breakdown import _print_breakdown, classify_frames
from scripts.sweep_verify_interval import _prepare_scratch_sample_dir

log = logging.getLogger(__name__)


def sweep_sample(
    config_path: str, base_overrides: list[str], sample_id: str,
    values: list[float], iou_threshold: float, include_stage5: bool,
) -> None:
    from aero_eyes.config import load_config
    from aero_eyes.stages.stage4 import run_stage4

    cfg = load_config(config_path, base_overrides)
    real_work_dir = Path(cfg.project.work_dir) / sample_id

    print(f"\n=== {sample_id} (IoU threshold={iou_threshold}) -- "
          f"sweeping stage4.kalman_motion_check.max_dist_ratio={values} ===")

    if not (real_work_dir / "detections.json").exists():
        print(f"  detections.json not found at {real_work_dir} -- run Stage 3 first.")
        return

    gt_file = cfg.data.gt.global_file
    try:
        gt = load_gt(gt_file, sample_id)
    except KeyError:
        print(f"  not found in {gt_file}, skipping.")
        return

    sweep_root = Path(cfg.project.work_dir) / "_kalman_max_dist_ratio_sweep"
    results: list[tuple[float, dict]] = []

    for value in values:
        scratch_work_dir = sweep_root / f"mdr_{value}"
        scratch_sample_dir = scratch_work_dir / sample_id
        _prepare_scratch_sample_dir(real_work_dir, scratch_sample_dir)

        # Fresh, fully-validated config per value -- the SAME override
        # mechanism the CLI itself uses, so this can never silently diverge
        # from what --set stage4.kalman_motion_check.max_dist_ratio=<value>
        # would really do. kalman_motion_check.enabled is forced true here
        # (after base_overrides, so it always wins) -- see module docstring.
        run_cfg = load_config(
            config_path,
            [*base_overrides, f"project.work_dir={scratch_work_dir}",
             "project.use_cache=false",
             "stage4.kalman_motion_check.enabled=true",
             f"stage4.kalman_motion_check.max_dist_ratio={value}"],
        )
        run_stage4(run_cfg, sample_id)

        raw_tracks = read_tracks(scratch_sample_dir / "tracks.json")
        pred_tube = {fi: b for fi, b in raw_tracks.items() if b is not None}
        r = classify_frames(pred_tube, gt, iou_threshold)
        results.append((value, r))
        _print_breakdown(f"max_dist_ratio={value}", r)

        if include_stage5:
            from aero_eyes.evaluate import _load_submission
            from aero_eyes.stages.stage5 import run_stage5

            run_stage5(run_cfg, sample_id)
            all_sub = _load_submission(scratch_sample_dir / cfg.data.submission.path_name)
            r5 = classify_frames(all_sub.get(sample_id, {}), gt, iou_threshold)
            _print_breakdown(f"max_dist_ratio={value} (after Stage 5)", r5)

    ranked = sorted(results, key=lambda vr: vr[1]["st_iou"], reverse=True)
    print("  Ranked by ST-IoU (best first): "
          + ", ".join(f"{v}={r['st_iou']:.4f}" for v, r in ranked))
    best_value = ranked[0][0]
    print(f"  -> best so far: max_dist_ratio={best_value}")
    if best_value == max(values):
        print(f"    note: {best_value} is the LARGEST value tried and still winning -- ST-IoU "
              "may keep improving past it (more lenient, fewer false alarms on real fast "
              "motion); consider sweeping larger values next.")
    elif best_value == min(values):
        print(f"    note: {best_value} is the SMALLEST value tried and still winning -- consider "
              "sweeping smaller values next (stricter, catches smaller jumps).")


def main():
    logging.basicConfig(level=logging.WARNING)
    p = argparse.ArgumentParser(
        description="Sweep stage4.kalman_motion_check.max_dist_ratio across several values and "
        "report each one's ST-IoU / coverage breakdown -- actually re-runs Stage 4 per value "
        "(see module docstring)."
    )
    p.add_argument("--config", required=True)
    p.add_argument("--sample", default=None, help="omit to sweep every sample in data_root")
    p.add_argument("--set", action="append", default=[],
                    help="base config overrides applied to every swept run "
                    "(in addition to the ones this script adds itself)")
    p.add_argument("--values", default="1.5,2,3,5,8",
                    help="comma-separated stage4.kalman_motion_check.max_dist_ratio values to "
                    "try (default: 1.5,2,3,5,8)")
    p.add_argument("--iou-threshold", type=float, default=0.5,
                    help="IoU >= this counts as MATCH rather than LOOSE (default: 0.5)")
    p.add_argument("--include-stage5", action="store_true",
                    help="Also run Stage 5 and report submission.json for every value (extra cost).")
    args = p.parse_args()

    values = [float(v) for v in args.values.split(",")]

    from aero_eyes.config import load_config
    cfg = load_config(args.config, args.set)

    if args.sample:
        sample_ids = [args.sample]
    else:
        data_root = Path(cfg.data.data_root)
        sample_ids = [d.name for d in sorted(data_root.iterdir()) if d.is_dir()]

    for sid in sample_ids:
        sweep_sample(args.config, args.set, sid, values, args.iou_threshold, args.include_stage5)


if __name__ == "__main__":
    main()
