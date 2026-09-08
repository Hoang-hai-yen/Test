"""Runner + diagnostic: sweep stage4.verify_interval across several values
and report each one's ST-IoU / MATCH-LOOSE-MISSING_PRED-MISSING_GT
breakdown, to pick the next value to try instead of hand-running the full
pipeline once per candidate value and re-typing the comparison yourself.

UNLIKE every other script under scripts/, this one is NOT read-only -- it
actually RE-RUNS Stage 4 (aero_eyes.stages.stage4.run_stage4) once per
swept value, since verify_interval changes tracking behavior itself, not
just how an already-produced artifact gets analyzed. Stage 3's own
detections.json is UNAFFECTED by verify_interval, so it's reused as-is
for every value -- Stage 3 itself is never re-run.

To avoid touching your real tracks.json/submission.json, each swept value
runs in an ISOLATED SCRATCH work_dir: every intermediate artifact under
work_dir/<sample_id>/ Stage 4 might need to read (detections.json,
detections_prerefine.json, prototype.npz, geco2_prototype.pt,
color_signature.npz, ...) is copied ONCE into
<work_dir>/_verify_interval_sweep/vi_<value>/<sample_id>/ (tracks.json/
submission.json/viz are deliberately NOT copied -- each value writes its
own fresh copy there), then Stage 4 runs with project.use_cache=false and
stage4.verify_interval=<value> against that copy. Your real
work_dir/<sample_id>/tracks.json is NEVER touched or read from. Each
value's scratch copy is left on disk afterward so you can point
check_tracker_coverage.py / check_st_iou_breakdown.py at a specific one
directly if a number here looks surprising.

For each value, reports classify_frames()'s own ST-IoU + MATCH/LOOSE/
MISSING_PRED/MISSING_GT breakdown (imported from check_st_iou_breakdown.py,
not reimplemented) against tracks.json (raw Stage 4). Stage 5 is NOT run
by default -- verify_interval's own effect on TRACKING is what's being
isolated here, and Stage 5's gap-fill/smoothing is a separate, orthogonal
concern; pass --include-stage5 to also run+report Stage 5's submission.json
for every value (extra cost: doubles the work per value).

Ends with a ranked summary (best ST-IoU first) and flags whether the best
value found sits at either end of --values, so you know whether to widen
the sweep range next rather than assuming the tested range already
contains the true optimum.

Usage:
    python -m scripts.sweep_verify_interval --config configs/config.yaml --sample LifeJacket_1 \\
        --values 0,5,10,15,20
    python -m scripts.sweep_verify_interval --config configs/config.yaml --sample LifeJacket_1 \\
        --values 20,25,30,40 --include-stage5
"""
from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

from aero_eyes.utils.io import load_gt, read_tracks
from scripts.check_st_iou_breakdown import _print_breakdown, classify_frames

log = logging.getLogger(__name__)

_SKIP_COPY = {"tracks.json", "submission.json", "viz"}


def _prepare_scratch_sample_dir(real_sample_dir: Path, scratch_sample_dir: Path) -> None:
    """Copy every intermediate artifact Stage 4 might need to read into an
    isolated scratch directory, so Stage 4 can run there without touching
    the real tracks.json/submission.json. Skips tracks.json/submission.json/
    viz themselves -- each swept run writes its own fresh copy."""
    if scratch_sample_dir.exists():
        shutil.rmtree(scratch_sample_dir)
    scratch_sample_dir.mkdir(parents=True, exist_ok=True)
    for item in real_sample_dir.iterdir():
        if item.name in _SKIP_COPY:
            continue
        if item.is_dir():
            shutil.copytree(item, scratch_sample_dir / item.name)
        else:
            shutil.copy2(item, scratch_sample_dir / item.name)


def sweep_sample(
    config_path: str, base_overrides: list[str], sample_id: str,
    values: list[int], iou_threshold: float, include_stage5: bool,
) -> None:
    from aero_eyes.config import load_config
    from aero_eyes.stages.stage4 import run_stage4

    cfg = load_config(config_path, base_overrides)
    real_work_dir = Path(cfg.project.work_dir) / sample_id

    print(f"\n=== {sample_id} (IoU threshold={iou_threshold}) -- "
          f"sweeping stage4.verify_interval={values} ===")

    if not (real_work_dir / "detections.json").exists():
        print(f"  detections.json not found at {real_work_dir} -- run Stage 3 first.")
        return

    gt_file = cfg.data.gt.global_file
    try:
        gt = load_gt(gt_file, sample_id)
    except KeyError:
        print(f"  not found in {gt_file}, skipping.")
        return

    sweep_root = Path(cfg.project.work_dir) / "_verify_interval_sweep"
    results: list[tuple[int, dict]] = []

    for value in values:
        scratch_work_dir = sweep_root / f"vi_{value}"
        scratch_sample_dir = scratch_work_dir / sample_id
        _prepare_scratch_sample_dir(real_work_dir, scratch_sample_dir)

        # Fresh, fully-validated config per value -- the SAME override
        # mechanism the CLI itself uses, so this can never silently diverge
        # from what --set stage4.verify_interval=<value> would really do.
        run_cfg = load_config(
            config_path,
            [*base_overrides, f"project.work_dir={scratch_work_dir}",
             f"stage4.verify_interval={value}", "project.use_cache=false"],
        )
        run_stage4(run_cfg, sample_id)

        raw_tracks = read_tracks(scratch_sample_dir / "tracks.json")
        pred_tube = {fi: b for fi, b in raw_tracks.items() if b is not None}
        r = classify_frames(pred_tube, gt, iou_threshold)
        results.append((value, r))
        _print_breakdown(f"verify_interval={value}", r)

        if include_stage5:
            from aero_eyes.evaluate import _load_submission
            from aero_eyes.stages.stage5 import run_stage5

            run_stage5(run_cfg, sample_id)
            all_sub = _load_submission(scratch_sample_dir / cfg.data.submission.path_name)
            r5 = classify_frames(all_sub.get(sample_id, {}), gt, iou_threshold)
            _print_breakdown(f"verify_interval={value} (after Stage 5)", r5)

    ranked = sorted(results, key=lambda vr: vr[1]["st_iou"], reverse=True)
    print("  Ranked by ST-IoU (best first): "
          + ", ".join(f"{v}={r['st_iou']:.4f}" for v, r in ranked))
    best_value = ranked[0][0]
    print(f"  -> best so far: verify_interval={best_value}")
    if best_value == max(values):
        print(f"    note: {best_value} is the LARGEST value tried and still winning -- ST-IoU "
              "may keep improving past it; consider sweeping larger values next.")
    elif best_value == min(values):
        print(f"    note: {best_value} is the SMALLEST value tried and still winning -- consider "
              "sweeping smaller (or 0) values next.")


def main():
    logging.basicConfig(level=logging.WARNING)
    p = argparse.ArgumentParser(
        description="Sweep stage4.verify_interval across several values and report each one's "
        "ST-IoU / coverage breakdown -- actually re-runs Stage 4 per value (see module docstring)."
    )
    p.add_argument("--config", required=True)
    p.add_argument("--sample", default=None, help="omit to sweep every sample in data_root")
    p.add_argument("--set", action="append", default=[],
                    help="base config overrides applied to every swept run "
                    "(in addition to the ones this script adds itself)")
    p.add_argument("--values", default="0,5,10,15,20",
                    help="comma-separated stage4.verify_interval values to try (default: 0,5,10,15,20)")
    p.add_argument("--iou-threshold", type=float, default=0.5,
                    help="IoU >= this counts as MATCH rather than LOOSE (default: 0.5)")
    p.add_argument("--include-stage5", action="store_true",
                    help="Also run Stage 5 and report submission.json for every value (extra cost).")
    args = p.parse_args()

    values = [int(v) for v in args.values.split(",")]

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
