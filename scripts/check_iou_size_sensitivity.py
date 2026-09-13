"""Diagnostic: is a low ST-IoU explained by genuinely bad localization, or by
the GT object simply being too SMALL for IoU to ever read high even with a
near-perfect box?

IoU is extremely sensitive to absolute pixel error on a small object: a 2px
offset on a 15x12px box (Helmet_0's own GT mean size) already destroys ~20%
of each side, while the same 2px offset on a 150x120px box is noise. So
"mean IoU=0.65 on MATCH frames" can mean either "the detector/tracker is
genuinely loose" or "the GT box is tiny and 0.65 is close to the best any
pixel-accurate box could achieve" -- this script tells the two apart.

For every frame where BOTH a prediction and GT exist, computes:
  gt_size   = sqrt(gt_w * gt_h)                          -- geometric-mean side, px
  offset    = mean(|dx1|, |dy1|, |dx2|, |dy2|)            -- per-frame corner error, px
  iou       = box_iou(pred, gt)

Buckets frames by gt_size (log-ish edges, override with --bucket-edges) and
reports, per bucket: how many frames, actual mean IoU, mean pixel offset,
and a CEILING IoU -- the IoU two equal, gt_size-sided boxes would have if
offset by that bucket's own mean pixel error along ONE axis only:
    ceiling = (s - o) / (s + o)   (s = bucket mean gt_size, o = bucket mean offset)
This is a deliberate SIMPLIFICATION (real boxes differ in both x/y and in
size, not just a single-axis shift of two identical squares) -- treat it as
an order-of-magnitude reference, not an exact bound. If actual mean IoU is
already close to the ceiling, the detector is basically pixel-accurate and
the low IoU is a size artifact, not a quality problem; a large gap between
actual and ceiling means there IS real room to tighten localization.

Also reports the Pearson correlation between log(gt_size) and per-frame IoU
(pooled across every matched frame) -- a strong positive correlation is
itself evidence that size, not detector quality, is driving the IoU spread.
And separately, the coverage rate (fraction of GT frames that got ANY
prediction) per bucket, since small objects can also be missed more often
by tracking -- a size effect on RECALL, distinct from the IoU-on-hit effect
above.

BIAS vs NOISE decomposition (the part that actually tells you whether
"1-2px average offset" is worth chasing): "mean offset" above is an
UNSIGNED average (|error|), which cannot tell a detector that is
consistently ~3px too wide on every frame (a fixable, systematic
calibration bug) apart from one whose error is genuinely random +/-3px
noise from frame to frame (much harder to reduce -- may already be near
the detector/annotation's intrinsic precision limit). This section computes
the SIGNED mean error per edge (dx1, dy1, dx2, dy2 = pred - gt) -- the
BIAS -- and its standard deviation -- the NOISE -- per bucket, plus:
  mean delta-width / delta-height  -- signed size bias (>0 = predicted box
    systematically wider/taller than GT, not just offset in position)
  ceiling if bias were fully corrected -- same ceiling formula, but using
    the NOISE magnitude instead of the raw (bias-inflated) mean |offset|,
    i.e. "what's achievable by fixing the systematic bug alone, leaving
    the random part untouched"
  verdict -- BIAS-dominated buckets are cheap, high-value fixes (recalibrate
    a fixed pixel/margin bug in the detector or box_refine step); NOISE-
    dominated buckets mean you're closer to the model/annotation's real
    precision limit and further gains need better localization quality,
    not calibration.

Usage:
    python -m scripts.check_iou_size_sensitivity --config configs/config.yaml --sample Helmet_0
    python -m scripts.check_iou_size_sensitivity --config configs/config.yaml   # pool all samples
    python -m scripts.check_iou_size_sensitivity --config configs/config.yaml --bucket-edges 8,16,32,64,128
"""
from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path

import numpy as np

from aero_eyes.evaluate import _load_submission
from aero_eyes.types import Box
from aero_eyes.utils.geometry import box_iou
from aero_eyes.utils.io import load_gt

log = logging.getLogger(__name__)

DEFAULT_BUCKET_EDGES = [8.0, 16.0, 32.0, 64.0, 128.0, 256.0]


def _bucket_label(lo: float, hi: float) -> str:
    lo_s = "0" if lo == 0.0 else f"{lo:.0f}"
    hi_s = "inf" if math.isinf(hi) else f"{hi:.0f}"
    return f"{lo_s}-{hi_s}px"


def collect_matched_frames(pred_tube: dict[int, Box], gt_tube: dict[int, Box]) -> list[dict]:
    """Per-frame records for every frame where BOTH pred and GT are present."""
    records = []
    for fi, gt_box in gt_tube.items():
        pred_box = pred_tube.get(fi)
        if pred_box is None:
            continue
        gt_w = gt_box.x2 - gt_box.x1
        gt_h = gt_box.y2 - gt_box.y1
        if gt_w <= 0 or gt_h <= 0:
            continue
        gt_size = math.sqrt(gt_w * gt_h)
        dx1 = pred_box.x1 - gt_box.x1
        dy1 = pred_box.y1 - gt_box.y1
        dx2 = pred_box.x2 - gt_box.x2
        dy2 = pred_box.y2 - gt_box.y2
        offset = (abs(dx1) + abs(dy1) + abs(dx2) + abs(dy2)) / 4.0
        iou = box_iou(pred_box, gt_box)
        records.append({
            "gt_size": gt_size, "offset": offset, "iou": iou,
            "dx1": dx1, "dy1": dy1, "dx2": dx2, "dy2": dy2,
            "d_width": dx2 - dx1, "d_height": dy2 - dy1,
        })
    return records


def collect_coverage(pred_tube: dict[int, Box], gt_tube: dict[int, Box]) -> list[dict]:
    """Per-GT-frame records of GT size + whether a prediction exists at all
    (for the separate "are small objects missed more often" question)."""
    records = []
    for fi, gt_box in gt_tube.items():
        gt_w = gt_box.x2 - gt_box.x1
        gt_h = gt_box.y2 - gt_box.y1
        if gt_w <= 0 or gt_h <= 0:
            continue
        records.append({
            "gt_size": math.sqrt(gt_w * gt_h),
            "covered": fi in pred_tube,
        })
    return records


def _ceiling_iou(s: float, o: float) -> float | None:
    """IoU of two s x s boxes offset by o along one axis. None if o >= s
    (boxes no longer overlap at all under this simplified model)."""
    if s <= 0 or o >= s:
        return None
    return (s - o) / (s + o)


def print_size_breakdown(records: list[dict], edges: list[float]) -> None:
    if not records:
        print("  (no matched frames)")
        return

    sizes = np.array([r["gt_size"] for r in records])
    ious = np.array([r["iou"] for r in records])
    offsets = np.array([r["offset"] for r in records])

    bounds = [0.0] + edges + [float("inf")]
    print(f"  {'bucket':<12}{'n':>7}{'mean IoU':>10}{'median IoU':>12}{'mean offset(px)':>17}{'ceiling IoU':>13}")
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        mask = (sizes >= lo) & (sizes < hi)
        n = int(mask.sum())
        if n == 0:
            print(f"  {_bucket_label(lo, hi):<12}{0:>7}   (no frames)")
            continue
        mean_iou = float(ious[mask].mean())
        median_iou = float(np.median(ious[mask]))
        mean_offset = float(offsets[mask].mean())
        mean_size = float(sizes[mask].mean())
        ceiling = _ceiling_iou(mean_size, mean_offset)
        ceiling_str = f"{ceiling:.3f}" if ceiling is not None else "n/a (offset>=size)"
        print(
            f"  {_bucket_label(lo, hi):<12}{n:>7}{mean_iou:>10.3f}{median_iou:>12.3f}"
            f"{mean_offset:>17.2f}{ceiling_str:>13}"
        )

    if len(sizes) >= 2 and sizes.std() > 0 and ious.std() > 0:
        log_sizes = np.log(sizes)
        r = float(np.corrcoef(log_sizes, ious)[0, 1])
        print(f"\n  Pearson r(log(gt_size), IoU) across {len(records)} matched frames = {r:.3f}")
        print(
            "    (closer to +1 = size explains more of the IoU spread; closer to 0 = "
            "IoU varies for reasons other than object size)"
        )


def print_bias_noise_breakdown(records: list[dict], edges: list[float]) -> None:
    """Split the per-bucket mean |offset| into a systematic BIAS component
    (signed mean error per edge -- a fixed calibration bug, cheap to fix)
    vs a random NOISE component (std of that error -- frame-to-frame
    jitter, expensive to reduce further). See module docstring."""
    if not records:
        return
    sizes = np.array([r["gt_size"] for r in records])
    dx1 = np.array([r["dx1"] for r in records])
    dy1 = np.array([r["dy1"] for r in records])
    dx2 = np.array([r["dx2"] for r in records])
    dy2 = np.array([r["dy2"] for r in records])
    d_width = np.array([r["d_width"] for r in records])
    d_height = np.array([r["d_height"] for r in records])

    print("\n  --- Bias (systematic, correctable) vs Noise (random, harder to reduce) ---")
    bounds = [0.0] + edges + [float("inf")]
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        mask = (sizes >= lo) & (sizes < hi)
        n = int(mask.sum())
        if n == 0:
            continue
        bias_vec = np.array([dx1[mask].mean(), dy1[mask].mean(), dx2[mask].mean(), dy2[mask].mean()])
        noise_vec = np.array([dx1[mask].std(), dy1[mask].std(), dx2[mask].std(), dy2[mask].std()])
        bias_mag = float(np.abs(bias_vec).mean())
        noise_mag = float(noise_vec.mean())
        mean_size = float(sizes[mask].mean())
        corrected_ceiling = _ceiling_iou(mean_size, noise_mag)
        corrected_str = f"{corrected_ceiling:.3f}" if corrected_ceiling is not None else "n/a"

        print(f"\n  {_bucket_label(lo, hi):<12}(n={n})")
        print(
            f"    bias (pred-gt, signed): dx1={bias_vec[0]:+.2f} dy1={bias_vec[1]:+.2f} "
            f"dx2={bias_vec[2]:+.2f} dy2={bias_vec[3]:+.2f}  |bias| avg={bias_mag:.2f}px"
        )
        print(
            f"    noise (std of error):   dx1={noise_vec[0]:.2f}  dy1={noise_vec[1]:.2f}  "
            f"dx2={noise_vec[2]:.2f}  dy2={noise_vec[3]:.2f}  avg={noise_mag:.2f}px"
        )
        print(
            f"    mean size bias: delta-width={d_width[mask].mean():+.2f}px  "
            f"delta-height={d_height[mask].mean():+.2f}px  "
            "(>0 = predicted box systematically WIDER/TALLER than GT, not just shifted)"
        )
        print(f"    ceiling IoU if bias were fully corrected (noise-only) = {corrected_str}")
        if bias_mag > noise_mag:
            print(
                f"    => BIAS-dominated ({bias_mag:.2f}px > noise {noise_mag:.2f}px): likely a fixed "
                "calibration bug (e.g. box_refine/mask padding, a systematic detector offset) -- "
                "cheap, high-value fix; correcting it alone should meaningfully raise IoU."
            )
        else:
            print(
                f"    => NOISE-dominated (std {noise_mag:.2f}px >= bias {bias_mag:.2f}px): error is "
                "mostly random frame-to-frame jitter, not a fixed offset -- likely closer to the "
                "detector/annotation's real precision limit; a simple calibration fix won't help much."
            )


def print_coverage_breakdown(records: list[dict], edges: list[float]) -> None:
    if not records:
        return
    sizes = np.array([r["gt_size"] for r in records])
    covered = np.array([r["covered"] for r in records])

    bounds = [0.0] + edges + [float("inf")]
    print(f"\n  {'bucket':<12}{'n GT frames':>13}{'coverage rate':>16}")
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        mask = (sizes >= lo) & (sizes < hi)
        n = int(mask.sum())
        if n == 0:
            continue
        rate = float(covered[mask].mean())
        print(f"  {_bucket_label(lo, hi):<12}{n:>13}{rate:>16.1%}")


def print_overall_summary(matched: list[dict], coverage: list[dict], edges: list[float]) -> None:
    """One final plain-language verdict for the whole run (pooled across
    whatever was checked -- one sample or all of them), instead of making
    the reader mentally combine every per-bucket line above."""
    print("\n=== SUMMARY ===")
    if not matched:
        print("  (no matched frames)")
        return

    sizes = np.array([r["gt_size"] for r in matched])
    ious = np.array([r["iou"] for r in matched])
    offsets = np.array([r["offset"] for r in matched])
    dx1 = np.array([r["dx1"] for r in matched])
    dy1 = np.array([r["dy1"] for r in matched])
    dx2 = np.array([r["dx2"] for r in matched])
    dy2 = np.array([r["dy2"] for r in matched])

    mean_size = float(sizes.mean())
    mean_iou = float(ious.mean())
    ceiling = _ceiling_iou(mean_size, float(offsets.mean()))

    bias_vec = np.array([dx1.mean(), dy1.mean(), dx2.mean(), dy2.mean()])
    noise_vec = np.array([dx1.std(), dy1.std(), dx2.std(), dy2.std()])
    bias_mag = float(np.abs(bias_vec).mean())
    noise_mag = float(noise_vec.mean())
    corrected_ceiling = _ceiling_iou(mean_size, noise_mag)

    # Tally, per size bucket, how many MATCHED frames fall in a
    # bias-dominated vs noise-dominated bucket (weighted by frame count,
    # not just "N buckets" -- a bucket with 1600 frames should outweigh one
    # with 6).
    bounds = [0.0] + edges + [float("inf")]
    n_bias_frames = n_noise_frames = 0
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        mask = (sizes >= lo) & (sizes < hi)
        n = int(mask.sum())
        if n == 0:
            continue
        b = np.array([dx1[mask].mean(), dy1[mask].mean(), dx2[mask].mean(), dy2[mask].mean()])
        nz = np.array([dx1[mask].std(), dy1[mask].std(), dx2[mask].std(), dy2[mask].std()])
        if float(np.abs(b).mean()) > float(nz.mean()):
            n_bias_frames += n
        else:
            n_noise_frames += n

    print(
        f"  {len(matched)} matched frames | mean GT size={mean_size:.1f}px | "
        f"mean IoU={mean_iou:.3f}"
    )
    print(f"  Ceiling given current offset       = {ceiling:.3f}" if ceiling is not None else
          "  Ceiling given current offset       = n/a")
    print(f"  Ceiling if bias were fully fixed   = {corrected_ceiling:.3f}" if corrected_ceiling is not None else
          "  Ceiling if bias were fully fixed   = n/a")
    if coverage:
        coverage_rate = float(np.mean([r["covered"] for r in coverage]))
        print(f"  Coverage (GT frames with ANY prediction) = {coverage_rate:.1%}")
    print(
        f"  {n_bias_frames}/{len(matched)} frames ({n_bias_frames / len(matched):.0%}) sit in a "
        f"BIAS-dominated size bucket; {n_noise_frames} ({n_noise_frames / len(matched):.0%}) in a "
        "NOISE-dominated one."
    )

    print()
    if ceiling is not None and mean_iou < ceiling - 0.03:
        print(
            f"  -> Actual mean IoU ({mean_iou:.3f}) is BELOW the size-imposed ceiling ({ceiling:.3f}): "
            "there is real, fixable headroom independent of how small the object is."
        )
    elif ceiling is not None:
        print(
            f"  -> Actual mean IoU ({mean_iou:.3f}) is already close to the size-imposed ceiling "
            f"({ceiling:.3f}): most of the remaining shortfall is the object's small size itself, not "
            "a quality bug -- don't expect much more without sub-pixel-level precision."
        )
    if n_bias_frames >= n_noise_frames:
        msg = (
            f"  -> PRIORITY: fix the systematic bias first (see dx1/dy1/dx2/dy2 signed means per "
            "bucket above -- likely a fixed calibration issue such as box_refine/mask padding or a "
            "detector regression offset). Cheap relative to model changes"
        )
        if corrected_ceiling is not None:
            msg += f", and alone would raise the achievable ceiling to ~{corrected_ceiling:.3f}."
        else:
            msg += "."
        print(msg)
    else:
        print(
            "  -> Error is mostly random frame-to-frame noise, not a fixed bias -- no cheap "
            "calibration fix available; further gains need a genuinely more precise detector/box_refine, "
            "or accepting the current size-imposed ceiling."
        )


def check_sample(cfg, sample_id: str, work_dir_override: Path | None = None) -> tuple[list[dict], list[dict]]:
    work_dir = (work_dir_override or Path(cfg.project.work_dir)) / sample_id
    submission_path = work_dir / cfg.data.submission.path_name

    gt_file = cfg.data.gt.global_file
    try:
        gt = load_gt(gt_file, sample_id)
    except KeyError:
        print(f"{sample_id}: not found in {gt_file}, skipping.")
        return [], []

    if not submission_path.exists():
        print(f"{sample_id}: submission not found at {submission_path}, skipping.")
        return [], []

    all_submissions = _load_submission(submission_path)
    pred_tube = all_submissions.get(sample_id, {})

    matched = collect_matched_frames(pred_tube, gt)
    coverage = collect_coverage(pred_tube, gt)
    return matched, coverage


def main():
    logging.basicConfig(level=logging.WARNING)
    p = argparse.ArgumentParser(
        description="Decompose IoU/ST-IoU shortfall into 'small object size ceiling' vs "
        "genuine localization/coverage problems."
    )
    p.add_argument("--config", required=True)
    p.add_argument("--sample", default=None, help="omit to pool every sample in data_root")
    p.add_argument("--set", action="append", default=[])
    p.add_argument(
        "--bucket-edges", default=None,
        help="comma-separated gt_size (px, sqrt(w*h)) bucket edges, e.g. 8,16,32,64,128 "
        f"(default: {','.join(str(int(e)) for e in DEFAULT_BUCKET_EDGES)})",
    )
    args = p.parse_args()

    from aero_eyes.config import load_config
    cfg = load_config(args.config, args.set)

    edges = (
        [float(x) for x in args.bucket_edges.split(",")]
        if args.bucket_edges else DEFAULT_BUCKET_EDGES
    )

    if args.sample:
        sample_ids = [args.sample]
    else:
        data_root = Path(cfg.data.data_root)
        sample_ids = [d.name for d in sorted(data_root.iterdir()) if d.is_dir()]

    all_matched: list[dict] = []
    all_coverage: list[dict] = []
    for sid in sample_ids:
        matched, coverage = check_sample(cfg, sid)
        if matched or coverage:
            print(f"\n=== {sid}: {len(matched)} matched frames, {len(coverage)} GT frames ===")
            print_size_breakdown(matched, edges)
            print_bias_noise_breakdown(matched, edges)
            print_coverage_breakdown(coverage, edges)
        all_matched.extend(matched)
        all_coverage.extend(coverage)

    if args.sample is None and len(sample_ids) > 1:
        print(f"\n=== POOLED across {len(sample_ids)} samples "
              f"({len(all_matched)} matched frames, {len(all_coverage)} GT frames) ===")
        print_size_breakdown(all_matched, edges)
        print_bias_noise_breakdown(all_matched, edges)
        print_coverage_breakdown(all_coverage, edges)

    print_overall_summary(all_matched, all_coverage, edges)


if __name__ == "__main__":
    main()
