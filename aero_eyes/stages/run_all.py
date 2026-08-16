"""End-to-end orchestrator: runs Stage 1 -> 5 in sequence.

    python -m aero_eyes.stages.run_all --config configs/config.yaml
    python -m aero_eyes.stages.run_all --config configs/config.yaml --sample Backpack_0

Each stage writes its own artifact, so a failed run can be resumed from any
stage with --from-stage N. The per-sample loop calls stages in sequence and
catches exceptions per sample so a single failure doesn't abort the entire batch.

When merge=True (default), combines every successful sample's submission.json
into <work_dir>/submission_all.json. Pass --no-merge to skip merging.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

log = logging.getLogger(__name__)


def run_all(cfg, sample_id: str | None = None, from_stage: int = 1, merge: bool = True) -> None:
    """Run the full pipeline for one or all samples.

    Stage list depends on cfg.pipeline.detector:
      "legacy" -> Stage1, Stage2, Stage3, Stage4, Stage5 (5 artifacts)
      "geco2"  -> merged Stage1+2+3 (stage123_geco2.py), Stage4, Stage5

    Stage numbers in --from-stage refer to the original 1-5 scheme
    (e.g. --from-stage 4 resumes at tracking either way).
    """
    from aero_eyes.stages.stage4 import run_stage4
    from aero_eyes.stages.stage5 import run_stage5

    # Determine which samples to run
    if sample_id is not None:
        sample_ids = [sample_id]
    else:
        data_root = Path(cfg.data.data_root)
        if not data_root.exists():
            raise FileNotFoundError(f"data_root not found: {data_root}")
        sample_ids = [d.name for d in sorted(data_root.iterdir()) if d.is_dir()]
        if not sample_ids:
            raise ValueError(f"No sample directories found under {data_root}")

    detector_mode = getattr(cfg.pipeline, "detector", "legacy") if hasattr(cfg, "pipeline") else "legacy"

    if detector_mode == "geco2":
        from aero_eyes.stages.stage123_geco2 import run_stage123_geco2
        stage_fns: list[tuple[int, object]] = [
            (1, run_stage123_geco2),
            (4, run_stage4),
            (5, run_stage5),
        ]
    else:
        from aero_eyes.stages.stage1 import run_stage1
        from aero_eyes.stages.stage2 import run_stage2
        from aero_eyes.stages.stage3 import run_stage3
        stage_fns = [
            (1, run_stage1),
            (2, run_stage2),
            (3, run_stage3),
            (4, run_stage4),
            (5, run_stage5),
        ]

    failed: list[str] = []
    successful_samples: list[str] = []

    for sid in sample_ids:
        log.info("=== Running pipeline for sample: %s (detector=%s, from stage %d) ===",
                 sid, detector_mode, from_stage)
        try:
            for stage_num, fn in stage_fns:
                if stage_num < from_stage:
                    log.debug("Skipping stage %d (--from-stage %d)", stage_num, from_stage)
                    continue
                log.info("--- Stage %d ---", stage_num)
                fn(cfg, sid)
            successful_samples.append(sid)
            log.info("=== Done: %s ===", sid)
        except Exception:
            log.exception("=== FAILED: %s -- continuing with remaining samples ===", sid)
            failed.append(sid)
            continue

    if failed:
        log.warning("run_all finished with %d/%d samples failed: %s",
                    len(failed), len(sample_ids), failed)

    if merge and successful_samples:
        try:
            from aero_eyes.utils.io import merge_submissions
            merged_path = Path(cfg.project.work_dir) / "submission_all.json"
            merge_submissions(cfg.project.work_dir, successful_samples, cfg.data.submission.path_name, merged_path)
            log.info("Merged %d submissions into %s", len(successful_samples), merged_path)
        except Exception:
            log.exception("Failed to merge submission files into submission_all.json")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    p = argparse.ArgumentParser(description="Run full AERO EYES pipeline end-to-end")
    p.add_argument("--config", required=True)
    p.add_argument("--sample", default=None, help="omit to run all samples")
    p.add_argument("--set", action="append", default=[], help="cfg override k=v")
    p.add_argument("--from-stage", type=int, default=1, dest="from_stage",
                   help="Resume from stage N (1-5)")
    p.add_argument("--no-merge", action="store_false", dest="merge",
                   help="Skip combining per-sample submission.json files into submission_all.json")
    args = p.parse_args()
    from aero_eyes.config import load_config
    cfg = load_config(args.config, args.set)
    run_all(cfg, sample_id=args.sample, from_stage=args.from_stage, merge=args.merge)


if __name__ == "__main__":
    main()