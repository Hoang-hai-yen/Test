"""End-to-end orchestrator: runs Stage 1 -> 5 in sequence.

    python -m aero_eyes.stages.run_all --config configs/config.yaml
    python -m aero_eyes.stages.run_all --config configs/config.yaml --sample Backpack_0

Each stage writes its own artifact, so a failed run can be resumed from any
stage with --from-stage N. The per-sample loop contains NO logic that
individual stages lack -- it only calls them in sequence. The one thing
added on top is a final merge step (see merge_submissions in
aero_eyes/utils/io.py): each sample's Stage 5 writes its OWN
submission.json, so after the loop this combines every sample's file into
one <work_dir>/submission_all.json, upserted by video_id. Pass --no-merge
to skip it (e.g. when only running/debugging a single sample).
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
    Stage numbers in --from-stage always refer to the ORIGINAL 1-5 scheme
    (e.g. --from-stage 4 resumes at tracking either way) so switching
    detector doesn't change what a given --from-stage value means.

    merge: after all requested samples finish, combine their individual
    submission.json files into <work_dir>/submission_all.json.
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
        # Skip dotfiles/dirs (e.g. .ipynb_checkpoints -- Jupyter creates this
        # inside whatever directory it's browsing/editing, including
        # data_root itself if a notebook was ever opened from there) -- not
        # a real sample, but bare iterdir() picks it up like any other dir.
        sample_ids = [
            d.name for d in sorted(data_root.iterdir())
            if d.is_dir() and not d.name.startswith(".")
        ]
        if not sample_ids:
            raise ValueError(f"No sample directories found under {data_root}")

    if cfg.pipeline.detector == "geco2":
        from aero_eyes.stages.stage123_geco2 import run_stage123_geco2
        # stage_num=1 covers what legacy Stage1-3 do combined.
        stage_fns: list[tuple[int, object]] = [(1, run_stage123_geco2), (4, run_stage4), (5, run_stage5)]
    else:
        from aero_eyes.stages.stage1 import run_stage1
        from aero_eyes.stages.stage2 import run_stage2
        from aero_eyes.stages.stage3 import run_stage3
        stage_fns = [(1, run_stage1), (2, run_stage2), (3, run_stage3), (4, run_stage4), (5, run_stage5)]

    for sid in sample_ids:
        log.info("=== Running pipeline for sample: %s (detector=%s, from stage %d) ===",
                 sid, cfg.pipeline.detector, from_stage)
        for stage_num, fn in stage_fns:
            if stage_num < from_stage:
                log.debug("Skipping stage %d (--from-stage %d)", stage_num, from_stage)
                continue
            log.info("--- Stage %d ---", stage_num)
            fn(cfg, sid)
        log.info("=== Done: %s ===", sid)

    if merge:
        from aero_eyes.utils.io import merge_submissions
        merged_path = Path(cfg.project.work_dir) / "submission_all.json"
        merge_submissions(cfg.project.work_dir, sample_ids, cfg.data.submission.path_name, merged_path)


def main():
    logging.basicConfig(level=logging.INFO)
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
