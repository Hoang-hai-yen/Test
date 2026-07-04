"""End-to-end orchestrator: runs Stage 1 -> 5 in sequence.

    python -m aero_eyes.stages.run_all --config configs/config.yaml
    python -m aero_eyes.stages.run_all --config configs/config.yaml --sample Backpack_0

Each stage writes its own artifact, so a failed run can be resumed
from any stage with --from-stage N. This wrapper contains NO logic
that individual stages lack.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

log = logging.getLogger(__name__)


def run_all(cfg, sample_id: str | None = None, from_stage: int = 1) -> None:
    """Run the full pipeline for one or all samples."""
    from aero_eyes.stages.stage1 import run_stage1
    from aero_eyes.stages.stage2 import run_stage2
    from aero_eyes.stages.stage3 import run_stage3
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

    stage_fns = [run_stage1, run_stage2, run_stage3, run_stage4, run_stage5]

    for sid in sample_ids:
        log.info("=== Running pipeline for sample: %s (from stage %d) ===", sid, from_stage)
        for stage_num, fn in enumerate(stage_fns, start=1):
            if stage_num < from_stage:
                log.debug("Skipping stage %d (--from-stage %d)", stage_num, from_stage)
                continue
            log.info("--- Stage %d ---", stage_num)
            fn(cfg, sid)
        log.info("=== Done: %s ===", sid)


def main():
    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser(description="Run full AERO EYES pipeline end-to-end")
    p.add_argument("--config", required=True)
    p.add_argument("--sample", default=None, help="omit to run all samples")
    p.add_argument("--set", action="append", default=[], help="cfg override k=v")
    p.add_argument("--from-stage", type=int, default=1, dest="from_stage",
                   help="Resume from stage N (1-5)")
    args = p.parse_args()
    from aero_eyes.config import load_config
    cfg = load_config(args.config, args.set)
    run_all(cfg, sample_id=args.sample, from_stage=args.from_stage)


if __name__ == "__main__":
    main()
