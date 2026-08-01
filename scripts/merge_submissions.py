"""Combine every sample's own submission.json (written by Stage 5) into one
file, without re-running the pipeline. Useful after running samples across
several sessions/machines, or to regenerate submission_all.json if it was
deleted.

run_all.py already does this automatically at the end of a full run (unless
--no-merge is passed) -- this script is for re-running just the merge step.

Usage:
    python -m scripts.merge_submissions --config configs/config.yaml
    python -m scripts.merge_submissions --config configs/config.yaml --sample BlackBox_0 --sample Person_1
    python -m scripts.merge_submissions --config configs/config.yaml --out runs/exp001/final.json
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

log = logging.getLogger(__name__)


def main():
    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser(description="Merge per-sample submission.json files into one")
    p.add_argument("--config", required=True)
    p.add_argument("--set", action="append", default=[], help="cfg override k=v")
    p.add_argument("--sample", action="append", default=None,
                    help="Restrict to specific sample_id(s); repeatable. Omit to merge all "
                         "subdirectories under work_dir.")
    p.add_argument("--out", default=None,
                    help="Output path (default: <work_dir>/submission_all.json)")
    args = p.parse_args()

    from aero_eyes.config import load_config
    from aero_eyes.utils.io import merge_submissions

    cfg = load_config(args.config, args.set)
    out_path = Path(args.out) if args.out else Path(cfg.project.work_dir) / "submission_all.json"

    merge_submissions(cfg.project.work_dir, args.sample, cfg.data.submission.path_name, out_path)


if __name__ == "__main__":
    main()
