"""Stage 3+4 — combined matching + tracking (cosine_rescore variant).

Used instead of separate run_stage3()/run_stage4() pipeline entries when
stage123_geco2.cosine_rescore.enabled=true (see run_all.py) -- mirrors the
merged Stage B (stage34.py) design from the thanhhuy branch: matching and
tracking are packaged as one atomic pipeline step so a --from-stage resume
that lands on tracking can never run it against a stale/missing
detections.json. Unlike that branch's stage34.py, this does NOT duplicate
Stage 3/4's logic inline -- it just calls the existing run_stage3()/
run_stage4() back to back, each of which still short-circuits via
cfg.project.use_cache internally, so this adds no cost when both artifacts
are already cached.

Reads:  candidates.json (+ .feats.npz), prototype.npz, video
Writes: detections.json, tracks.json
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

log = logging.getLogger(__name__)


def run_stage34(cfg, sample_id: str) -> Path:
    """Run Stage 3 (cosine matching) then Stage 4 (tracking) back to back.
    Returns path to tracks.json."""
    from aero_eyes.stages.stage3 import run_stage3
    from aero_eyes.stages.stage4 import run_stage4

    run_stage3(cfg, sample_id)
    return run_stage4(cfg, sample_id)


def main():
    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser(description="Stage 3+4 — cosine matching + tracking")
    p.add_argument("--config", required=True)
    p.add_argument("--sample", required=True)
    p.add_argument("--set", action="append", default=[])
    args = p.parse_args()
    from aero_eyes.config import load_config
    cfg = load_config(args.config, args.set)
    run_stage34(cfg, args.sample)


if __name__ == "__main__":
    main()
