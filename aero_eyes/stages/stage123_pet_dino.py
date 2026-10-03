"""Stage 1+2+3 replacement -- PET-DINO text-prompted detector (CVPR 2026,
arXiv:2604.00503). Selected via config: pipeline.detector: pet_dino.

See Stage123PetDinoConfig's own docstring (aero_eyes/config.py) for the
full setup/scope caveats -- in particular: this needs a SEPARATE,
heavyweight MMDetection-based dependency stack (not part of this project's
default requirements.txt) and aero_eyes.models.pet_dino_detector has NOT
been smoke-tested against a real checkpoint in this environment.
Deliberately scoped to a MINIMAL per-keyframe detection loop, mirroring
aero_eyes.stages.stage123_gdino.run_stage123_gdino's own BASE path (plain
box_threshold, no online_adaptive_threshold/online_fusion/
cascade_verification/clip_tiebreak equivalents) -- a first, comparable
baseline to test PET-DINO's own raw detection quality against, not a full
port of stage123_gdino's whole FP-filtering suite onto an as-yet-unverified
new detector. The one exception is the opt-in color_postfilter
(stage123_pet_dino.color_postfilter), shared with stage123_gdino/geco2.

Flow:  text prompt (resolve_text_prompt)
       -> per-keyframe PET-DINO forward pass
          -> box_threshold -> NMS -> top-K
          -> [color_postfilter, if enabled]
       -> detections.json (same schema Stage 3 writes, so Stage 4/5 need
          no changes to consume it)

Reads:  cfg.data video (+ this sample's text prompt)
Writes: <work_dir>/<sample_id>/detections.json
        <work_dir>/<sample_id>/color_postfilter.jsonl (if color_postfilter.enabled)
Viz:    <work_dir>/<sample_id>/viz/stage123_pet_dino/ (when save_visualizations=true)
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

from aero_eyes.types import Detection

log = logging.getLogger(__name__)


def resolve_text_prompt(cfg, sample_id: str) -> str:
    """Same 3-way precedence as stage123_gdino.resolve_text_prompt
    (<data_root>/<sample_id>/<prompt_file_name> > text_prompts[sample_id] >
    default_text_prompt), against stage123_pet_dino's own config fields --
    duplicated rather than shared, see Stage123PetDinoConfig's own
    docstring for why. Raises if none of the three yields non-empty text."""
    p = cfg.stage123_pet_dino
    prompt_path = Path(cfg.data.data_root) / sample_id / p.prompt_file_name
    if prompt_path.exists():
        text = prompt_path.read_text(encoding="utf-8").strip()
        if text:
            return text
        log.warning("[Stage123-PETDINO] %s: %s exists but is empty -- falling through.", sample_id, prompt_path)
    text = p.text_prompts.get(sample_id, "").strip()
    if text:
        return text
    text = p.default_text_prompt.strip()
    if text:
        return text
    raise ValueError(
        f"stage123_pet_dino: no text prompt for sample '{sample_id}' -- set "
        f"stage123_pet_dino.default_text_prompt, or stage123_pet_dino.text_prompts['{sample_id}'], "
        f"or create {prompt_path}."
    )


def _locate_video(cfg, sample_id: str) -> Path:
    data_root = Path(cfg.data.data_root)
    video_dir = data_root / sample_id
    video_files = list(video_dir.glob(cfg.data.video_glob))
    if not video_files:
        raise FileNotFoundError(f"No video matching '{cfg.data.video_glob}' found in {video_dir}.")
    return video_files[0]


def run_stage123_pet_dino(cfg, sample_id: str) -> Path:
    """Run the merged PET-DINO stage for one sample. Returns path to detections.json."""
    from aero_eyes.models.pet_dino_detector import PetDinoDetector
    from aero_eyes.utils.io import write_detections
    from aero_eyes.utils.video import frame_iterator, keyframe_indices, video_info

    t0 = time.time()
    work_dir = Path(cfg.project.work_dir) / sample_id
    work_dir.mkdir(parents=True, exist_ok=True)

    det_path = work_dir / "detections.json"
    if cfg.project.use_cache and det_path.exists():
        log.info("[Stage123-PETDINO] %s: using cached detections at %s", sample_id, det_path)
        return det_path

    text_prompt = resolve_text_prompt(cfg, sample_id)
    log.info("[Stage123-PETDINO] %s: text prompt = %r", sample_id, text_prompt)
    detector = PetDinoDetector(cfg)

    video_path = _locate_video(cfg, sample_id)
    info = video_info(video_path)
    total_frames = info["total_frames"]
    log.info("[Stage123-PETDINO] %s: video=%s (%d frames)", sample_id, video_path.name, total_frames)

    kf_indices = set(keyframe_indices(total_frames, cfg.stage123_pet_dino.keyframe_interval))
    viz_dir = work_dir / "viz" / "stage123_pet_dino"
    save_viz = cfg.runtime.save_visualizations

    # Optional color post-filter -- same build_color_signature/
    # apply_color_postfilter pair stage123_gdino uses (reference photos
    # segmented with stage1.segmentation, like stage123_gdino).
    cpf_cfg = cfg.stage123_pet_dino.color_postfilter
    color_sig = None
    color_stats: list = []
    color_records: list[dict] = []
    if cpf_cfg.enabled:
        from aero_eyes.stages.stage123_geco2 import apply_color_postfilter, build_color_signature

        color_sig = build_color_signature(
            cfg, sample_id, work_dir, cpf_cfg, cfg.stage1.segmentation,
            cache_name="color_signature_pet_dino.npz", log_prefix="Stage123-PETDINO",
        )

    detections: dict[int, list[Detection]] = {}
    for frame_idx, frame_bgr in frame_iterator(video_path):
        if frame_idx not in kf_indices:
            continue
        boxes = detector.detect_frame(frame_bgr, text_prompt)
        if color_sig is not None and boxes:
            frame_color_records: list[dict] = []
            boxes = apply_color_postfilter(
                frame_bgr, boxes, color_sig, cpf_cfg, stats_out=color_stats, records=frame_color_records,
            )
            for r in frame_color_records:
                r["frame_idx"] = frame_idx
            color_records.extend(frame_color_records)
            if save_viz:
                from aero_eyes.utils import viz as vizmod
                vizmod.save_color_postfilter(frame_bgr, frame_color_records, frame_idx, viz_dir / "color")
        result_dets = [
            Detection(frame_idx=frame_idx, box=b, similarity=b.score, source="detect")
            for b in boxes
        ]
        detections[frame_idx] = result_dets
        log.debug("[Stage123-PETDINO] frame %d: %d detections", frame_idx, len(result_dets))
        if save_viz:
            from aero_eyes.utils import viz as vizmod
            vizmod.save_stage3_detections(
                frame_bgr, [d.box for d in result_dets], [d.similarity for d in result_dets],
                frame_idx, viz_dir,
            )

    write_detections(detections, det_path, threshold=cfg.stage123_pet_dino.box_threshold)

    if color_records:
        import json

        from aero_eyes.stages.stage123_geco2 import log_color_postfilter_stats

        log_color_postfilter_stats("Stage123-PETDINO", sample_id, color_stats, cpf_cfg)
        color_log_path = work_dir / "color_postfilter.jsonl"
        with open(color_log_path, "w", encoding="utf-8") as f:
            for r in color_records:
                f.write(json.dumps(r) + "\n")
        log.info("[Stage123-PETDINO] %s: color_postfilter dropped %d/%d box(es), per-box records -> %s",
                 sample_id, sum(not r["kept"] for r in color_records), len(color_records), color_log_path)

    elapsed = time.time() - t0
    log.info("[Stage123-PETDINO] %s done in %.1fs -> %s (%d detection frames)",
              sample_id, elapsed, det_path, len(detections))
    return det_path


def main():
    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser(description="Run Stage 1+2+3 (PET-DINO) for one sample")
    p.add_argument("--config", required=True)
    p.add_argument("--sample", required=True)
    p.add_argument("--set", action="append", default=[], help="cfg override k=v")
    args = p.parse_args()
    from aero_eyes.config import load_config
    cfg = load_config(args.config, args.set)
    run_stage123_pet_dino(cfg, args.sample)


if __name__ == "__main__":
    main()
