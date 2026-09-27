"""Stage 1+2+3 replacement — Grounding DINO text-prompted detector.

Selected via config: pipeline.detector: grounding_dino  (default stays
"legacy"; see also pipeline.detector: geco2 for the image-exemplar
alternative in stage123_geco2.py).

Unlike stage1/stage123_geco2, this detector is OPEN-VOCABULARY and
TEXT-prompted -- it never looks at data.refs_subdir's reference photos. The
text prompt for a sample comes from resolve_text_prompt() below (a
per-sample prompt.txt file, then stage123_gdino.text_prompts, then
stage123_gdino.default_text_prompt).

Flow:  text prompt (resolve_text_prompt)
       -> per-keyframe Grounding DINO forward pass
          -> box_threshold/text_threshold -> NMS -> top-K
       -> detections.json (same schema Stage 3 writes, so Stage 4/5 need
          no changes to consume it)

Reads:  cfg.data video (+ this sample's text prompt)
Writes: <work_dir>/<sample_id>/detections.json
Viz:    <work_dir>/<sample_id>/viz/stage123_gdino/ (when save_visualizations=true)
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

from aero_eyes.types import Detection

log = logging.getLogger(__name__)


def _locate_video(cfg, sample_id: str) -> Path:
    data_root = Path(cfg.data.data_root)
    video_dir = data_root / sample_id
    video_files = list(video_dir.glob(cfg.data.video_glob))
    if not video_files:
        raise FileNotFoundError(f"No video matching '{cfg.data.video_glob}' found in {video_dir}.")
    return video_files[0]


def resolve_text_prompt(cfg, sample_id: str) -> str:
    """Precedence: <data_root>/<sample_id>/<prompt_file_name> (if present)
    > stage123_gdino.text_prompts[sample_id] > stage123_gdino.
    default_text_prompt. Raises if none of the three yields non-empty text
    -- there is no safe default prompt to fall back to silently."""
    g = cfg.stage123_gdino
    prompt_path = Path(cfg.data.data_root) / sample_id / g.prompt_file_name
    if prompt_path.exists():
        text = prompt_path.read_text(encoding="utf-8").strip()
        if text:
            return text
        log.warning("[Stage123-GDINO] %s: %s exists but is empty -- falling through.", sample_id, prompt_path)
    text = g.text_prompts.get(sample_id, "").strip()
    if text:
        return text
    text = g.default_text_prompt.strip()
    if text:
        return text
    raise ValueError(
        f"stage123_gdino: no text prompt for sample '{sample_id}' -- set "
        f"stage123_gdino.default_text_prompt, or stage123_gdino.text_prompts['{sample_id}'], "
        f"or create {prompt_path}."
    )


def run_stage123_gdino(cfg, sample_id: str) -> Path:
    """Run the merged Grounding DINO stage for one sample. Returns path to detections.json."""
    from aero_eyes.models.grounding_dino_detector import GroundingDinoDetector
    from aero_eyes.utils.io import write_detections
    from aero_eyes.utils.video import frame_iterator, keyframe_indices, video_info

    t0 = time.time()
    work_dir = Path(cfg.project.work_dir) / sample_id
    work_dir.mkdir(parents=True, exist_ok=True)

    det_path = work_dir / "detections.json"
    if cfg.project.use_cache and det_path.exists():
        log.info("[Stage123-GDINO] %s: using cached detections at %s", sample_id, det_path)
        return det_path

    text_prompt = resolve_text_prompt(cfg, sample_id)
    log.info("[Stage123-GDINO] %s: text prompt = %r", sample_id, text_prompt)
    detector = GroundingDinoDetector(cfg)

    video_path = _locate_video(cfg, sample_id)
    info = video_info(video_path)
    total_frames = info["total_frames"]
    log.info("[Stage123-GDINO] %s: video=%s (%d frames)", sample_id, video_path.name, total_frames)

    kf_indices = set(keyframe_indices(total_frames, cfg.stage123_gdino.keyframe_interval))
    viz_dir = work_dir / "viz" / "stage123_gdino"
    save_viz = cfg.runtime.save_visualizations

    detections: dict[int, list[Detection]] = {}
    for frame_idx, frame_bgr in frame_iterator(video_path):
        if frame_idx not in kf_indices:
            continue
        boxes = detector.detect_frame(frame_bgr, text_prompt)
        result_dets = [
            Detection(frame_idx=frame_idx, box=b, similarity=b.score, source="detect")
            for b in boxes
        ]
        detections[frame_idx] = result_dets
        log.debug("[Stage123-GDINO] frame %d: %d detections", frame_idx, len(result_dets))
        if save_viz:
            from aero_eyes.utils import viz as vizmod
            vizmod.save_stage3_detections(
                frame_bgr, [d.box for d in result_dets], [d.similarity for d in result_dets],
                frame_idx, viz_dir,
            )

    idf_cfg = cfg.stage123_gdino.isolated_detection_filter
    if idf_cfg.enabled:
        from aero_eyes.stages.stage3 import find_isolated_keyframes
        isolated = find_isolated_keyframes(
            {fi: max(d.similarity for d in dets) for fi, dets in detections.items() if dets},
            cfg.stage123_gdino.keyframe_interval, idf_cfg,
        )
        for fi in isolated:
            detections[fi] = []
        if isolated:
            log.info(
                "[Stage123-GDINO] %s: isolated_detection_filter (max_gap=%d x %d frames, "
                "keep_conf_threshold=%s) dropped %d isolated keyframe(s): %s",
                sample_id, idf_cfg.max_gap_intervals, cfg.stage123_gdino.keyframe_interval,
                idf_cfg.keep_conf_threshold, len(isolated), sorted(isolated),
            )

    write_detections(detections, det_path, threshold=cfg.stage123_gdino.box_threshold)

    elapsed = time.time() - t0
    log.info("[Stage123-GDINO] %s done in %.1fs -> %s (%d detection frames)",
              sample_id, elapsed, det_path, len(detections))
    return det_path


def run_stage12_gdino_candidates(cfg, sample_id: str) -> Path:
    """Stage 1+2 replacement (cosine_rescore variant) -- Grounding DINO as a
    CANDIDATE generator instead of the final word. Used instead of
    run_stage123_gdino when stage123_gdino.cosine_rescore.enabled=true.

    Differs from run_stage123_gdino in exactly one way: Grounding DINO's own
    box_threshold/topk_per_keyframe are replaced with the looser
    cosine_rescore.candidate_* values (so real detections aren't filtered
    out before Stage 3 gets to see them), each surviving candidate crop is
    embedded with stage1.feature_extractor (an INSTANCE-level signal
    Grounding DINO's own category-level score can't provide -- see
    GDinoCosineRescoreConfig's own docstring), and the result is written to
    candidates.json (+ .feats.npz) in the same schema Stage 2 writes,
    instead of straight to detections.json.
    aero_eyes.stages.stage3.run_stage3 then does the actual threshold/NMS/
    top-K filtering that produces detections.json for Stage 4/5.

    Reads:  cfg.data reference images + video (+ this sample's text prompt)
    Writes: <work_dir>/<sample_id>/prototype.npz (via stage1.run_stage1)
            <work_dir>/<sample_id>/candidates.json (+ .feats.npz)
            <work_dir>/<sample_id>/viz/stage123_gdino/candidates/frame_XXXXXX.jpg
              (raw candidates + Grounding DINO's own score per box, when
              runtime.save_visualizations=true)
    """
    from aero_eyes.models.features import build_feature_extractor
    from aero_eyes.models.grounding_dino_detector import GroundingDinoDetector
    from aero_eyes.stages.stage1 import run_stage1
    from aero_eyes.stages.stage2 import _write_candidates_with_features
    from aero_eyes.utils.video import frame_iterator, keyframe_indices, video_info

    t0 = time.time()
    work_dir = Path(cfg.project.work_dir) / sample_id
    work_dir.mkdir(parents=True, exist_ok=True)

    cand_path = work_dir / "candidates.json"
    if cfg.project.use_cache and cand_path.exists():
        log.info("[Stage12-GDINO] %s: using cached candidates at %s", sample_id, cand_path)
        return cand_path

    text_prompt = resolve_text_prompt(cfg, sample_id)
    log.info("[Stage12-GDINO] %s: text prompt = %r", sample_id, text_prompt)

    # stage1.feature_extractor's prototype -- an independent, instance-level
    # signal Stage 3's cosine matching will check candidates against.
    run_stage1(cfg, sample_id)
    extractor = build_feature_extractor(cfg)

    detector = GroundingDinoDetector(cfg)
    cr = cfg.stage123_gdino.cosine_rescore
    # Loosen Grounding DINO's own cut so real detections survive through to
    # Stage 3's cosine matching -- see GDinoCosineRescoreConfig docstring.
    detector.box_threshold = cr.candidate_box_threshold
    detector.topk_per_keyframe = cr.candidate_topk_per_keyframe

    video_path = _locate_video(cfg, sample_id)
    info = video_info(video_path)
    total_frames = info["total_frames"]
    log.info("[Stage12-GDINO] %s: video=%s (%d frames)", sample_id, video_path.name, total_frames)

    kf_indices = set(keyframe_indices(total_frames, cfg.stage123_gdino.keyframe_interval))
    cand_viz_dir = (
        work_dir / "viz" / "stage123_gdino" / "candidates" if cfg.runtime.save_visualizations else None
    )

    candidates: dict[int, list[Detection]] = {}
    for frame_idx, frame_bgr in frame_iterator(video_path):
        if frame_idx not in kf_indices:
            continue
        boxes = detector.detect_frame(frame_bgr, text_prompt)
        if boxes:
            feats = extractor.extract_crops(
                frame_bgr, boxes,
                pad_ratio=cfg.stage2.candidate.feature_crop_pad,
                batch_size=cfg.runtime.batch_size,
            )
        else:
            feats = None

        frame_dets: list[Detection] = []
        for i, box in enumerate(boxes):
            d = Detection(frame_idx=frame_idx, box=box, similarity=0.0, source="candidate")
            d._feature = feats[i]  # type: ignore[attr-defined]
            frame_dets.append(d)
        candidates[frame_idx] = frame_dets
        log.debug("[Stage12-GDINO] frame %d: %d candidates", frame_idx, len(frame_dets))
        if cand_viz_dir is not None and boxes:
            from aero_eyes.utils import viz as vizmod
            vizmod.save_stage2_keyframe(frame_bgr, boxes, None, frame_idx, cand_viz_dir)

    _write_candidates_with_features(candidates, cand_path, placeholder_features=False)

    elapsed = time.time() - t0
    log.info("[Stage12-GDINO] %s done in %.1fs -> %s (%d keyframes)",
              sample_id, elapsed, cand_path, len(candidates))
    if cand_viz_dir is not None:
        log.info("[Stage12-GDINO] %s: raw candidate frames (%d with >=1 box) saved to %s",
                 sample_id, sum(1 for d in candidates.values() if d), cand_viz_dir)
    return cand_path


def main():
    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser(description="Run Stage 1+2+3 (Grounding DINO) for one sample")
    p.add_argument("--config", required=True)
    p.add_argument("--sample", required=True)
    p.add_argument("--set", action="append", default=[], help="cfg override k=v")
    args = p.parse_args()
    from aero_eyes.config import load_config
    cfg = load_config(args.config, args.set)
    run_stage123_gdino(cfg, args.sample)


if __name__ == "__main__":
    main()
