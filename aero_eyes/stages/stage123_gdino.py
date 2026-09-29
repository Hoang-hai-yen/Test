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

Reads:  cfg.data video (+ this sample's text prompt); also cfg.data
          refs_subdir's reference photos when color_postfilter.enabled OR
          cosine_rescore/online_fusion.enabled (all opt-in, off by default
          -- the base text-prompted path never touches them)
Writes: <work_dir>/<sample_id>/detections.json
        <work_dir>/<sample_id>/cascade_verification.jsonl (one line per
          Pass-1 box, only when stage123_gdino.cascade_verification.enabled
          -- see cascade_verify_boxes's own docstring)
        <work_dir>/<sample_id>/color_signature_gdino.npz (cached reference
          color histogram, only when color_postfilter.enabled)
        <work_dir>/<sample_id>/color_postfilter.jsonl (one line per box
          evaluated against the reference color signature, only when
          color_postfilter.enabled -- see apply_color_postfilter's own
          `records` argument)
        <work_dir>/<sample_id>/clip_tiebreak.jsonl (one line per box
          re-checked by CLIP because its Pass-1 score was tied with
          another box's, only when clip_tiebreak.enabled -- see
          clip_tiebreak_boxes's own docstring)
Viz:    <work_dir>/<sample_id>/viz/stage123_gdino/ (when save_visualizations=true)
        <work_dir>/<sample_id>/viz/stage123_gdino/cascade/ (Pass-1 vs Pass-2
          score per box, green=kept/red=dropped -- only when
          cascade_verification.enabled AND save_visualizations=true)
        <work_dir>/<sample_id>/viz/stage123_gdino/color/ (Hue+Sat/Value/
          effective similarity per box, green=kept/red=dropped -- only
          when color_postfilter.enabled AND save_visualizations=true)
"""
from __future__ import annotations

import argparse
import logging
import time
from collections import deque
from pathlib import Path

import numpy as np

from aero_eyes.types import Detection

log = logging.getLogger(__name__)


class GDinoOnlineAdaptiveThreshold:
    """Causal, streaming-compatible per-video threshold for Grounding DINO's
    own box score -- see GDinoOnlineAdaptiveThresholdConfig's own docstring
    (aero_eyes/config.py) for the full rationale. Mirrors
    aero_eyes.stages.stage3.OnlineAdaptiveThreshold's own contract:
    threshold_for_next_frame() decides using ONLY scores observed at
    STRICTLY EARLIER keyframes; observe() then feeds THIS keyframe's own
    raw scores in, called AFTER its own accept/reject decision was already
    made against the pre-update window."""

    def __init__(self, cfg_oat):
        self.cfg = cfg_oat
        self.history: deque = deque(maxlen=cfg_oat.window_size)

    def threshold_for_next_frame(self) -> float:
        if len(self.history) < self.cfg.min_samples:
            return self.cfg.abs_floor
        scores = np.array(self.history)
        raw_threshold = float(scores.mean() + self.cfg.z_score * scores.std())
        return max(self.cfg.abs_floor, raw_threshold)

    def observe(self, raw_scores: np.ndarray) -> None:
        self.history.extend(raw_scores.tolist())


class CausalRunningStats:
    """Causal, streaming-compatible running mean/std over a bounded window
    -- used by stage123_gdino.online_fusion (GDinoOnlineFusionConfig) to
    z-score TWO differently-scaled score streams (Grounding DINO's own box
    confidence, DINOv3 cosine similarity) against each OTHER's raw units
    before combining them.

    Rationale: a plain multiplicative/weighted combination of two raw
    scores is dominated by whichever one happens to have the larger
    ABSOLUTE margin between target and clutter in RAW units, regardless of
    which one is actually more informative -- e.g. if Grounding DINO scores
    0.65 (target) vs 0.30 (clutter) but DINOv3 cosine scores 0.55 vs 0.50
    (much smaller raw margin, common when the ground-to-aerial domain gap
    compresses cosine similarity), a raw product/sum is effectively decided
    by Grounding DINO alone -- DINOv3's identity signal gets diluted to
    near a constant multiplier, defeating the entire reason it was added
    (catching a same-category-but-wrong-identity confuser Grounding DINO's
    own category-level score cannot separate). Standardizing each stream to
    its OWN running mean/std first (z-score) means each contributes
    according to how many of ITS OWN standard deviations it deviates by --
    a small raw margin that is still statistically decisive (low noise)
    keeps its full weight; a small raw margin that is ALSO noisy (i.e.
    DINOv3 genuinely isn't discriminative in this domain) correctly
    collapses toward zero instead of silently doing nothing while looking
    like it's contributing.

    z_score(value) uses ONLY values observed via observe() so far (a keyframe
    must call z_score() for its OWN candidates BEFORE observe(), same
    causality contract as GDinoOnlineAdaptiveThreshold) -- returns 0.0
    during cold start (fewer than min_samples observed, or a degenerate
    zero-variance window), a neutral value that neither helps nor hurts a
    downstream weighted combination while there isn't enough history to
    trust mean/std yet.
    """

    def __init__(self, window_size: int, min_samples: int):
        self.min_samples = min_samples
        self.history: deque = deque(maxlen=window_size)

    def z_score(self, value: float) -> float:
        if len(self.history) < self.min_samples:
            return 0.0
        arr = np.array(self.history)
        std = float(arr.std())
        if std < 1e-8:
            return 0.0
        return float((value - arr.mean()) / std)

    def observe(self, values) -> None:
        self.history.extend(np.atleast_1d(values).tolist())


def online_fusion_detect_frame(
    frame_bgr, text_prompt: str, detector, extractor, prototype,
    gdino_stats: CausalRunningStats, cosine_stats: CausalRunningStats,
    fused_threshold_tracker: GDinoOnlineAdaptiveThreshold, cfg_fusion, cfg,
):
    """One keyframe of stage123_gdino.online_fusion -- see
    GDinoOnlineFusionConfig's own docstring for the full rationale. Reads
    every tracker to make THIS keyframe's decision, then updates them
    AFTER, so no keyframe's own candidates can influence their own
    decision (same causality contract as GDinoOnlineAdaptiveThreshold).
    Returns (boxes, threshold_used) -- the latter only for logging.

    gdino_stats observes EVERY raw candidate's score (cheap); cosine_stats
    only observes the pooled (cfg_fusion.candidate_pool) candidates that
    actually got embedded -- see the config's own docstring for why this
    asymmetry is accepted, not an oversight.
    """
    from aero_eyes.types import Box

    boxes_xyxy, gdino_scores = detector.raw_boxes_and_scores(
        frame_bgr, text_prompt, box_threshold=0.0, text_threshold=detector.text_threshold,
    )
    threshold = fused_threshold_tracker.threshold_for_next_frame()
    if len(boxes_xyxy) == 0:
        gdino_stats.observe(gdino_scores)
        return [], threshold

    pool_n = min(cfg_fusion.candidate_pool, len(boxes_xyxy))
    top_idx = np.argsort(gdino_scores)[::-1][:pool_n]
    pooled_boxes_xyxy = boxes_xyxy[top_idx]
    pooled_gdino_scores = gdino_scores[top_idx]

    pooled_boxes = [
        Box(x1=float(b[0]), y1=float(b[1]), x2=float(b[2]), y2=float(b[3])) for b in pooled_boxes_xyxy
    ]
    feats = extractor.extract_crops(
        frame_bgr, pooled_boxes,
        pad_ratio=cfg.stage2.candidate.feature_crop_pad, batch_size=cfg.runtime.batch_size,
    )
    cosine_sims = feats @ prototype

    fused_scores = np.array([
        cfg_fusion.weight * gdino_stats.z_score(float(g)) + (1.0 - cfg_fusion.weight) * cosine_stats.z_score(float(c))
        for g, c in zip(pooled_gdino_scores, cosine_sims)
    ])
    keep = fused_scores >= threshold
    result = detector.filter_boxes(pooled_boxes_xyxy[keep], fused_scores[keep], frame_bgr.shape[:2])

    # Update every tracker AFTER the decision -- causality.
    gdino_stats.observe(gdino_scores)
    cosine_stats.observe(cosine_sims)
    fused_threshold_tracker.observe(fused_scores)
    return result, threshold


def cascade_verify_boxes(
    detector, frame_bgr, text_prompt: str, boxes: list, cfg_cascade, records: list | None = None,
) -> list:
    """stage123_gdino.cascade_verification -- Pass-2 "zoom-in" re-check of
    already-decided Pass-1 boxes -- see GDinoCascadeVerificationConfig's own
    docstring (aero_eyes/config.py) for the full rationale. For each box,
    crops a padded region around it and re-runs Grounding DINO on JUST that
    crop with the same text prompt; keeps the box only if the crop pass's
    own best score is at least cfg_cascade.min_score_ratio * the box's
    Pass-1 score (and >= min_absolute_score). Does NOT touch a surviving
    box's own score -- pure accept/reject, so callers keep using each box's
    original Pass-1 score afterward.

    max_zoom caps the crop's effective magnification relative to Pass 1's
    own full-frame resize -- see GDinoCascadeVerificationConfig's own
    "IMPORTANT confound" docstring section for why this exists: without it,
    a tiny box's crop gets magnified far more by the model's fixed-input-
    size resize than the box was within the full frame, and that extra
    magnification is mostly interpolation, not genuine detail -- confirmed
    in practice to inflate FALSE positives' scores rather than collapse
    them. Implemented via crop_with_pad's own min_side argument: the crop's
    shorter side is floored at min(frame_h, frame_w) / max_zoom, overriding
    pad_ratio whenever the two disagree (max_zoom<=0 disables the cap).

    When `records` is given (a list the caller owns), appends one dict per
    INPUT box -- {x1,y1,x2,y2,pass1_score,pass2_score,ratio,zoom,kept} --
    for EVERY box, not just survivors, so a caller can inspect why a box
    was kept/dropped (e.g. write it to cascade_verification.jsonl, or feed
    it to aero_eyes.utils.viz.save_cascade_verification) instead of only
    seeing the post-filter box count."""
    from aero_eyes.utils.geometry import crop_with_pad

    h, w = frame_bgr.shape[:2]
    min_side = (min(h, w) / cfg_cascade.max_zoom) if cfg_cascade.max_zoom > 0 else 0.0

    kept = []
    for box in boxes:
        crop = crop_with_pad(frame_bgr, box, cfg_cascade.pad_ratio, min_side=min_side)
        _, pass2_scores = detector.raw_boxes_and_scores(
            crop, text_prompt, box_threshold=0.0, text_threshold=detector.text_threshold,
        )
        pass2_score = float(pass2_scores.max()) if len(pass2_scores) else 0.0
        keep = (
            pass2_score >= cfg_cascade.min_absolute_score
            and pass2_score >= cfg_cascade.min_score_ratio * box.score
        )
        if keep:
            kept.append(box)
        if records is not None:
            ch, cw = crop.shape[:2]
            records.append({
                "x1": box.x1, "y1": box.y1, "x2": box.x2, "y2": box.y2,
                "pass1_score": box.score, "pass2_score": pass2_score,
                "ratio": (pass2_score / box.score) if box.score > 0 else None,
                "zoom": (min(h, w) / min(ch, cw)) if min(ch, cw) > 0 else None,
                "kept": keep,
            })
    return kept


def clip_tiebreak_boxes(
    clip_extractor, text_feat: np.ndarray, frame_bgr, boxes: list, cfg_tie, records: list | None = None,
) -> list:
    """stage123_gdino.clip_tiebreak -- when several surviving boxes in a
    keyframe have Grounding-DINO scores too close together to trust the
    ranking, re-check just that TIED group against CLIP's own image-text
    similarity (an independent signal -- see GDinoClipTiebreakConfig's own
    docstring, aero_eyes/config.py). `text_feat` is the sample's text
    prompt, ALREADY encoded once via clip_extractor.encode_text (doesn't
    change across keyframes, so callers should encode it once, not per
    call). No-op (returns boxes unchanged) when fewer than 2 boxes are
    "tied" (within cfg_tie.margin of the group's own top score) -- a
    keyframe with a single clear winner never pays for a CLIP forward
    pass. Drops a tied candidate whose own CLIP similarity falls more than
    cfg_tie.drop_margin below the tied group's best CLIP similarity;
    non-tied candidates are returned untouched regardless of their own
    CLIP similarity.

    When `records` is given, appends one dict per box IN THE TIED GROUP --
    {x1,y1,x2,y2,pass1_score,clip_sim,kept} -- so a caller can inspect the
    tie-break itself. Boxes outside the tied group are not recorded (they
    were never re-checked)."""
    if len(boxes) < 2:
        return boxes
    top_score = max(b.score for b in boxes)
    tied = [b for b in boxes if top_score - b.score <= cfg_tie.margin]
    if len(tied) < 2:
        return boxes

    from aero_eyes.utils.geometry import crop_with_pad

    crops = [crop_with_pad(frame_bgr, b, cfg_tie.pad_ratio) for b in tied]
    img_feats = clip_extractor.extract(crops, batch_size=len(crops))
    clip_sims = img_feats @ text_feat
    best_sim = float(clip_sims.max())

    dropped_ids = set()
    for box, sim in zip(tied, clip_sims):
        keep = float(sim) >= best_sim - cfg_tie.drop_margin
        if not keep:
            dropped_ids.add(id(box))
        if records is not None:
            records.append({
                "x1": box.x1, "y1": box.y1, "x2": box.x2, "y2": box.y2,
                "pass1_score": box.score, "clip_sim": float(sim), "kept": keep,
            })
    return [b for b in boxes if id(b) not in dropped_ids]


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

    oat_cfg = cfg.stage123_gdino.online_adaptive_threshold
    fusion_cfg = cfg.stage123_gdino.online_fusion
    cascade_cfg = cfg.stage123_gdino.cascade_verification
    cpf_cfg = cfg.stage123_gdino.color_postfilter
    tie_cfg = cfg.stage123_gdino.clip_tiebreak
    online_threshold = GDinoOnlineAdaptiveThreshold(oat_cfg) if oat_cfg.enabled and not fusion_cfg.enabled else None
    if oat_cfg.enabled and fusion_cfg.enabled:
        log.warning(
            "[Stage123-GDINO] %s: both online_adaptive_threshold and online_fusion are enabled -- "
            "online_fusion's own threshold on the FUSED score decides accept/reject; "
            "online_adaptive_threshold's raw-GDINO-score threshold is unused.", sample_id,
        )

    extractor = prototype = None
    gdino_stats = cosine_stats = fused_threshold_tracker = None
    if fusion_cfg.enabled:
        from aero_eyes.models.features import build_feature_extractor
        from aero_eyes.stages.stage1 import run_stage1
        from aero_eyes.utils.io import read_prototype

        run_stage1(cfg, sample_id)
        extractor = build_feature_extractor(cfg)
        prototype, _, _ = read_prototype(work_dir / cfg.stage1.prototype.cache_name)
        gdino_stats = CausalRunningStats(fusion_cfg.window_size, fusion_cfg.min_samples)
        cosine_stats = CausalRunningStats(fusion_cfg.window_size, fusion_cfg.min_samples)
        fused_threshold_tracker = GDinoOnlineAdaptiveThreshold(fusion_cfg.threshold)

    # Cheap, pure-OpenCV color check for the same "same-shape/-category,
    # different-color" blind spot stage123_geco2.color_postfilter already
    # guards against -- see Stage123GDinoConfig.color_postfilter's own
    # docstring. Built ONCE (reference photos don't change per keyframe).
    color_sig = None
    color_segmenter = None
    if cpf_cfg.enabled:
        from aero_eyes.stages.stage123_geco2 import build_color_signature

        color_sig = build_color_signature(
            cfg, sample_id, work_dir, cpf_cfg, cfg.stage1.segmentation,
            cache_name="color_signature_gdino.npz", log_prefix="Stage123-GDINO",
        )
        # Dense (full-frame-context) segmentation for CANDIDATE crops --
        # see ColorPostfilterConfig.candidate_segmentation_enabled's own
        # docstring for the full rationale (docs/attribute_taxonomy_plan.md
        # SS9.1/SS9.2/SS9.8 -- background contamination was the largest
        # measured color error source). Reuses stage1.segmentation's own
        # configured model (same one build_color_signature just used for
        # the reference photos) -- a SECOND instance, since the reference
        # one has no per-frame state to share and dense segmentation needs
        # its own set_frame()/segment_box_cached() call pattern.
        if cpf_cfg.candidate_segmentation_enabled and cfg.stage1.segmentation.enabled:
            from aero_eyes.models.segmentation import build_segmenter

            color_segmenter = build_segmenter(cfg.stage1.segmentation, cfg)
        elif cpf_cfg.candidate_segmentation_enabled:
            log.warning(
                "[Stage123-GDINO] %s: color_postfilter.candidate_segmentation_enabled=true but "
                "stage1.segmentation.enabled=false -- dense candidate segmentation needs a "
                "segmenter model; falling back to candidate_inset_ratio for every candidate.",
                sample_id,
            )

    # CLIP's own image-text similarity, reserved for keyframes where
    # Grounding DINO's own box scores are too close together to trust the
    # ranking -- see GDinoClipTiebreakConfig's own docstring. Both the CLIP
    # instance and the text prompt's own embedding are built ONCE (the
    # prompt never changes across keyframes).
    clip_extractor = clip_text_feat = None
    if tie_cfg.enabled:
        from aero_eyes.models.features import CLIPFeatureExtractor

        clip_extractor = CLIPFeatureExtractor(variant=tie_cfg.variant, device=cfg.device())
        clip_text_feat = clip_extractor.encode_text([text_prompt])[0]

    detections: dict[int, list[Detection]] = {}
    cascade_records: list[dict] = []
    color_stats: list[tuple[float, float, float]] = []
    color_records: list[dict] = []
    clip_tiebreak_records: list[dict] = []
    for frame_idx, frame_bgr in frame_iterator(video_path):
        if frame_idx not in kf_indices:
            continue
        if fusion_cfg.enabled:
            boxes, threshold = online_fusion_detect_frame(
                frame_bgr, text_prompt, detector, extractor, prototype,
                gdino_stats, cosine_stats, fused_threshold_tracker, fusion_cfg, cfg,
            )
        elif online_threshold is not None:
            # ONE forward pass, box_threshold=0 (see everything) -- reused
            # both for THIS keyframe's decision (against the running
            # threshold, built from strictly earlier keyframes only) and to
            # extend the running window for FUTURE keyframes, in that
            # order, so no keyframe's own scores can influence its own
            # threshold. text_threshold stays at its own configured value.
            boxes_xyxy, scores = detector.raw_boxes_and_scores(
                frame_bgr, text_prompt, box_threshold=0.0, text_threshold=detector.text_threshold,
            )
            threshold = online_threshold.threshold_for_next_frame()
            keep = scores >= threshold
            boxes = detector.filter_boxes(boxes_xyxy[keep], scores[keep], frame_bgr.shape[:2])
            online_threshold.observe(scores)
        else:
            threshold = cfg.stage123_gdino.box_threshold
            boxes = detector.detect_frame(frame_bgr, text_prompt)
        if color_sig is not None and boxes:
            from aero_eyes.stages.stage123_geco2 import apply_color_postfilter

            pre_n = len(boxes)
            frame_color_records: list[dict] = []
            boxes = apply_color_postfilter(
                frame_bgr, boxes, color_sig, cpf_cfg, stats_out=color_stats, segmenter=color_segmenter,
                records=frame_color_records,
            )
            for r in frame_color_records:
                r["frame_idx"] = frame_idx
            color_records.extend(frame_color_records)
            if len(boxes) != pre_n:
                log.debug(
                    "[Stage123-GDINO] frame %d: color_postfilter dropped %d/%d box(es)",
                    frame_idx, pre_n - len(boxes), pre_n,
                )
            if save_viz:
                from aero_eyes.utils import viz as vizmod
                vizmod.save_color_postfilter(
                    frame_bgr, frame_color_records, frame_idx, viz_dir / "color",
                )
        if cascade_cfg.enabled and boxes:
            pre_n = len(boxes)
            frame_cascade_records: list[dict] = []
            boxes = cascade_verify_boxes(
                detector, frame_bgr, text_prompt, boxes, cascade_cfg, records=frame_cascade_records,
            )
            for r in frame_cascade_records:
                r["frame_idx"] = frame_idx
            cascade_records.extend(frame_cascade_records)
            if len(boxes) != pre_n:
                log.debug(
                    "[Stage123-GDINO] frame %d: cascade_verification dropped %d/%d box(es)",
                    frame_idx, pre_n - len(boxes), pre_n,
                )
            if save_viz:
                from aero_eyes.utils import viz as vizmod
                vizmod.save_cascade_verification(
                    frame_bgr, frame_cascade_records, frame_idx, viz_dir / "cascade",
                )
        if tie_cfg.enabled and len(boxes) >= 2:
            pre_n = len(boxes)
            frame_tie_records: list[dict] = []
            boxes = clip_tiebreak_boxes(
                clip_extractor, clip_text_feat, frame_bgr, boxes, tie_cfg, records=frame_tie_records,
            )
            for r in frame_tie_records:
                r["frame_idx"] = frame_idx
            clip_tiebreak_records.extend(frame_tie_records)
            if len(boxes) != pre_n:
                log.debug(
                    "[Stage123-GDINO] frame %d: clip_tiebreak dropped %d/%d tied box(es)",
                    frame_idx, pre_n - len(boxes), pre_n,
                )
        result_dets = [
            Detection(frame_idx=frame_idx, box=b, similarity=b.score, source="detect")
            for b in boxes
        ]
        detections[frame_idx] = result_dets
        log.debug("[Stage123-GDINO] frame %d: %d detections (threshold=%.3f)", frame_idx, len(result_dets), threshold)
        if save_viz:
            from aero_eyes.utils import viz as vizmod
            vizmod.save_stage3_detections(
                frame_bgr, [d.box for d in result_dets], [d.similarity for d in result_dets],
                frame_idx, viz_dir,
            )

    if cascade_cfg.enabled and cascade_records:
        import json

        cascade_log_path = work_dir / "cascade_verification.jsonl"
        with open(cascade_log_path, "w", encoding="utf-8") as f:
            for r in cascade_records:
                f.write(json.dumps(r) + "\n")
        n_dropped = sum(1 for r in cascade_records if not r["kept"])
        ratios = [r["ratio"] for r in cascade_records if r["ratio"] is not None]
        log.info(
            "[Stage123-GDINO] %s: cascade_verification dropped %d/%d box(es) total "
            "(pass2/pass1 ratio mean=%.3f, median=%.3f) -> %s",
            sample_id, n_dropped, len(cascade_records),
            float(np.mean(ratios)) if ratios else float("nan"),
            float(np.median(ratios)) if ratios else float("nan"),
            cascade_log_path,
        )

    if cpf_cfg.enabled and color_stats:
        arr = np.array(color_stats)  # columns: sim_hs, sim_v, effective_sim, overexposed_fraction
        log.info(
            "[Stage123-GDINO] %s: color_postfilter similarity stats over %d candidates "
            "(min_similarity=%.2f) -- sim_hs p10/p50/p90=%.3f/%.3f/%.3f, "
            "sim_v p10/p50/p90=%.3f/%.3f/%.3f, effective_sim p10/p50/p90=%.3f/%.3f/%.3f, "
            "%% below min_similarity=%.1f%%, %% overexposed(>=%.0f%% clipped)=%.1f%%",
            sample_id, len(color_stats), cpf_cfg.min_similarity,
            *np.percentile(arr[:, 0], [10, 50, 90]),
            *np.percentile(arr[:, 1], [10, 50, 90]),
            *np.percentile(arr[:, 2], [10, 50, 90]),
            100.0 * float((arr[:, 2] < cpf_cfg.min_similarity).mean()),
            cpf_cfg.overexposure_ramp_frac * 100.0,
            100.0 * float((arr[:, 3] >= cpf_cfg.overexposure_ramp_frac).mean()),
        )

    if cpf_cfg.enabled and color_records:
        import json

        color_log_path = work_dir / "color_postfilter.jsonl"
        with open(color_log_path, "w", encoding="utf-8") as f:
            for r in color_records:
                f.write(json.dumps(r) + "\n")
        log.info(
            "[Stage123-GDINO] %s: color_postfilter per-box records (%d box(es)) -> %s",
            sample_id, len(color_records), color_log_path,
        )

    if tie_cfg.enabled and clip_tiebreak_records:
        import json

        tie_log_path = work_dir / "clip_tiebreak.jsonl"
        with open(tie_log_path, "w", encoding="utf-8") as f:
            for r in clip_tiebreak_records:
                f.write(json.dumps(r) + "\n")
        n_dropped = sum(1 for r in clip_tiebreak_records if not r["kept"])
        log.info(
            "[Stage123-GDINO] %s: clip_tiebreak re-checked %d tied box(es) across the run, "
            "dropped %d -> %s",
            sample_id, len(clip_tiebreak_records), n_dropped, tie_log_path,
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

    # online_adaptive_threshold varies per keyframe -- no single scalar
    # applies to the whole video, so record None (informational only, same
    # convention run_stage123_geco2 uses for its own per-frame-relative path).
    recorded_threshold = (
        None if (online_threshold is not None or fusion_cfg.enabled) else cfg.stage123_gdino.box_threshold
    )
    write_detections(detections, det_path, threshold=recorded_threshold)

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

    if cfg.stage123_gdino.online_adaptive_threshold.enabled:
        log.warning(
            "[Stage12-GDINO] %s: stage123_gdino.online_adaptive_threshold.enabled=true has no "
            "effect here -- this path already loosens Grounding DINO's own cutoff via "
            "cosine_rescore.candidate_box_threshold and lets Stage 3's cosine matching decide "
            "the final threshold instead.", sample_id,
        )

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
