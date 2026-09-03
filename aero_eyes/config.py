"""Typed configuration schema + loader.

Loads configs/config.yaml into validated Pydantic models.
Supports CLI overrides:  --set stage2.proposal_model=fastsam_s
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Literal, Optional

import yaml
from pydantic import BaseModel, field_validator, model_validator


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------

class ProjectConfig(BaseModel):
    name: str = "aero_eyes"
    work_dir: str = "./runs/exp001"
    use_cache: bool = True
    seed: int = 42


class GTConfig(BaseModel):
    global_file: str = "annotations (1).json"
    box_format: Literal["xyxy", "xywh", "cxcywh"] = "xyxy"
    normalized: bool = False
    frame_index_base: int = 0
    absent_encoding: Literal["omit", "null_box", "empty_list"] = "omit"
    one_object_per_video: bool = True


class SubmissionConfig(BaseModel):
    path_name: str = "submission.json"
    box_format: Literal["xyxy", "xywh", "cxcywh"] = "xyxy"
    normalized: bool = False
    frame_index_base: int = 0
    absent_encoding: Literal["omit", "null_box", "empty_list"] = "omit"


class DataConfig(BaseModel):
    data_root: str = "./data"
    refs_subdir: str = "refs"
    video_glob: str = "*.mp4"
    num_references: int = 3
    gt: GTConfig = GTConfig()
    submission: SubmissionConfig = SubmissionConfig()


class RuntimeConfig(BaseModel):
    device: str = "auto"
    num_workers: int = 4
    batch_size: int = 16
    log_level: str = "INFO"
    save_visualizations: bool = True


class SegmentationConfig(BaseModel):
    enabled: bool = True
    model: str = "mobilesam"
    weights: Optional[str] = None
    fallback_if_missing: str = "passthrough"
    min_area_frac: float = 0.05
    max_area_frac: float = 0.95
    # Among SAM's 3 candidates, restrict the "prefer largest" pick to those
    # scoring within this ratio of the best score -- prevents a low-confidence
    # but merely area-plausible candidate (background bleeding into the mask)
    # from winning just for being big. 1.0 = only the single best-scoring
    # candidate is eligible (falls back to old highest-score behavior).
    score_ratio_floor: float = 0.85
    # Reject a candidate whose mask touches the true image border more than
    # this fraction of edge pixels -- the box prompt is inset 5% from the
    # edges, so a correctly-segmented subject essentially never reaches the
    # real border; a background plane (ground, wall, sky) commonly does.
    max_border_touch_frac: float = 0.02
    # Center-point prompt (in addition to the box prompt) assumes the
    # geometric center pixel is foreground -- breaks down for ring/donut-
    # shaped objects (e.g. a life ring) whose center is a HOLLOW interior
    # (background), which can bias SAM's mask proposals toward confused/
    # leaky boundaries (confirmed empirically: life-ring reference photos
    # showed both border-touching passthrough failures AND loose/over-
    # inclusive masks on the candidate that WAS accepted). Set false to
    # prompt with the box alone for object shapes like this.
    use_point_prompt: bool = True
    # What to do with the non-mask (background) region of a reference image:
    #   mean_fill -- flat mean-color fill (old/default behavior). Cheapest,
    #                but a large flat, textureless region is far outside what
    #                the backbone (pretrained on natural photos) ever saw --
    #                empirically this can push the exemplar token into an
    #                unnatural part of feature space.
    #   keep_real -- leave the reference photo's real background untouched.
    #                Only the tight mask bbox is used to pick the RoI-Align
    #                region, so background never gets pooled into the token,
    #                but the backbone still sees a natural image overall.
    #   blur      -- strong Gaussian blur of the real background: keeps
    #                natural color/texture statistics but discards fine
    #                detail that could otherwise cause spurious background
    #                matches.
    background_mode: Literal["mean_fill", "keep_real", "blur"] = "mean_fill"
    blur_sigma: float = 25.0  # Gaussian sigma (px) used when background_mode == "blur"
    # When the segmenter's own mask area ratio (mask pixels / total pixels)
    # falls outside [min_valid_mask_ratio, max_valid_mask_ratio], replace it
    # with a safe rectangular center-crop mask (center_fallback_ratio of the
    # frame -- see aero_eyes.utils.geometry.center_box_mask) instead of
    # passing the implausible mask straight through. An almost-empty mask is
    # likely pure segmentation noise; an almost-full mask is effectively a
    # whole-image passthrough that lets background bleed straight into the
    # exemplar/prototype and corrupts matching downstream. Reference photos
    # are always close-up shots with the target centered, so a center-crop
    # is a reasonable stand-in for "the object" in either failure case.
    # Off by default -- does not change existing runs unless opted in.
    center_crop_fallback: bool = False
    min_valid_mask_ratio: float = 0.03
    max_valid_mask_ratio: float = 0.92
    center_fallback_ratio: float = 0.75


class FeatureExtractorConfig(BaseModel):
    model: Literal["dinov2", "dinov3", "clip", "siglip", "ensemble"] = "dinov2"
    dinov2_variant: Literal["vits14", "vitb14", "vitl14", "vitg14"] = "vitb14"
    # DINOv3 weights are gated on HuggingFace (facebook/dinov3-*) -- request
    # access on the model page and set HF_TOKEN before using this.
    dinov3_variant: Literal["vits16", "vitb16", "vitl16"] = "vitb16"
    clip_variant: str = "vit-b/32"   # "vit-b/32" (512-d) or "vit-l/14" (768-d)
    # SigLIP: open access (no gating), vision-only encoder.
    siglip_variant: Literal["base", "large", "so400m"] = "base"
    weights: Optional[str] = None
    image_size: int = 224


class PrototypeConfig(BaseModel):
    fusion: Literal["mean", "max", "concat_then_pca"] = "mean"
    l2_normalize: bool = True
    cache_name: str = "prototype.npz"


class AerialSimConfig(BaseModel):
    """Degrade reference images to look more like a distant aerial capture
    before feature extraction, to shrink the domain gap between crisp
    close-up references and the drone's actual view of the object."""
    enabled: bool = False
    downscale_factor: float = 1.0  # e.g. 0.25 = shrink to 1/4 then upscale back (simulate distance)
    blur_ksize: int = 0  # Gaussian blur kernel size in px, 0 = off (simulate motion/optical blur)


class DinoDomainCalibrationConfig(BaseModel):
    """DINOv2 analog of stage123_geco2.domain_calibration: shifts the fused
    `prototype` (and each per-ref vector, when multi_reference_embedding is
    active) toward the mean DINOv2 embedding of several RAW frames sampled
    from the query video -- an estimate of this video's own general scene/
    lighting domain (color temperature, exposure, compression, motion blur),
    independent of where the target object actually is in those frames.

    Different from -- and complementary to -- stage3.dynamic_prototype:
    that mechanism shifts the prototype toward high-confidence CANDIDATE
    CROPS (object-focused, but only available/reliable once matching has
    already found some plausible hits). This one shifts toward whole video
    frames (background-heavy, but available immediately, before any
    matching happens, and captures broad scene-level lighting/exposure
    differences a handful of object crops might not fully represent). Can
    be used together with dynamic_prototype or on its own.

    Disabled by default -- prototype.npz is built exactly as before this
    option existed unless explicitly turned on.
    """
    enabled: bool = False
    num_sample_frames: int = 5
    # 0 = no change, 1 = appearance fully replaced by the video's own mean
    # embedding (almost certainly too aggressive -- the object's own
    # identity would be washed out by generic background/scene content).
    strength: float = 0.3


class Stage1Config(BaseModel):
    segmentation: SegmentationConfig = SegmentationConfig()
    feature_extractor: FeatureExtractorConfig = FeatureExtractorConfig()
    prototype: PrototypeConfig = PrototypeConfig()
    aerial_sim: AerialSimConfig = AerialSimConfig()
    domain_calibration: DinoDomainCalibrationConfig = DinoDomainCalibrationConfig()
    # Crop each reference image to its MobileSAM tight mask box (expanded by
    # crop_context_margin) BEFORE resizing to feature_extractor.image_size
    # -- keeps 100% real pixels, no masking/fill, just a tighter field of
    # view than the whole reference photo. Since the resize always
    # renormalizes the (now smaller) image's longer side back up to
    # image_size, the object ends up occupying a LARGER fraction of the
    # final canvas than it would from the whole uncropped photo. Mirrors
    # stage123_geco2.crop_to_object -- see
    # aero_eyes/utils/geometry.py::crop_to_object. Requires
    # segmentation.enabled (needs the tight mask box). Off by default --
    # does not change existing runs unless opted in.
    crop_to_object: bool = False
    crop_context_margin: float = 0.5


class SAHIConfig(BaseModel):
    use_sahi: bool = True
    tile: list[int] = [640, 640]
    overlap: float = 0.25


class Yolov11nConfig(BaseModel):
    weights: str = "yolo11n.pt"
    conf: float = 0.05
    iou: float = 0.5
    max_det: int = 300
    classes: Optional[Any] = None


class FastSamSConfig(BaseModel):
    weights: str = "FastSAM-s.pt"
    conf: float = 0.2
    iou: float = 0.7
    imgsz: int = 640


class CandidateConfig(BaseModel):
    min_box_area: float = 16.0
    max_candidates_per_keyframe: int = 400
    feature_crop_pad: float = 0.10


class Stage2Config(BaseModel):
    keyframe_interval: int = 8
    sahi: SAHIConfig = SAHIConfig()
    proposal_model: str = "yolov11n"
    yolov11n: Yolov11nConfig = Yolov11nConfig()
    fastsam_s: FastSamSConfig = FastSamSConfig()
    candidate: CandidateConfig = CandidateConfig()

    @field_validator("proposal_model")
    @classmethod
    def check_proposal_model(cls, v: str) -> str:
        allowed = {"yolov11n", "fastsam_s"}
        if v not in allowed:
            raise ValueError(
                f"stage2.proposal_model must be one of {allowed}; got '{v}'. "
                "YOLOv8 is explicitly NOT allowed."
            )
        return v


class CalibrateConfig(BaseModel):
    enabled: bool = False
    target_metric: str = "st_iou"
    search_range: list[float] = [0.40, 0.75]
    steps: int = 8


class DynamicPrototypeConfig(BaseModel):
    """Optional 2-pass matching: after the initial similarity pass, pick the
    candidates scoring above an ADAPTIVE (percentile-based) threshold of
    THIS sample's own score distribution -- not a fixed cutoff, since a
    fixed one only fires for "easy" targets whose scores are already high
    (a "hard" target's scores may never clear a fixed bar, so the mechanism
    never activates for it) -- and blend their mean feature into the
    prototype, then re-score. Repeated for `rounds` passes so the prototype
    drifts toward this specific video's own appearance of the target.
    Disabled by default: plain single-pass cosine matching against the
    Stage 1 prototype, unchanged from before this option existed.
    """
    enabled: bool = False
    rounds: int = 2
    alpha: float = 0.3  # blend weight of the new high-confidence mean feature into the prototype
    high_conf_percentile: float = 90.0  # percentile of THIS sample's score distribution
    high_conf_abs_floor: float = 0.15   # absolute floor, so a low-scoring sample doesn't update from noise
    min_support: int = 2  # minimum high-confidence candidates required to update; else stop early


class Stage3Config(BaseModel):
    similarity: Literal["cosine", "l1", "l2"] = "cosine"
    match_threshold: float = 0.55
    nms_iou: float = 0.5
    topk_per_keyframe: int = 5
    # When cross-domain gap is large, absolute threshold fails.
    # global_topk: cap on how many candidates to keep globally (applied AFTER filtering).
    # None = no cap.  Recommended: 30–100 when domain gap is large.
    global_topk: Optional[int] = None
    # adaptive_threshold: compute per-video threshold as mean + z_score * std.
    # Robust to domain gap — adapts to the actual similarity distribution.
    # Replaces match_threshold when enabled.
    adaptive_threshold: bool = False
    adaptive_z_score: float = 2.0   # higher = fewer FP, lower = more recall (see configs/config.yaml for the sweep)
    adaptive_min_floor: float = 0.05  # hard floor: never accept sim below this
    calibrate: CalibrateConfig = CalibrateConfig()
    dynamic_prototype: DynamicPrototypeConfig = DynamicPrototypeConfig()


class BuiltinTrackerConfig(BaseModel):
    algorithm: Literal["csrt", "kcf", "mosse"] = "csrt"


class LiteTrackConfig(BaseModel):
    onnx_path: Optional[str] = None
    input_size: int = 256


class DetectionConfirmationConfig(BaseModel):
    """Guards against a SINGLE spurious detection getting amplified into a
    long false track: a detector "hit" (whether the initial keyframe scan
    or a re-detect after track loss) is not trusted until `required_hits`
    consecutive hits agree spatially (IoU >= iou_threshold). Only then does
    Stage 4 initialize/re-initialize the tracker from it.

    Matters most for GeCo2 (pipeline.detector=geco2): its per-frame score
    is threshold RELATIVE to that frame's own max, so it structurally
    always returns >=1 box -- on data where the score doesn't separate
    "target present" from "target absent" (see
    scripts/check_geco2_score_separation.py), a single stray keyframe hit
    can spawn a tracker.builtin (CSRT) track that survives up to
    max_track_age frames, which stage5.min_tube_length (typically 2) is far
    too small to catch since the false track isn't short. Requiring N
    agreeing hits before trusting a detection attacks that amplification
    directly, independent of whether the detector's raw score is separable.

    Applies identically to the legacy and geco2 detectors (Stage 4's
    tracking loop is shared).
    """
    enabled: bool = False
    required_hits: int = 2
    iou_threshold: float = 0.3


class Stage4Config(BaseModel):
    tracker: str = "builtin"
    builtin: BuiltinTrackerConfig = BuiltinTrackerConfig()
    litetrack: LiteTrackConfig = LiteTrackConfig()
    tracker_conf_threshold: float = 0.40
    max_track_age: int = 30
    confirm_detections: DetectionConfirmationConfig = DetectionConfirmationConfig()
    # Every verify_interval frames of ACTIVE tracking (builtin/litetrack,
    # not tracker=none), re-embed the currently-tracked crop with DINOv2 and
    # cross-check it against the prototype -- the real correctness check
    # BuiltinTracker's own confidence cannot provide (it returns a fixed
    # 0.9 placeholder on any OpenCV-reported success; see trackers.py). If
    # the re-embedded crop no longer matches, forces the same re-detect path
    # used when confidence/age fail, even though OpenCV still reports
    # tracking as nominally successful -- catches silent drift instead of
    # letting it persist for the full max_track_age.
    #
    # Independent of confirm_detections above -- that guards against
    # trusting a single SPURIOUS detection before a track ever starts;
    # this guards against a track that started fine but DRIFTED after the
    # fact. Both can be enabled together.
    #
    # Needs a DINOv2 prototype.npz to re-embed against: always available on
    # the legacy pipeline; on pipeline.detector=geco2 only when
    # stage123_geco2.cosine_rescore.enabled built one (see stage1.run_stage1)
    # -- otherwise this silently has no effect (logged once) rather than
    # erroring, since plain GeCo2 has no DINOv2 embedding space to check
    # against.
    #
    # 0 (default) = disabled -- reproduces the exact original tracking
    # logic (confidence/age only), unchanged.
    verify_interval: int = 0

    @field_validator("tracker")
    @classmethod
    def check_tracker(cls, v: str) -> str:
        allowed = {"builtin", "litetrack", "none"}
        if v not in allowed:
            raise ValueError(f"stage4.tracker must be one of {allowed}; got '{v}'.")
        return v


class TemporalSmoothingConfig(BaseModel):
    enabled: bool = True
    method: Literal["ema", "none"] = "ema"
    ema_alpha: float = 0.6


class Stage5Config(BaseModel):
    temporal_smoothing: TemporalSmoothingConfig = TemporalSmoothingConfig()
    min_tube_length: int = 2
    fill_short_gaps: int = 3


class SyntheticViewpointAugConfig(BaseModel):
    enabled: bool = True
    method: Literal["homography", "perspective_warp"] = "homography"
    num_synth_views: int = 6
    pitch_range_deg: list[float] = [40.0, 85.0]
    fold_into_prototype: bool = True


class DomainPrompterConfig(BaseModel):
    enabled: bool = True
    num_prompts: int = 4
    strength: float = 0.3


class CheapBoostersConfig(BaseModel):
    multi_scale_scan: bool = True
    scales: list[float] = [0.75, 1.0, 1.5]
    tuned_nms: bool = True
    multi_reference_embedding: bool = True
    # How per-reference-image similarity scores are pooled into one score,
    # when multi_reference_embedding is active (see stage3.py's use_multi_ref).
    #   mean (default, unchanged from before this option existed) -- a
    #     candidate that matches ONE ref very well but the other two poorly
    #     (e.g. the object was photographed from 3 different angles, and
    #     this candidate's viewing angle only resembles 1 of them) gets its
    #     good score DILUTED by the two weak ones.
    #   max -- take the single best-matching ref's score per candidate
    #     instead of averaging all of them. Keeps a genuinely good match
    #     from a single well-aligned reference view from being dragged down
    #     by refs shot from a different angle/lighting than this candidate.
    multi_ref_pooling: Literal["mean", "max"] = "mean"


class MaxAccuracyConfig(BaseModel):
    synthetic_viewpoint_aug: SyntheticViewpointAugConfig = SyntheticViewpointAugConfig()
    domain_prompter: DomainPrompterConfig = DomainPrompterConfig()


class AccuracyConfig(BaseModel):
    mode: Literal["baseline", "cheap_boosters", "max_accuracy"] = "baseline"
    cheap_boosters: CheapBoostersConfig = CheapBoostersConfig()
    max_accuracy: MaxAccuracyConfig = MaxAccuracyConfig()


class EvalConfig(BaseModel):
    metric: str = "st_iou"
    spatial_iou_type: str = "standard"
    report_per_video: bool = True


class PipelineConfig(BaseModel):
    # "legacy"  = Stage1 (DINOv2 prototype) -> Stage2 (YOLO/FastSAM proposals)
    #             -> Stage3 (cosine matching), as three separate artifacts.
    # "geco2"   = single merged stage (stage123_geco2.py) using the vendored
    #             GECO2/ few-shot exemplar detector in place of all three.
    #             Stage 4/5 are unchanged either way.
    detector: Literal["legacy", "geco2"] = "legacy"


class ScaleCalibrationConfig(BaseModel):
    """Fixes the ground-to-aerial SIZE mismatch that ref_downscale_factor
    cannot fix: GECO2/utils/data.py::resize_and_pad always re-normalizes the
    WHOLE image's longer side back to stage123_geco2.image_size, so any
    uniform pre-shrink of the reference photo (what ref_downscale_factor
    does) gets exactly cancelled out by that re-normalization -- the
    object's box-to-photo ratio is intrinsic to how the photo was framed
    and is scale-invariant under uniform resize. The only lever that
    actually changes that ratio is changing how much the object fills a
    canvas (crop tighter/looser) -- see
    aero_eyes/stages/stage123_geco2.py::_build_scale_calibrated_canvas,
    which builds a synthetic canvas sized so the object occupies the same
    fraction of the canvas as it's expected to occupy in the query video
    frame after ITS OWN resize_and_pad.
    """
    enabled: bool = False
    # Expected apparent size(s) [width, height] in pixels of the object AS
    # IT APPEARS IN THE RAW VIDEO FRAME (before any resize/pad) -- e.g.
    # estimated from flight altitude/GSD, or eyeballed on a sample frame.
    # Required when enabled=true; there is no safe default (a wrong value
    # actively hurts -- it recreates the same kind of scale mismatch this
    # feature exists to remove, just in a different direction).
    #
    # Accepts EITHER a single [w, h] pair (shorthand, normalized to [[w, h]]
    # below -- exactly the original single-scale behavior) OR a list of
    # [w, h] pairs, e.g. [[18, 15], [26, 22], [34, 29]], to hedge against
    # uncertainty in the true apparent scale (altitude/zoom varies shot to
    # shot, or the estimate is a rough eyeball guess). See multi_scale_mode
    # below for how more than one scale is actually consumed.
    expected_object_px: Optional[list[list[float]]] = None
    # "first" (default): only expected_object_px[0] is used -- exactly the
    #   original single-canvas-per-reference-image behavior, unaffected by
    #   any extra scales listed.
    # "all": build ONE calibrated canvas PER (reference image, scale) pair
    #   and feed every one of them into GeCo2Detector.encode_exemplars as
    #   its own exemplar entry -- each contributes its own appearance token
    #   (RoI-Align pooled from that canvas) and, when use_shape_token=true,
    #   its own shape token (that scale's own calibrated box (w,h) -- shape
    #   tokens naturally come out different per scale with no extra code,
    #   since shape_or_objectness is computed from each canvas's own box).
    #   Total exemplar count becomes num_refs * num_scales; all of them are
    #   concatenated into the same K/V sequence cross-attention already
    #   treats as a flat set, so nothing downstream (calibrate_prototype,
    #   Stage 4 re-detect, etc.) needs to change. Costs num_refs*num_scales
    #   backbone forward passes instead of num_refs.
    multi_scale_mode: Literal["first", "all"] = "first"
    # Extra padding kept around the tight mask box, as a fraction of the
    # object's own size, before that (object+margin) footprint is calibrated
    # to match expected_object_px -- gives the model a bit of surrounding
    # context instead of the object filling the canvas edge-to-edge.
    context_margin: float = 0.5

    @field_validator("expected_object_px", mode="before")
    @classmethod
    def _normalize_expected_object_px(cls, v: Any) -> Any:
        """Accept a flat [w, h] pair (the original single-scale shape) as
        shorthand for [[w, h]] -- keeps existing configs setting
        expected_object_px: [22, 18] working unchanged."""
        if (
            v is not None
            and len(v) == 2
            and all(isinstance(x, (int, float)) for x in v)
        ):
            return [v]
        return v

    @field_validator("expected_object_px")
    @classmethod
    def check_expected_object_px(cls, v: Optional[list[list[float]]]) -> Optional[list[list[float]]]:
        if v is None:
            return v
        if len(v) == 0:
            raise ValueError("scale_calibration.expected_object_px must have at least one [width, height] entry")
        for entry in v:
            if len(entry) != 2:
                raise ValueError("scale_calibration.expected_object_px entries must each be [width, height]")
        return v


class DomainCalibrationConfig(BaseModel):
    """Shifts exemplar APPEARANCE tokens (not shape tokens) toward the
    feature-space region the backbone actually produces for this video's
    own frames. Even with correct scale and a natural background
    (background_mode != mean_fill), running the backbone on an isolated
    reference photo vs. on a real video frame are two independent forward
    passes with two different self-attention contexts -- see
    GeCo2Detector.estimate_domain_shift / calibrate_prototype. This
    computes the video's own mean token (from a few sampled frames,
    unpaired/unlabeled) and nudges each exemplar's appearance token toward
    it, blended by `strength`.
    """
    enabled: bool = False
    num_sample_frames: int = 5
    strength: float = 1.0  # 0 = no change, 1 = fully match the video's own mean token


class ColorPostfilterConfig(BaseModel):
    """Cheap post-detection filter for GeCo2's blind spot: it's a few-shot
    COUNTING model matching shape/texture via its vision backbone -- it has
    no explicit color signal, so same-silhouette-different-color objects
    are a common false positive. Compares each candidate box's color
    against the reference object's own color signature (computed once from
    the MobileSAM-masked reference photos, cached to color_signature.npz).
    Pure OpenCV, no extra model, no finetuning -- see aero_eyes/utils/
    color.py and stage123_geco2.py::build_color_signature /
    apply_color_postfilter.

    Falls back to the WHOLE reference photo's color (diluted by
    background) if segmentation.enabled=false -- still works, just less
    precise; a warning is logged when that happens.

    TWO signals are compared and blended by color_confidence (see
    saturation_value_confidence in aero_eyes/utils/color.py):
      - Hue+Saturation histogram (brightness/value ignored -- robust to
        lighting differences between the reference photo and the video
        frame) -- reliable for colorful objects, but Hue is
        unstable/noisy for near-achromatic (black/white/gray) ones.
      - Value/brightness histogram -- the ONE property that reliably
        separates black from white/gray, exactly where Hue+Saturation
        carries no signal. More lighting-sensitive than Hue+Saturation,
        so it's down-weighted (not solely relied on) for colorful objects.
    color_confidence (0=achromatic, 1=colorful) linearly blends the two:
    effective_similarity = confidence*sim_hue_sat + (1-confidence)*sim_value.

    EMPIRICALLY CONFIRMED (not just theoretical), in this order:
    (1) a black-ish reference object (mean saturation=60.1, value=121.3)
    saw ST-IoU DROP even with a correctly-sized histogram (0.4264 ->
    0.3902) using Hue+Saturation alone; (2) blending in Value at low
    confidence was added specifically because, even after that fix, the
    detector still visibly confused a similarly-shaped WHITE object in the
    output video -- Hue+Saturation structurally cannot catch that (both
    black and white can have arbitrary/unstable Hue), only Value can.
    """
    enabled: bool = False
    # Deliberately COARSE (not the ~30x32 "whole photo" tutorial default):
    # candidate crops here can be as small as ~20x10px (~200 pixels) --
    # empirically confirmed a 30x32=960-bin histogram from that few pixels
    # is severely under-sampled, so even a GENUINELY correct-color match
    # only scored ~0.49 similarity (barely above min_similarity's default
    # floor, easily pushed below it by real-world noise) while 12x8=96
    # bins scored ~0.83 on the identical case -- with NO loss of
    # discrimination against a truly different color (both still scored
    # ~0.0). Re-validate with your own crop sizes if you raise these.
    hue_bins: int = 12
    sat_bins: int = 8
    # Bins for the separate Value/brightness histogram (see class
    # docstring) -- kept coarse for the same small-crop-sample-size reason
    # as hue_bins/sat_bins above.
    value_bins: int = 8
    metric: Literal["bhattacharyya", "correlation"] = "bhattacharyya"
    # Candidates scoring below this similarity (roughly 0..1, higher = more
    # similar) against EVERY reference photo are dropped outright. This is
    # the ONLY mechanism that should filter by color -- see `reweight`
    # below for why letting color CHANGE surviving candidates' scores is
    # dangerous. 0.0 = color_postfilter becomes a pure no-op.
    min_similarity: float = 0.3
    # DEFAULT FALSE -- EMPIRICALLY CONFIRMED HARMFUL, not just theoretical.
    # If true, surviving candidates' scores are multiplied by their color
    # similarity. This sounds like a harmless "soft penalty", but
    # aero_eyes/stages/stage4.py picks the keyframe candidate to
    # (re)initialize the tracker from via `max(dets, key=lambda d:
    # d.similarity)` -- i.e. it re-runs argmax over EXACTLY this score.
    # Reweighting by a noisy signal (Value/brightness is lighting-sensitive
    # -- see the class docstring) can flip WHICH candidate wins that argmax
    # even when zero candidates are ever hard-dropped, silently swapping in
    # a wrong box at a keyframe that then persists via tracking for up to
    # max_track_age frames. Confirmed on real data: with reweight=true,
    # min_similarity=0.0 (no hard-drop at all, i.e. IDENTICAL candidate
    # sets survive at every keyframe as min_similarity=0.3) produced the
    # exact same degraded ST-IoU as min_similarity=0.3 -- proving 100% of
    # the harm came from the score multiplication itself, not from
    # anything being removed. Leave false; only min_similarity above
    # should ever change which candidates survive.
    reweight: bool = False
    # Color-trust ramp: below min_ref_saturation, confidence=0 (color
    # signal fully suppressed -- catches near-WHITE/gray objects); at/above
    # saturation_full_confidence, confidence=1 (full effect); linearly
    # interpolated in between. mean saturation = 0-255, averaged over the
    # masked object pixels across all 3 ref photos.
    #
    # 65.0 (not the naive-looking 40.0): saturation=(max-min)/max is a
    # RATIO, so for genuinely dark/near-black pixels small absolute sensor
    # noise gets amplified into a spuriously HIGH saturation reading -- an
    # actual black reference object in this codebase's own test data
    # measured mean_saturation=60.1, which sat ABOVE a 40.0 floor and so
    # still leaked ~22% confidence onto Hue+Saturation (a channel this
    # class's own docstring calls unreliable for dark objects) instead of
    # relying on Value as intended. 65.0 sits just above that observed
    # noise floor so a genuinely-black reference reliably lands at
    # confidence=0 (Value only); re-check the real mean_saturation logged
    # by build_color_signature for YOUR reference object if black/white
    # discrimination still looks off, and raise further if it's still
    # landing above this floor.
    min_ref_saturation: float = 65.0
    saturation_full_confidence: float = 130.0
    # Same ramp, triggered by mean HSV value/brightness -- catches
    # near-BLACK objects. Needed because saturation=(max-min)/max is a
    # RATIO: for dark pixels, small absolute sensor noise gets amplified
    # into a spuriously HIGH saturation reading, so the saturation ramp
    # alone can under-react to dark objects (confirmed: a synthetic
    # near-black pixel with only +-4/255 noise computed mean saturation
    # ~50, above min_ref_saturation's default). Overall confidence used is
    # the MINIMUM of the saturation ramp and this value ramp.
    min_ref_value: float = 50.0
    value_full_confidence: float = 160.0
    # Shrink each CANDIDATE box inward by this fraction of its own
    # width/height (on each side) before sampling its color histogram --
    # e.g. 0.15 keeps only the middle 70%x70% of the box. A rectangular
    # detector box's edges/corners commonly include background the
    # (usually non-rectangular) real object doesn't cover; unlike the
    # reference photos (masked by MobileSAM to pure object pixels, see
    # build_color_signature), a video candidate box has no per-candidate
    # segmentation to strip that background out, so its color histogram
    # gets diluted by whatever's at the edges. This hurts achromatic
    # (black/white) discrimination specifically MORE than chromatic colors:
    # background rarely shares a colorful object's distinct HUE, but
    # commonly sits at a MID brightness that pulls both a black and a white
    # candidate's Value histogram toward each other. 0.0 = no-op (samples
    # the whole box, original behavior).
    candidate_inset_ratio: float = 0.15


class Geco2CosineRescoreConfig(BaseModel):
    """Optional extra matching pass inserted between GeCo2 detection and
    Stage 4 tracking: instead of GeCo2's own score alone deciding
    detections.json (score_threshold_ratio/score_threshold_abs/nms_iou/
    topk_per_keyframe above), GeCo2 first produces a WIDER per-keyframe
    candidate set (this config's own looser threshold/topk below), each
    candidate crop is embedded with a separate DINOv2 prototype (built the
    same way legacy stage1.py does, from the same 3 reference images), and
    aero_eyes.stages.stage3.run_stage3's cosine matching (optionally with
    stage3.dynamic_prototype) does the final threshold/NMS/top-K filtering
    that writes detections.json. GeCo2's cross-attention score and DINOv2's
    cosine similarity are independent signals from different backbones, so
    this is a genuine second opinion rather than re-deriving what GeCo2
    already scored.

    Disabled by default: run_stage123_geco2 alone decides detections.json
    exactly as before this option existed (original behavior, unchanged).
    """
    enabled: bool = False
    # Looser than stage123_geco2.score_threshold_ratio/topk_per_keyframe --
    # this stage only needs to not throw away the true positive; Stage 3's
    # cosine matching (+ dynamic_prototype, if enabled) does the real cut.
    candidate_score_threshold_ratio: float = 0.15
    candidate_topk_per_keyframe: int = 15


class GlobalAdaptiveThresholdConfig(BaseModel):
    """Optional alternative to GeCo2's default per-frame-relative decision
    (score_threshold_ratio/score_threshold_abs above): a keyframe with no
    real target still has a "best" box by construction (score is thresholded
    RELATIVE to that frame's own max), so per-frame-relative thresholding
    structurally always keeps something on every frame -- across a whole
    video that means stray boxes on every frame that has no real target.

    Instead: Pass 1 pools RAW (unfiltered) per-location scores across EVERY
    keyframe in the whole video first; Pass 2 computes ONE global threshold
    = max(abs_floor, mean + z_score*std) over that pooled distribution
    (capped at the video's own observed max so the statistical estimate
    never rejects the single best real score), then applies it to every
    keyframe -- same style of fix as stage3.adaptive_threshold, applied to
    GeCo2's own score instead of DINOv2 cosine similarity.

    Costs a second pass over the video's keyframes, but reuses each frame's
    already-computed raw backbone output from Pass 1 (see
    GeCo2Detector.forward_scores/filter_boxes_by_threshold) -- does NOT
    double the number of GeCo2 backbone forward passes.

    Disabled by default -- score_threshold_ratio/score_threshold_abs decide
    detections.json exactly as before this option existed. Only applies to
    run_stage123_geco2 (the default geco2 path); has no effect when
    stage123_geco2.cosine_rescore.enabled (that path's own Stage 3 cosine
    matching decides the final threshold instead).
    """
    enabled: bool = False
    z_score: float = 1.0
    abs_floor: float = 0.15


class Stage123Geco2Config(BaseModel):
    """Only used when pipeline.detector == 'geco2'. Requires the vendored
    GECO2/ repo's own dependencies (hydra-core, omegaconf, its sam2 package)
    installed, and pretrained weights downloaded -- see GECO2/README.md.
    """
    repo_path: str = "./GECO2"
    weights_path: str = "./GECO2/CNTQG_multitrain_ca44.pth"
    # Same MobileSAM foreground masking as stage1.segmentation (background
    # filled with the ref image's own mean color) -- applied before
    # ref_downscale_factor. Reuses the same SegmentationConfig shape/defaults.
    segmentation: SegmentationConfig = SegmentationConfig()
    image_size: int = 1024
    emb_dim: int = 256
    kernel_dim: int = 3
    reduction: int = 16
    keyframe_interval: int = 8
    # Per-frame relative threshold: keep detections with score >
    # box_v.max() * score_threshold_ratio (GeCo2's own score scale is not
    # comparable across frames, so this can't be a fixed absolute cutoff
    # like stage3.match_threshold -- see GECO2/demo_gradio.py's threshold
    # slider, default 0.33, for the reference implementation this mirrors).
    score_threshold_ratio: float = 0.33
    # Absolute floor on a frame's OWN max score (box_v.max()), independent of
    # score_threshold_ratio above -- GeCo2 was trained/evaluated on FSC147
    # where every image guarantees >=1 instance of the counted class, so the
    # relative-only ratio structurally cannot express "target absent this
    # frame" (it always keeps >=1 box whenever max score > 0). If the frame's
    # peak score doesn't clear this floor, detect_frame() returns no boxes
    # for that frame at all. 0.0 = disabled (old always-detects-something
    # behavior). Calibrate with scripts/check_geco2_score_separation.py on
    # your own present/absent-labeled frames -- do NOT guess a value blind.
    score_threshold_abs: float = 0.0
    nms_iou: float = 0.5
    topk_per_keyframe: int = 5
    prototype_cache_name: str = "geco2_prototype.pt"
    # Shrink each reference image before encoding it as an exemplar, to
    # narrow the ground-to-aerial domain gap (close-up ref photos are
    # otherwise much crisper/larger-looking than how the object actually
    # appears in the drone video). 1.0 = no-op (default). The shrunk image
    # still gets upscaled back up to stage123_geco2.image_size by
    # resize_and_pad -- so the effective blur amount depends on how the
    # shrunk size compares to image_size, not just this factor alone (a
    # given factor blurs a low-res ref photo far more than a high-res one).
    # NOTE: proven no-op on final object SIZE on the model's canvas (see
    # ScaleCalibrationConfig docstring) -- it only affects blur/detail level.
    # Use scale_calibration below to actually fix apparent-size mismatch.
    ref_downscale_factor: float = 1.0
    # Crop each reference image to its MobileSAM tight mask box (expanded by
    # crop_context_margin) BEFORE resize_and_pad -- keeps 100% real pixels,
    # no masking/fill (unlike background_mode), just a tighter field of view
    # than the whole reference photo. Since resize_and_pad always renormalizes
    # the (now smaller) image's longer side back up to image_size, the object
    # ends up occupying a LARGER fraction of the 1024 canvas than it would
    # from the whole uncropped photo -- so RoI-Align pools from more
    # feature-map cells at each pyramid level, giving a higher-resolution
    # appearance token. Unlike scale_calibration below, this needs NO oracle
    # knowledge of the deployment video's apparent object size -- it is
    # purely a function of the reference photo's own (already-computed)
    # object bounds. Requires segmentation.enabled (needs the tight mask
    # box). See aero_eyes/utils/geometry.py::crop_to_object.
    crop_to_object: bool = False
    crop_context_margin: float = 0.5
    scale_calibration: ScaleCalibrationConfig = ScaleCalibrationConfig()
    domain_calibration: DomainCalibrationConfig = DomainCalibrationConfig()
    # Diagnostic/ablation toggle: box size feeds the exemplar prototype
    # through TWO independent paths -- (1) shape_or_objectness(w,h) -> a
    # dedicated shape token, and (2) the box coordinates that define the
    # RoI-Align pooling region for the appearance token (main/l1/l2).
    # use_shape_token=false disables ONLY path (1) -- it does NOT fix path
    # (2) (a wrong-scaled box still pools the wrong region for appearance).
    # Combine with scale_calibration.enabled to test all 4 combinations:
    #   use_shape_token=true,  scale_calibration=false -- current default
    #   use_shape_token=false, scale_calibration=false -- isolates path (1)
    #   use_shape_token=true,  scale_calibration=true  -- both paths fixed
    #   use_shape_token=false, scale_calibration=true  -- path (2) fixed, path (1) removed
    use_shape_token: bool = True
    color_postfilter: ColorPostfilterConfig = ColorPostfilterConfig()
    cosine_rescore: Geco2CosineRescoreConfig = Geco2CosineRescoreConfig()
    global_adaptive_threshold: GlobalAdaptiveThresholdConfig = GlobalAdaptiveThresholdConfig()

    @model_validator(mode="after")
    def check_scale_calibration(self) -> "Stage123Geco2Config":
        if self.scale_calibration.enabled:
            if not self.scale_calibration.expected_object_px:
                raise ValueError(
                    "stage123_geco2.scale_calibration.enabled=true requires "
                    "stage123_geco2.scale_calibration.expected_object_px=[w,h] "
                    "(estimated object size in the RAW video frame, pixels)."
                )
            if not self.segmentation.enabled:
                raise ValueError(
                    "stage123_geco2.scale_calibration.enabled=true requires "
                    "stage123_geco2.segmentation.enabled=true (scale calibration builds "
                    "its canvas around the MobileSAM tight mask box)."
                )
        return self


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------

class AeroEyesConfig(BaseModel):
    project: ProjectConfig = ProjectConfig()
    data: DataConfig = DataConfig()
    runtime: RuntimeConfig = RuntimeConfig()
    pipeline: PipelineConfig = PipelineConfig()
    stage1: Stage1Config = Stage1Config()
    stage2: Stage2Config = Stage2Config()
    stage3: Stage3Config = Stage3Config()
    stage4: Stage4Config = Stage4Config()
    stage5: Stage5Config = Stage5Config()
    stage123_geco2: Stage123Geco2Config = Stage123Geco2Config()
    accuracy: AccuracyConfig = AccuracyConfig()
    eval: EvalConfig = EvalConfig()

    @model_validator(mode="after")
    def check_litetrack_path(self) -> "AeroEyesConfig":
        if self.stage4.tracker == "litetrack":
            if not self.stage4.litetrack.onnx_path:
                raise ValueError(
                    "stage4.tracker is 'litetrack' but stage4.litetrack.onnx_path is not set. "
                    "Download the LiteTrack-B4 ONNX weights and set "
                    "stage4.litetrack.onnx_path=/path/to/litetrack.onnx in your config."
                )
        return self

    def sample_work_dir(self, sample_id: str) -> Path:
        return Path(self.project.work_dir) / sample_id

    def device(self) -> str:
        if self.runtime.device != "auto":
            return self.runtime.device
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base."""
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def _parse_override(s: str) -> tuple[list[str], str]:
    """Parse 'a.b.c=value' into (['a','b','c'], 'value')."""
    m = re.match(r"^([\w.]+)=(.*)$", s, re.DOTALL)
    if not m:
        raise ValueError(f"Invalid override '{s}'; expected dotted.key=value")
    keys = m.group(1).split(".")
    raw = m.group(2)
    # Try to coerce to Python primitive types
    if raw.lower() == "true":
        value: Any = True
    elif raw.lower() == "false":
        value = False
    elif raw.lower() in ("null", "none", "~"):
        value = None
    else:
        try:
            value = int(raw)
        except ValueError:
            try:
                value = float(raw)
            except ValueError:
                # Try JSON (handles lists like [640,640] and dicts)
                if raw.startswith(("[", "{")):
                    try:
                        import json as _json
                        value = _json.loads(raw)
                    except Exception:
                        value = raw
                else:
                    value = raw
    return keys, value


def _set_nested(d: dict, keys: list[str], value: Any) -> None:
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def load_config(path: str | Path, overrides: list[str] | None = None) -> AeroEyesConfig:
    """Load config.yaml, apply CLI overrides, validate and return typed config."""
    with open(path) as f:
        raw: dict = yaml.safe_load(f) or {}

    if overrides:
        for ov in overrides:
            keys, value = _parse_override(ov)
            _set_nested(raw, keys, value)

    return AeroEyesConfig.model_validate(raw)
