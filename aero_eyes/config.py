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


class Stage1Config(BaseModel):
    segmentation: SegmentationConfig = SegmentationConfig()
    feature_extractor: FeatureExtractorConfig = FeatureExtractorConfig()
    prototype: PrototypeConfig = PrototypeConfig()
    aerial_sim: AerialSimConfig = AerialSimConfig()


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
    # Expected apparent size [width, height] in pixels of the object AS IT
    # APPEARS IN THE RAW VIDEO FRAME (before any resize/pad) -- e.g.
    # estimated from flight altitude/GSD, or eyeballed on a sample frame.
    # Required when enabled=true; there is no safe default (a wrong value
    # actively hurts -- it recreates the same kind of scale mismatch this
    # feature exists to remove, just in a different direction).
    expected_object_px: Optional[list[float]] = None
    # Extra padding kept around the tight mask box, as a fraction of the
    # object's own size, before that (object+margin) footprint is calibrated
    # to match expected_object_px -- gives the model a bit of surrounding
    # context instead of the object filling the canvas edge-to-edge.
    context_margin: float = 0.5

    @field_validator("expected_object_px")
    @classmethod
    def check_expected_object_px(cls, v: Optional[list[float]]) -> Optional[list[float]]:
        if v is not None and len(v) != 2:
            raise ValueError("scale_calibration.expected_object_px must be [width, height]")
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
    are a common false positive. Compares each candidate box's HSV
    hue+saturation histogram (brightness/value ignored -- robust to
    lighting differences between the close-up reference photo and the
    video frame) against the reference object's own color signature
    (computed once from the MobileSAM-masked reference photos, cached to
    color_signature.npz). Pure OpenCV, no extra model, no finetuning -- see
    aero_eyes/utils/color.py and stage123_geco2.py::build_color_signature /
    apply_color_postfilter.

    Falls back to the WHOLE reference photo's color (diluted by
    background) if segmentation.enabled=false -- still works, just less
    precise; a warning is logged when that happens.

    IMPORTANT (empirically confirmed, not just theoretical): Hue is
    unstable/noisy for near-achromatic (black/white/gray) objects -- a
    black-colored reference object saw ST-IoU DROP (0.4264 -> 0.3560, with
    reweight-only / no hard-drop) when this was enabled, while a
    saturated-color (orange) object saw it IMPROVE (0.5170 -> 0.5448) under
    the identical settings. min_ref_saturation below auto-disables this
    filter for low-saturation reference objects instead of silently
    hurting accuracy on them.
    """
    enabled: bool = False
    hue_bins: int = 30
    sat_bins: int = 32
    metric: Literal["bhattacharyya", "correlation"] = "bhattacharyya"
    # Candidates scoring below this similarity (roughly 0..1, higher = more
    # similar) against EVERY reference photo are dropped outright.
    # 0.0 = never hard-drop (rely on reweight only).
    min_similarity: float = 0.3
    # If true, surviving candidates' scores are multiplied by their color
    # similarity (soft penalty) and the list is re-sorted by the new score
    # -- lets a borderline-color match still surface if GeCo2's own
    # confidence is otherwise much higher than competing candidates.
    reweight: bool = True
    # Auto-disables the ENTIRE color_postfilter (returns to plain GeCo2
    # output) for a sample whose reference object's own mean HSV saturation
    # (0-255, averaged over the masked object pixels across all 3 ref
    # photos) is below this floor -- catches near-WHITE/gray objects.
    # NOTE: not sufficient alone for DARK objects -- see min_ref_value.
    # The actual computed value is always logged at build time (even when
    # not disabling), so inspect real numbers across your samples first.
    min_ref_saturation: float = 40.0
    # Same auto-disable, triggered by LOW mean HSV value/brightness --
    # catches near-BLACK objects. Needed because saturation = (max-min)/max
    # is a RATIO: for dark pixels, small absolute sensor noise gets
    # amplified into a spuriously HIGH saturation reading, so
    # min_ref_saturation alone can miss dark objects entirely (confirmed:
    # a synthetic near-black pixel with only +-4/255 noise computed mean
    # saturation ~50, above the min_ref_saturation default).
    min_ref_value: float = 50.0


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
