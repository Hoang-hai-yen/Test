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


class DenseScanConfig(BaseModel):
    """Fallback candidate source: DINOv2 patch-similarity scan against the
    Stage-1 prototype, run alongside (never instead of) the proposal_model
    above. Only fires when the proposal model starves a keyframe -- targets
    cases like a flat/small object (e.g. an ID card) that YOLOv11n/FastSAM-s
    essentially never proposes a box for from a top-down drone view. Only
    supported when stage1.feature_extractor.model == 'dinov2' (the only
    extractor with a patch-token grid method)."""
    enabled: bool = False
    # Only run the dense scan on a keyframe if the proposal path found fewer
    # than this many candidates -- keeps the extra ViT forward passes rare on
    # videos where YOLO/FastSAM already works.
    trigger_min_proposals: int = 3
    # Pre-filter bar for patch cosine similarity vs the prototype. Kept low
    # on purpose: Stage 3's own (usually adaptive) match_threshold is the
    # real accept/reject decision -- this only needs to avoid missing the
    # target region entirely.
    sim_threshold: float = 0.12
    # Minimum connected-component size (in ViT patches, 14px each) to keep a
    # blob as a candidate box; filters single-patch noise.
    min_blob_patches: int = 2
    max_dense_candidates_per_keyframe: int = 20
    # FPN-style multi-scale fusion: instead of a single-resolution patch
    # grid, extract grids at several input resolutions and fuse them
    # top-down (coarse -> fine) before scoring. Targets blob-precision, not
    # just recall -- a validated failure mode (see configs/config.yaml) is
    # that a single loose-threshold scale merges genuinely-matching patches
    # with noise into an oversized, imprecise bounding box. Opt-in until
    # validated on Kaggle -- roughly triples the DINOv2 forward-pass cost of
    # every dense-scan trigger.
    use_fpn_pyramid: bool = False


class Stage2Config(BaseModel):
    keyframe_interval: int = 8
    sahi: SAHIConfig = SAHIConfig()
    proposal_model: str = "yolov11n"
    yolov11n: Yolov11nConfig = Yolov11nConfig()
    fastsam_s: FastSamSConfig = FastSamSConfig()
    candidate: CandidateConfig = CandidateConfig()
    dense_scan: DenseScanConfig = DenseScanConfig()

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


class Stage4Config(BaseModel):
    tracker: str = "builtin"
    builtin: BuiltinTrackerConfig = BuiltinTrackerConfig()
    litetrack: LiteTrackConfig = LiteTrackConfig()
    tracker_conf_threshold: float = 0.40
    max_track_age: int = 30
    # OpenCV trackers (csrt/kcf/mosse/mil) report a fixed placeholder
    # confidence on every "successful" update -- they cannot tell drift from
    # a correct lock, so tracker_conf_threshold alone almost never fires.
    # Every verify_interval frames while tracking, re-embed the tracked crop
    # with DINOv2 and compare it against the prototype; treat the track as
    # lost (trigger re-detect) if similarity falls below the match threshold.
    # Set to 0 to disable (old behaviour: trust the tracker until max_track_age).
    verify_interval: int = 5

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


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------

class AeroEyesConfig(BaseModel):
    project: ProjectConfig = ProjectConfig()
    data: DataConfig = DataConfig()
    runtime: RuntimeConfig = RuntimeConfig()
    stage1: Stage1Config = Stage1Config()
    stage2: Stage2Config = Stage2Config()
    stage3: Stage3Config = Stage3Config()
    stage4: Stage4Config = Stage4Config()
    stage5: Stage5Config = Stage5Config()
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
    elif raw.lower() in ("null", "~"):
        value = None
        # NOTE: intentionally NOT treating the string "none" as an alias for
        # null here. Several fields in this schema use "none" as a real
        # enum value (stage4.tracker: "none" = detect every frame,
        # stage2.proposal_model-style literals elsewhere) -- coercing it to
        # Python None broke `--set stage4.tracker=none` with a pydantic
        # "Input should be a valid string" error. Use "null" or "~" for an
        # actual null override.
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
