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
    score_ratio_floor: float = 0.85
    max_border_touch_frac: float = 0.02
    background_mode: Literal["mean_fill", "keep_real", "blur"] = "mean_fill"
    blur_sigma: float = 25.0


class FeatureExtractorConfig(BaseModel):
    model: Literal["dinov2", "dinov3", "clip", "siglip", "ensemble"] = "dinov2"
    dinov2_variant: Literal["vits14", "vitb14", "vitl14", "vitg14"] = "vitb14"
    dinov3_variant: Literal["vits16", "vitb16", "vitl16"] = "vitb16"
    clip_variant: str = "vit-b/32"
    siglip_variant: Literal["base", "large", "so400m"] = "base"
    weights: Optional[str] = None
    image_size: int = 224


class PrototypeConfig(BaseModel):
    fusion: Literal["mean", "max", "concat_then_pca"] = "mean"
    l2_normalize: bool = True
    cache_name: str = "prototype.npz"


class AerialSimConfig(BaseModel):
    enabled: bool = False
    downscale_factor: float = 1.0
    blur_ksize: int = 0


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
    model_config = {'extra': 'allow'}
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
    model_config = {'extra': 'allow'}
    similarity: Literal["cosine", "l1", "l2"] = "cosine"
    match_threshold: float = 0.55
    nms_iou: float = 0.5
    topk_per_keyframe: int = 5
    global_topk: Optional[int] = None
    adaptive_threshold: bool = False
    adaptive_z_score: float = 2.0
    adaptive_min_floor: float = 0.05
    calibrate: CalibrateConfig = CalibrateConfig()


class BuiltinTrackerConfig(BaseModel):
    algorithm: Literal["csrt", "kcf", "mosse"] = "csrt"


class LiteTrackConfig(BaseModel):
    onnx_path: Optional[str] = None
    input_size: int = 256


class DetectionConfirmationConfig(BaseModel):
    enabled: bool = False
    required_hits: int = 2
    iou_threshold: float = 0.3


class Stage4Config(BaseModel):
    model_config = {'extra': 'allow'}
    tracker: str = "builtin"
    builtin: BuiltinTrackerConfig = BuiltinTrackerConfig()
    litetrack: LiteTrackConfig = LiteTrackConfig()
    tracker_conf_threshold: float = 0.40
    max_track_age: int = 30
    verify_interval: int = 5
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
    detector: Literal['legacy', 'geco2', 'merged'] = 'legacy'


class ScaleCalibrationConfig(BaseModel):
    enabled: bool = False
    expected_object_px: Optional[list[float]] = None
    context_margin: float = 0.5

    @field_validator("expected_object_px")
    @classmethod
    def check_expected_object_px(cls, v: Optional[list[float]]) -> Optional[list[float]]:
        if v is not None and len(v) != 2:
            raise ValueError("scale_calibration.expected_object_px must be [width, height]")
        return v


class DomainCalibrationConfig(BaseModel):
    enabled: bool = False
    num_sample_frames: int = 5
    strength: float = 1.0


class Stage123Geco2Config(BaseModel):
    repo_path: str = "./GECO2"
    weights_path: str = "./GECO2/CNTQG_multitrain_ca44.pth"
    segmentation: SegmentationConfig = SegmentationConfig()
    image_size: int = 1024
    emb_dim: int = 256
    kernel_dim: int = 3
    reduction: int = 16
    keyframe_interval: int = 8
    score_threshold_ratio: float = 0.33
    score_threshold_abs: float = 0.0
    nms_iou: float = 0.5
    topk_per_keyframe: int = 5
    prototype_cache_name: str = "geco2_prototype.pt"
    ref_downscale_factor: float = 1.0
    scale_calibration: ScaleCalibrationConfig = ScaleCalibrationConfig()
    domain_calibration: DomainCalibrationConfig = DomainCalibrationConfig()

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
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def _parse_override(s: str) -> tuple[list[str], str]:
    m = re.match(r"^([\w.]+)=(.*)$", s, re.DOTALL)
    if not m:
        raise ValueError(f"Invalid override '{s}'; expected dotted.key=value")
    keys = m.group(1).split(".")
    raw = m.group(2)
    if raw.lower() == "true":
        value: Any = True
    elif raw.lower() == "false":
        value = False
    elif raw.lower() in ("null", "~"):
        value = None
    else:
        try:
            value = int(raw)
        except ValueError:
            try:
                value = float(raw)
            except ValueError:
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