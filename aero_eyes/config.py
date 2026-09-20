"""Typed configuration schema + loader.

Loads configs/config.yaml into validated Pydantic models.
Supports CLI overrides:  --set stage2.proposal_model=fastsam_s
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Literal, Optional

import yaml
from pydantic import BaseModel, field_validator, model_validator

# Pure infra escape-hatch for a known cuDNN issue seen on some GPU/driver
# combos ("Unable to find a valid cuDNN algorithm to run convolution" /
# "GET was unable to find an engine..."), NOT a real modeling choice --
# deliberately an env var, not a config.yaml field, since it has nothing to
# do with the experiment being run. Disabling cuDNN falls back to a slower
# but much more reliable conv implementation. Set AERO_EYES_DISABLE_CUDNN=1
# in the shell BEFORE running any aero_eyes command if you hit that error.
#
# Applied at IMPORT time (not inside Config.device()) -- a stage can run its
# own model (e.g. stage1.py's MobileSAMSegmenter) and fire the first CUDA
# conv of the whole process before anything ever calls cfg.device(), so
# setting torch.backends.cudnn.enabled=False only there arrives too late for
# that first call and the env var silently does nothing for it.
if os.environ.get("AERO_EYES_DISABLE_CUDNN"):
    try:
        import torch
        torch.backends.cudnn.enabled = False
    except ImportError:
        pass


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
    # "mobilesam" (default): lightest/fastest of the 3, proven in production
    #   here. "weights" below is its checkpoint path.
    # "fastsam": reuses stage2.fastsam_s's weights/conf/iou/imgsz (NOT
    #   "weights" below, which is ignored for this model) -- has NO
    #   prompt-conditioned decoder, so it can only SELECT among masks its
    #   own "segment everything" pass already produced for this image,
    #   never generate a NEW one conditioned on where it's prompted (same
    #   ceiling noted on box_refine.method=fastsam_dense: a small/thin
    #   object merged with a neighbor or missed outright in that pass
    #   can't be recovered here either).
    # "sam2": a full standalone SAM2 (same one box_refine.method=
    #   sam2_native uses) -- "weights" below is ignored, needs
    #   stage123_geco2.repo_path's vendored GECO2/sam2 package + its own
    #   deps (hydra-core, omegaconf). Heavier/slower than mobilesam per
    #   reference image (3 images/sample here, so this adds up).
    # fastsam/sam2 are NOT YET VALIDATED for this reference-image masking
    # role (only proven so far as box_refine methods, a different job) --
    # compare against the mobilesam baseline on your own footage before
    # trusting either here.
    model: Literal["mobilesam", "fastsam", "sam2"] = "mobilesam"
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
    # Opt-out: min_area_frac/max_area_frac/max_border_touch_frac above are
    # calibrated against MobileSAM's own behavior -- with a newly-wired
    # model (model="fastsam"/"sam2"), or just to see what the model
    # actually proposed before any of this project's own heuristics get a
    # say, set this false to skip BOTH final rejection checks entirely and
    # return whichever mask segment()'s own candidate-selection picked, no
    # matter its area or how much it touches the border (never falls back
    # to the all-ones passthrough for THIS reason -- inference failures/
    # the model being unavailable still do). The candidate-selection step
    # itself (isolate a connected component, prefer the largest among
    # confident candidates) still runs; this only skips the pass/fail
    # gate applied to whatever it picked.
    reject_implausible_mask: bool = True
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


class ProjectionHeadConfig(BaseModel):
    """Optional small trainable head applied ON TOP of the frozen backbone
    embedding (whichever stage1.feature_extractor.model is selected), to
    close the domain gap between close-up reference photos and tiny aerial
    crops WITHOUT fine-tuning the backbone itself -- see
    scripts/train_projection_head.py for how weights_path is produced.

    Disabled by default: build_feature_extractor() returns the raw backbone
    extractor unchanged unless this is explicitly turned on with a valid
    weights_path. Applies everywhere that extractor is used (Stage 1
    prototype build, Stage 3/Stage12-GeCo2 candidate features, Stage 4
    verify_interval re-check) since they all go through the same factory.
    """
    enabled: bool = False
    weights_path: Optional[str] = None
    # Must match the architecture the checkpoint at weights_path was
    # actually trained with -- see train_projection_head.py's --output-dim/
    # --hidden-dim. Kept here (not read off the checkpoint alone) so a
    # config typo mismatching the checkpoint fails loudly at load time.
    output_dim: int = 256
    hidden_dim: Optional[int] = None   # null = single Linear layer, no hidden layer


class FeatureExtractorConfig(BaseModel):
    model: Literal["dinov2", "dinov3", "clip", "siglip", "ensemble"] = "dinov2"
    dinov2_variant: Literal["vits14", "vitb14", "vitl14", "vitg14"] = "vitb14"
    # DINOv2 "with registers" (torch.hub dinov2_{variant}_reg / HF
    # facebook/dinov2-with-registers-*): Meta found a handful of patch tokens
    # in the original DINOv2 get repurposed internally as a "scratch pad" for
    # global information the model needs but has nowhere else to put, which
    # shows up as high-norm artifact tokens polluting the attention/feature
    # maps. Adding a few dedicated register tokens (ignored for the CLS
    # output used here) gives the model that scratch space directly, per
    # Meta's own ablations producing cleaner features -- same CLS token
    # output dim as the non-register variant (see DINOv2FeatureExtractor.
    # _DIMS), so this is a drop-in swap, not a separate model size to
    # reconfigure downstream. False (default) = original DINOv2, unchanged.
    dinov2_use_registers: bool = False
    # DINOv3 architecture size. Weights are gated on HuggingFace
    # (facebook/dinov3-*) -- request access on the model page and set
    # HF_TOKEN before using dinov3_source=huggingface below.
    dinov3_variant: Literal["vits16", "vitb16", "vitl16"] = "vitb16"
    # WHICH pretraining run's weights to load, same architecture either way:
    # "lvd1689m" (default) = Meta's large natural-image corpus. "sat493m" =
    # a satellite-imagery pretraining run -- likely a better domain match
    # for aerial/drone footage than the natural-image default. Applies to
    # BOTH sources below (huggingface builds the repo id from it; for
    # kaggle, dinov3_kaggle_model_id is a free-form string you supply
    # yourself, so it's on you to point it at a checkpoint whose own
    # architecture/dataset matches this + dinov3_variant -- this field is
    # bookkeeping/logging only in that case, not enforced). Not every
    # (dinov3_variant, dinov3_pretrain_dataset) combination is necessarily
    # published on HuggingFace -- an unavailable one 404s straight from
    # `from_pretrained`.
    dinov3_pretrain_dataset: Literal["lvd1689m", "sat493m"] = "lvd1689m"
    # "huggingface" (default): facebook/dinov3-{dinov3_variant}-pretrain-
    #   {dinov3_pretrain_dataset} via transformers.AutoModel, gated -- see
    #   dinov3_variant's own comment.
    # "kaggle": loads a raw Meta DINOv3 checkpoint (a bare .pth/.pt state
    #   dict from Meta's OWN dinov3 codebase, NOT a transformers-format
    #   folder) via kagglehub.model_download(dinov3_kaggle_model_id), then
    #   torch.hub.load("facebookresearch/dinov3", ..., weights=<that file>)
    #   -- lets you point at a pretraining variant mirrored on Kaggle
    #   without needing HuggingFace gating (e.g. when a HF repo for it
    #   isn't published, or you just don't have HF access).
    #   Needs the `kagglehub` package installed and Kaggle API credentials
    #   configured (~/.kaggle/kaggle.json or KAGGLE_USERNAME/KAGGLE_KEY).
    dinov3_source: Literal["huggingface", "kaggle"] = "huggingface"
    # Required when dinov3_source == "kaggle", e.g.
    # "yadavdamodar/dinov3-vitl16-pretrain-sat493m/pyTorch/default" -- its
    # own architecture/dataset must match dinov3_variant/
    # dinov3_pretrain_dataset above (this string is the actual weight
    # source; those two fields aren't validated against it).
    dinov3_kaggle_model_id: Optional[str] = None
    clip_variant: str = "vit-b/32"   # "vit-b/32" (512-d) or "vit-l/14" (768-d)
    # SigLIP: open access (no gating), vision-only encoder.
    siglip_variant: Literal["base", "large", "so400m"] = "base"
    weights: Optional[str] = None
    image_size: int = 224
    projection_head: ProjectionHeadConfig = ProjectionHeadConfig()


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
    # min_support only checks COUNT -- a handful of near-duplicate frames
    # (e.g. 3 consecutive keyframes of the same unmoving pose) clears it just
    # as easily as 3 genuinely different views, but only the latter is
    # actually safe to generalize from. When enabled, a round's high-
    # confidence picks must ALSO span at least min_frame_span frames
    # (max(frame_idx) - min(frame_idx) among the picks) before being trusted
    # -- cheap proxy for "the target's appearance actually varies across
    # these picks" without needing an embedding-space diversity metric.
    # Rounds failing this (like the count check) stop the loop early instead
    # of updating the prototype from an unrepresentative slice. False
    # (default) = unchanged, count-only gate.
    require_diverse_picks: bool = False
    min_frame_span: int = 30


class Stage3Config(BaseModel):
    # Dev/debug convenience: candidates.json's companion candidates.feats.npz
    # is written by Stage 2 (see aero_eyes/stages/stage2.py::
    # _write_candidates_with_features) using whatever stage1.feature_extractor
    # was active THEN -- Stage 2's own cache check only looks at whether
    # candidates.json exists, with no awareness of which extractor produced
    # it, so switching feature_extractor.model/variant without also clearing
    # that cache leaves a STALE, wrong-dimension .feats.npz that crashes
    # Stage 3's `feats @ ref` with a cryptic shape-mismatch error. The correct
    # fix is normally to invalidate the cache (project.use_cache=false, or
    # delete candidates.json so Stage 2 reruns) -- but Stage 2 also reruns
    # SAHI/proposal-model box detection, which doesn't depend on the
    # extractor at all and can be the expensive part on a long video. When
    # true, Stage 3 instead re-extracts features for the EXISTING candidate
    # boxes (no box detection re-run) using the CURRENT feature_extractor,
    # then overwrites candidates.json + candidates.feats.npz in place -- so a
    # later run (with this back off) sees a cache that's actually consistent
    # with the extractor now configured. False (default) = unchanged, use
    # whatever features are already cached.
    recompute_candidate_features: bool = False
    # Rejects a degenerate, near-zero-AREA candidate box (from
    # candidates.json, whichever pipeline produced it) before it can
    # occupy one of THIS stage's own topk_per_keyframe slots below -- same
    # AREA-not-min-side-length rationale as
    # stage123_geco2.min_box_area_enabled (see that field's own docstring:
    # this project's own GT survey found a real object's thinnest side can
    # legitimately be ~2px at the frame edge, but no real GT box has area
    # <= 16px^2). Reads candidates.json as already written -- no
    # candidate-generation stage needs rerunning to retune this threshold,
    # unlike stage123_geco2's own version.
    #
    # Deliberately does NOT replace stage123_geco2.min_box_area_enabled:
    # that one runs BEFORE stage123_geco2.cosine_rescore.
    # candidate_topk_per_keyframe caps the RAW candidate pool -- a
    # degenerate box surviving that cap crowds out a real candidate
    # PERMANENTLY (it never reaches candidates.json at all), which this
    # later filter cannot recover. Running both is the safe choice; this
    # one alone only protects this stage's own topk_per_keyframe cap, not
    # the earlier candidate-generation one.
    min_box_area_enabled: bool = False
    min_box_area: int = 24
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
    # The mean/std used above are computed on all_sims AS-IS -- when
    # accuracy.cheap_boosters.multi_reference_embedding pools scores across
    # per_ref_features (max, typically) and stage3.dynamic_prototype has
    # APPENDED extra reference vectors derived from a narrow, self-selected
    # candidate subset, a handful of OTHER candidates that happen to match
    # those new (narrow) references well get inflated scores via max-pooling
    # -- which drags mean/std up and raises the threshold for EVERYONE,
    # including true positives whose own score never changed (they just
    # don't match the narrow references any better than before). When
    # enabled, mean/std (or median/MAD, see adaptive_threshold_robust below)
    # are computed from the similarity distribution BEFORE dynamic_prototype
    # ran (i.e. against the original per-reference-photo vectors only) --
    # stable, not draggable by whatever dynamic_prototype appends later --
    # while candidate ACCEPTANCE still uses the full (original + dynamic)
    # pooled scores, so dynamic_prototype's recall benefit is kept, just not
    # its ability to also move the goalpost. No-op when dynamic_prototype is
    # disabled (the two distributions are identical then). False (default)
    # = threshold stats computed on the same (possibly dynamic-prototype-
    # affected) distribution used for acceptance, as before this option
    # existed.
    adaptive_threshold_anchor_to_original_refs: bool = False
    # mean+std is sensitive to a small number of outlier-high scores --
    # exactly what a narrow dynamic_prototype addition (or any other skew)
    # produces. median + z_score*MAD (median absolute deviation, scaled by
    # 1.4826 so it's comparable to std under a roughly normal distribution)
    # is far less moved by a handful of outliers. Combines with
    # adaptive_threshold_anchor_to_original_refs above (independent knobs --
    # anchoring picks WHICH distribution to summarize, this picks HOW to
    # summarize it). False (default) = mean/std, unchanged.
    adaptive_threshold_robust: bool = False
    # "z_score" (default, unchanged): threshold = center + adaptive_z_score
    #   * spread, per adaptive_threshold_robust above. A single fixed z
    #   multiplier can't serve every video at once: a LOW-FP sample's
    #   all_sims is mostly true positives already clustered near the top,
    #   so center is already high and mean/median+z*std OVERSHOOTS,
    #   cutting real positives that are merely "average" within that
    #   already-good population (needs a LOWER z) -- while a HIGH-FP
    #   sample's center is dragged down by the FP mass, so the same z
    #   barely clears it, letting FPs through (needs a HIGHER z). The
    #   right z isn't a property of the similarity metric, it's a property
    #   of how much of all_sims is background vs. signal -- which varies
    #   per video, so no single z serves both regimes.
    # "otsu": Otsu's method (the classic image-binarization algorithm,
    #   applied directly to the real-valued all_sims distribution instead
    #   of an 8-bit pixel histogram) -- finds the split maximizing
    #   between-class separation, with NO z multiplier to hand-tune at
    #   all. Self-adapts to how separated the FP/TP populations actually
    #   are for this video, instead of assuming a fixed offset from center.
    # "gmm": fits 1- and 2-component 1D Gaussian mixtures to all_sims. If
    #   the 2-component fit is genuinely bimodal (better BIC AND the 2
    #   component means separated by >= adaptive_gmm_min_separation_std
    #   pooled std), thresholds at the analytic crossing point between the
    #   2 fitted Gaussians -- the natural FP/TP decision boundary. If NOT
    #   bimodal (e.g. a low-FP sample where all_sims is really just one
    #   cluster of mostly true positives, nothing resembling a second
    #   background cluster), forcing a 2-cluster split onto it is
    #   meaningless -- falls back to adaptive_gmm_fallback_percentile
    #   instead (a permissive cut: a unimodal all_sims here usually means
    #   "most of these candidates are already good").
    # Both "otsu"/"gmm" fall back to "z_score" (with a warning) when
    # all_sims has fewer than adaptive_threshold_min_samples points -- too
    # little data for a distribution-SHAPE method to be reliable (same
    # class of problem as topk_fusion's min_window_for_zscore).
    # NOT YET VALIDATED against this project's own footage -- compare
    # against the z_score baseline (e.g. scripts/sweep_zscore_loocv.py-
    # style before/after) on your own videos before trusting either in
    # production, same as every other opt-in accuracy knob in this project.
    adaptive_threshold_method: Literal["z_score", "otsu", "gmm"] = "z_score"
    adaptive_threshold_min_samples: int = 20
    adaptive_otsu_bins: int = 256
    adaptive_gmm_min_separation_std: float = 1.5
    adaptive_gmm_fallback_percentile: float = 20.0  # keep roughly the top (100 - this)% when no real bimodality is found
    # EXPERIMENTAL, opt-in: real-time-deployment-compatible variant of
    # adaptive_threshold. Everything above (adaptive_threshold_method's
    # z_score/otsu/gmm) needs the WHOLE video's all_sims computed up front
    # -- fine for offline/batch evaluation on a pre-recorded video file,
    # but incompatible with a live feed where keyframe N+1's candidates
    # don't exist yet when keyframe N needs a decision NOW. When true, the
    # threshold for keyframe N is instead computed from a RUNNING WINDOW
    # of only the STRICTLY-PRIOR keyframes' own similarity scores (the most
    # recent adaptive_threshold_online_window samples -- see stage3.py's
    # OnlineAdaptiveThreshold) -- reuses compute_adaptive_threshold's
    # z_score/otsu/gmm dispatch unchanged, just fed a causal buffer instead
    # of the whole video. Cold start (fewer than adaptive_threshold_min_
    # samples observed so far) falls back to adaptive_min_floor, same
    # philosophy as stage123_geco2.dynamic_prototype's own cold start.
    # Mutually exclusive with stage3.dynamic_prototype below (also a
    # whole-video, 2-pass batch mechanism) -- forced off with a warning if
    # both are enabled, since dynamic_prototype's own rounds need the
    # entire video's candidates just as much as the batch threshold does.
    # NOT YET VALIDATED -- an online/causal threshold is inherently less
    # stable than one computed on the whole video (same cold-start
    # tradeoff stage123_geco2.dynamic_prototype already accepts): compare
    # against the batch baseline on recorded footage before trusting it
    # for a live feed.
    adaptive_threshold_online: bool = False
    adaptive_threshold_online_window: int = 200
    calibrate: CalibrateConfig = CalibrateConfig()
    dynamic_prototype: DynamicPrototypeConfig = DynamicPrototypeConfig()


class BuiltinTrackerConfig(BaseModel):
    algorithm: Literal["csrt", "kcf", "mosse"] = "csrt"


class LiteTrackConfig(BaseModel):
    # LiteTrack's real network is 2 separate graphs (see
    # aero_eyes/models/trackers.py module docstring for why one ONNX file
    # isn't enough), both produced by LiteTrack/tracking/export_litetrack_onnx.py
    # from a real trained checkpoint (e.g. LiteTrack_ep0300.pth.tar).
    onnx_path_z: Optional[str] = None   # template crop -> template_feats (run once per track init)
    onnx_path_x: Optional[str] = None   # template_feats + search crop -> response/size/offset maps (every tracked frame)
    # Must match the exported checkpoint's own experiment yaml (TEST.* /
    # MODEL.BACKBONE.STRIDE) -- defaults here are LiteTrack's B4 config.
    template_size: int = 128
    search_size: int = 256
    template_factor: float = 2.0
    search_factor: float = 4.0
    stride: int = 16


class CoTrackerConfig(BaseModel):
    """stage4.tracker=cotracker3 -- see aero_eyes/models/trackers.py's
    CoTrackerTracker docstring for the windowed-recompute adaptation this
    project uses (CoTracker3 tracks POINTS, not boxes, and its real
    "online" predictor needs FUTURE frames of context before it can emit a
    result for a given frame -- incompatible with a per-frame-causal
    Tracker.update() like CSRT/LiteTrack's. This class instead calls the
    OFFLINE predictor's simple full-clip forward repeatedly on a small
    growing/sliding buffer, always reading out the box for the buffer's
    LAST frame, trading CoTracker's own incremental-state efficiency for a
    synchronous per-frame answer). Experimental -- see that module for the
    known trade-offs before relying on this in place of builtin/litetrack.
    """
    # torch.hub entrypoint name under facebookresearch/co-tracker. The
    # offline predictor's forward signature (model(video, queries=...) over
    # one self-contained clip) is what CoTrackerTracker's windowed-recompute
    # design actually calls -- see class docstring. Only override to
    # "cotracker3_online" if you also adapt CoTrackerTracker._run_model to
    # that predictor's own step-wise/is_first_step calling convention.
    variant: str = "cotracker3_offline"
    device: str = "auto"
    # NxN grid of query points sampled inside the box at (re-)init/re-anchor.
    grid_size: int = 5
    # Frames buffered before the tracker re-anchors (fresh grid re-sampled
    # from the current box, buffer reset) -- bounds memory/compute for long
    # tracks at the cost of losing point identity across the reset.
    window_len: int = 16
    # Re-run the model every Nth frame; on skipped frames the last box is
    # held as-is (still appended to the buffer) -- the cheap knob for
    # trading tracking granularity against compute cost.
    recompute_stride: int = 1
    # Fraction of the grid's query points that must still be "visible"
    # (CoTracker's own occlusion prediction) on the current frame for the
    # fitted box to be trusted -- this IS a real confidence signal, unlike
    # BuiltinTracker's fixed 0.9 placeholder (see trackers.py).
    min_visible_ratio: float = 0.3
    # Trim this total percentage (half from each tail) of visible points'
    # x/y coordinates before fitting the box, so a few residual outlier
    # points among the "visible" ones don't blow the box out.
    outlier_trim_pct: float = 10.0


class CosineArbitrationConfig(BaseModel):
    """backward_tracking.validate_against_boundary.cosine_arbitration --
    EXPERIMENTAL, opt-in secondary arbitration for the exact moment
    validate_against_boundary's motion check already found the backward
    segment implausible against the boundary box. Without this, that
    disagreement always means "discard the whole backward segment" (the
    boundary box is trusted unconditionally, since it's independently
    detected/tracked, and the backward segment never was). This second-
    guesses that default: when BOTH objects are re-embedded with DINOv2
    and scored against the SAME prototype (the same signal
    stage4.verify_interval uses), the one with the HIGHER cosine similarity
    is treated as more likely the real target.

    Deliberately experimental/off by default -- this project's own
    diagnostics repeatedly found cosine similarity poorly separated and
    domain-gap-sensitive on this footage (see
    keep_tracking_on_missed_keyframe's own docstring for why THAT feature
    avoids cosine entirely). This may make backward_tracking's outcomes
    WORSE, not better, on some videos -- that's exactly why it needs its
    own config gate instead of always being on, so it can be A/B compared
    (e.g. via scripts/check_stage_prf1_progression.py) before trusting it.

    enabled: false (default) = current behavior, motion disagreement always
    discards the backward segment.

    override_boundary_on_win: what happens when the backward segment's
    score WINS (is higher):
      false -- keep the backward segment, but leave the boundary box
        untouched (both survive, side by side -- the disagreement is
        simply no longer treated as disqualifying for the backward side).
      true  -- ALSO discard the boundary box itself (tracks[fi] = None),
        treating the backward segment's win as evidence the boundary was
        actually the wrong object -- a stronger, riskier claim (overrides
        an independently-produced box instead of just no longer
        discarding the backward one).
    When the backward segment's score does NOT win (lower or equal, or
    either crop fails to embed), falls back to the same behavior as
    enabled=false -- discard the backward segment.

    pooling ("mean" default / "max"): how a candidate crop's score is
    combined across per_ref_features when accuracy.cheap_boosters.
    multi_reference_embedding is active. "match_stage3" (recommended if
    you already run Stage 3 with multi_ref_pooling=max) reads
    accuracy.cheap_boosters.multi_ref_pooling directly, so arbitration
    scores things the SAME way Stage 3's own matching did; the plain
    "mean"/"max" values pin it regardless of that setting. Default "mean"
    reproduces this function's original (pre-arbitration) behavior,
    unchanged, since stage4.verify_interval already relies on mean and
    this option must not silently change that.

    use_adaptive_prototype: false (default) scores against ONLY the
    original 3 reference-photo vectors (prototype.npz, from Stage 1,
    unaffected by Stage 3). true scores against prototype_adapted.npz
    instead -- the SAME references PLUS whatever stage3.dynamic_prototype
    appended while matching this video (per_ref_features only ever grows
    via .append there, so this is "original 3 + adaptive extras combined",
    never original-only vs. adaptive-only). Needs
    stage3.dynamic_prototype.enabled to have actually produced that file;
    falls back to the original prototype.npz (with a logged warning) if
    it's missing.
    """
    enabled: bool = False
    override_boundary_on_win: bool = False
    pooling: Literal["mean", "max", "match_stage3"] = "mean"
    use_adaptive_prototype: bool = False


class BackwardTrackingConfig(BaseModel):
    """stage4.backward_tracking -- recovers frames where the object was
    genuinely present but not yet DETECTED: an object entering the frame is
    often too degraded (motion blur, partial visibility) for the detector
    to lock onto for the first few keyframes, and the SAME shape of problem
    can happen mid-video after a track-loss episode (object reappears but
    isn't re-detected immediately) -- both leave a gap of real presence
    reported as absent, purely because no track existed yet to report it.

    Unlike stage4.keep_tracking_on_missed_keyframe (which extends an
    ALREADY-active track through a gap), this recovers frames BEFORE any
    track existed at all, by running a SEPARATE tracker instance BACKWARD
    in time from the first confirmed box of each NEW track segment --
    every False->True transition of the active-tracking state, not just
    the video's very first lock, since a re-lock after a mid-video
    track-loss episode has the exact same shape of problem. This project's
    trackers (builtin/litetrack) have no inherent notion of time
    direction: template/filter state only depends on the (frame, box) pair
    given at init() and the immediately preceding reported position, so
    running them on frames in decreasing index order is mechanically
    identical to running forward -- see aero_eyes/models/trackers.py.

    Stops recovering backward as soon as it hits whichever comes first: a
    frame already covered by a PREVIOUS track segment (never overwrites
    it), frame 0, max_backward_frames frames back, or the backward
    tracker's own confidence (stage4.tracker_conf_threshold) dropping too
    low to trust further -- so a genuinely-absent stretch before the
    object truly entered the frame is not filled in.

    Needs a bounded rolling buffer of recently-read frames (bounded by
    max_backward_frames) kept in memory during the forward pass to supply
    the backward tracker with pixels for frames already read past -- see
    run_stage4's `recent_frames`. No effect when stage4.tracker == "none"
    (NoneTracker re-detects every frame independently; there is no
    continuous tracker state to run backward).

    validate_against_boundary: "stops at a frame already covered by a
    PREVIOUS track segment" above means the backward run never OVERWRITES
    that frame's box -- but by itself that says nothing about whether the
    backward segment it silently stitched onto that boundary is actually
    the SAME object. If the backward tracker drifted onto a confuser
    partway through (nothing forces it to be right just because it hasn't
    lost confidence yet), the recovered frames would sit right next to a
    real, independent box (a real keyframe detection, or the previous
    segment's own last tracked frame) that may be far away from it --
    exactly the failure mode this option catches.

    When enabled, a BoxDriftCheck-style linear trend is fit from the
    backward segment's OWN recovered positions as it goes (same method as
    stage4.kalman_motion_check and keep_tracking_on_missed_keyframe's own
    validate_against_next_keyframe -- see aero_eyes/utils/
    motion_drift_check.py::BoxDriftCheck). The moment backward recovery
    hits that existing boundary box, it's checked against this trend
    instead of being trusted blindly; if implausible, the ENTIRE backward-
    recovered segment from this call is discarded (every frame it filled
    in reverts to absent) rather than kept as a likely-wrong track stitched
    onto a real one. Deliberately NOT appearance/cosine-based, same
    reasoning as keep_tracking_on_missed_keyframe's own validation (domain
    gap + dynamic_prototype drift make cosine similarity unreliable on its
    own for this project's footage).

    False (default) = disabled -- a backward-recovered segment is always
    kept once produced, whatever it lands next to, unchanged from before
    this option existed.
    """
    enabled: bool = False
    max_backward_frames: int = 30
    validate_against_boundary: bool = False
    # Same semantics/defaults as stage4.kalman_motion_check /
    # keep_tracking_on_missed_keyframe's own fields -- see those for what
    # each one means; applied here to the backward-recovered segment's own
    # trajectory instead.
    window_frames: int = 10
    max_dist_ratio: float = 3.0
    # Only consulted when validate_against_boundary's motion check ALREADY
    # flagged a disagreement -- see CosineArbitrationConfig's own docstring.
    cosine_arbitration: CosineArbitrationConfig = CosineArbitrationConfig()


class KeepTrackingOnMissedKeyframeConfig(BaseModel):
    """stage4.keep_tracking_on_missed_keyframe -- without this, a KEYFRAME
    with zero surviving detections (GeCo2/Stage3 found nothing there, e.g.
    a single false-negative frame sandwiched between two keyframes that
    both DID detect the object) unconditionally kills an already-active
    track: stage4.py never even calls tracker.update() for that keyframe
    or any frame up to the NEXT one, so the whole gap comes back as absent
    even though the tracker's own state (from the PRECEDING keyframe)
    might still be tracking the object correctly. This is different from
    every other failure mode in this file: those all judge an ACTIVE
    track's own claim (conf/age/cosine/drift); this one discards a live
    track purely because a SEPARATE detector call at this one frame came
    up empty, without ever asking the tracker itself.

    When enabled, a keyframe with no detection is treated like any other
    non-keyframe frame WHEN a track is already active: tracker.update()
    runs as usual, still subject to every other check (conf threshold,
    max_track_age, kalman_motion_check, verify_interval + absence_check).
    Has no effect when no track is active yet (nothing to fall back on).

    validate_against_next_keyframe: tolerating a missed keyframe is only
    safe if the track being extended through it was actually correct --
    without a check, this also lets a track that was ALREADY wrong (locked
    onto a confuser at some earlier keyframe) survive a missed keyframe
    that would otherwise have reset it, extending the wrong track instead
    of a right one. verify_interval's DINOv2 cosine check could catch that,
    but is deliberately NOT relied on here: on this project's footage,
    domain gap (reference photos vs. drone frames) and dynamic_prototype
    drift already make cosine similarity an unreliable signal on its own
    (see stage3's own adaptive-threshold machinery for how much tuning that
    needed) -- exactly the failure mode this feature would be most exposed
    to if it leaned on the same signal.

    Instead, once a kept-through segment reaches the next INDEPENDENT box
    (a real keyframe detection, or a successful re-detect), that box is
    checked for motion-plausibility against a linear trend fitted from the
    kept segment's OWN tracked positions -- same method and config shape as
    stage4.kalman_motion_check (see aero_eyes/utils/motion_drift_check.py::
    BoxDriftCheck), just applied retroactively to one pending segment
    instead of flagging every frame live. If the independent box lands
    implausibly far from where that trend predicts, every frame in the
    kept-through segment is retroactively marked absent instead of keeping
    a track that most likely drifted onto the wrong object. Real fast
    motion is tolerated (the trend is fit from the object's OWN recent
    trajectory, not a fixed position), the same way kalman_motion_check
    tolerates it.

    True (default whenever this feature is enabled) -- the safety net is
    what makes tolerating a missed keyframe defensible in the first place.
    Set False to reproduce "always keep the segment, never retroactively
    check it" for comparison/debugging.
    """
    enabled: bool = False
    validate_against_next_keyframe: bool = True
    # Caps how many CONSECUTIVE missed keyframes one open segment tolerates
    # before giving up on it (same "give up" path as kt_cfg.enabled=False)
    # -- deliberately separate from stage4.max_track_age, which bounds
    # elapsed FRAMES since the last confirmed re-anchor. track_age is frozen
    # for the whole time a segment is open (see run_stage4's own comment
    # next to `consecutive_missed_keyframes`), so max_track_age no longer
    # has any way to bound this tolerance -- without a dedicated limit here,
    # an object that's genuinely gone would let the tracker coast on stale
    # motion forever. The two limits used to overlap by coincidence (a
    # frame-count ceiling divided by keyframe_interval implicitly capped
    # consecutive misses too) in a way that shifted with keyframe_interval
    # and was never a deliberate design choice -- this makes the intended
    # limit explicit and independent of that config's value.
    max_consecutive_missed_keyframes: int = 3
    # Same semantics/defaults as stage4.kalman_motion_check's own fields --
    # see KalmanMotionCheckConfig for what each one means; applied here to
    # the kept-through segment's trajectory instead of every live frame.
    window_frames: int = 10
    max_dist_ratio: float = 3.0
    # EXPERIMENTAL, opt-in secondary arbitration for the exact moment the
    # motion check above ALREADY flagged a disagreement -- same mechanism
    # and same CosineArbitrationConfig as backward_tracking.
    # validate_against_boundary uses (see that class's own docstring for
    # the full rationale/caveats: deliberately off by default, this
    # project's own diagnostics repeatedly found cosine poorly separated
    # on this footage). Differs in ONE way: here the "boundary" object (the
    # independent detection at the CURRENT frame) hasn't been written to
    # tracks[] yet when this runs, so override_boundary_on_win=true means
    # REJECTING that detection for this one frame (reports absent) rather
    # than overwriting an already-recorded box -- the kept-through track
    # itself is left running untouched, and kept_segment_start stays open
    # for a later independent detection to resolve.
    cosine_arbitration: CosineArbitrationConfig = CosineArbitrationConfig()


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


class AbsenceCheckConfig(BaseModel):
    """stage4.absence_check -- refines what verify_interval's cosine check
    does once it already fails (sim < match_threshold): distinguishes
    "borderline/drifted, still worth a re-detect attempt" from "similarity
    is so far below match_threshold the object has almost certainly left
    the frame, don't bother re-detecting". match_threshold itself was
    calibrated to separate "is this candidate the right object" during
    Stage 3 matching -- it was never calibrated as a presence/absence
    boundary, so a fixed cutoff there conflates two different questions.

    Without this, every verify_interval failure -- however low the
    similarity -- still triggers a full re-detect attempt (GeCo2/YOLO+
    DINOv2). That costs compute either way, but the real harm is when it
    SUCCEEDS at finding some box (via its own separate scoring, not
    cosine) even though the real object is genuinely gone -- extending a
    post_departure_drift run instead of ending it (see
    scripts/check_tracker_coverage.py's post_departure_drift attribution).

    Needs stage4.verify_interval > 0 (same DINOv2 prototype/extractor) to
    have any effect -- this only fires once verify_interval's own check has
    already failed.
    """
    enabled: bool = False
    # absence_threshold = match_threshold * absence_ratio. Expressed as a
    # RATIO (not an absolute cosine cutoff) since match_threshold itself
    # can be adaptive per video (stage3.adaptive_threshold) -- a fixed
    # absolute absence value would need separate re-tuning per video/
    # threshold regime, a ratio automatically tracks match_threshold.
    # Lower = stricter (only the most extreme mismatches skip re-detect,
    # closer to today's always-re-detect behavior); higher = more lenient
    # (skips re-detect more readily, risks giving up on a merely-drifted
    # track that a re-detect could have recovered).
    absence_ratio: float = 0.5


class KalmanMotionCheckConfig(BaseModel):
    """stage4.kalman_motion_check -- a cheap per-frame drift-plausibility
    check, complementary to verify_interval's (appearance-based) cosine
    check. verify_interval can only catch a confuser that LOOKS different
    from the prototype; it is blind to a confuser that looks similar but
    sits somewhere the track's own recent trajectory could not plausibly
    have led to. See aero_eyes/utils/motion_drift_check.py for the check
    itself and why this borrows ByteTrack's core idea instead of the whole
    (multi-object, per-frame-detection) framework.

    v1 of this (single-step constant-velocity Kalman filter) was swept
    empirically (scripts/sweep_kalman_max_dist_ratio.py) and found NET
    HARMFUL at every ratio strict enough to ever trigger, on real footage
    with plenty of genuine frame-to-frame acceleration (drone camera +
    falling/tumbling objects) -- it mistook real motion for drift far more
    often than it caught actual confuser locks. v2 (current) fits a robust
    linear trend over window_frames PAST positions instead of trusting just
    the immediately preceding frame, to damp that false-alarm rate -- NOT
    yet validated the same way; re-sweep before trusting this in production.

    Runs EVERY frame of active tracking (not gated by verify_interval's own
    cadence), since fitting a short linear trend is far cheaper than a
    DINOv2 embed -- if it already flags a frame, verify_interval's own
    (more expensive) cosine check for that same frame is skipped, since
    track_ok is already False by then.
    """
    enabled: bool = False
    # How far (in units of the reported box's own diagonal) the box's
    # center may land from where the fitted trend predicted, before being
    # judged an implausible departure from the track's own recent
    # trajectory. Lower = stricter (catches smaller departures, but more
    # likely to flag genuine fast/erratic real motion as drift).
    max_dist_ratio: float = 3.0
    # How many past frames the linear trend is fit over. Larger = smoother
    # (more resistant to single-frame noise, but slower to notice a real
    # direction change); needs >=3 to fit a trend at all -- below that the
    # check is a no-op (always plausible) until enough history accumulates.
    window_frames: int = 10


class Stage4Config(BaseModel):
    tracker: str = "builtin"
    builtin: BuiltinTrackerConfig = BuiltinTrackerConfig()
    litetrack: LiteTrackConfig = LiteTrackConfig()
    cotracker: CoTrackerConfig = CoTrackerConfig()
    tracker_conf_threshold: float = 0.40
    max_track_age: int = 30
    confirm_detections: DetectionConfirmationConfig = DetectionConfirmationConfig()

    keep_tracking_on_missed_keyframe: KeepTrackingOnMissedKeyframeConfig = KeepTrackingOnMissedKeyframeConfig()
    backward_tracking: BackwardTrackingConfig = BackwardTrackingConfig()

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

    absence_check: AbsenceCheckConfig = AbsenceCheckConfig()

    # When pipeline.detector=geco2, GeCo2's own re-detect score (relative
    # per-frame, not cosine -- see geco2_detector.py) sometimes locks onto a
    # confuser object instead of correctly reporting "not found" (observed
    # as unrelated_false_positive runs via check_tracker_coverage.py). When
    # this is enabled, every GeCo2 re-detect (NoneTracker's per-frame loop
    # AND the active-tracker's re-detect-on-track-loss fallback) additionally
    # embeds each GeCo2 candidate box with DINOv2 and drops any candidate
    # whose cosine similarity to the prototype falls below the SAME
    # match_threshold Stage 3 used (adaptive z-score value when
    # stage3.adaptive_threshold is enabled, else the fixed config default --
    # see stage4.py's match_threshold loading) -- the best-scoring GeCo2 box
    # among the survivors is returned, or None if none survive. No effect on
    # the legacy pipeline (already cosine-gated) or when verify_interval's
    # own prototype.npz isn't available (needs stage123_geco2.cosine_rescore
    # .enabled, same requirement as verify_interval above).
    #
    # False (default) = disabled -- GeCo2 re-detect behavior unchanged.
    geco2_redetect_cosine_filter: bool = False

    kalman_motion_check: KalmanMotionCheckConfig = KalmanMotionCheckConfig()

    @field_validator("tracker")
    @classmethod
    def check_tracker(cls, v: str) -> str:
        allowed = {"builtin", "litetrack", "cotracker3", "none"}
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


class AutoScaleCalibrationConfig(BaseModel):
    """Per-sample automatic replacement for hand-tuning ref_downscale_factor
    (and, jointly, crop_context_margin): builds one candidate exemplar
    prototype per (crop_margin, downscale_factor) combination, scores each
    candidate's quality against a handful of frames sampled from THIS
    sample's own video, then does a quality-weighted SOFT BLEND of the
    per-candidate appearance tokens (default) or hard-selects the single
    best one. The blend happens BEFORE cross-attention, producing exactly
    one token per ref per pyramid level -- identical in shape to today's
    single-factor behavior, so nothing downstream changes and there is no
    train/inference token-count mismatch (unlike ref_downscale_levels,
    which flat-concatenates every scale into the K/V sequence -- a
    configuration GeCo2 was never trained on; see
    aero_eyes/models/geco2_finetune_data.py::sample_ref_downscale_factor).

    Both axes matter, not just ref_downscale_factor's blur/detail axis:
    reading GECO2/utils/data.py::resize_and_pad shows the exemplar's
    canvas-relative SIZE (controlled here via crop_context_margin, same
    mechanism as Stage123Geco2Config.crop_to_object) also changes the
    backbone's receptive-field/self-attention context around the object,
    independent of blur -- and this project's own docs/
    GECO2_baseline_scale_calibration_results*.md sweeps (16 videos, two
    disjoint sets) already show no single fixed size works across videos
    (optimal ratio to the object's true size ranges ~1.1x-2.9x depending on
    the object). Requires segmentation.enabled (crop_context_margin needs a
    tight mask box, same as crop_to_object).

    Mutually exclusive with ref_downscale_factor/ref_downscale_levels: when
    enabled, this OVERRIDES both (a warning is logged if either is set to a
    non-default value) -- see aero_eyes/stages/stage123_geco2.py::
    build_exemplar_prototype.

    NOT YET VALIDATED -- compare against your best manually-tuned
    ref_downscale_factor per sample (scripts/compare_auto_scale_vs_fixed.py)
    before trusting this in production.
    """
    enabled: bool = False
    # Candidate crop_context_margin values (see Stage123Geco2Config.
    # crop_context_margin's own docstring) -- larger margin = object
    # occupies a SMALLER fraction of the final canvas.
    candidate_crop_margins: list[float] = [0.5, 1.0, 2.0, 4.0]
    # Candidate ref_downscale_factor values (see Stage123Geco2Config.
    # ref_downscale_factor's own docstring) -- log-spaced by convention,
    # matching sample_ref_downscale_factor's log-uniform training
    # distribution.
    candidate_downscale_factors: list[float] = [1.0, 0.5, 0.25, 0.125, 0.0625, 0.03]
    # Frames sampled (deterministically, evenly spaced via np.linspace --
    # NOT randomly, so re-running with an unchanged config/video reproduces
    # an identical calibration and stays safe under the existing
    # project.use_cache contract) from the sample's own video to score each
    # candidate. One-time cost per sample, folded into the cached
    # prototype file: up to max_candidates * num_probe_frames extra query-
    # side backbone passes, plus max_candidates * num_references extra
    # ref-side backbone passes.
    num_probe_frames: int = 12
    # "soft" (default, recommended): quality-weighted average of every
    #   candidate's appearance token -- see class docstring.
    # "hard": one-hot select the single best-scoring candidate (closest
    #   analog to manually picking one (crop_margin, factor) pair, but
    #   chosen automatically per sample instead of by hand).
    selection_mode: Literal["hard", "soft"] = "soft"
    # "auto" (default): use gt_iou when cfg.data.gt.global_file has GT for
    #   this sample_id, else fall back to self_supervised_margin. Forcing
    #   "gt_iou" is only meaningful for offline dev/calibration on labeled
    #   data -- real deployment on an unlabeled video always falls back.
    # "gt_iou": mean IoU between the top-1 predicted box and GT, over this
    #   sample's own GT-present frames.
    # "self_supervised_margin": mean over probe frames of
    #   (max(raw_scores) - mean(raw_scores)) / (std(raw_scores) + eps) -- a
    #   well-matched exemplar scale should produce one strong, spatially
    #   localized peak. KNOWN RISK: a confidently-wrong high-scoring
    #   background patch can also produce a high, peaky margin -- this is a
    #   heuristic proxy, not a correctness guarantee. Prefer gt_iou
    #   whenever any GT exists.
    quality_metric: Literal["auto", "gt_iou", "self_supervised_margin"] = "auto"
    # Softmax temperature applied to per-candidate z-scored quality when
    # selection_mode="soft" -- lower = closer to hard selection, higher =
    # closer to a uniform blend across all candidates.
    temperature: float = 0.5
    # Caps len(candidate_crop_margins) * len(candidate_downscale_factors) --
    # the full grid can get expensive fast; candidates are sampled evenly
    # from the grid (not truncated from one end) when the product exceeds
    # this, with a warning logged.
    max_candidates: int = 12
    eps: float = 1e-6


class Geco2LearnedScaleFusionConfig(BaseModel):
    """Track B (docs/GECO2_scale_domain_gap_plan.md): inference-side use of
    a checkpoint finetuned with `--num-ref-scale-variants > 1`
    (scripts/train_geco2_aeroeyes.py) -- i.e. one that actually has a
    trained `scale_fusion_gates` submodule (aero_eyes/models/
    geco2_scale_fusion.py::ScaleFusionGate, one per pyramid level).

    Mirrors ref_downscale_levels' mechanism for BUILDING candidate
    exemplars (one entry per (ref image, factor in candidate_factors)) but
    FUSES them with the trained, query-conditioned gate instead of flat-
    concatenating into the K/V sequence -- see GeCo2Detector.
    encode_exemplars_fused. candidate_factors should cover roughly the same
    range the checkpoint was trained on (--ref-downscale-lo/hi).

    The gate needs a QUERY-image feature to condition on, but this
    happens once per sample (like encode_exemplars), before any specific
    query frame is known -- num_context_frames sample frames from the
    sample's own video are averaged into one proxy query context (same
    "sample a few frames, average" pattern as domain_calibration), trading
    true per-frame adaptivity for keeping the existing "build the
    prototype once, reuse across the whole video" architecture. Revisit
    with a genuinely per-frame version (fusing inside the per-keyframe
    loop instead) if this proxy underperforms.

    Mutually exclusive with ref_downscale_factor/ref_downscale_levels/
    auto_scale_calibration/scale_calibration -- overrides them all when
    enabled. Requires a checkpoint that actually has scale_fusion_gates
    weights (stage123_geco2.weights_path) -- a base (non-Track-B) checkpoint
    has none, so this always starts from a freshly-initialized (untrained,
    useless) gate in that case; a startup warning is logged if no
    "scale_fusion_gates." keys are found in the loaded checkpoint.

    NOT YET VALIDATED -- no Track B checkpoint has been trained/evaluated
    yet (requires a GPU, see scripts/train_geco2_aeroeyes.py). Compare
    against auto_scale_calibration and the manually-tuned baseline via
    scripts/compare_auto_scale_vs_fixed.py before trusting this.
    """
    enabled: bool = False
    candidate_factors: list[float] = [1.0, 0.5, 0.25, 0.125, 0.0625, 0.03]
    num_context_frames: int = 5
    scale_gate_hidden_dim: int = 64


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


class Geco2DynamicPrototypeTopKFusionConfig(BaseModel):
    """stage123_geco2.dynamic_prototype.topk_fusion -- ONLY wired into
    run_stage12_geco2_candidates (stage123_geco2.cosine_rescore.enabled=true)
    -- see this field's own placement docstring for why the plain
    run_stage123_geco2 path and cross_check_source="hiera" are deliberately
    excluded.

    Problem this addresses: dynamic_prototype normally offers ONLY
    boxes[0] (GeCo2's own highest-scoring candidate) per keyframe to the
    consecutive-hit confirmer -- if a confuser happens to outscore the
    real target THAT keyframe, the real target (even if present at rank
    2/3/...) is never even looked at, so a persistently-confusable scene
    can starve the confirmer of real hits indefinitely.

    When enabled, every keyframe considers ALL of GeCo2's surviving
    candidates (however many cosine_rescore.candidate_topk_per_keyframe
    kept -- no separate K parameter here) instead of just boxes[0]:
    computes a fused_score for each (see below), and offers the box with
    the HIGHEST fused_score to the confirmer -- so a real target can still
    win selection even when GeCo2's own raw score alone would have picked
    a confuser instead.

    fused_score_i = cosine_weight * cosine_z_i + (1 - cosine_weight) * geco2_z_i

    Both cosine_z_i (feature_extractor cosine to the reference prototype,
    same pooling as cross_check_threshold's own check) and geco2_z_i
    (GeCo2's own raw per-box score) are Z-SCORED against a RUNNING window
    of this SAME sample's own past chosen-candidate values (running_window
    keyframes), not compared to any fixed absolute number -- this is what
    makes the mechanism self-calibrating to a given reference-photo-set's
    OWN domain gap instead of needing a hand-tuned absolute cosine cutoff
    per video (the exact problem that forced cross_check_threshold down to
    an unreliable ~0.2 on some footage). Once a candidate is confirmed by
    the consecutive-hit gate, it's accepted if fused_score_i >=
    acceptance_z_threshold (0.0 = "at or above this sample's own running
    average"), REPLACING cross_check_threshold's plain absolute-cosine gate
    for the confirmed candidate's acceptance decision.

    Cold start: the Z-score has no meaning until running_window has
    accumulated at least min_window_for_zscore samples -- until then,
    offer_topk() falls back to exactly today's boxes[0]-only, plain
    cross_check_threshold behavior (still recording raw cosine/score into
    the running window so it warms up), so a run isn't left doing nothing
    for its first min_window_for_zscore keyframes.

    Why NOT wired into run_stage123_geco2 or cross_check_source="hiera":
    fused_score needs EVERY surviving candidate's cosine BEFORE knowing
    which one will be selected (selection itself depends on it) -- unlike
    today's design, which only ever embeds/cross-checks the ALREADY-
    confirmed single candidate (rare, event-driven). run_stage12_geco2_
    candidates already embeds every surviving candidate anyway (for
    candidates.json), so this costs nothing extra there; run_stage123_geco2
    embeds nothing today and would need K new feature_extractor calls per
    keyframe (not just on confirmation) to support this. cross_check_
    source="hiera" would need K encode_exemplars() GeCo2 backbone passes
    per keyframe instead of the current design's 1-per-confirmation --
    defeats the entire "keep this infrequent" premise the class's own
    docstring establishes. Both are silently ignored (topk_fusion has no
    effect) rather than erroring, with a one-time warning, if enabled
    together with either.
    """
    enabled: bool = False
    # w in the formula above; GeCo2's own weight is implicitly (1 - this).
    cosine_weight: float = 0.5
    # How many past keyframes' CHOSEN-candidate raw cosine/geco2 values to
    # keep for the running Z-score baseline (a simple deque, oldest evicted
    # once exceeded) -- larger is a more stable baseline but slower to
    # adapt if the object's own appearance/domain gap shifts partway
    # through the video (e.g. lighting change).
    running_window: int = 50
    # Cold-start guard: below this many accumulated samples, the Z-score
    # baseline is too noisy to trust (a window of 1-2 points has an
    # arbitrary std) -- offer_topk() behaves like plain offer() on boxes[0]
    # until the window warms up past this.
    min_window_for_zscore: int = 5
    # fused_score threshold to ACCEPT a confirmed candidate (0.0 = at or
    # above this sample's own running average of past chosen candidates).
    acceptance_z_threshold: float = 0.0
    # --------------------------------------------------------------------
    # The 3 knobs below each guard against the SAME failure mode: the
    # running_window baseline above is built from THIS SAME mechanism's own
    # past picks, so a persistently-confusable scene can get its confuser's
    # cosine/geco2 values baked into the baseline as "normal", which then
    # makes that SAME confuser (or a similar one) easily clear
    # acceptance_z_threshold on every later frame -- the baseline
    # rationalizes the confuser instead of catching it. Each is independent
    # and defaults to off (today's behavior unchanged) so they can be A/B
    # tested one at a time.
    # --------------------------------------------------------------------
    # false (default): Z-score baseline is the temporal running_window
    # history described above (unchanged behavior).
    # true: for a keyframe with >= intra_frame_min_boxes surviving
    # candidates, compute cosine_mean/std and geco2_mean/std from THIS
    # FRAME's OWN candidates instead of the temporal history -- every box
    # in one frame shares the same domain (same lighting/scale/moment), so
    # comparing a candidate against its own frame's siblings cancels out
    # domain gap entirely without needing any accumulated history, and
    # can't be poisoned by a confuser chosen in a PAST frame. Falls back to
    # the temporal history baseline (old behavior) for a frame with fewer
    # than intra_frame_min_boxes candidates, since 1-2 points can't
    # estimate a meaningful std either way.
    intra_frame_baseline: bool = False
    # Minimum surviving candidates a keyframe must have for
    # intra_frame_baseline's per-frame stats to be trusted -- below this,
    # falls back to the temporal running_window baseline for that frame
    # only (does not disable intra_frame_baseline for later frames).
    intra_frame_min_boxes: int = 3
    # false (default): the temporal running_window history (used as
    # intra_frame_baseline's fallback, and as the sole baseline when
    # intra_frame_baseline is off) records the CHOSEN candidate's raw
    # cosine/geco2 values on EVERY offer_topk() call, whether or not that
    # candidate ever passes consecutive-hit confirmation or the acceptance
    # gate -- so a repeatedly-argmax-winning confuser that never actually
    # gets appended still shapes the baseline.
    # true: only record a candidate's raw values once it's actually been
    # appended to the dynamic prototype -- a stricter, slower-to-warm-up
    # history that only ever reflects candidates this same mechanism has
    # already trusted, instead of everything it merely glanced at.
    history_update_on_append_only: bool = False
    # false (default): no absolute floor -- a candidate can be accepted
    # via fused_score alone even if its raw cosine is effectively noise
    # (e.g. the baseline itself has drifted down with it).
    # true: additionally require the candidate's raw (pre-Z-score) cosine
    # to be >= min_absolute_cosine_floor, regardless of how favorably it
    # Z-scores against the (possibly drifted) baseline -- same backstop
    # role as stage3.dynamic_prototype's own adaptive_min_floor.
    min_absolute_cosine_floor_enabled: bool = False
    min_absolute_cosine_floor: float = 0.05
    # Opt-in, only takes effect when min_absolute_cosine_floor_enabled is
    # also true (and per-ref vectors are available -- same requirement and
    # fallback as dynamic_prototype.cross_check_threshold_self_calibrate,
    # which this shares its ref-vs-ref self-similarity ceiling with).
    # Replaces the hand-set min_absolute_cosine_floor above with
    # ref_self_sim * min_absolute_cosine_floor_self_calibrate_ratio. Use a
    # LOWER ratio than cross_check_threshold_self_calibrate_ratio's --
    # this floor is meant as a sanity backstop under fused_score
    # acceptance, not the acceptance bar itself.
    min_absolute_cosine_floor_self_calibrate: bool = False
    min_absolute_cosine_floor_self_calibrate_ratio: float = 0.3


class Geco2DynamicPrototypeConfig(BaseModel):
    """stage123_geco2.dynamic_prototype -- ONLINE/incremental analog of
    stage3.dynamic_prototype for GeCo2's OWN exemplar tokens, for exactly
    the reason that batch version can't be ported as-is: stage3's version
    is a 2-pass mechanism (match the WHOLE video's candidates once, pick a
    percentile-of-the-whole-video's worth of high scorers, blend, re-score)
    -- but stage123_geco2.py processes keyframes in ONE sequential pass, so
    there is no "whole video" distribution to compute a percentile from
    until the video is already done (too late to have helped the frames
    already processed).

    Instead of a batch percentile, this appends new exemplar tokens to the
    prototype's own K/V token sequence AS THE VIDEO IS PROCESSED (GeCo2's
    adapt_features already treats prototype["main"/"l1"/"l2"] as an
    arbitrary-length token sequence -- see GeCo2Detector.encode_exemplars
    -- so appending more tokens needs no architecture change, just more
    entries along that same dimension), using a SLIDING WINDOW
    (max_tokens, FIFO eviction of the OLDEST appended token -- never the
    original reference-image tokens) instead of "the whole video minus a
    percentile cut".

    Trustworthiness problem: GeCo2's own score is RELATIVE-only (see
    ScoreConfig-adjacent fields' own docstrings, e.g. score_threshold_abs)
    -- a confidently-scored box can still be a confuser, and there is no
    reliable "percentile of a small streaming sample" to fall back on
    early in a video the way stage3's batch version can. So a candidate is
    NOT trusted on GeCo2's own score alone -- it must ALSO be confirmed by
    BOTH:
      1. min_consecutive_hits consecutive keyframes whose best box
         spatially agrees (same mechanism as stage4.confirm_detections'
         own DetectionConfirmer -- see aero_eyes.utils.detection_confirm)
         -- a single spurious high-scoring frame can't poison the
         prototype on its own.
      2. cross_check_source's cosine similarity to a reference embedding
         clearing cross_check_threshold -- see that field's own docstring
         for the "feature_extractor" vs "hiera" tradeoff.
    Only once BOTH agree does the candidate's exemplar tokens get appended.

    Not yet benchmarked -- compare with scripts/check_box_refine_effect.py-
    style before/after runs on your own footage before trusting it, same
    as every other opt-in accuracy knob in this project.
    """
    enabled: bool = False
    # Sliding-window cap on how many EXTRA (appended) exemplar token-sets
    # are kept -- FIFO eviction of the oldest once exceeded. The original
    # reference-image tokens are never evicted, only what this mechanism
    # itself added. Cross-attention cost scales with total token count, so
    # this also bounds the extra compute dynamic_prototype adds per frame.
    max_tokens: int = 5
    # false (default): once max_tokens is reached, a new accepted candidate
    # STILL gets appended, evicting the OLDEST appended token (FIFO) --
    # the active set keeps drifting to reflect the most RECENT accepted
    # appearances of the target.
    # true: once max_tokens is reached, stop accepting new tokens entirely
    # -- the set locked in first stays fixed for the rest of the video,
    # instead of being replaced by whatever gets accepted later. Useful
    # when early-video acceptances are trusted more than later ones (e.g.
    # the target's appearance is expected to stay stable, and you'd rather
    # keep a known-good early set than risk a later confuser slipping in
    # and evicting it) -- the flip side of FIFO's own tradeoff (a later
    # confuser can push out a genuinely good early token; freezing removes
    # that risk but also removes FIFO's ability to adapt to a target whose
    # appearance genuinely drifts over the video).
    freeze_when_full: bool = False
    # IMPORTANT: "consecutive" here means consecutive PROCESSED KEYFRAMES,
    # not consecutive video frames -- GeCo2's own detect_frame() only ever
    # runs at stage123_geco2.keyframe_interval spacing (default 8), and
    # this tracker is fed exactly those same keyframes' best boxes, so two
    # "consecutive" hits are actually keyframe_interval RAW FRAMES apart
    # (e.g. ~0.27s at 30fps with the default interval=8). That's enough
    # time for a small/rotating object (a handheld ID card, not a car or
    # person) to move/turn far more than adjacent-frame tracking would,
    # so consecutive_hits_iou is a STRICTER bar than its number suggests --
    # confirmed in practice: a run can sit at n_confirmed==0 in
    # log_summary() indefinitely even with a perfectly good detector, if
    # the target rotates/moves enough between keyframes to keep missing
    # this threshold. Lower consecutive_hits_iou (or keyframe_interval
    # itself, at the cost of more GeCo2 forward passes per video) if
    # log_summary() shows offers >> confirmed.
    #
    # Confusingly similar-sounding but NOT the same cadence as
    # stage4.confirm_detections' own DetectionConfirmer instance, which
    # this shares its CODE with (aero_eyes.utils.detection_confirm) but not
    # its calling cadence -- that one runs on EVERY video frame when
    # stage4.tracker=none (genuine adjacent-frame comparisons), and only at
    # keyframes otherwise. This tracker has no non-keyframe granularity to
    # fall back to at all, since stage123_geco2.py never runs GeCo2 between
    # keyframes.
    min_consecutive_hits: int = 2
    consecutive_hits_iou: float = 0.5
    # "feature_extractor" (default, RECOMMENDED): embeds the candidate crop
    #   with stage1.feature_extractor (whatever model is configured there --
    #   dinov2/dinov3/clip/siglip/ensemble) and compares against Stage 1's
    #   OWN prototype.npz (the SAME reference-image embedding
    #   stage4.geco2_redetect_cosine_filter already cross-checks GeCo2
    #   re-detects against) -- a genuinely INDEPENDENT model from GeCo2's
    #   own Hiera backbone, so it can catch a confuser that fools GeCo2's
    #   own score without sharing GeCo2's own blind spots.
    # "hiera": reuses GeCo2's OWN backbone feature (an appearance token
    #   from GeCo2Detector.encode_exemplars applied to the candidate crop)
    #   compared against the CURRENT effective prototype's own exemplar
    #   tokens, avoiding a second model/extra load entirely. NOT actually
    #   independent -- GeCo2's own score is ALREADY effectively a learned
    #   similarity between the SAME two things in the SAME Hiera space
    #   (see adapt_features), so a confuser that fools the score is likely
    #   to also score high here (correlated failure, not an independent
    #   second opinion). Exposed for you to A/B test yourself -- don't
    #   assume it works as well as "feature_extractor" without checking.
    cross_check_source: Literal["feature_extractor", "hiera"] = "feature_extractor"
    cross_check_threshold: float = 0.5
    # Opt-in, only takes effect when cross_check_source="feature_extractor"
    # and Stage 1 saved per-ref vectors (accuracy.cheap_boosters.
    # multi_reference_embedding=true) -- silently falls back to the plain
    # cross_check_threshold above (one-time warning) otherwise. Instead of
    # a hand-tuned absolute cosine cutoff (has to be re-tuned per video's
    # own domain gap -- the exact problem that forces it down to an
    # unreliable ~0.2 on some footage), computes the MIN pairwise cosine
    # among the 3 reference images' OWN embeddings (an empirical ceiling on
    # "how similar do two genuinely-matching crops of THIS domain look",
    # measured once per sample from data already on hand) and uses
    # ref_self_sim * cross_check_threshold_self_calibrate_ratio as the
    # EFFECTIVE threshold in its place. If the domain gap is severe enough
    # that even the 3 refs don't look alike to feature_extractor, the
    # threshold drops right along with them -- no manual per-video retune.
    # Governs offer()'s own gate AND offer_topk()'s cold-start/fallback
    # gate (topk_fusion's own warmed-up fused_score/acceptance_z_threshold
    # gate is unaffected -- see min_absolute_cosine_floor_self_calibrate on
    # topk_fusion for the equivalent there).
    cross_check_threshold_self_calibrate: bool = False
    cross_check_threshold_self_calibrate_ratio: float = 0.7
    # Opt-in second full sweep over the video after pass 1 finishes: pass 1
    # runs exactly as described above (online accumulation via offer()),
    # and if it accepted at least one dynamic token, pass 2 re-detects
    # every keyframe from frame 1 again using that FINAL prototype --
    # frozen, no more offer()/consecutive-hit confirmation -- instead of
    # pass 1's prototype which only grows richer as the video progresses.
    # This targets the specific weakness of an online/incremental
    # mechanism: EARLY frames only ever saw the 3 original reference
    # images, so a target whose on-video appearance differs a lot from
    # those refs (different angle/lighting/scale) can be missed right at
    # the start even though by the end of pass 1 the accumulated tokens
    # would have caught it easily. Pass 2's detections/candidates and
    # (if runtime.save_visualizations) viz REPLACE pass 1's outright --
    # this is meant as the final answer, not a merge of both passes.
    # Costs roughly 2x the per-video GeCo2 forward-pass time. No-op
    # (skipped, pass 1's result stands) when pass 1 accepted zero dynamic
    # tokens, since pass 2 would then be identical to pass 1.
    second_pass: bool = False
    topk_fusion: Geco2DynamicPrototypeTopKFusionConfig = Geco2DynamicPrototypeTopKFusionConfig()


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
    # GeCo2's box regression head has no guarantee against a degenerate,
    # near-zero-area box (e.g. one edge collapsing to a couple pixels) --
    # unlike the legacy pipeline's stage2.candidate.min_box_area, nothing
    # here rejects one, and a tiny/sliver crop later chokes feature
    # extractors (transformers' image processor logs "channel dimension is
    # ambiguous" for a near-square tiny shape and has to guess). Surveyed
    # this project's own GT (annotations_converted.json, 14233 boxes across
    # 10 videos): smallest real box area is 24px^2, while a real box's
    # THINNEST SIDE can legitimately be as low as 2px (an object clipped at
    # the frame edge, entering/leaving frame) -- so filtering by min side
    # length would reject real objects; filtering by min AREA does not
    # (confirmed 0 real GT boxes at or below area 16). min_box_area's
    # default (24) matches that smallest observed real box exactly -- tune
    # down if your own dataset has smaller real objects, but validate
    # against your own GT first (see scripts/check_box_size_bias.py-style
    # analysis) rather than guessing.
    min_box_area_enabled: bool = False
    min_box_area: int = 24
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
    # Applies whether or not segmentation.enabled -- with it off, shrinks
    # the RAW (unmasked) reference image instead of the masked/background-
    # filled/cropped one.
    ref_downscale_factor: float = 1.0
    # Multi-blur appearance-token ensemble (opt-in, config toggle since we
    # don't yet know if it helps): when set (non-empty), OVERRIDES
    # ref_downscale_factor above -- instead of shrinking each ref image by
    # ONE fixed factor, builds one exemplar entry PER (ref image, factor in
    # this list), all concatenated into a single exemplar token sequence
    # (exactly like scale_calibration.multi_scale_mode="all" already does
    # for canvas size -- adapt_features attends over the prototype as a
    # flat K/V sequence regardless of token count, so this needs no model
    # changes). Lets cross-attention pick whichever blur/detail level best
    # matches a given query object's own apparent scale, instead of a
    # single hand-picked ref_downscale_factor that may only suit one
    # altitude/distance. null (default) = old single-factor behavior,
    # unchanged. Combines with scale_calibration.multi_scale_mode="all" if
    # both are enabled (their entries stack).
    ref_downscale_levels: Optional[list[float]] = None
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
    auto_scale_calibration: AutoScaleCalibrationConfig = AutoScaleCalibrationConfig()
    learned_scale_fusion: Geco2LearnedScaleFusionConfig = Geco2LearnedScaleFusionConfig()
    scale_calibration: ScaleCalibrationConfig = ScaleCalibrationConfig()
    domain_calibration: DomainCalibrationConfig = DomainCalibrationConfig()
    dynamic_prototype: Geco2DynamicPrototypeConfig = Geco2DynamicPrototypeConfig()
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
class AdaptiveContextMarginConfig(BaseModel):
    """box_refine.adaptive_context_margin -- box_refine.context_margin is
    ONE flat value applied to every box regardless of size, but the right
    amount of margin is size-dependent: a large, well-defined object (e.g.
    a motorbike) benefits from a generous margin (more room for SAM/GrabCut
    to find the true boundary), while a tiny/ambiguous object (e.g. a
    helmet, ~15x12px) is more likely to have that same margin sweep in a
    nearby confuser or background clutter (see BoxRefineConfig.
    min_iou_with_original's own docstring for this exact failure mode) --
    confirmed in practice: context_margin=0.5 helped a motorbike sample a
    lot but hurt a helmet sample's recall, while context_margin=0.0 was
    the better choice for the helmet sample specifically.

    When enabled, the EFFECTIVE margin used for a given box is
    context_margin scaled by how the box's own size (sqrt(w*h), the same
    geometric-mean-side metric scripts/check_iou_size_sensitivity.py uses)
    falls between min_size_px and max_size_px:
      size <= min_size_px  -> effective margin = context_margin * min_ratio
      size >= max_size_px  -> effective margin = context_margin (unscaled)
      in between            -> linearly interpolated
    So a tiny box automatically gets little/no margin (avoiding the
    confuser-sweep risk) while a large box still gets the full configured
    margin -- one context_margin value serves both object sizes instead of
    having to hand-pick a per-sample value.

    Applies to context_margin everywhere it's used: the "sam"/"grabcut"
    crop margin (refine_box) AND the "sam_dense" prompt-expansion margin
    (refine_boxes_dense) -- same underlying risk in both.

    False (default) = disabled, context_margin is used as-is for every box
    regardless of size, unchanged from before this option existed.

    relative_to_sample_median: min_size_px/max_size_px above compare a box
    against a FIXED, absolute pixel scale -- they can't tell a genuinely
    tiny object (small in every frame of its own video -- shrinking its
    margin is correct) apart from a normally larger object that THIS one
    box just badly undersizes (e.g. GeCo2 detected only ~40px of a
    motorbike that measures ~150px everywhere else in the same video --
    shrinking its margin here is exactly backwards: SAM needs MORE room to
    reach the true boundary, not less). When enabled, the caller also
    computes this SAMPLE's own reference size (its median confident-
    detection size for the same video) and passes it down; if a box's own
    size falls below reference_size * relative_undersize_ratio, the normal
    min_size_px/max_size_px shrink is skipped for that box and the FULL,
    unscaled context_margin is used instead -- regardless of where the
    box's absolute size would otherwise land on that curve. Only affects a
    box that's anomalously small RELATIVE TO ITS OWN SAMPLE; a sample whose
    typical size is itself small (e.g. the helmet case above) is unaffected
    since its boxes stay close to its own reference size.

    False (default) = disabled -- only the absolute min/max_size_px curve
    applies, unchanged from before this option existed.
    """
    enabled: bool = False
    min_size_px: float = 20.0   # box geometric-mean side (px) at/below which margin -> min_ratio
    max_size_px: float = 100.0  # box geometric-mean side (px) at/above which margin -> full context_margin
    relative_to_sample_median: bool = False
    # Below this fraction of the sample's own reference size, a box is
    # treated as anomalously undersized rather than genuinely small.
    relative_undersize_ratio: float = 0.5
    min_ratio: float = 0.0      # fraction of context_margin used at/below min_size_px (0.0 = no margin at all)


class BoxRefineConfig(BaseModel):
    """Sharpens an imprecise detection/tracking box to tightly fit the
    actual object silhouette, via a lightweight per-box segmentation pass.
    Addresses the "box not tight" (localization imprecision) component of
    ST-IoU loss identified via scripts/check_st_iou_breakdown.py --
    distinct from the tracking-COVERAGE component stage4.verify_interval
    addresses (this doesn't help a box that's simply MISSING, only one
    that IS present but loosely placed).

    method:
      sam        -- MobileSAM prompted directly with a small CROP around
                    the box (reuses the same weights already used for
                    Stage 1/GeCo2 reference segmentation, via
                    stage1.segmentation.weights). Empirically confirmed
                    UNRELIABLE on small/low-res candidate boxes (as small
                    as ~20x10px): the crop is too tiny/low-info for SAM to
                    segment consistently -- kept for comparison, prefer
                    "sam_dense" below.
      sam_dense  -- MobileSAM, but encodes the WHOLE frame ONCE (no crop)
                    and reuses that single embedding for every box prompt
                    on it -- same "encode once, reuse for every box"
                    principle GeCo2's own SAM2-based sam_mask module uses
                    on its own dense backbone features (GeCo2's Hiera
                    features can't literally be reused with MobileSAM's
                    SAM1-style decoder -- different training distribution
                    -- so this re-encodes with MobileSAM's own encoder
                    instead, avoiding a second SAM2 checkpoint download).
                    Costs one MobileSAM forward pass per KEYFRAME (not per
                    box) instead of per box -- usually cheaper than "sam"
                    when topk_per_keyframe > 1, and avoids the small-crop
                    reliability problem "sam" has.
      grabcut    -- classic OpenCV GrabCut seeded from the box. No model,
                    no weights, much cheaper, noticeably lower quality
                    than either SAM method.
      sam2_dense -- GeCo2's OWN SAM2-based mask refinement
                    (GECO2/models/sam_mask.py::MaskProcessor, the same
                    submodule CNT.forward itself calls internally but this
                    pipeline's detect_frame() otherwise skips). Only
                    available when pipeline.detector == "geco2" (needs a
                    GeCo2Detector + its cached exemplar prototype -- see
                    stage123_geco2.prototype_cache_name). Refines using
                    GeCo2's OWN dense Hiera backbone features -- no crop or
                    second encoder, unlike "sam"/"sam_dense" -- at the cost
                    of one extra GeCo2 backbone forward pass per refined
                    FRAME (shared across every box on it, not per box) and
                    a one-time download of Meta's public pretrained SAM2
                    checkpoint (sam2_hiera_base_plus.pt, ~300+MB) on first
                    use. Not yet benchmarked on this project's dataset the
                    way "sam"/"grabcut" were (see the IMPORTANT note
                    below) -- compare with scripts/check_box_refine_effect.py
                    and check_box_size_bias.py before trusting it.
      sam2_native -- a GENUINELY independent SAM2: its OWN image encoder
                    (Hiera-base-plus) AND OWN mask decoder, both from the
                    SAME public checkpoint, run via a fresh encode pass on
                    each refined FRAME (aero_eyes.models.segmentation.
                    SAM2Segmenter) -- unlike "sam2_dense", which decodes
                    from GeCo2's OWN detection-backbone features instead of
                    running SAM2's encoder itself. In THIS project's GeCo2
                    checkpoint the two are numerically close to equivalent
                    (GECO2/train.sh passes --backbone_lr 0, so GeCo2's
                    backbone stays frozen at that SAME checkpoint's original
                    weights -- see GeCo2DynamicPrototypeTracker's neighbor
                    discussion in aero_eyes/models/geco2_detector.py) --
                    this method exists to verify that empirically (compare
                    against sam2_dense on the same footage) rather than
                    assume it, and to keep working correctly as an honestly
                    independent baseline regardless of what any FUTURE
                    GeCo2 checkpoint's backbone_lr was trained with. Needs
                    the vendored GECO2/sam2 package's own hydra-core/
                    omegaconf dependencies (see GECO2/install.sh) in
                    addition to pipeline.detector=geco2's own requirements.
                    Only available when pipeline.detector == "geco2" (reuses
                    the vendored GECO2/sam2 package -- see
                    stage123_geco2.repo_path). Costs one FULL SAM2 encoder
                    forward pass per refined FRAME on top of whatever
                    detector/tracker already ran -- heavier per-frame than
                    "sam_dense" (MobileSAM's much smaller encoder) and than
                    "sam2_dense" (reuses GeCo2's already-computed features
                    for free). Not yet benchmarked -- compare with
                    scripts/check_box_refine_effect.py before trusting it.
      fastsam_dense -- FastSAM-s (stage2.fastsam_s weights) "segment
                    everything" run ONCE per frame, then whichever
                    already-produced instance mask best matches a given
                    box (by IoU, or by point-containment when
                    use_center_point_prompt is on) is picked afterward --
                    see aero_eyes.models.segmentation.FastSAMSegmenter's
                    own docstring. Unlike sam/sam_dense/sam2_dense's
                    promptable decoders, FastSAM CANNOT generate a new mask
                    conditioned on the box -- it can only select among
                    whatever the everything-pass already segmented, so it
                    hits a ceiling those methods don't when the true
                    object wasn't cleanly its own instance there (merged
                    with a neighbor, or missed outright -- a known weak
                    point on small objects). Not yet benchmarked -- compare
                    with scripts/check_box_refine_effect.py before trusting it.

    IMPORTANT (measured on this dataset, not just theoretical): whether
    ANY of these methods helps or hurts ST-IoU is highly dependent on
    whether the ORIGINAL box already runs larger or smaller than its GT
    box on that particular video -- shrinking an already-undersized box
    makes it worse; shrinking an oversized one helps. There is no way to
    know that direction at real inference time (no GT then). See
    scripts/check_box_size_bias.py and check_box_refine_effect.py before
    trusting this on a new dataset -- averaged across this project's 6
    labeled samples, method="sam" was roughly NET-NEUTRAL (helped 3,
    hurt 3), not a reliable universal win.

    apply_in_stage3: refine the FINAL box Stage 3's cosine matching picked,
      once per keyframe (see aero_eyes/stages/stage3.py) -- cheap (bounded
      by keyframe count x topk_per_keyframe), safe to try first.
    apply_in_stage4: ALSO refine periodically during Stage 4 tracking,
      piggybacked on the SAME cadence as stage4.verify_interval (no
      separate interval here -- only takes effect when
      stage4.verify_interval > 0), re-initializing the tracker from the
      refined box so both drift AND box shape get corrected together. More
      expensive: runs during tracking, not just at keyframes.

    Disabled by default -- boxes are used exactly as the detector/tracker
    produced them, unchanged.
    """
    enabled: bool = False
    method: Literal["sam", "sam_dense", "grabcut", "sam2_dense", "fastsam_dense", "sam2_native"] = "sam"
    # Padding kept around the original box, as a fraction of the box's own
    # width/height. For "sam"/"grabcut": how much extra context to include
    # when CROPPING the region that gets segmented, so the segmenter isn't
    # starved of surrounding context right at the box edge. For "sam_dense":
    # expands the box PROMPT itself before querying SAM's (already
    # full-frame-encoded) embedding -- without this, SAM's box-conditioned
    # decoder tends to stay close to whatever box it's given, so an
    # UNDERSIZED detector box (e.g. only ~60% of the true object) rarely
    # gets expanded back out even with min_iou_with_original=0.0 (confirmed
    # in practice -- see MobileSAMSegmenter.segment_box_cached's own
    # docstring). Also honored by "sam2_dense" via GeCo2Detector.
    # sam2_refine_boxes's own wrapper (see its docstring) -- 0.0 (default)
    # keeps that method's original, unexpanded box-only prompting.
    context_margin: float = 0.2
    adaptive_context_margin: AdaptiveContextMarginConfig = AdaptiveContextMarginConfig()
    # Affects method="sam"/"sam_dense" (MobileSAM) AND "sam2_dense" (via
    # GeCo2Detector.sam2_refine_boxes's own wrapper around GECO2's
    # MaskProcessor -- see that method's docstring; nothing inside GECO2/
    # itself is touched). Also passes the ORIGINAL (pre-margin) box's own
    # center as a positive point prompt, alongside the (possibly
    # margin-expanded) box prompt -- a bigger context_margin gives SAM room
    # to reach the true boundary of an undersized box, but also more
    # background/confuser area it could latch onto instead; the center
    # point pins down WHICH blob in that wider region is the target,
    # without needing to shrink the margin. Off by default: this project's
    # OWN reference-image segmentation (stage1.segmentation's
    # use_point_prompt) found a center point unreliable for ring/donut-
    # shaped objects (hollow center = background, not foreground, biasing
    # SAM toward leaked/confused masks) -- only enable this if none of your
    # tracked object classes are shaped like that. Not yet benchmarked on
    # this project's own dataset -- compare with
    # scripts/check_box_refine_effect.py before trusting it, same as every
    # other box_refine.method choice (see this class's own IMPORTANT note
    # above).
    use_center_point_prompt: bool = False
    # method="sam2_dense" ONLY. GECO2's own MaskProcessor (GECO2/models/
    # sam_mask.py) always takes a HARD-CODED mask candidate (index 2 of the
    # 4 SAM2 mask_decoder produces when multimask_output=True) regardless
    # of that candidate's own predicted-IoU score -- a choice tuned for
    # GECO2's own counting-task validation path, not necessarily the best
    # one for refining an arbitrary tracked box on this project's own
    # footage. When true, GeCo2Detector.sam2_refine_boxes's wrapper instead
    # picks whichever of the 3 multimask candidates SAM2 itself scored
    # highest per box -- same "trust the model's own confidence" principle
    # MobileSAMSegmenter.segment_box_cached's own `scores.argmax()` already
    # uses for "sam_dense". False (default) = GECO2's original index-2
    # choice, unchanged. Not yet benchmarked -- compare with
    # scripts/check_box_refine_effect.py before trusting it.
    sam2_dense_select_best_mask: bool = False
    apply_in_stage3: bool = True
    apply_in_stage4: bool = False
    # Reject a refined box whose IoU with the ORIGINAL (pre-refine) box
    # falls below this -- guards against the segmenter latching onto a
    # sub-part, a nearby confuser, or background clutter within the padded
    # crop instead of the intended object (confirmed to happen in
    # practice: on small/low-res candidate boxes -- as small as ~20x10px,
    # see ColorPostfilterConfig's docstring for the same small-crop issue
    # elsewhere -- SAM sometimes segments an entirely different region,
    # silently replacing a decent box with a much worse one and TANKING
    # ST-IoU rather than improving it). 0.0 = no sanity check (accept
    # whatever the segmenter returns, even a wildly different region).
    min_iou_with_original: float = 0.3


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
    box_refine: BoxRefineConfig = BoxRefineConfig()

    @model_validator(mode="after")
    def check_litetrack_path(self) -> "AeroEyesConfig":
        if self.stage4.tracker == "litetrack":
            missing = [
                f for f in ("onnx_path_z", "onnx_path_x")
                if not getattr(self.stage4.litetrack, f)
            ]
            if missing:
                raise ValueError(
                    f"stage4.tracker is 'litetrack' but stage4.litetrack.{missing[0]} is not set. "
                    "Export both ONNX graphs from a trained checkpoint with "
                    "LiteTrack/tracking/export_litetrack_onnx.py and set "
                    "stage4.litetrack.onnx_path_z / onnx_path_x in your config."
                )
        return self

    def sample_work_dir(self, sample_id: str) -> Path:
        return Path(self.project.work_dir) / sample_id

    def device(self) -> str:
        # AERO_EYES_DISABLE_CUDNN is applied at module-import time above,
        # not here -- see that comment for why.
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
