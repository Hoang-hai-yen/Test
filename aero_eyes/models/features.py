"""Feature extractors for Stage B (prototype encoding + candidate matching).

Supported models:
  dinov2   — DINOv2 ViT-S/14 or ViT-B/14 (384 or 768-d). pooling="cls"
             (default) is the single global CLS token, as before.
             pooling="multiscale_attn" is NOT YET VALIDATED -- see
             DINOv2FeatureExtractor's own docstring. Optionally the "with
             registers" variant (dinov2_use_registers) -- same output dim,
             cleaner attention/features per Meta's ablations.
  dinov3   — DINOv3 ViT-S/16, ViT-B/16 or ViT-L/16 (384/768/1024-d). Same
             pooling="cls"/"multiscale_attn" choice as dinov2 above (see
             DINOv3FeatureExtractor's own docstring; multiscale_attn
             requires dinov3_source="huggingface").
             dinov3_source="huggingface" (default): weights gated on
             HuggingFace -- request access and set HF_TOKEN before use.
             dinov3_source="kaggle": loads a raw Meta .pth checkpoint (e.g. a
             satellite-pretrained variant) via kagglehub.model_download,
             no HuggingFace gating needed.
  clip     — CLIP ViT-B/32, visual encoder (512-d)
  siglip   — SigLIP vision encoder (base/large/so400m), pooled output
             (768/1024/1152-d). Open access, no gating.
  ensemble — DINOv2 (default) or DINOv3 + CLIP concatenated then
             L2-normalized (dino_dim + clip_dim, e.g. 1280 for vitb14/
             vitb16 + vit-b/32) -- see FeatureExtractorConfig.
             ensemble_dino_model.
  fgclip   — FG-CLIP (base/large), a CLIP variant fine-tuned with ~10M hard
             fine-grained negative pairs (512/768-d). NOT YET VALIDATED.
  radio    — NVIDIA RADIO / C-RADIO, a backbone distilled from multiple
             teacher VFMs at once (DINOv2/v3 + CLIP/SigLIP2 + SAM/SAM3),
             pooled "summary" embedding. NOT YET VALIDATED.
  siglip2  — SigLIP2 (base/so400m), adds Global-Local + Masked Prediction
             losses over SigLIP for better local/dense semantics. Standard
             transformers classes, no trust_remote_code. NOT YET VALIDATED.
  evaclip  — EVA02-CLIP-B/16 (~150M) via the open_clip_torch library (a new
             dependency). NOT YET VALIDATED.
  dinotxt  — dino.txt: a text encoder LiT-aligned to a FROZEN DINOv2
             ViT-L/14 (w/ registers) backbone -- adds language/semantic
             grounding without leaving the DINO family. Same torch.hub
             mechanism as model="dinov2". NOT YET VALIDATED -- see
             DinoTxtFeatureExtractor's own docstring for details that could
             not be independently verified without a live download.

All extractors return L2-normalized float32 feature vectors.
"""
from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from aero_eyes.types import Box
from aero_eyes.utils.geometry import crop_with_pad

log = logging.getLogger(__name__)

# ImageNet normalization (DINOv2)
_DINO_MEAN = (0.485, 0.456, 0.406)
_DINO_STD  = (0.229, 0.224, 0.225)


# ---------------------------------------------------------------------------
# Preprocessing helpers
# ---------------------------------------------------------------------------

def _preprocess_dino(img_bgr: np.ndarray, image_size: int = 224) -> torch.Tensor:
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb).resize((image_size, image_size), Image.BICUBIC)
    mean = np.array(_DINO_MEAN, dtype=np.float32)
    std  = np.array(_DINO_STD,  dtype=np.float32)
    arr  = np.array(img_pil, dtype=np.float32) / 255.0
    arr  = (arr - mean) / std
    return torch.from_numpy(arr.transpose(2, 0, 1).copy())


def _bgr_to_pil(img_bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))


def _ensure_transformers_onnx_shim() -> None:
    """FG-CLIP's own remote code (qihoo360/fg-clip-*'s modeling_clip.py,
    executed via trust_remote_code=True) does `from transformers.onnx
    import OnnxConfig` at import time -- a module removed from recent
    transformers releases (ONNX export tooling moved to the separate
    `optimum` package), causing a hard ModuleNotFoundError before
    FGCLIPFeatureExtractor ever gets a chance to run, independent of
    anything in THIS project. OnnxConfig is only referenced there for
    legacy/typing purposes and is never actually instantiated for a
    vision-only forward pass (get_image_features()), so a minimal stub
    class satisfies the import without needing to downgrade transformers
    itself (which risks breaking DINOv3/SigLIP2's own from_pretrained calls
    elsewhere in this file that may need a newer version). No-op if
    transformers.onnx already imports fine AND exposes OnnxConfig (older
    transformers installs, or a future release that restores it) -- never
    overrides a real module.

    Checks sys.modules directly rather than relying solely on a bare
    `import transformers.onnx` statement's own ImportError -- transformers'
    top-level package uses a lazy-module system whose submodule resolution
    does not reliably short-circuit via sys.modules the way a plain
    package's would, making a bare import statement alone unpredictable to
    reason about (and to test) across transformers versions.
    """
    import sys
    existing = sys.modules.get("transformers.onnx")
    if existing is not None and hasattr(existing, "OnnxConfig"):
        return
    try:
        import transformers.onnx as real_onnx
        if hasattr(real_onnx, "OnnxConfig"):
            return
    except ImportError:
        pass
    import types
    shim = types.ModuleType("transformers.onnx")
    shim.OnnxConfig = type("OnnxConfig", (), {})
    sys.modules["transformers.onnx"] = shim


def _ensure_fgclip_subconfigs(config: Any, hf_name: str) -> None:
    """FG-CLIP's own remote code (qihoo360/fg-clip-*, loaded via
    trust_remote_code) does not always convert its `text_config`/
    `vision_config` sub-fields from the plain dict read out of config.json
    into CLIPTextConfig/CLIPVisionConfig instances -- something
    transformers.CLIPConfig's own __init__ has historically done
    automatically. Another version-skew symptom in FG-CLIP's hosted code
    (same category as _ensure_transformers_onnx_shim above), surfacing as
    "config.text_config is expected to be of type CLIPTextConfig but is of
    type <class 'dict'>" from modeling_fgclip.py's own constructor.

    IMPORTANT: modeling_fgclip.py's isinstance check is against ITS OWN
    vendored CLIPTextConfig/CLIPVisionConfig classes (bundled alongside
    modeling_fgclip.py in the same trust_remote_code download), NOT
    transformers.CLIPTextConfig -- same name, but a DIFFERENT class object,
    so constructing with the standard library's class still fails the
    check (confirmed: the error message literally names
    "transformers.models.clip.configuration_clip.CLIPTextConfig" as the
    WRONG type once that fix was tried). get_class_from_dynamic_module
    fetches the exact class modeling_fgclip.py itself resolves, from the
    same cached module -- guaranteed identity match regardless of where
    FG-CLIP's bundle actually defines/imports it from.

    Patches both sub-configs IN PLACE unless already an instance of the
    correct (vendored) class; a no-op once the underlying bug is fixed
    upstream, so this stays harmless if it ever gets fixed.
    """
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    CLIPTextConfig = get_class_from_dynamic_module("modeling_fgclip.CLIPTextConfig", hf_name)
    CLIPVisionConfig = get_class_from_dynamic_module("modeling_fgclip.CLIPVisionConfig", hf_name)

    text_config = getattr(config, "text_config", None)
    if text_config is not None and not isinstance(text_config, CLIPTextConfig):
        kwargs = text_config if isinstance(text_config, dict) else text_config.to_dict()
        config.text_config = CLIPTextConfig(**kwargs)

    vision_config = getattr(config, "vision_config", None)
    if vision_config is not None and not isinstance(vision_config, CLIPVisionConfig):
        kwargs = vision_config if isinstance(vision_config, dict) else vision_config.to_dict()
        config.vision_config = CLIPVisionConfig(**kwargs)


def _fix_fgclip_position_ids(model: Any) -> None:
    """FG-CLIP's own vendored modeling_clip.py registers the vision
    embeddings' `position_ids` buffer as a PERSISTENT buffer (an old,
    long-since-fixed pattern in upstream transformers' own CLIP
    implementation -- current transformers marks this buffer
    persistent=False precisely because it's a deterministic
    torch.arange(num_positions) sequence, never a trained value, and
    should never be read from a checkpoint's state_dict). Because it's
    persistent here, loading FG-CLIP's checkpoint OVERWRITES the correct
    arange sequence with whatever raw (and in practice, garbage/
    uninitialized-looking) values happen to be stored under that key in
    the checkpoint file -- confirmed empirically on this project's own
    diagnostic run: position_ids.max() came back as 352951806590982, wildly
    outside the valid [0, num_positions) range, causing an out-of-bounds
    CUDA embedding-lookup crash inside self.position_embedding(
    self.position_ids) that looked (from the crash site alone) like an
    image-size/config mismatch but was neither -- config.vision_config's
    image_size/patch_size and the actual processor output were BOTH
    confirmed correct and mutually consistent (196 patches + 1 CLS = 197,
    exactly matching the position_embedding table's own size) before this
    was found to be the real cause.

    Recomputes the buffer as the correct, checkpoint-independent
    torch.arange(num_positions) sequence, in place, for whichever vision
    tower attribute path this loaded model actually has (varies across
    forks/versions of this vendored code -- checked defensively rather
    than assumed to be at one single fixed path). A no-op (silently) if
    none of the checked paths exist, so this doesn't newly break a future
    FG-CLIP release that fixes persistence upstream or restructures its
    module layout.
    """
    candidates = [
        getattr(getattr(model, "vision_model", None), "embeddings", None),
        getattr(getattr(getattr(model, "vision_model", None), "vision_model", None), "embeddings", None),
    ]
    for embeddings in candidates:
        position_ids = getattr(embeddings, "position_ids", None)
        if position_ids is None:
            continue
        num_positions = position_ids.shape[-1]
        correct = torch.arange(num_positions, device=position_ids.device).expand(1, -1)
        embeddings.position_ids = correct
        log.info(
            "FG-CLIP: reset a persistent (checkpoint-corrupted) position_ids buffer "
            "to the correct arange(%d) sequence (was max=%s)",
            num_positions, int(position_ids.max()) if position_ids.numel() else "n/a",
        )
        return
    log.warning(
        "FG-CLIP: could not find a vision embeddings.position_ids buffer to fix -- "
        "if the position-embedding out-of-bounds crash recurs, this model's module "
        "layout has changed and this fix needs updating."
    )


# ---------------------------------------------------------------------------
# Multi-scale attention-weighted pooling (pooling="multiscale_attn", shared
# between DINOv2FeatureExtractor and DINOv3FeatureExtractor)
# ---------------------------------------------------------------------------
#
# ViT analog of DAVE's detect-and-verify backbone (DAVE/models/backbone.py),
# which concatenates ResNet layer2+3+4 conv features instead of using a
# single global vector -- multi-scale spatial detail the final layer alone
# doesn't preserve. There is no CNN "layer2/3/4" in a ViT, so the closest
# equivalent is concatenating patch-token features from several transformer
# DEPTHS (early/mid/late) instead of only the final CLS token.
#
# Each selected depth's patch tokens are pooled using THAT layer's own
# CLS-token attention as weights, instead of a naive average -- the original
# DINO paper's well-documented finding is that CLS-token attention highlights
# salient foreground regions (Caron et al., "Emerging Properties in
# Self-Supervised Vision Transformers", arXiv:2104.14294), an
# already-computed foreground signal this reuses for free. Patches the model
# itself is NOT attending to (background/clutter texture) contribute less to
# the resulting embedding than they would under global-average pooling.

def _select_multiscale_layers(num_layers: int) -> list[int]:
    """Pick ~3 transformer depths (1-indexed into `hidden_states`, i.e. "the
    representation after block i") at roughly 50%/75%/100% of the
    backbone's depth -- the ViT analog of DAVE's ResNet layer2+3+4
    concatenation, spanning early/mid/late representations instead of a
    single final-layer vector."""
    layers = sorted({max(1, min(num_layers, round(f * num_layers))) for f in (0.5, 0.75, 1.0)})
    return layers


def _multiscale_attn_pool(
    hidden_states: tuple[torch.Tensor, ...],
    attentions: tuple[torch.Tensor, ...],
    scale_layers: list[int],
    num_patches: int,
) -> torch.Tensor:
    """Concatenate, across `scale_layers` (1-indexed depths into
    hidden_states), each layer's patch tokens pooled by that layer's own
    CLS-token attention (averaged over heads). Returns
    [B, D*len(scale_layers)], NOT yet L2-normalized (the caller does that
    after this call, same as every other extractor in this module).

    num_patches must be the EXACT patch count for the actual processed
    image (compute it from the real pixel_values tensor shape and the
    architecture's fixed patch_size, e.g. (H // patch_size) * (W //
    patch_size) -- never from a configured image_size field, since the HF
    image processor used upstream may resize to its own default rather
    than consulting it). num_prefix_tokens = seq_len - num_patches then
    covers CLS (+ any register tokens) correctly regardless of whether
    registers are enabled, since it's derived from the real tensor shape
    rather than assumed.
    """
    if attentions is None or attentions[0] is None:
        raise RuntimeError(
            "pooling='multiscale_attn' needs output_attentions=True to return real "
            "attention tensors -- got None. This backbone must be loaded with "
            "attn_implementation='eager' (recent transformers releases default to "
            "sdpa/flash-attn backends, which don't return attentions)."
        )
    pooled_per_layer = []
    for layer_idx in scale_layers:
        h = hidden_states[layer_idx]                        # [B, seq, D]
        num_prefix = h.shape[1] - num_patches                # CLS (+ any register tokens)
        if num_prefix < 1:
            raise RuntimeError(
                f"multiscale_attn: layer {layer_idx} has seq_len={h.shape[1]} but "
                f"num_patches={num_patches} leaves no room for a CLS token -- "
                "patch_size/pixel_values shape mismatch."
            )
        patch_tokens = h[:, num_prefix:, :]                  # [B, N, D]
        attn = attentions[layer_idx - 1].mean(dim=1)         # [B, seq, seq], avg over heads
        weights = attn[:, 0, num_prefix:]                    # [B, N] -- CLS's attention to each patch
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        pooled = (patch_tokens * weights.unsqueeze(-1)).sum(dim=1)  # [B, D]
        pooled_per_layer.append(pooled)
    return torch.cat(pooled_per_layer, dim=-1)


def _patch_grid_count(pixel_values: torch.Tensor, patch_size: int) -> int:
    """Exact patch count for an already-processed [B, 3, H, W] tensor, given
    the architecture's fixed patch_size -- see _multiscale_attn_pool's own
    docstring for why this must come from the real tensor, not a config
    field."""
    h, w = pixel_values.shape[-2], pixel_values.shape[-1]
    if h % patch_size != 0 or w % patch_size != 0:
        raise RuntimeError(
            f"multiscale_attn: processed image size {h}x{w} is not divisible by "
            f"patch_size={patch_size} -- the image processor must output patch-aligned "
            "dimensions for attention-weighted patch pooling to align correctly."
        )
    return (h // patch_size) * (w // patch_size)


# ---------------------------------------------------------------------------
# DINOv2
# ---------------------------------------------------------------------------

class DINOv2FeatureExtractor:
    """Batched DINOv2 ViT-S/14 or ViT-B/14, returns L2-normalized features.

    pooling="cls" (default): single global CLS token, as before.

    pooling="multiscale_attn": NOT YET VALIDATED -- see this module's
    "Multi-scale attention-weighted pooling" section above for the
    mechanism. Forces the HuggingFace backend (skips the torch.hub attempt
    entirely) since it needs output_hidden_states=True/
    output_attentions=True with attn_implementation="eager" -- the raw
    torch.hub checkpoint doesn't expose per-layer attention the same way.
    Output dim becomes len(scale_layers)x (typically 3x) the single-CLS
    dim (concatenated, then L2-normalized).
    """

    _HF_MAP_NO_REG = {
        "vits14": "facebook/dinov2-small", "vitb14": "facebook/dinov2-base",
        "vitl14": "facebook/dinov2-large", "vitg14": "facebook/dinov2-giant",
    }
    _HF_MAP_REG = {
        "vits14": "facebook/dinov2-with-registers-small", "vitb14": "facebook/dinov2-with-registers-base",
        "vitl14": "facebook/dinov2-with-registers-large", "vitg14": "facebook/dinov2-with-registers-giant",
    }
    _PATCH_SIZE = 14

    def __init__(
        self, variant: str = "vitb14", device: str = "auto", image_size: int = 224,
        use_registers: bool = False, pooling: str = "cls",
    ):
        if pooling not in ("cls", "multiscale_attn"):
            raise ValueError(f"Unknown DINOv2 pooling '{pooling}'. Must be 'cls' or 'multiscale_attn'.")
        self.variant       = variant
        self.image_size    = image_size
        self.use_registers = use_registers
        self.pooling       = pooling
        self.device        = _resolve_device(device)
        self.processor: Any = None
        self._scale_layers: list[int] = []
        if pooling == "multiscale_attn":
            self.model, self.processor = self._load_hf(variant, eager_attn=True)
            self._scale_layers = _select_multiscale_layers(self.model.config.num_hidden_layers)
        else:
            self.model = self._load(variant)
        self.model.eval().to(self.device)
        log.info(
            "DINOv2 %s%s pooling=%s on %s  (dim=%d)", variant,
            " (with registers)" if use_registers else "", pooling, self.device, self._dim(),
        )

    def _hf_repo(self, variant: str) -> str:
        hf_map = self._HF_MAP_REG if self.use_registers else self._HF_MAP_NO_REG
        if variant not in hf_map:
            raise ValueError(f"Unknown DINOv2 variant '{variant}'. Must be one of {list(hf_map)}.")
        return hf_map[variant]

    def _load(self, variant: str) -> Any:
        hub_name = f"dinov2_{variant}" + ("_reg" if self.use_registers else "")
        try:
            m = torch.hub.load("facebookresearch/dinov2", hub_name, pretrained=True)
            return m
        except Exception as e:
            log.warning("torch.hub failed (%s) → HuggingFace", e)
        from transformers import AutoModel
        m = AutoModel.from_pretrained(self._hf_repo(variant))
        m._hf = True
        return m

    def _load_hf(self, variant: str, eager_attn: bool = False):
        from transformers import AutoImageProcessor, AutoModel
        hf_name = self._hf_repo(variant)
        processor = AutoImageProcessor.from_pretrained(hf_name)
        # output_attentions=True needs attn_implementation="eager" on recent
        # transformers releases -- sdpa/flash-attn backends silently return
        # None for attentions instead.
        kwargs = {"attn_implementation": "eager"} if eager_attn else {}
        model = AutoModel.from_pretrained(hf_name, **kwargs)
        model._hf = True
        return model, processor

    @torch.no_grad()
    def extract(self, images: list[np.ndarray], batch_size: int = 16) -> np.ndarray:
        if not images:
            return np.zeros((0, self._dim()), dtype=np.float32)
        if self.pooling == "multiscale_attn":
            return self._extract_multiscale_attn(images, batch_size)
        tensors = [_preprocess_dino(im, self.image_size) for im in images]
        out: list[np.ndarray] = []
        for i in range(0, len(tensors), batch_size):
            batch = torch.stack(tensors[i:i+batch_size]).to(self.device).float()
            if getattr(self.model, "_hf", False):
                feats = self.model(pixel_values=batch).last_hidden_state[:, 0]
            else:
                feats = self.model(batch)
            out.append(F.normalize(feats, dim=-1).cpu().numpy())
        return np.concatenate(out, axis=0).astype(np.float32)

    @torch.no_grad()
    def _extract_multiscale_attn(self, images: list[np.ndarray], batch_size: int) -> np.ndarray:
        pil_imgs = [_bgr_to_pil(im) for im in images]
        out: list[np.ndarray] = []
        for i in range(0, len(pil_imgs), batch_size):
            batch_pil = pil_imgs[i:i+batch_size]
            inputs = self.processor(images=batch_pil, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model(**inputs, output_hidden_states=True, output_attentions=True)
            num_patches = _patch_grid_count(inputs["pixel_values"], self._PATCH_SIZE)
            pooled = _multiscale_attn_pool(
                outputs.hidden_states, outputs.attentions, self._scale_layers, num_patches,
            )
            out.append(F.normalize(pooled, dim=-1).cpu().numpy())
        return np.concatenate(out, axis=0).astype(np.float32)

    def extract_crops(self, frame_bgr: np.ndarray, boxes: list[Box],
                      pad_ratio: float = 0.10, batch_size: int = 16) -> np.ndarray:
        if not boxes:
            return np.zeros((0, self._dim()), dtype=np.float32)
        return self.extract([crop_with_pad(frame_bgr, b, pad_ratio) for b in boxes], batch_size)

    _DIMS = {"vits14": 384, "vitb14": 768, "vitl14": 1024, "vitg14": 1536}

    def _dim(self) -> int:
        base = self._DIMS.get(self.variant, 768)
        if self.pooling == "multiscale_attn":
            return base * max(len(self._scale_layers), 1)
        return base

    # Keep old name for compatibility
    def _feature_dim(self) -> int:
        return self._dim()


# ---------------------------------------------------------------------------
# DINOv3
# ---------------------------------------------------------------------------

class DINOv3FeatureExtractor:
    """DINOv3 ViT-S/16, ViT-B/16 or ViT-L/16, returns L2-normalized CLS tokens.

    pretrain_dataset picks WHICH pretraining run's weights to load, same
    architecture either way: "lvd1689m" (default) is Meta's large natural-
    image corpus; "sat493m" is a satellite-imagery pretraining run --
    likely a better domain match for aerial/drone footage than the natural-
    image default. Applies to BOTH sources below (huggingface builds the
    repo id from it; kaggle_model_id is a free-form string you supply
    yourself, so it's on you to pick one whose own architecture/dataset
    matches variant/pretrain_dataset -- this field is mostly bookkeeping/
    logging in that case, not something this class enforces).

    source="huggingface" (default): weights are gated on HuggingFace
    (facebook/dinov3-{variant}-pretrain-{pretrain_dataset}) -- request
    access on the model page, then set HF_TOKEN before running, or
    `from_pretrained` will fail with a 401/403. Not every
    (variant, pretrain_dataset) combination is necessarily published --
    an unavailable one surfaces as a 404 straight from `from_pretrained`.

    source="kaggle": loads a raw Meta DINOv3 checkpoint (a bare .pth/.pt
    state dict from Meta's OWN dinov3 codebase, NOT a transformers-format
    folder) via kagglehub.model_download(kaggle_model_id), then
    torch.hub.load("facebookresearch/dinov3", ..., weights=<that file>) --
    lets a pretraining variant mirrored on Kaggle be used without
    HuggingFace gating (e.g. when a HF repo for it isn't published, or you
    just don't have HF access). Needs `kagglehub` installed and Kaggle API
    credentials configured. Unlike the huggingface path, there is no
    AutoImageProcessor here -- preprocessing reuses _preprocess_dino (same
    ImageNet-style normalization DINOv2's own torch.hub path uses), since
    the raw hub model expects a plain tensor, not transformers' processor
    output.

    pooling="cls" (default): single global CLS/pooler_output token, as
    before. pooling="multiscale_attn" is NOT YET VALIDATED -- see this
    module's "Multi-scale attention-weighted pooling" section above for
    the mechanism. Requires source="huggingface" (raises at construction
    otherwise) -- the kaggle raw checkpoint path has no
    output_hidden_states/output_attentions API. Output dim becomes
    len(scale_layers)x (typically 3x) the single-CLS dim.
    """

    _ARCHS = ("vits16", "vitb16", "vitl16")
    _PRETRAIN_DATASETS = ("lvd1689m", "sat493m")
    _DIMS = {"vits16": 384, "vitb16": 768, "vitl16": 1024}
    _PATCH_SIZE = 16

    def __init__(
        self, variant: str = "vitb16", device: str = "auto",
        source: str = "huggingface", pretrain_dataset: str = "lvd1689m",
        kaggle_model_id: str | None = None, image_size: int = 224,
        pooling: str = "cls",
    ):
        if variant not in self._ARCHS:
            raise ValueError(f"Unknown DINOv3 variant '{variant}'. Must be one of {self._ARCHS}.")
        if pretrain_dataset not in self._PRETRAIN_DATASETS:
            raise ValueError(
                f"Unknown DINOv3 pretrain_dataset '{pretrain_dataset}'. "
                f"Must be one of {self._PRETRAIN_DATASETS}."
            )
        if source not in ("huggingface", "kaggle"):
            raise ValueError(f"Unknown DINOv3 source '{source}'. Must be 'huggingface' or 'kaggle'.")
        if pooling not in ("cls", "multiscale_attn"):
            raise ValueError(f"Unknown DINOv3 pooling '{pooling}'. Must be 'cls' or 'multiscale_attn'.")
        if pooling == "multiscale_attn" and source != "huggingface":
            raise ValueError(
                "DINOv3 pooling='multiscale_attn' requires source='huggingface' -- the kaggle "
                "raw checkpoint path doesn't expose output_hidden_states/output_attentions."
            )
        self.variant          = variant
        self.pretrain_dataset = pretrain_dataset
        self.source           = source
        self.image_size       = image_size
        self.pooling          = pooling
        self.device           = _resolve_device(device)
        self.processor        = None
        self._scale_layers: list[int] = []
        if source == "huggingface":
            self.model, self.processor = self._load_huggingface(
                variant, pretrain_dataset, eager_attn=(pooling == "multiscale_attn"),
            )
        else:
            self.model = self._load_kaggle(variant, kaggle_model_id)
        self.model.eval().to(self.device)
        if pooling == "multiscale_attn":
            self._scale_layers = _select_multiscale_layers(self.model.config.num_hidden_layers)
        log.info(
            "DINOv3 %s pretrain=%s (source=%s) pooling=%s on %s  (dim=%d)",
            variant, pretrain_dataset, source, pooling, self.device, self._dim(),
        )

    def _load_huggingface(self, variant: str, pretrain_dataset: str, eager_attn: bool = False):
        from transformers import AutoImageProcessor, AutoModel
        hf_name = f"facebook/dinov3-{variant}-pretrain-{pretrain_dataset}"
        processor = AutoImageProcessor.from_pretrained(hf_name)
        # output_attentions=True needs attn_implementation="eager" on recent
        # transformers releases -- sdpa/flash-attn backends silently return
        # None for attentions instead.
        kwargs = {"attn_implementation": "eager"} if eager_attn else {}
        model = AutoModel.from_pretrained(hf_name, **kwargs)
        return model, processor

    def _load_kaggle(self, variant: str, kaggle_model_id: str | None):
        if not kaggle_model_id:
            raise ValueError(
                "stage1.feature_extractor.dinov3_source='kaggle' needs "
                "dinov3_kaggle_model_id set (e.g. "
                "'yadavdamodar/dinov3-vitl16-pretrain-sat493m/pyTorch/default')."
            )
        try:
            import kagglehub
        except ImportError:
            raise RuntimeError(
                "kagglehub not installed. Run: pip install kagglehub"
            )
        from pathlib import Path
        local_dir = kagglehub.model_download(kaggle_model_id)
        ckpt_files = sorted(Path(local_dir).rglob("*.pth")) + sorted(Path(local_dir).rglob("*.pt"))
        if not ckpt_files:
            raise FileNotFoundError(
                f"No .pth/.pt checkpoint found under kagglehub download at {local_dir} "
                f"(kaggle_model_id='{kaggle_model_id}')."
            )
        weights_path = str(ckpt_files[0])
        log.info("DINOv3 kaggle: loading checkpoint %s", weights_path)
        return torch.hub.load(
            "facebookresearch/dinov3", f"dinov3_{variant}",
            source="github", weights=weights_path,
        )

    @torch.no_grad()
    def extract(self, images: list[np.ndarray], batch_size: int = 16) -> np.ndarray:
        if not images:
            return np.zeros((0, self._dim()), dtype=np.float32)
        if self.pooling == "multiscale_attn":
            return self._extract_multiscale_attn(images, batch_size)
        if self.source == "kaggle":
            # Raw torch.hub model, plain tensor in/CLS tensor out -- same
            # calling convention as DINOv2FeatureExtractor's own hub path.
            tensors = [_preprocess_dino(im, self.image_size) for im in images]
            out: list[np.ndarray] = []
            for i in range(0, len(tensors), batch_size):
                batch = torch.stack(tensors[i:i+batch_size]).to(self.device).float()
                feats = self.model(batch)
                out.append(F.normalize(feats, dim=-1).cpu().numpy())
            return np.concatenate(out, axis=0).astype(np.float32)
        pil_imgs = [_bgr_to_pil(im) for im in images]
        out: list[np.ndarray] = []
        for i in range(0, len(pil_imgs), batch_size):
            batch_pil = pil_imgs[i:i+batch_size]
            inputs = self.processor(images=batch_pil, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            pooled = self.model(**inputs).pooler_output
            out.append(F.normalize(pooled, dim=-1).cpu().numpy())
        return np.concatenate(out, axis=0).astype(np.float32)

    @torch.no_grad()
    def _extract_multiscale_attn(self, images: list[np.ndarray], batch_size: int) -> np.ndarray:
        pil_imgs = [_bgr_to_pil(im) for im in images]
        out: list[np.ndarray] = []
        for i in range(0, len(pil_imgs), batch_size):
            batch_pil = pil_imgs[i:i+batch_size]
            inputs = self.processor(images=batch_pil, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model(**inputs, output_hidden_states=True, output_attentions=True)
            num_patches = _patch_grid_count(inputs["pixel_values"], self._PATCH_SIZE)
            pooled = _multiscale_attn_pool(
                outputs.hidden_states, outputs.attentions, self._scale_layers, num_patches,
            )
            out.append(F.normalize(pooled, dim=-1).cpu().numpy())
        return np.concatenate(out, axis=0).astype(np.float32)

    def extract_crops(self, frame_bgr: np.ndarray, boxes: list[Box],
                      pad_ratio: float = 0.10, batch_size: int = 16) -> np.ndarray:
        if not boxes:
            return np.zeros((0, self._dim()), dtype=np.float32)
        return self.extract([crop_with_pad(frame_bgr, b, pad_ratio) for b in boxes], batch_size)

    def _dim(self) -> int:
        base = self._DIMS.get(self.variant, 768)
        if self.pooling == "multiscale_attn":
            return base * max(len(self._scale_layers), 1)
        return base

    def _feature_dim(self) -> int:
        return self._dim()


# ---------------------------------------------------------------------------
# CLIP
# ---------------------------------------------------------------------------

class CLIPFeatureExtractor:
    """CLIP visual encoder — returns L2-normalized image embeddings (512-d)."""

    _VARIANT_MAP = {
        "vit-b/32": "openai/clip-vit-base-patch32",   # 512-d
        "vit-l/14": "openai/clip-vit-large-patch14",  # 768-d
    }

    def __init__(self, variant: str = "vit-b/32", device: str = "auto"):
        self.variant = variant
        self.device  = _resolve_device(device)
        self.model, self.processor = self._load(variant)
        self.model.eval().to(self.device)
        log.info("CLIP %s on %s  (dim=%d)", variant, self.device, self._dim())

    def _load(self, variant: str):
        try:
            from transformers import CLIPModel, CLIPProcessor
        except ImportError:
            raise RuntimeError(
                "transformers not installed. Run: pip install transformers"
            )
        hf_name = self._VARIANT_MAP.get(variant, variant)
        model     = CLIPModel.from_pretrained(hf_name)
        processor = CLIPProcessor.from_pretrained(hf_name)
        return model, processor

    @torch.no_grad()
    def extract(self, images: list[np.ndarray], batch_size: int = 16) -> np.ndarray:
        if not images:
            return np.zeros((0, self._dim()), dtype=np.float32)
        pil_imgs = [_bgr_to_pil(im) for im in images]
        out: list[np.ndarray] = []
        for i in range(0, len(pil_imgs), batch_size):
            batch_pil = pil_imgs[i:i+batch_size]
            inputs = self.processor(images=batch_pil, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            # Some transformers releases return a ModelOutput (not a plain
            # tensor) from get_image_features() due to an upstream API
            # regression. Call the stable vision_model + visual_projection
            # submodules directly instead of depending on that method.
            pooled = self.model.vision_model(pixel_values=inputs["pixel_values"]).pooler_output
            feats = self.model.visual_projection(pooled)
            out.append(F.normalize(feats, dim=-1).cpu().numpy())
        return np.concatenate(out, axis=0).astype(np.float32)

    def extract_crops(self, frame_bgr: np.ndarray, boxes: list[Box],
                      pad_ratio: float = 0.10, batch_size: int = 16) -> np.ndarray:
        if not boxes:
            return np.zeros((0, self._dim()), dtype=np.float32)
        return self.extract([crop_with_pad(frame_bgr, b, pad_ratio) for b in boxes], batch_size)

    def _dim(self) -> int:
        return {"vit-b/32": 512, "vit-l/14": 768}.get(self.variant, 512)

    def _feature_dim(self) -> int:
        return self._dim()


# ---------------------------------------------------------------------------
# SigLIP
# ---------------------------------------------------------------------------

class SiglipFeatureExtractor:
    """SigLIP vision encoder only (no text tower), returns L2-normalized pooled tokens."""

    _VARIANT_MAP = {
        "base":    "google/siglip-base-patch16-224",
        "large":   "google/siglip-large-patch16-256",
        "so400m":  "google/siglip-so400m-patch14-384",
    }
    _DIMS = {"base": 768, "large": 1024, "so400m": 1152}

    def __init__(self, variant: str = "base", device: str = "auto"):
        if variant not in self._VARIANT_MAP:
            raise ValueError(f"Unknown SigLIP variant '{variant}'. Must be one of {list(self._VARIANT_MAP)}.")
        self.variant = variant
        self.device  = _resolve_device(device)
        self.model, self.processor = self._load(variant)
        self.model.eval().to(self.device)
        log.info("SigLIP %s on %s  (dim=%d)", variant, self.device, self._dim())

    def _load(self, variant: str):
        from transformers import AutoImageProcessor, SiglipVisionModel
        hf_name = self._VARIANT_MAP[variant]
        processor = AutoImageProcessor.from_pretrained(hf_name)
        model     = SiglipVisionModel.from_pretrained(hf_name)
        return model, processor

    @torch.no_grad()
    def extract(self, images: list[np.ndarray], batch_size: int = 16) -> np.ndarray:
        if not images:
            return np.zeros((0, self._dim()), dtype=np.float32)
        pil_imgs = [_bgr_to_pil(im) for im in images]
        out: list[np.ndarray] = []
        for i in range(0, len(pil_imgs), batch_size):
            batch_pil = pil_imgs[i:i+batch_size]
            inputs = self.processor(images=batch_pil, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            pooled = self.model(**inputs).pooler_output
            out.append(F.normalize(pooled, dim=-1).cpu().numpy())
        return np.concatenate(out, axis=0).astype(np.float32)

    def extract_crops(self, frame_bgr: np.ndarray, boxes: list[Box],
                      pad_ratio: float = 0.10, batch_size: int = 16) -> np.ndarray:
        if not boxes:
            return np.zeros((0, self._dim()), dtype=np.float32)
        return self.extract([crop_with_pad(frame_bgr, b, pad_ratio) for b in boxes], batch_size)

    def _dim(self) -> int:
        return self._DIMS.get(self.variant, 768)

    def _feature_dim(self) -> int:
        return self._dim()


# ---------------------------------------------------------------------------
# Ensemble (DINOv2 + CLIP concat → L2-normalize)
# ---------------------------------------------------------------------------

class EnsembleFeatureExtractor:
    """Concatenates a DINO family model (DINOv2 or DINOv3) + CLIP features
    then L2-normalizes.

    dim = DINO_dim + CLIP_dim  (e.g. 768 + 512 = 1280 for DINOv2 vitb14 +
    CLIP vit-b/32; DINOv3 vitb16 + CLIP vit-b/32 is also 768 + 512 = 1280).

    dino_model="dinov3" is NOT YET VALIDATED -- see
    FeatureExtractorConfig.ensemble_dino_model's own docstring
    (aero_eyes/config.py) for the rationale (CLIP's semantic/categorical
    training objective as a complement to DINO's texture-clustering one,
    for cases where DINO alone confuses the target with texturally-similar
    background clutter).
    """

    def __init__(
        self,
        dinov2_variant: str = "vitb14",
        clip_variant: str = "vit-b/32",
        device: str = "auto",
        image_size: int = 224,
        dinov2_use_registers: bool = False,
        dino_model: str = "dinov2",
        dinov3_variant: str = "vitb16",
        dinov3_source: str = "huggingface",
        dinov3_pretrain_dataset: str = "lvd1689m",
        dinov3_kaggle_model_id: str | None = None,
        dinov2_pooling: str = "cls",
        dinov3_pooling: str = "cls",
    ):
        if dino_model == "dinov3":
            self.dino = DINOv3FeatureExtractor(
                variant=dinov3_variant, device=device, source=dinov3_source,
                pretrain_dataset=dinov3_pretrain_dataset, kaggle_model_id=dinov3_kaggle_model_id,
                image_size=image_size, pooling=dinov3_pooling,
            )
        elif dino_model == "dinov2":
            self.dino = DINOv2FeatureExtractor(
                dinov2_variant, device, image_size, dinov2_use_registers, pooling=dinov2_pooling,
            )
        else:
            raise ValueError(f"Unknown ensemble dino_model '{dino_model}'. Must be 'dinov2' or 'dinov3'.")
        self.clip = CLIPFeatureExtractor(clip_variant, device)
        log.info("Ensemble %s+CLIP  dim=%d", dino_model.upper(), self._dim())

    def extract(self, images: list[np.ndarray], batch_size: int = 16) -> np.ndarray:
        if not images:
            return np.zeros((0, self._dim()), dtype=np.float32)
        d = self.dino.extract(images, batch_size)  # [N, D1]
        c = self.clip.extract(images, batch_size)  # [N, D2]
        combined = np.concatenate([d, c], axis=-1)  # [N, D1+D2]
        norms = np.linalg.norm(combined, axis=-1, keepdims=True).clip(min=1e-8)
        return (combined / norms).astype(np.float32)

    def extract_crops(self, frame_bgr: np.ndarray, boxes: list[Box],
                      pad_ratio: float = 0.10, batch_size: int = 16) -> np.ndarray:
        if not boxes:
            return np.zeros((0, self._dim()), dtype=np.float32)
        crops = [crop_with_pad(frame_bgr, b, pad_ratio) for b in boxes]
        return self.extract(crops, batch_size)

    def _dim(self) -> int:
        return self.dino._dim() + self.clip._dim()

    def _feature_dim(self) -> int:
        return self._dim()


# ---------------------------------------------------------------------------
# FG-CLIP (fine-grained CLIP, hard-negative contrastive training)
# ---------------------------------------------------------------------------

class FGCLIPFeatureExtractor:
    """FG-CLIP visual encoder (arXiv:2505.05071) -- vision-only, returns
    L2-normalized image embeddings.

    Unlike vanilla CLIP/SigLIP (both already in this project, and
    empirically UNDERPERFORMING DINOv2/DINOv3 on this project's own aerial
    footage per its own A/B testing), FG-CLIP is fine-tuned with ~10M hard
    fine-grained negative pairs specifically to separate near-duplicate
    instances that share a broad category/appearance -- a different
    failure mode than the broad category-vs-category alignment vanilla
    CLIP/SigLIP optimize for. Candidate for when the confusers are
    texturally close to the target (dry leaves, plastic sheeting, white
    paper) rather than semantically distinct categories.

    NOT YET VALIDATED on this project's own footage.
    """

    _VARIANT_MAP = {
        "base":  "qihoo360/fg-clip-base",   # 512-d
        "large": "qihoo360/fg-clip-large",  # 768-d
    }
    _DIMS = {"base": 512, "large": 768}
    # FG-CLIP's own official usage example manually resizes every image to
    # (_IMAGE_SIZE, _IMAGE_SIZE) BEFORE calling the image processor, rather
    # than relying on AutoImageProcessor's own resizing -- see extract()'s
    # own comment for why this isn't optional.
    _IMAGE_SIZE = 224

    def __init__(self, variant: str = "base", device: str = "auto"):
        if variant not in self._VARIANT_MAP:
            raise ValueError(f"Unknown FG-CLIP variant '{variant}'. Must be one of {list(self._VARIANT_MAP)}.")
        self.variant = variant
        self.device  = _resolve_device(device)
        self.model, self.processor = self._load(variant)
        self.model.eval().to(self.device)
        log.info("FG-CLIP %s on %s  (dim=%d)", variant, self.device, self._dim())

    def _load(self, variant: str):
        try:
            from transformers import AutoConfig, AutoImageProcessor, AutoModelForCausalLM
        except ImportError:
            raise RuntimeError(
                "transformers not installed. Run: pip install transformers"
            )
        _ensure_transformers_onnx_shim()
        hf_name = self._VARIANT_MAP[variant]
        # FG-CLIP ships custom modeling code (not a stock CLIPModel), so it
        # needs trust_remote_code -- registered under AutoModelForCausalLM
        # per the model's own official usage example, despite being used
        # here purely as a vision encoder (only get_image_features() below
        # is called, never text generation).
        processor = AutoImageProcessor.from_pretrained(hf_name)
        # Load+patch the config BEFORE from_pretrained builds the model --
        # see _ensure_fgclip_subconfigs's own docstring for why this is
        # needed (another version-skew symptom in FG-CLIP's hosted code,
        # like _ensure_transformers_onnx_shim above).
        config = AutoConfig.from_pretrained(hf_name, trust_remote_code=True)
        _ensure_fgclip_subconfigs(config, hf_name)
        model = AutoModelForCausalLM.from_pretrained(hf_name, config=config, trust_remote_code=True)
        _fix_fgclip_position_ids(model)
        return model, processor

    @torch.no_grad()
    def extract(self, images: list[np.ndarray], batch_size: int = 16) -> np.ndarray:
        if not images:
            return np.zeros((0, self._dim()), dtype=np.float32)
        # Confirmed on this project's own GPU run: skipping this manual
        # resize (i.e. trusting AutoImageProcessor's own resizing, the
        # pattern every OTHER extractor in this module uses) produces a
        # patch grid that doesn't match modeling_fgclip.py's position-
        # embedding table size -- NOT a clear shape-mismatch error at the
        # input, but a CUDA device-side assert deep inside the vision
        # encoder ("indexSelectLargeIndex ... srcIndex < srcSelectDimSize").
        pil_imgs = [_bgr_to_pil(im).resize((self._IMAGE_SIZE, self._IMAGE_SIZE), Image.BICUBIC) for im in images]
        out: list[np.ndarray] = []
        for i in range(0, len(pil_imgs), batch_size):
            batch_pil = pil_imgs[i:i+batch_size]
            pixel_values = self.processor.preprocess(batch_pil, return_tensors="pt")["pixel_values"]
            pixel_values = pixel_values.to(self.device)
            feats = self.model.get_image_features(pixel_values)
            out.append(F.normalize(feats, dim=-1).cpu().numpy())
        return np.concatenate(out, axis=0).astype(np.float32)

    def extract_crops(self, frame_bgr: np.ndarray, boxes: list[Box],
                      pad_ratio: float = 0.10, batch_size: int = 16) -> np.ndarray:
        if not boxes:
            return np.zeros((0, self._dim()), dtype=np.float32)
        return self.extract([crop_with_pad(frame_bgr, b, pad_ratio) for b in boxes], batch_size)

    def _dim(self) -> int:
        return self._DIMS.get(self.variant, 512)

    def _feature_dim(self) -> int:
        return self._dim()


# ---------------------------------------------------------------------------
# RADIO / C-RADIO (NVIDIA multi-teacher distilled hybrid backbone)
# ---------------------------------------------------------------------------

class RadioFeatureExtractor:
    """NVIDIA RADIO / C-RADIO (arXiv:2312.06709 AM-RADIO, arXiv:2412.07679
    RADIOv2.5) -- a single backbone distilled from multiple teacher VFMs at
    once (DINOv2/DINOv3, CLIP/SigLIP2, SAM/SAM3). Returns the model's
    pooled "summary" output (CLS-token analog), L2-normalized.

    variant is the torch.hub `version` string, e.g. "c-radio_v3-b"
    (default -- the smallest C-RADIO tier, closest in scale to this
    project's DINOv2 ViT-B/14 baseline, NVIDIA Open Model License,
    commercial use allowed). Plain "radio-*"/"e-radio" checkpoints are
    released under NSCLv1 (non-commercial only) -- prefer a "c-radio_*"
    variant for anything beyond research use.

    Output dimension is NOT hardcoded from a static table (unlike the
    other extractors above) -- NVIDIA's own docs don't publish a clean
    per-variant summary-dim table, so it's probed once at construction
    time with a real forward pass instead of risking a wrong guess that
    would silently corrupt prototype/RMD-score dimensionality downstream.

    NOT YET VALIDATED for retrieval/re-identification -- all published
    RADIO evidence (vs. single-teacher DINOv2/SAM/CLIP baselines) is from
    segmentation/classification/VQA benchmarks, not retrieval. Candidate
    for testing whether SAM's boundary/localization-aware teacher signal,
    combined with DINOv2/v3's own spatial features, sharpens this
    project's object-vs-background-clutter embedding.
    """

    _KNOWN_VERSIONS = (
        "c-radio_v3-b", "c-radio_v3-l", "c-radio_v3-h", "c-radio_v3-g",
        "c-radio_v4-so400m", "c-radio_v4-h",
        "radio-b", "radio-l", "radio-g", "e-radio",
    )
    _NON_COMMERCIAL_PREFIXES = ("radio-", "e-radio")

    def __init__(self, variant: str = "c-radio_v3-b", device: str = "auto", image_size: int = 224):
        if variant not in self._KNOWN_VERSIONS:
            raise ValueError(f"Unknown RADIO variant '{variant}'. Must be one of {self._KNOWN_VERSIONS}.")
        if variant.startswith(self._NON_COMMERCIAL_PREFIXES):
            log.warning(
                "RADIO variant '%s' is released under NSCLv1 (non-commercial use only) -- "
                "use a 'c-radio_*' variant instead for commercial deployment.", variant,
            )
        self.variant    = variant
        self.image_size = image_size
        self.device     = _resolve_device(device)
        self.model      = torch.hub.load("NVlabs/RADIO", "radio_model", version=variant, progress=True)
        self.model.eval().to(self.device)
        self._cached_dim = self._probe_dim()
        log.info("RADIO %s on %s  (dim=%d)", variant, self.device, self._cached_dim)

    @torch.no_grad()
    def _probe_dim(self) -> int:
        dummy = torch.zeros(1, 3, self.image_size, self.image_size, device=self.device)
        dummy = self._to_supported_resolution(dummy)
        summary, _ = self.model(dummy)
        return int(summary.shape[-1])

    def _to_supported_resolution(self, batch: torch.Tensor) -> torch.Tensor:
        res = self.model.get_nearest_supported_resolution(*batch.shape[-2:])
        if tuple(res) == tuple(batch.shape[-2:]):
            return batch
        return F.interpolate(batch, res, mode="bilinear", align_corners=False)

    @torch.no_grad()
    def extract(self, images: list[np.ndarray], batch_size: int = 16) -> np.ndarray:
        if not images:
            return np.zeros((0, self._dim()), dtype=np.float32)
        out: list[np.ndarray] = []
        for i in range(0, len(images), batch_size):
            batch_imgs = images[i:i+batch_size]
            tensors = []
            for im in batch_imgs:
                pil = _bgr_to_pil(im).resize((self.image_size, self.image_size), Image.BICUBIC)
                # RADIO expects [0, 1]-range NCHW float tensors and does its
                # own internal mean/std normalization -- no ImageNet-style
                # preprocessing here (see NVlabs/RADIO README), just scale.
                arr = np.array(pil, dtype=np.float32) / 255.0
                tensors.append(torch.from_numpy(arr.transpose(2, 0, 1).copy()))
            batch = torch.stack(tensors).to(self.device)
            batch = self._to_supported_resolution(batch)
            summary, _ = self.model(batch)
            out.append(F.normalize(summary, dim=-1).cpu().numpy())
        return np.concatenate(out, axis=0).astype(np.float32)

    def extract_crops(self, frame_bgr: np.ndarray, boxes: list[Box],
                      pad_ratio: float = 0.10, batch_size: int = 16) -> np.ndarray:
        if not boxes:
            return np.zeros((0, self._dim()), dtype=np.float32)
        return self.extract([crop_with_pad(frame_bgr, b, pad_ratio) for b in boxes], batch_size)

    def _dim(self) -> int:
        return self._cached_dim

    def _feature_dim(self) -> int:
        return self._dim()


# ---------------------------------------------------------------------------
# SigLIP2 (Google DeepMind, arXiv:2502.14786)
# ---------------------------------------------------------------------------

class Siglip2FeatureExtractor:
    """SigLIP2 vision encoder (arXiv:2502.14786) -- vision-only, returns
    L2-normalized image embeddings.

    Adds a Global-Local Loss + Masked Prediction Loss on top of SigLIP's
    original sigmoid image-text loss, specifically to improve LOCAL/dense
    semantics rather than just global category alignment -- the second-
    strongest evidenced fine-grained/near-duplicate discrimination family
    found (after FG-CLIP) for this project's object-vs-clutter problem, and
    used in NVIDIA's own production video-analytics stack for cosine-
    similarity re-identification.

    Unlike FG-CLIP (qihoo360/fg-clip-*, this project's own earlier,
    troubled integration -- 4 separate remote-code compatibility bugs
    fixed in turn), SigLIP2 is a STANDARD, first-party transformers
    architecture: loaded via plain AutoModel/AutoProcessor, no
    trust_remote_code, no vendored/community modeling code -- materially
    lower integration risk.

    Uses model.get_image_features(**inputs) -- the officially documented
    API (transformers docs' own dedicated example calls exactly this),
    NOT a hand-pooled hidden_state -- SigLIP2's checkpoints (even the
    fixed-resolution "FixRes" ones used here) are patch-count-based
    internally (see Siglip2VisionConfig.num_patches), so get_image_features
    is the one call guaranteed to handle whatever pixel_values/
    pixel_attention_mask/spatial_shapes the processor actually produced,
    without this class needing to special-case that.

    Only "base" (google/siglip2-base-patch16-224) and "so400m"
    (google/siglip2-so400m-patch14-384) are wired -- the two checkpoint
    ids directly confirmed to exist at implementation time; other sizes
    (large/giant) were not independently verified and are deliberately
    left out rather than guessed.

    Output dimension is probed once at construction (real forward pass on
    a dummy image) rather than hardcoded -- the transformers docs read
    while implementing this did not surface a clean per-variant output_dim
    table, and this project has already been burned once (FG-CLIP) by
    trusting an assumed number instead of a measured one.

    NOT YET VALIDATED on this project's own footage.
    """

    _VARIANT_MAP = {
        "base":   "google/siglip2-base-patch16-224",
        "so400m": "google/siglip2-so400m-patch14-384",
    }

    def __init__(self, variant: str = "base", device: str = "auto"):
        if variant not in self._VARIANT_MAP:
            raise ValueError(f"Unknown SigLIP2 variant '{variant}'. Must be one of {list(self._VARIANT_MAP)}.")
        self.variant = variant
        self.device  = _resolve_device(device)
        self.model, self.processor = self._load(variant)
        self.model.eval().to(self.device)
        self._cached_dim = self._probe_dim()
        log.info("SigLIP2 %s on %s  (dim=%d)", variant, self.device, self._cached_dim)

    def _load(self, variant: str):
        try:
            from transformers import AutoModel, AutoProcessor
        except ImportError:
            raise RuntimeError(
                "transformers not installed. Run: pip install transformers"
            )
        hf_name = self._VARIANT_MAP[variant]
        processor = AutoProcessor.from_pretrained(hf_name)
        model = AutoModel.from_pretrained(hf_name)
        return model, processor

    @torch.no_grad()
    def _probe_dim(self) -> int:
        dummy = Image.new("RGB", (224, 224))
        inputs = self.processor(images=[dummy], return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        feats = self.model.get_image_features(**inputs)
        return int(feats.shape[-1])

    @torch.no_grad()
    def extract(self, images: list[np.ndarray], batch_size: int = 16) -> np.ndarray:
        if not images:
            return np.zeros((0, self._dim()), dtype=np.float32)
        pil_imgs = [_bgr_to_pil(im) for im in images]
        out: list[np.ndarray] = []
        for i in range(0, len(pil_imgs), batch_size):
            batch_pil = pil_imgs[i:i+batch_size]
            inputs = self.processor(images=batch_pil, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            feats = self.model.get_image_features(**inputs)
            out.append(F.normalize(feats, dim=-1).cpu().numpy())
        return np.concatenate(out, axis=0).astype(np.float32)

    def extract_crops(self, frame_bgr: np.ndarray, boxes: list[Box],
                      pad_ratio: float = 0.10, batch_size: int = 16) -> np.ndarray:
        if not boxes:
            return np.zeros((0, self._dim()), dtype=np.float32)
        return self.extract([crop_with_pad(frame_bgr, b, pad_ratio) for b in boxes], batch_size)

    def _dim(self) -> int:
        return self._cached_dim

    def _feature_dim(self) -> int:
        return self._dim()


# ---------------------------------------------------------------------------
# EVA02-CLIP (BAAI, via open_clip_torch)
# ---------------------------------------------------------------------------

class EVACLIPFeatureExtractor:
    """EVA02-CLIP-B/16 (BAAI) -- vision-only, returns L2-normalized image
    embeddings. Vision tower initializes from an EVA (self-supervised
    masked-image-modeling) backbone before CLIP-style contrastive image-
    text fine-tuning -- a hybrid of self-supervised + contrastive training,
    unlike vanilla CLIP/SigLIP (purely contrastive).

    Loaded via `open_clip_torch` (LAION's library) -- CONFIRMED (checked
    open_clip's own model registry/pretrained-tag table while implementing
    this) rather than assumed: model name "EVA02-B-16", pretrained tag
    "merged2b_s8b_b131k" (weights auto-fetched from the `timm/` HF hub
    namespace). This is NOT the narrow, sparsely-maintained BAAI `eva_clip`
    package (a different, higher-risk dependency this project deliberately
    avoided), and NOT transformers/trust_remote_code -- open_clip is a
    stable, widely-used, actively-maintained library, though it IS a NEW
    dependency for this project (not required by any other extractor here).

    Weaker DIRECT evidence for fine-grained/near-duplicate discrimination
    than FG-CLIP/SigLIP2 -- only large-scale zero-shot classification
    numbers were found for the EVA-CLIP family (and only for the 18B
    flagship, not this ~150M base size), not a comparable FG-OVD-style
    benchmark. Included as a lower-integration-risk alternative to try
    empirically, not because of stronger cited evidence.

    Output dimension is probed once at construction (real forward pass on
    a dummy image) rather than hardcoded.

    NOT YET VALIDATED on this project's own footage.
    """

    _OPEN_CLIP_NAME = "EVA02-B-16"
    _PRETRAINED_TAG = "merged2b_s8b_b131k"

    def __init__(self, variant: str = "base", device: str = "auto"):
        if variant != "base":
            raise ValueError(f"Unknown EVA-CLIP variant '{variant}'. Only 'base' is wired.")
        self.variant = variant
        self.device  = _resolve_device(device)
        self.model, self.preprocess = self._load()
        self.model.eval().to(self.device)
        self._cached_dim = self._probe_dim()
        log.info("EVA02-CLIP-B/16 on %s  (dim=%d)", self.device, self._cached_dim)

    def _load(self):
        try:
            import open_clip
        except ImportError:
            raise RuntimeError(
                "open_clip_torch not installed. Run: pip install open_clip_torch"
            )
        model, _, preprocess = open_clip.create_model_and_transforms(
            self._OPEN_CLIP_NAME, pretrained=self._PRETRAINED_TAG,
        )
        return model, preprocess

    @torch.no_grad()
    def _probe_dim(self) -> int:
        dummy = Image.new("RGB", (224, 224))
        batch = self.preprocess(dummy).unsqueeze(0).to(self.device)
        feats = self.model.encode_image(batch)
        return int(feats.shape[-1])

    @torch.no_grad()
    def extract(self, images: list[np.ndarray], batch_size: int = 16) -> np.ndarray:
        if not images:
            return np.zeros((0, self._dim()), dtype=np.float32)
        pil_imgs = [_bgr_to_pil(im) for im in images]
        out: list[np.ndarray] = []
        for i in range(0, len(pil_imgs), batch_size):
            batch_pil = pil_imgs[i:i+batch_size]
            batch = torch.stack([self.preprocess(im) for im in batch_pil]).to(self.device)
            feats = self.model.encode_image(batch)
            out.append(F.normalize(feats, dim=-1).cpu().numpy())
        return np.concatenate(out, axis=0).astype(np.float32)

    def extract_crops(self, frame_bgr: np.ndarray, boxes: list[Box],
                      pad_ratio: float = 0.10, batch_size: int = 16) -> np.ndarray:
        if not boxes:
            return np.zeros((0, self._dim()), dtype=np.float32)
        return self.extract([crop_with_pad(frame_bgr, b, pad_ratio) for b in boxes], batch_size)

    def _dim(self) -> int:
        return self._cached_dim

    def _feature_dim(self) -> int:
        return self._dim()


# ---------------------------------------------------------------------------
# dino.txt / "DINOv2 Meets Text" (arXiv:2412.16334)
# ---------------------------------------------------------------------------

class DinoTxtFeatureExtractor:
    """dino.txt (arXiv:2412.16334, "DINOv2 Meets Text") -- vision-only,
    returns L2-normalized image embeddings.

    Adds a text encoder trained via LiT (Locked-image Text tuning) to
    align with a FROZEN DINOv2 ViT-L/14 (with registers) backbone --
    retrofits language/semantic grounding onto DINOv2 while keeping its
    dense/pixel-level task quality, rather than switching away from the
    DINO family entirely (unlike CLIP/SigLIP2/FG-CLIP/EVA-CLIP, which are
    all different architectures/training paradigms). Directly targets this
    project's own original framing of DINOv2's limitation -- pure self-
    supervised texture clustering, no notion of "this is an object" vs
    "this is background clutter" -- while other candidates in this module
    trade away DINO's dense-feature strength to get that notion.

    Loaded via the SAME torch.hub mechanism as this project's own
    model="dinov2" option (facebookresearch/dinov2 repo) -- no
    trust_remote_code, no new dependency, same trust level as an
    already-working extractor in this file.

    HONESTY NOTE on things that could NOT be independently verified
    without a live download (no GPU/network available while implementing
    this): (1) the exact return type of torch.hub.load(...) for this
    entrypoint -- handled defensively below for both a bare model and a
    (model, tokenizer) tuple; (2) the exact image preprocessing this
    checkpoint expects -- reuses _preprocess_dino (the same ImageNet-style
    normalization this project's own DINOv2 torch.hub path already uses),
    a reasoned assumption (the vision tower is a frozen, unmodified
    DINOv2), not a confirmed one; (3) the exact output embedding
    dimension -- probed dynamically rather than hardcoded, same pattern as
    RadioFeatureExtractor/Siglip2FeatureExtractor/EVACLIPFeatureExtractor
    above, specifically BECAUSE of this uncertainty.

    Confirmed (not assumed): the hub entrypoint name itself, and
    encode_image(images, normalize=True) as the method to call.

    NOT YET VALIDATED on this project's own footage -- validate the
    preprocessing assumption above FIRST if results look wrong, before
    concluding the model itself is a poor fit.
    """

    _HUB_ENTRYPOINT = "dinov2_vitl14_reg4_dinotxt_tet1280d20h24l"

    def __init__(self, variant: str = "default", device: str = "auto", image_size: int = 224):
        self.variant    = variant
        self.image_size = image_size
        self.device     = _resolve_device(device)
        self.model      = self._load()
        self.model.eval().to(self.device)
        self._cached_dim = self._probe_dim()
        log.info("dino.txt (%s) on %s  (dim=%d)", self._HUB_ENTRYPOINT, self.device, self._cached_dim)

    def _load(self):
        loaded = torch.hub.load("facebookresearch/dinov2", self._HUB_ENTRYPOINT)
        # Defensive: the confirmed usage pattern is `model, tokenizer =
        # entrypoint()` when called directly as a Python function (per the
        # dinov3 repo's own inference notebook for the analogous DINOv3
        # entrypoint) -- torch.hub.load's own return convention for this
        # specific entrypoint could not be independently verified without a
        # live download, so both shapes are handled rather than assumed.
        if isinstance(loaded, tuple):
            return loaded[0]
        return loaded

    @torch.no_grad()
    def _probe_dim(self) -> int:
        dummy = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        feats = self._encode([dummy])
        return int(feats.shape[-1])

    @torch.no_grad()
    def _encode(self, images: list[np.ndarray]) -> torch.Tensor:
        tensors = [_preprocess_dino(im, self.image_size) for im in images]
        batch = torch.stack(tensors).to(self.device).float()
        return self.model.encode_image(batch, normalize=True)

    @torch.no_grad()
    def extract(self, images: list[np.ndarray], batch_size: int = 16) -> np.ndarray:
        if not images:
            return np.zeros((0, self._dim()), dtype=np.float32)
        out: list[np.ndarray] = []
        for i in range(0, len(images), batch_size):
            batch_imgs = images[i:i+batch_size]
            feats = self._encode(batch_imgs)
            # encode_image(..., normalize=True) already L2-normalizes --
            # F.normalize again is a defensive no-op if so, not a
            # correctness risk either way.
            out.append(F.normalize(feats, dim=-1).cpu().numpy())
        return np.concatenate(out, axis=0).astype(np.float32)

    def extract_crops(self, frame_bgr: np.ndarray, boxes: list[Box],
                      pad_ratio: float = 0.10, batch_size: int = 16) -> np.ndarray:
        if not boxes:
            return np.zeros((0, self._dim()), dtype=np.float32)
        return self.extract([crop_with_pad(frame_bgr, b, pad_ratio) for b in boxes], batch_size)

    def _dim(self) -> int:
        return self._cached_dim

    def _feature_dim(self) -> int:
        return self._dim()


# ---------------------------------------------------------------------------
# Candidate-crop background masking wrapper (opt-in, stage1.
# feature_extractor.candidate_background_masking)
# ---------------------------------------------------------------------------

class MaskedCropFeatureExtractor:
    """Wraps any base extractor's extract_crops(), running the SAME
    segmentation + background-fill primitive stage1.segmentation already
    uses for reference photos (aero_eyes.utils.geometry.
    apply_background_mode) on each VIDEO CANDIDATE crop first -- see
    FeatureExtractorConfig.candidate_background_masking's own docstring
    (aero_eyes/config.py) for the asymmetry this addresses (reference
    exemplars are background-masked; candidate crops never were).

    Only extract_crops() is wrapped -- extract() passes straight through
    to the base extractor unchanged, since that method takes ALREADY-
    PREPARED images (e.g. reference photos, which get their own masking
    earlier in a separate pipeline) with no box/frame to derive a fresh
    mask from here.

    A per-crop segmentation failure (exception, or the segmenter's own
    "implausible mask" rejection re-raised) falls back to the UNMASKED
    crop rather than dropping the candidate outright -- masking is a
    quality improvement attempt, not a correctness requirement, and one
    candidate's crop being harder to segment (e.g. tiny/degenerate box)
    must not crash or silently vanish the whole batch.
    """

    def __init__(self, base, segmenter, background_mode: str, blur_sigma: float):
        self.base = base
        self.segmenter = segmenter
        self.background_mode = background_mode
        self.blur_sigma = blur_sigma

    def extract(self, images: list[np.ndarray], batch_size: int = 16) -> np.ndarray:
        return self.base.extract(images, batch_size)

    def extract_crops(self, frame_bgr: np.ndarray, boxes: list[Box],
                      pad_ratio: float = 0.10, batch_size: int = 16) -> np.ndarray:
        if not boxes:
            return np.zeros((0, self._dim()), dtype=np.float32)
        from aero_eyes.utils.geometry import apply_background_mode

        masked_crops = []
        for b in boxes:
            crop = crop_with_pad(frame_bgr, b, pad_ratio)
            try:
                mask = self.segmenter.segment(crop)
                masked_crops.append(apply_background_mode(crop, mask, self.background_mode, self.blur_sigma))
            except Exception:
                log.debug("MaskedCropFeatureExtractor: segmentation failed on a candidate crop, using it unmasked", exc_info=True)
                masked_crops.append(crop)
        return self.base.extract(masked_crops, batch_size)

    def _dim(self) -> int:
        return self.base._dim()

    def _feature_dim(self) -> int:
        return self.base._feature_dim()


# ---------------------------------------------------------------------------
# Projection head wrapper (opt-in, stage1.feature_extractor.projection_head)
# ---------------------------------------------------------------------------

class ProjectedFeatureExtractor:
    """Wraps any of the extractors above, applying a trained ProjectionHead
    to its raw output -- see aero_eyes.models.projection_head and
    scripts/train_projection_head.py. The base extractor stays frozen;
    only the (much smaller) head was trained, so this is a drop-in
    replacement everywhere build_feature_extractor() is used (Stage 1
    prototype build, Stage 3/Stage12-GeCo2 candidate features, Stage 4
    verify_interval) -- callers never need to know a projection is active.
    """

    def __init__(self, base, weights_path: str, device: str = "cpu"):
        from aero_eyes.models.projection_head import ProjectionHead

        self.base = base
        self.device = device
        self.head = ProjectionHead.load(weights_path, device=device)
        if self.head.in_dim != base._feature_dim():
            raise ValueError(
                f"Projection head at {weights_path} expects input dim {self.head.in_dim}, "
                f"but the base extractor ({type(base).__name__}) produces {base._feature_dim()}-d "
                "features. Was this head trained against a different stage1.feature_extractor.model?"
            )

    @torch.no_grad()
    def extract(self, images: list[np.ndarray], batch_size: int = 16) -> np.ndarray:
        raw = self.base.extract(images, batch_size)
        if raw.shape[0] == 0:
            return np.zeros((0, self._dim()), dtype=np.float32)
        # base.extract() always returns a CPU numpy array (every extractor's
        # own .extract() ends in .cpu().numpy()) -- must move onto the
        # head's own device before this forward pass, or a non-CPU device
        # (e.g. runtime.device="cuda:1") crashes with a device-mismatch
        # RuntimeError inside the Linear layer.
        projected = self.head(torch.from_numpy(raw).float().to(self.device))
        return projected.cpu().numpy().astype(np.float32)

    def extract_crops(self, frame_bgr: np.ndarray, boxes: list[Box],
                      pad_ratio: float = 0.10, batch_size: int = 16) -> np.ndarray:
        if not boxes:
            return np.zeros((0, self._dim()), dtype=np.float32)
        return self.extract([crop_with_pad(frame_bgr, b, pad_ratio) for b in boxes], batch_size)

    def _dim(self) -> int:
        return self.head.out_dim

    def _feature_dim(self) -> int:
        return self._dim()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_feature_extractor(cfg):
    """Build the feature extractor specified by cfg.stage1.feature_extractor."""
    fe  = cfg.stage1.feature_extractor
    dev = cfg.device()

    if fe.model == "dinov2":
        base = DINOv2FeatureExtractor(
            variant       = fe.dinov2_variant,
            device        = dev,
            image_size    = fe.image_size,
            use_registers = fe.dinov2_use_registers,
            pooling       = fe.dinov2_pooling,
        )
    elif fe.model == "dinov3":
        base = DINOv3FeatureExtractor(
            variant          = fe.dinov3_variant,
            device           = dev,
            source           = fe.dinov3_source,
            pretrain_dataset = fe.dinov3_pretrain_dataset,
            kaggle_model_id  = fe.dinov3_kaggle_model_id,
            image_size       = fe.image_size,
            pooling          = fe.dinov3_pooling,
        )
    elif fe.model == "clip":
        base = CLIPFeatureExtractor(
            variant = fe.clip_variant,
            device  = dev,
        )
    elif fe.model == "siglip":
        base = SiglipFeatureExtractor(
            variant = fe.siglip_variant,
            device  = dev,
        )
    elif fe.model == "ensemble":
        base = EnsembleFeatureExtractor(
            dinov2_variant = fe.dinov2_variant,
            clip_variant   = fe.clip_variant,
            device         = dev,
            image_size     = fe.image_size,
            dinov2_use_registers = fe.dinov2_use_registers,
            dino_model              = fe.ensemble_dino_model,
            dinov3_variant          = fe.dinov3_variant,
            dinov3_source           = fe.dinov3_source,
            dinov3_pretrain_dataset = fe.dinov3_pretrain_dataset,
            dinov3_kaggle_model_id  = fe.dinov3_kaggle_model_id,
            dinov2_pooling          = fe.dinov2_pooling,
            dinov3_pooling          = fe.dinov3_pooling,
        )
    elif fe.model == "fgclip":
        base = FGCLIPFeatureExtractor(
            variant = fe.fgclip_variant,
            device  = dev,
        )
    elif fe.model == "radio":
        base = RadioFeatureExtractor(
            variant    = fe.radio_variant,
            device     = dev,
            image_size = fe.image_size,
        )
    elif fe.model == "siglip2":
        base = Siglip2FeatureExtractor(
            variant = fe.siglip2_variant,
            device  = dev,
        )
    elif fe.model == "evaclip":
        base = EVACLIPFeatureExtractor(
            variant = fe.evaclip_variant,
            device  = dev,
        )
    elif fe.model == "dinotxt":
        base = DinoTxtFeatureExtractor(
            device     = dev,
            image_size = fe.image_size,
        )
    else:
        raise ValueError(
            f"Unknown feature extractor model '{fe.model}'. "
            "Must be 'dinov2', 'dinov3', 'clip', 'siglip', 'ensemble', 'fgclip', 'radio', "
            "'siglip2', 'evaclip', or 'dinotxt'."
        )

    cbm = fe.candidate_background_masking
    if cbm.enabled:
        from aero_eyes.models.segmentation import build_segmenter
        log.info(
            "Feature extractor: wrapping %s with candidate-crop background masking "
            "(model=%s, background_mode=%s)", fe.model, cbm.model, cbm.background_mode,
        )
        base = MaskedCropFeatureExtractor(
            base, build_segmenter(cbm, cfg), cbm.background_mode, cbm.blur_sigma,
        )

    ph = fe.projection_head
    if ph.enabled:
        if not ph.weights_path:
            raise ValueError(
                "stage1.feature_extractor.projection_head.enabled=true but weights_path is not "
                "set. Train one first with scripts/train_projection_head.py, then point "
                "weights_path at the saved .pt file."
            )
        log.info("Feature extractor: wrapping %s with projection head from %s", fe.model, ph.weights_path)
        return ProjectedFeatureExtractor(base, ph.weights_path, device=dev)
    return base


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"
