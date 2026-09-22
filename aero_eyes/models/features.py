"""Feature extractors for Stage B (prototype encoding + candidate matching).

Supported models:
  dinov2   — DINOv2 ViT-S/14 or ViT-B/14, CLS token (384 or 768-d).
             Optionally the "with registers" variant (dinov2_use_registers) --
             same output dim, cleaner attention/features per Meta's ablations.
  dinov3   — DINOv3 ViT-S/16, ViT-B/16 or ViT-L/16, CLS token (384/768/1024-d).
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


# ---------------------------------------------------------------------------
# DINOv2
# ---------------------------------------------------------------------------

class DINOv2FeatureExtractor:
    """Batched DINOv2 ViT-S/14 or ViT-B/14, returns L2-normalized CLS tokens."""

    def __init__(
        self, variant: str = "vitb14", device: str = "auto", image_size: int = 224,
        use_registers: bool = False,
    ):
        self.variant       = variant
        self.image_size    = image_size
        self.use_registers = use_registers
        self.device        = _resolve_device(device)
        self.model         = self._load(variant)
        self.model.eval().to(self.device)
        log.info(
            "DINOv2 %s%s on %s  (dim=%d)", variant,
            " (with registers)" if use_registers else "", self.device, self._dim(),
        )

    def _load(self, variant: str) -> Any:
        hub_name = f"dinov2_{variant}" + ("_reg" if self.use_registers else "")
        try:
            m = torch.hub.load("facebookresearch/dinov2", hub_name, pretrained=True)
            return m
        except Exception as e:
            log.warning("torch.hub failed (%s) → HuggingFace", e)
        if self.use_registers:
            hf_map = {
                "vits14": "facebook/dinov2-with-registers-small", "vitb14": "facebook/dinov2-with-registers-base",
                "vitl14": "facebook/dinov2-with-registers-large", "vitg14": "facebook/dinov2-with-registers-giant",
            }
        else:
            hf_map = {
                "vits14": "facebook/dinov2-small", "vitb14": "facebook/dinov2-base",
                "vitl14": "facebook/dinov2-large", "vitg14": "facebook/dinov2-giant",
            }
        if variant not in hf_map:
            raise ValueError(f"Unknown DINOv2 variant '{variant}'. Must be one of {list(hf_map)}.")
        from transformers import AutoModel
        m = AutoModel.from_pretrained(hf_map[variant])
        m._hf = True
        return m

    @torch.no_grad()
    def extract(self, images: list[np.ndarray], batch_size: int = 16) -> np.ndarray:
        if not images:
            return np.zeros((0, self._dim()), dtype=np.float32)
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

    def extract_crops(self, frame_bgr: np.ndarray, boxes: list[Box],
                      pad_ratio: float = 0.10, batch_size: int = 16) -> np.ndarray:
        if not boxes:
            return np.zeros((0, self._dim()), dtype=np.float32)
        return self.extract([crop_with_pad(frame_bgr, b, pad_ratio) for b in boxes], batch_size)

    _DIMS = {"vits14": 384, "vitb14": 768, "vitl14": 1024, "vitg14": 1536}

    def _dim(self) -> int:
        return self._DIMS.get(self.variant, 768)

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
    """

    _ARCHS = ("vits16", "vitb16", "vitl16")
    _PRETRAIN_DATASETS = ("lvd1689m", "sat493m")
    _DIMS = {"vits16": 384, "vitb16": 768, "vitl16": 1024}

    def __init__(
        self, variant: str = "vitb16", device: str = "auto",
        source: str = "huggingface", pretrain_dataset: str = "lvd1689m",
        kaggle_model_id: str | None = None, image_size: int = 224,
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
        self.variant          = variant
        self.pretrain_dataset = pretrain_dataset
        self.source           = source
        self.image_size       = image_size
        self.device           = _resolve_device(device)
        self.processor        = None
        if source == "huggingface":
            self.model, self.processor = self._load_huggingface(variant, pretrain_dataset)
        else:
            self.model = self._load_kaggle(variant, kaggle_model_id)
        self.model.eval().to(self.device)
        log.info(
            "DINOv3 %s pretrain=%s (source=%s) on %s  (dim=%d)",
            variant, pretrain_dataset, source, self.device, self._dim(),
        )

    def _load_huggingface(self, variant: str, pretrain_dataset: str):
        from transformers import AutoImageProcessor, AutoModel
        hf_name = f"facebook/dinov3-{variant}-pretrain-{pretrain_dataset}"
        processor = AutoImageProcessor.from_pretrained(hf_name)
        model     = AutoModel.from_pretrained(hf_name)
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
    ):
        if dino_model == "dinov3":
            self.dino = DINOv3FeatureExtractor(
                variant=dinov3_variant, device=device, source=dinov3_source,
                pretrain_dataset=dinov3_pretrain_dataset, kaggle_model_id=dinov3_kaggle_model_id,
                image_size=image_size,
            )
        elif dino_model == "dinov2":
            self.dino = DINOv2FeatureExtractor(dinov2_variant, device, image_size, dinov2_use_registers)
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
            from transformers import AutoImageProcessor, AutoModelForCausalLM
        except ImportError:
            raise RuntimeError(
                "transformers not installed. Run: pip install transformers"
            )
        hf_name = self._VARIANT_MAP[variant]
        # FG-CLIP ships custom modeling code (not a stock CLIPModel), so it
        # needs trust_remote_code -- registered under AutoModelForCausalLM
        # per the model's own official usage example, despite being used
        # here purely as a vision encoder (only get_image_features() below
        # is called, never text generation).
        processor = AutoImageProcessor.from_pretrained(hf_name)
        model = AutoModelForCausalLM.from_pretrained(hf_name, trust_remote_code=True)
        return model, processor

    @torch.no_grad()
    def extract(self, images: list[np.ndarray], batch_size: int = 16) -> np.ndarray:
        if not images:
            return np.zeros((0, self._dim()), dtype=np.float32)
        pil_imgs = [_bgr_to_pil(im) for im in images]
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
        )
    elif fe.model == "dinov3":
        base = DINOv3FeatureExtractor(
            variant          = fe.dinov3_variant,
            device           = dev,
            source           = fe.dinov3_source,
            pretrain_dataset = fe.dinov3_pretrain_dataset,
            kaggle_model_id  = fe.dinov3_kaggle_model_id,
            image_size       = fe.image_size,
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
    else:
        raise ValueError(
            f"Unknown feature extractor model '{fe.model}'. "
            "Must be 'dinov2', 'dinov3', 'clip', 'siglip', 'ensemble', 'fgclip', or 'radio'."
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
