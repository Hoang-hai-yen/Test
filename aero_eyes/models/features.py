"""Feature extractors for Stage B (prototype encoding + candidate matching).

Supported models:
  dinov2   — DINOv2 ViT-S/14 or ViT-B/14, CLS token (384 or 768-d)
  dinov3   — DINOv3 ViT-S/16, ViT-B/16 or ViT-L/16, CLS token (384/768/1024-d).
             Weights are gated on HuggingFace -- request access and set
             HF_TOKEN before use.
  clip     — CLIP ViT-B/32, visual encoder (512-d)
  siglip   — SigLIP vision encoder (base/large/so400m), pooled output
             (768/1024/1152-d). Open access, no gating.
  ensemble — DINOv2 + CLIP concatenated then L2-normalized (1280 or 896-d)
  vdt      — View-Decoupled Transformer (VDT) based on ViT-Base (768-d).
             Custom loaded from weights file. 

All extractors return L2-normalized float32 feature vectors.
"""
from __future__ import annotations

import logging
from typing import Any
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

from aero_eyes.types import Box
from aero_eyes.utils.geometry import crop_with_pad

log = logging.getLogger(__name__)

# ImageNet normalization (DINOv2 & VDT)
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
# VDT (View-Decoupled Transformer)
# ---------------------------------------------------------------------------

class VDTFeatureExtractor:
    """View-Decoupled Transformer (VDT) based on ViT-Base.
    
    Loads a custom .pth file. The input image size is strictly 256x128 
    as per the paper. Only the meta_token (identity) is returned.
    """
    def __init__(self, weights_path: str, device: str = "auto"):
        self.device = _resolve_device(device)
        self.weights_path = weights_path
        self.model = self._load_model()
        self.model.eval().to(self.device)
        
        # VDT requires 256x128 resolution
        self.transform = T.Compose([
            T.Resize((256, 128)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        log.info(f"VDT loaded on {self.device} from {self.weights_path} (dim=768)")

    def _load_model(self):
        if not self.weights_path or not os.path.exists(self.weights_path):
            raise FileNotFoundError(f"VDT weights not found at {self.weights_path}")
        
        # Import your actual VDT model class here
        # For example, assuming it's in aero_eyes.models.vdt_model:
        try:
            from aero_eyes.models.vdt_model import build_vdt_model
            model = build_vdt_model()
        except ImportError:
            log.warning("Could not import build_vdt_model from aero_eyes.models.vdt_model. Falling back to timm vit_base_patch16_224 as a placeholder.")
            import timm
            model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=0, img_size=(256, 128))

        state_dict = torch.load(self.weights_path, map_location=self.device)
        if "module." in list(state_dict.keys())[0]:
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        
        model.load_state_dict(state_dict, strict=False)
        return model

    @torch.no_grad()
    def extract(self, images: list[np.ndarray], batch_size: int = 16) -> np.ndarray:
        if not images:
            return np.zeros((0, self._dim()), dtype=np.float32)
        
        pil_imgs = [_bgr_to_pil(im) for im in images]
        tensors = [self.transform(img) for img in pil_imgs]
        
        out = []
        for i in range(0, len(tensors), batch_size):
            batch = torch.stack(tensors[i:i+batch_size]).to(self.device)
            outputs = self.model(batch)
            
            # Extract meta token (t_m). Assume output is tuple (t_m, t_v) or just t_m
            if isinstance(outputs, tuple):
                meta_token = outputs[0]
            else:
                meta_token = outputs
                
            out.append(F.normalize(meta_token, p=2, dim=1).cpu().numpy())
            
        return np.concatenate(out, axis=0).astype(np.float32)

    def extract_crops(self, frame_bgr: np.ndarray, boxes: list[Box],
                      pad_ratio: float = 0.10, batch_size: int = 16) -> np.ndarray:
        if not boxes:
            return np.zeros((0, self._dim()), dtype=np.float32)
        return self.extract([crop_with_pad(frame_bgr, b, pad_ratio) for b in boxes], batch_size)

    def _dim(self) -> int:
        return 768

    def _feature_dim(self) -> int:
        return self._dim()

# ---------------------------------------------------------------------------
# DINOv2
# ---------------------------------------------------------------------------

class DINOv2FeatureExtractor:
    """Batched DINOv2 ViT-S/14 or ViT-B/14, returns L2-normalized CLS tokens."""

    def __init__(self, variant: str = "vitb14", device: str = "auto", image_size: int = 224):
        self.variant    = variant
        self.image_size = image_size
        self.device     = _resolve_device(device)
        self.model      = self._load(variant)
        self.model.eval().to(self.device)
        log.info("DINOv2 %s on %s  (dim=%d)", variant, self.device, self._dim())

    def _load(self, variant: str) -> Any:
        try:
            m = torch.hub.load("facebookresearch/dinov2", f"dinov2_{variant}", pretrained=True)
            return m
        except Exception as e:
            log.warning("torch.hub failed (%s) → HuggingFace", e)
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

    Weights are gated on HuggingFace (facebook/dinov3-*-pretrain-lvd1689m):
    request access on the model page, then set HF_TOKEN before running, or
    `from_pretrained` will fail with a 401/403.
    """

    _VARIANT_MAP = {
        "vits16": "facebook/dinov3-vits16-pretrain-lvd1689m",
        "vitb16": "facebook/dinov3-vitb16-pretrain-lvd1689m",
        "vitl16": "facebook/dinov3-vitl16-pretrain-lvd1689m",
    }
    _DIMS = {"vits16": 384, "vitb16": 768, "vitl16": 1024}

    def __init__(self, variant: str = "vitb16", device: str = "auto"):
        if variant not in self._VARIANT_MAP:
            raise ValueError(f"Unknown DINOv3 variant '{variant}'. Must be one of {list(self._VARIANT_MAP)}.")
        self.variant = variant
        self.device  = _resolve_device(device)
        self.model, self.processor = self._load(variant)
        self.model.eval().to(self.device)
        log.info("DINOv3 %s on %s  (dim=%d)", variant, self.device, self._dim())

    def _load(self, variant: str):
        from transformers import AutoImageProcessor, AutoModel
        hf_name = self._VARIANT_MAP[variant]
        processor = AutoImageProcessor.from_pretrained(hf_name)
        model     = AutoModel.from_pretrained(hf_name)
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
    """Concatenates DINOv2 + CLIP features then L2-normalizes.

    dim = DINOv2_dim + CLIP_dim  (e.g. 768 + 512 = 1280 for vitb14 + vit-b/32)
    """

    def __init__(
        self,
        dinov2_variant: str = "vitb14",
        clip_variant: str = "vit-b/32",
        device: str = "auto",
        image_size: int = 224,
    ):
        self.dino = DINOv2FeatureExtractor(dinov2_variant, device, image_size)
        self.clip = CLIPFeatureExtractor(clip_variant, device)
        log.info("Ensemble DINOv2+CLIP  dim=%d", self._dim())

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
# Factory
# ---------------------------------------------------------------------------

def build_feature_extractor(cfg) -> DINOv2FeatureExtractor | DINOv3FeatureExtractor | CLIPFeatureExtractor | SiglipFeatureExtractor | EnsembleFeatureExtractor | VDTFeatureExtractor:
    """Build the feature extractor specified by cfg.stage1.feature_extractor."""
    fe  = cfg.stage1.feature_extractor
    dev = cfg.device()

    if fe.model == "dinov2":
        return DINOv2FeatureExtractor(
            variant    = fe.dinov2_variant,
            device     = dev,
            image_size = fe.image_size,
        )
    if fe.model == "dinov3":
        return DINOv3FeatureExtractor(
            variant = fe.dinov3_variant,
            device  = dev,
        )
    if fe.model == "clip":
        return CLIPFeatureExtractor(
            variant = fe.clip_variant,
            device  = dev,
        )
    if fe.model == "siglip":
        return SiglipFeatureExtractor(
            variant = fe.siglip_variant,
            device  = dev,
        )
    if fe.model == "ensemble":
        return EnsembleFeatureExtractor(
            dinov2_variant = fe.dinov2_variant,
            clip_variant   = fe.clip_variant,
            device         = dev,
            image_size     = fe.image_size,
        )
    if fe.model == "vdt":
        return VDTFeatureExtractor(
            weights_path = fe.vdt_weights,
            device = dev
        )
    raise ValueError(
        f"Unknown feature extractor model '{fe.model}'. "
        "Must be 'dinov2', 'dinov3', 'clip', 'siglip', 'ensemble', or 'vdt'."
    )


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"
