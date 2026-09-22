"""Smoke test for stage1.feature_extractor.{dinov2,dinov3}_pooling=
"multiscale_attn" -- builds the extractor directly and runs it on a couple
of images, with NO project sample data / prototype.npz / video needed. Use
this to confirm the new pooling mode loads and produces the expected
output shape before touching the real pipeline (Stage 1/3).

Requires network access to download the HuggingFace checkpoint the first
time (facebook/dinov2-base for --model dinov2, gated
facebook/dinov3-vitb16-pretrain-lvd1689m for --model dinov3 -- request
access on the model page and set HF_TOKEN first).

Usage:
    python -m scripts.check_multiscale_attn_pooling --model dinov2
    python -m scripts.check_multiscale_attn_pooling --model dinov3
    python -m scripts.check_multiscale_attn_pooling --model dinov2 --image path/to/crop.jpg
"""
from __future__ import annotations

import argparse
import logging

import cv2
import numpy as np

log = logging.getLogger(__name__)


def _load_image(path: str | None) -> np.ndarray:
    if path is None:
        # Deterministic synthetic image (not a real object) -- enough to
        # confirm the extractor runs and produces the right shape; doesn't
        # tell you anything about matching QUALITY, just plumbing.
        rng = np.random.default_rng(0)
        return rng.integers(0, 255, size=(128, 96, 3), dtype=np.uint8)
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["dinov2", "dinov3"], default="dinov2")
    p.add_argument("--variant", default=None, help="e.g. vitb14 (dinov2) / vitb16 (dinov3); default = smallest")
    p.add_argument("--image", default=None, help="path to a real crop; default = a synthetic dummy image")
    p.add_argument("--device", default="auto")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    from aero_eyes.models.features import DINOv2FeatureExtractor, DINOv3FeatureExtractor

    img1 = _load_image(args.image)
    img2 = _load_image(args.image)  # same or a second real crop if you pass --image twice manually

    if args.model == "dinov2":
        ext_cls = DINOv2FeatureExtractor
        variant = args.variant or "vits14"
        ext = ext_cls(variant=variant, device=args.device, pooling="multiscale_attn")
    else:
        ext_cls = DINOv3FeatureExtractor
        variant = args.variant or "vits16"
        ext = ext_cls(variant=variant, device=args.device, pooling="multiscale_attn")

    print(f"\nBuilt {args.model} variant={variant} pooling=multiscale_attn")
    print(f"scale_layers = {ext._scale_layers}")
    print(f"_dim()       = {ext._dim()}  (single-CLS dim x {len(ext._scale_layers)})")

    feats = ext.extract([img1, img2])
    print(f"\nextract() output shape = {feats.shape}")
    assert feats.shape == (2, ext._dim()), "output shape doesn't match _dim() -- BUG"
    norms = np.linalg.norm(feats, axis=-1)
    print(f"L2 norms (should be ~1.0) = {norms}")
    assert np.allclose(norms, 1.0, atol=1e-4), "output isn't L2-normalized -- BUG"

    sim = float(feats[0] @ feats[1])
    print(f"\ncosine(img1, img2) = {sim:.4f}  (=1.0 if --image omitted, since img1==img2)")
    print("\nOK -- multiscale_attn pooling runs end-to-end.")


if __name__ == "__main__":
    main()
