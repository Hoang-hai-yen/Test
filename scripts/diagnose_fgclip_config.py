"""One-off diagnostic for the FG-CLIP position-embedding out-of-bounds crash
(RuntimeError: CUDA error: device-side assert triggered, at
self.position_embedding(self.position_ids) inside modeling_clip.py).

Prints exactly what image_size/patch_size/num_position_embeddings end up on
config.vision_config at each stage of the loading path this project uses
(aero_eyes.models.features.FGCLIPFeatureExtractor._load), to find WHERE a
mismatch is introduced -- config.json itself is confirmed correct
(image_size=224, patch_size=16) but the crash persists, so the bug is
somewhere in the AutoConfig.from_pretrained(trust_remote_code=True) /
_ensure_fgclip_subconfigs reconstruction path, not the input image size.

Usage:
    python -m scripts.diagnose_fgclip_config
"""
from __future__ import annotations


def _describe(label: str, vision_config) -> None:
    print(f"\n--- {label} ---")
    print(f"  type: {type(vision_config)}")
    for attr in ("image_size", "patch_size", "num_positions", "hidden_size"):
        print(f"  {attr}: {getattr(vision_config, attr, '<missing>')}")
    # Also compute what the position-embedding table SHOULD be sized for,
    # given whatever image_size/patch_size this config object thinks it has.
    img = getattr(vision_config, "image_size", None)
    patch = getattr(vision_config, "patch_size", None)
    if img and patch:
        num_patches = (img // patch) ** 2
        print(f"  implied num_patches (img//patch)**2: {num_patches}  (+1 CLS = {num_patches + 1} positions expected)")


def main() -> None:
    from transformers import AutoConfig

    hf_name = "qihoo360/fg-clip-base"

    # Same prerequisite FGCLIPFeatureExtractor._load() applies before ANY
    # AutoConfig/AutoModel call for this repo -- forgetting this (as this
    # diagnostic script's first version did) reproduces the ORIGINAL
    # transformers.onnx ModuleNotFoundError, unrelated to the position-
    # embedding bug this script is actually trying to isolate.
    from aero_eyes.models.features import _ensure_transformers_onnx_shim
    _ensure_transformers_onnx_shim()

    # ---- Stage 1: raw AutoConfig.from_pretrained, BEFORE this project's own fix ----
    config = AutoConfig.from_pretrained(hf_name, trust_remote_code=True)
    _describe("AutoConfig.from_pretrained (raw, before _ensure_fgclip_subconfigs)", config.vision_config)
    print(f"\n  config.vision_config is a dict? {isinstance(config.vision_config, dict)}")

    # ---- Stage 2: after this project's own _ensure_fgclip_subconfigs fix ----
    from aero_eyes.models.features import _ensure_fgclip_subconfigs
    _ensure_fgclip_subconfigs(config, hf_name)
    _describe("AFTER _ensure_fgclip_subconfigs", config.vision_config)

    # ---- Stage 3: actually construct the model and inspect its REAL position_embedding weight shape ----
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(hf_name, config=config, trust_remote_code=True)
    pos_emb = model.vision_model.embeddings.position_embedding
    print("\n--- Actual constructed model's position_embedding module ---")
    print(f"  num_embeddings (the REAL table size): {pos_emb.num_embeddings}")
    print(f"  embedding_dim: {pos_emb.embedding_dim}")
    print(f"  position_ids buffer shape: {tuple(model.vision_model.embeddings.position_ids.shape)}")
    print(f"  position_ids max value: {int(model.vision_model.embeddings.position_ids.max())}")

    # ---- Stage 4: what does OUR processor + manual 224x224 resize actually produce? ----
    from transformers import AutoImageProcessor
    from PIL import Image
    processor = AutoImageProcessor.from_pretrained(hf_name)
    dummy = Image.new("RGB", (224, 224))
    pixel_values = processor.preprocess([dummy], return_tensors="pt")["pixel_values"]
    print("\n--- Our actual processor output ---")
    print(f"  pixel_values.shape: {tuple(pixel_values.shape)}")
    h, w = pixel_values.shape[-2], pixel_values.shape[-1]
    patch = getattr(config.vision_config, "patch_size", 16)
    produced_patches = (h // patch) * (w // patch)
    print(f"  implied produced patches (h//patch * w//patch): {produced_patches}  (+1 CLS = {produced_patches + 1})")
    print(
        f"\n=== DIAGNOSIS: table size={pos_emb.num_embeddings} vs. patches we actually feed it "
        f"(+1 CLS)={produced_patches + 1} -- {'MISMATCH -- THIS IS THE BUG' if pos_emb.num_embeddings != produced_patches + 1 else 'match, bug is elsewhere'} ==="
    )


if __name__ == "__main__":
    main()
