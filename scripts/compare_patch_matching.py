"""Compare CLS cosine vs patch-token Chamfer / OT scores on hand-picked crops.

For diagnosing the "blank paper scores like the ID card" failure: give it the
reference images and a few candidate crops (a true object crop, a blank-paper
crop, ...) and it prints, per crop, the CLS cosine next to the patch scores for
several settings. A good setting widens the gap between the true crop and the
blank/confuser crops.

    python scripts/compare_patch_matching.py --refs data/IDCard_1/refs \
        --crops crops/true_1.jpg crops/blank_paper.jpg \
        --layers "-1" "6,9,-1" --long-sides 224 448 --methods chamfer ot
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import cv2
import numpy as np

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _collect(paths: list[str]) -> list[Path]:
    out: list[Path] = []
    for p in map(Path, paths):
        out.extend(sorted(f for f in p.iterdir() if f.suffix.lower() in EXTS) if p.is_dir() else [p])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--refs", nargs="+", required=True, help="reference image files or dirs")
    ap.add_argument("--crops", nargs="+", required=True, help="candidate crop files or dirs")
    ap.add_argument("--variant", default="vitb16", choices=["vits16", "vitb16", "vitl16"])
    ap.add_argument("--pretrain-dataset", default="lvd1689m", choices=["lvd1689m", "sat493m"])
    ap.add_argument("--lora", default=None, help="optional LoRA weights path")
    ap.add_argument("--layers", nargs="+", default=["-1"], help='layer sets, e.g. "-1" "6,9,-1"')
    ap.add_argument("--long-sides", nargs="+", type=int, default=[224])
    ap.add_argument("--methods", nargs="+", default=["chamfer", "ot"], choices=["chamfer", "ot"])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--reuse-cls-preprocess", action="store_true",
                    help="patch path uses --preprocess-mode/--candidate-preprocess-mode/--image-size "
                         "like the CLS path (--long-sides is then ignored)")
    ap.add_argument("--preprocess-mode", default="stretch", choices=["stretch", "resize_then_crop", "pad_to_square"])
    ap.add_argument("--candidate-preprocess-mode", default=None,
                    choices=["stretch", "resize_then_crop", "pad_to_square"])
    ap.add_argument("--image-size", type=int, default=224)
    args = ap.parse_args()

    from aero_eyes.config import PatchMatchingConfig
    from aero_eyes.models.features import DINOv3FeatureExtractor
    from aero_eyes.models.patch_match import PatchMatcher

    extractor = DINOv3FeatureExtractor(
        variant=args.variant, device=args.device, pretrain_dataset=args.pretrain_dataset,
        lora_weights_path=args.lora, image_size=args.image_size,
        preprocess_mode=args.preprocess_mode, candidate_preprocess_mode=args.candidate_preprocess_mode,
    )
    ref_paths, crop_paths = _collect(args.refs), _collect(args.crops)
    refs = [cv2.imread(str(p)) for p in ref_paths]
    crops = [cv2.imread(str(p)) for p in crop_paths]

    # CLS baseline: mean cosine over refs
    ref_cls, crop_cls = extractor.extract(refs), extractor.extract(crops)
    cls_scores = (crop_cls @ ref_cls.T).mean(axis=1)

    rows: dict[str, np.ndarray] = {"cls": cls_scores}
    long_sides = [args.image_size] if args.reuse_cls_preprocess else args.long_sides
    for layers_s, long_side, method in itertools.product(args.layers, long_sides, args.methods):
        cfg = PatchMatchingConfig(
            enabled=True, method=method, long_side=long_side,
            layers=[int(x) for x in layers_s.split(",")],
            reuse_cls_preprocess=args.reuse_cls_preprocess,
        )
        matcher = PatchMatcher(extractor, cfg)
        matcher.set_references(refs)
        rows[f"{method} L[{layers_s}] {long_side}px"] = matcher.score_crops(crops).mean(axis=1)

    width = max(len(k) for k in rows)
    print(f"\n{'setting'.ljust(width)}  " + "  ".join(p.name[:18].rjust(18) for p in crop_paths))
    for name, vals in rows.items():
        print(f"{name.ljust(width)}  " + "  ".join(f"{v:18.4f}" for v in vals))


if __name__ == "__main__":
    main()
