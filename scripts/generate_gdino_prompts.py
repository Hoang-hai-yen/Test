"""One-off helper: auto-generate stage123_gdino prompt.txt files from each
sample's own reference photos (cfg.data.refs_subdir), using RAM++
(Recognize Anything Plus Model, arXiv:2310.15200) to tag each photo, then
picking the tags that AGREE across most of a sample's reference photos as
the best guess for "the object itself" (as opposed to background/lighting
tags, which differ photo to photo since the 3 ref photos are different
angles/crops of the same object -- see _majority_tags's own docstring).

Deliberately NOT wired into stage123_gdino.py itself: this writes a plain
prompt.txt per sample (the SAME file resolve_text_prompt() already reads
with top priority, see aero_eyes/stages/stage123_gdino.py) so a human
reviews/edits the auto-generated prompt BEFORE it ever reaches the real
detection run -- an unreviewed auto-caption silently driving an
open-vocabulary detector is exactly the kind of failure this project's own
conventions try to avoid (see e.g. Stage123GDinoConfig's own "NOT YET
VALIDATED" framing). Running this script only ever produces a plain text
file for you to look at; it never touches config.yaml or detections.json.

Setup (NOT a project-wide dependency -- only needed to run this script):
    pip install git+https://github.com/xinyu1205/recognize-anything.git
    # download ram_plus_swin_large_14m.pth (~3GB) from
    # https://huggingface.co/xinyu1205/recognize-anything-plus-model

Usage:
    python -m scripts.generate_gdino_prompts --config configs/config.yaml \\
        --ram-checkpoint /path/to/ram_plus_swin_large_14m.pth
    # single sample, preview only (no file written):
    python -m scripts.generate_gdino_prompts --config configs/config.yaml \\
        --ram-checkpoint /path/to/ram_plus_swin_large_14m.pth \\
        --sample Backpack_0 --dry-run
"""
from __future__ import annotations

import argparse
import logging
import math
from collections import Counter
from pathlib import Path

log = logging.getLogger(__name__)

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _list_sample_ids(cfg) -> list[str]:
    data_root = Path(cfg.data.data_root)
    if not data_root.exists():
        raise FileNotFoundError(f"data_root not found: {data_root}")
    return [d.name for d in sorted(data_root.iterdir()) if d.is_dir() and not d.name.startswith(".")]


def _load_ref_image_paths(cfg, sample_id: str) -> list[Path]:
    refs_dir = Path(cfg.data.data_root) / sample_id / cfg.data.refs_subdir
    paths = sorted(p for p in (refs_dir.iterdir() if refs_dir.is_dir() else []) if p.suffix.lower() in _IMG_EXTS)
    return paths[: cfg.data.num_references]


def _majority_tags(tags_per_image: list[list[str]], min_agree_frac: float) -> tuple[list[str], bool]:
    """Keeps tags that appear in at least ceil(min_agree_frac * n_images) of
    the reference photos' own tag lists -- the idea being that the OBJECT's
    own tags should recur across every photo (different angle, same object),
    while background/lighting/incidental tags mostly don't. Order: most
    frequent first, ties broken by first-seen order across images.

    Falls back to the plain union of every tag (still deduplicated,
    first-seen order) if NOTHING clears the threshold -- returns
    (tags, used_fallback) so the caller can warn loudly rather than writing
    a low-confidence guess silently. Never raises -- an empty result here
    just means every image produced zero tags."""
    n = len(tags_per_image)
    threshold = max(1, math.ceil(min_agree_frac * n))
    counts: Counter[str] = Counter()
    first_seen: dict[str, int] = {}
    order = 0
    for tags in tags_per_image:
        for t in tags:
            if t not in first_seen:
                first_seen[t] = order
                order += 1
            counts[t] += 1
    ranked = sorted(counts, key=lambda t: (-counts[t], first_seen[t]))
    majority = [t for t in ranked if counts[t] >= threshold]
    if majority:
        return majority, False
    return ranked, True


def _build_ram_plus(checkpoint: str, image_size: int, device: str):
    try:
        from ram import get_transform
        from ram import inference_ram as inference
        from ram.models import ram_plus
    except ImportError as e:
        raise RuntimeError(
            "The `ram` package isn't installed -- run `pip install "
            "git+https://github.com/xinyu1205/recognize-anything.git` and download "
            "ram_plus_swin_large_14m.pth from "
            "https://huggingface.co/xinyu1205/recognize-anything-plus-model first. "
            f"(original error: {e})"
        ) from e
    model = ram_plus(pretrained=checkpoint, image_size=image_size, vit="swin_l")
    model.eval()
    model = model.to(device)
    transform = get_transform(image_size=image_size)
    return model, transform, inference


def tags_for_image(model, transform, inference, image_path: Path, device: str) -> list[str]:
    from PIL import Image

    img = transform(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
    english_tags, _chinese_tags = inference(img, model)
    # RAM/RAM++'s own convention: one string, tags separated by " | ".
    return [t.strip().lower() for t in english_tags.split("|") if t.strip()]


def generate_prompt_for_sample(cfg, sample_id: str, model, transform, inference, device: str, min_agree_frac: float):
    """Returns (prompt_text, per_image_tags, used_fallback) or None if the
    sample has no reference images to tag."""
    ref_paths = _load_ref_image_paths(cfg, sample_id)
    if not ref_paths:
        log.warning("[gdino-prompts] %s: no reference images found under %s -- skipped.",
                    sample_id, Path(cfg.data.data_root) / sample_id / cfg.data.refs_subdir)
        return None
    per_image_tags = [tags_for_image(model, transform, inference, p, device) for p in ref_paths]
    tags, used_fallback = _majority_tags(per_image_tags, min_agree_frac)
    if not tags:
        log.warning("[gdino-prompts] %s: RAM++ returned zero tags for every reference image -- skipped.", sample_id)
        return None
    prompt_text = " ".join(tags) + "."
    return prompt_text, per_image_tags, used_fallback


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser(description="Auto-generate stage123_gdino prompt.txt files via RAM++ tagging")
    p.add_argument("--config", required=True)
    p.add_argument("--sample", default=None, help="omit to process every sample under data.data_root")
    p.add_argument("--ram-checkpoint", required=True, help="path to ram_plus_swin_large_14m.pth")
    p.add_argument("--image-size", type=int, default=384)
    p.add_argument("--min-agree-frac", type=float, default=0.5,
                   help="keep a tag if it appears in at least this fraction of a sample's ref photos (default 0.5 = majority)")
    p.add_argument("--overwrite", action="store_true",
                   help="overwrite an existing prompt.txt (default: skip samples that already have one, "
                        "so a hand-edited prompt is never clobbered silently)")
    p.add_argument("--dry-run", action="store_true", help="print what would be written, write nothing")
    p.add_argument("--device", default=None, help="cpu | cuda (default: cfg.device())")
    args = p.parse_args()

    from aero_eyes.config import load_config
    cfg = load_config(args.config)
    device = args.device or cfg.device()

    sample_ids = [args.sample] if args.sample else _list_sample_ids(cfg)
    if not sample_ids:
        raise ValueError(f"No samples found under {cfg.data.data_root}")

    log.info("[gdino-prompts] loading RAM++ (%s) on %s ...", args.ram_checkpoint, device)
    model, transform, inference = _build_ram_plus(args.ram_checkpoint, args.image_size, device)

    for sample_id in sample_ids:
        prompt_path = Path(cfg.data.data_root) / sample_id / cfg.stage123_gdino.prompt_file_name
        if prompt_path.exists() and not args.overwrite:
            log.info("[gdino-prompts] %s: %s already exists -- skipped (pass --overwrite to replace it).",
                      sample_id, prompt_path)
            continue

        result = generate_prompt_for_sample(cfg, sample_id, model, transform, inference, device, args.min_agree_frac)
        if result is None:
            continue
        prompt_text, per_image_tags, used_fallback = result

        log.info("[gdino-prompts] %s:", sample_id)
        for img_tags in per_image_tags:
            log.info("    ref tags: %s", ", ".join(img_tags))
        if used_fallback:
            log.warning(
                "    no tag reached the %.0f%% agreement threshold across %d ref photos -- "
                "falling back to the full tag union. Review this prompt carefully before use.",
                args.min_agree_frac * 100, len(per_image_tags),
            )
        log.info("    -> prompt: %r", prompt_text)

        if args.dry_run:
            log.info("    (dry-run, not written)")
            continue
        prompt_path.write_text(prompt_text + "\n", encoding="utf-8")
        log.info("    written to %s", prompt_path)

    log.info(
        "[gdino-prompts] done. REVIEW each prompt.txt before running "
        "pipeline.detector=grounding_dino -- RAM++ tags are auto-generated guesses, not validated prompts."
    )


if __name__ == "__main__":
    main()
