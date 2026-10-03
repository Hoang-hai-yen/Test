"""Learned color-GROUP classifier (MobileNetV4 via timm) -- an optional,
learned counterpart to the training-free histogram signal in
aero_eyes/utils/color.py. Trained by scripts/train_color_classifier.py.

The fine-grained colors of the Roboflow ColorClassifier dataset (8 colors,
plus a locally added beige folder) are merged into 4 coarse groups
(COLOR_GROUP_PRESETS). Deliberate: the hard boundaries between
adjacent hues (red/orange/yellow under harsh sun, blue/purple/black in
shade) are exactly where the fine-grained label is unreliable on this
project's footage (docs/attribute_taxonomy_plan.md SS9.10), so the
classifier is only asked for distinctions that survive lighting changes.

Callers should compare the SOFTMAX VECTORS of the reference object and a
candidate (see group_probability_similarity), not argmax labels -- a
borderline object split between two groups should not flip from
"match" to "no match" on a tiny probability change.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

# group name -> fine-grained dataset folder names merged into it. Selected
# at train time (scripts/train_color_classifier.py --groups); the checkpoint
# stores the group names it was trained with.
COLOR_GROUP_PRESETS: dict[str, dict[str, tuple[str, ...]]] = {
    "default": {
        "warm": ("red", "orange", "yellow"),
        "dark_cool": ("blue", "black", "purple"),
        # beige (cardboard/kraft): its close-up refs read warm-ish but its drone
        # crops read white -- grouping it with white keeps both on the same side.
        "white": ("white", "beige"),
        "green": ("green",),
    },
    # Red objects (e.g. Helmet) read as purple in drone footage far more often
    # than as blue/black, so purple is moved next to red.
    "purple_warm": {
        "warm": ("red", "orange", "yellow", "purple"),
        "dark_cool": ("blue", "black"),
        "white": ("white", "beige"),
        "green": ("green",),
    },
}
COLOR_GROUPS: dict[str, tuple[str, ...]] = COLOR_GROUP_PRESETS["default"]
GROUP_NAMES: list[str] = list(COLOR_GROUPS)
FINE_TO_GROUP: dict[str, int] = {
    fine: gi for gi, fines in enumerate(COLOR_GROUPS.values()) for fine in fines
}

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class ColorGroupDataset(Dataset):
    """Wraps a folder-format split (`<split_dir>/<fine_color>/*.jpg`, as
    exported by Roboflow) and remaps each fine color to its group index in
    self.group_names. Subfolders not listed in `groups` are skipped, so an
    unexpected extra class in a future dataset version doesn't silently
    get mislabeled.

    groups: group -> fine colors mapping (default: COLOR_GROUPS).
    repeat: fine color -> how many times each of its images is listed, to
    oversample a small folder (e.g. {"beige": 8}). Each repeat goes through
    the random train transform independently, so repeats act as extra
    augmented copies rather than identical duplicates.

    Yields (transformed PIL->tensor image, group index).
    """

    def __init__(self, split_dir: str | Path, transform=None,
                 groups: dict[str, tuple[str, ...]] | None = None, repeat: dict[str, int] | None = None):
        self.split_dir = Path(split_dir)
        self.transform = transform
        groups = groups if groups is not None else COLOR_GROUPS
        self.group_names = list(groups)
        fine_to_group = {fine: gi for gi, fines in enumerate(groups.values()) for fine in fines}
        repeat = repeat or {}
        self.samples: list[tuple[Path, int]] = []
        self.skipped_dirs: list[str] = []
        for class_dir in sorted(p for p in self.split_dir.iterdir() if p.is_dir()):
            name = class_dir.name.lower()
            group = fine_to_group.get(name)
            if group is None:
                self.skipped_dirs.append(class_dir.name)
                continue
            files = [f for f in sorted(class_dir.iterdir()) if f.suffix.lower() in _IMG_EXTS]
            self.samples += [(f, group) for f in files] * max(1, repeat.get(name, 1))
        if not self.samples:
            raise FileNotFoundError(f"No images for any of {sorted(fine_to_group)} under {self.split_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, group = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, group

    def group_counts(self) -> np.ndarray:
        return np.bincount([g for _, g in self.samples], minlength=len(self.group_names))


class ColorGroupClassifier:
    """Inference wrapper around a checkpoint written by
    scripts/train_color_classifier.py. Takes BGR crops (OpenCV convention,
    same as the rest of the pipeline) and returns per-group probabilities.
    """

    def __init__(self, model: torch.nn.Module, img_size: int, mean, std,
                 group_names: list[str], device: str | torch.device = "cpu"):
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.img_size = img_size
        self.group_names = group_names
        self._mean = torch.tensor(mean, dtype=torch.float32, device=self.device).view(1, 3, 1, 1)
        self._std = torch.tensor(std, dtype=torch.float32, device=self.device).view(1, 3, 1, 1)

    @classmethod
    def from_checkpoint(cls, path: str | Path, device: str | torch.device = "cpu") -> "ColorGroupClassifier":
        import timm

        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        model = timm.create_model(ckpt["model_name"], pretrained=False, num_classes=len(ckpt["group_names"]))
        model.load_state_dict(ckpt["state_dict"])
        return cls(model, ckpt["img_size"], ckpt["mean"], ckpt["std"], ckpt["group_names"], device)

    def _preprocess(self, crops_bgr: list[np.ndarray]) -> torch.Tensor:
        # Plain stretch-resize, matching the dataset's own "Resize to 224x224
        # (Stretch)" preprocessing -- no aspect-preserving letterbox.
        batch = np.stack([
            cv2.resize(cv2.cvtColor(c, cv2.COLOR_BGR2RGB), (self.img_size, self.img_size),
                       interpolation=cv2.INTER_LINEAR)
            for c in crops_bgr
        ])
        x = torch.from_numpy(batch).to(self.device).permute(0, 3, 1, 2).float().div_(255.0)
        return (x - self._mean) / self._std

    @torch.inference_mode()
    def predict_proba(self, crops_bgr: list[np.ndarray], batch_size: int = 64) -> np.ndarray:
        """(N, n_groups) softmax probabilities. Empty crops (zero-area box)
        get a uniform distribution instead of crashing cv2.resize."""
        n_groups = len(self.group_names)
        out = np.full((len(crops_bgr), n_groups), 1.0 / n_groups, dtype=np.float32)
        valid = [i for i, c in enumerate(crops_bgr) if c is not None and c.size > 0]
        for start in range(0, len(valid), batch_size):
            idx = valid[start:start + batch_size]
            logits = self.model(self._preprocess([crops_bgr[i] for i in idx]))
            out[idx] = logits.float().softmax(dim=1).cpu().numpy()
        return out


_CLASSIFIER_CACHE: dict[tuple[str, str], ColorGroupClassifier] = {}


def load_color_classifier(path: str | Path, device: str = "auto") -> ColorGroupClassifier:
    """Load a checkpoint once per (resolved path, device) and reuse it --
    build_color_signature runs once per stage per sample (Stage 1+2+3 and
    again in Stage 4), so re-reading the weights each time is wasted work.
    device "auto" picks CUDA when available."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    p = Path(path).resolve()
    if not p.is_file():
        raise FileNotFoundError(
            f"color classifier checkpoint not found: {p} -- train one with "
            f"scripts/train_color_classifier.py or fix color_postfilter.classifier_weights_path"
        )
    key = (str(p), device)
    if key not in _CLASSIFIER_CACHE:
        _CLASSIFIER_CACHE[key] = ColorGroupClassifier.from_checkpoint(p, device)
    return _CLASSIFIER_CACHE[key]


def group_probability_similarity(p_ref: np.ndarray, p_cand: np.ndarray) -> np.ndarray:
    """Bhattacharyya coefficient sum_k sqrt(p_ref[k] * p_cand[k]) in [0,1]
    (1 = identical distributions) -- the same similarity family as
    utils/color.histogram_similarity's default, so the two signals live on
    comparable scales if fused. p_ref: (G,), p_cand: (N, G) -> (N,)."""
    return np.sqrt(np.clip(p_cand, 0, None) * np.clip(p_ref, 0, None)[None, :]).sum(axis=1)
