"""Unit tests for scripts/generate_gdino_prompts.py -- exercises the pure
tag-aggregation logic (_majority_tags) and the per-sample orchestration
(generate_prompt_for_sample) with a fake RAM++ model/inference function, so
none of this needs the real `ram` package or its checkpoint downloaded."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "generate_gdino_prompts.py"
_spec = importlib.util.spec_from_file_location("generate_gdino_prompts", _MODULE_PATH)
gp = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = gp
_spec.loader.exec_module(gp)


# ---------------------------------------------------------------------------
# _majority_tags
# ---------------------------------------------------------------------------

def test_majority_tags_keeps_tags_agreeing_across_majority_of_images():
    tags_per_image = [
        ["backpack", "black", "floor"],
        ["backpack", "black", "table"],
        ["backpack", "red strap", "wall"],
    ]
    tags, used_fallback = gp._majority_tags(tags_per_image, min_agree_frac=0.5)
    assert used_fallback is False
    # "backpack" (3/3) and "black" (2/3) clear a 0.5 threshold (ceil(0.5*3)=2); the
    # per-image-unique tags (floor/table/red strap/wall) do not.
    assert tags == ["backpack", "black"]


def test_majority_tags_orders_by_frequency_then_first_seen():
    tags_per_image = [["a", "b"], ["b", "a"], ["b", "c"]]
    tags, used_fallback = gp._majority_tags(tags_per_image, min_agree_frac=0.0)
    assert used_fallback is False
    # b:3, a:2, c:1 -- frequency desc, ties (none here) by first-seen order.
    assert tags == ["b", "a", "c"]


def test_majority_tags_falls_back_to_union_when_nothing_agrees():
    tags_per_image = [["only_in_1"], ["only_in_2"], ["only_in_3"]]
    tags, used_fallback = gp._majority_tags(tags_per_image, min_agree_frac=1.0)
    assert used_fallback is True
    assert set(tags) == {"only_in_1", "only_in_2", "only_in_3"}


def test_majority_tags_empty_input_returns_empty_no_crash():
    tags, used_fallback = gp._majority_tags([[], [], []], min_agree_frac=0.5)
    assert tags == []
    assert used_fallback is True


# ---------------------------------------------------------------------------
# generate_prompt_for_sample (fake RAM++ model/inference, no real deps)
# ---------------------------------------------------------------------------

class _FakeCfgData:
    def __init__(self, data_root, refs_subdir="object_images", num_references=3):
        self.data_root = str(data_root)
        self.refs_subdir = refs_subdir
        self.num_references = num_references


class _FakeCfg:
    def __init__(self, data_root):
        self.data = _FakeCfgData(data_root)


def _make_sample_with_ref_images(tmp_path, sample_id: str, n: int = 3) -> Path:
    refs_dir = tmp_path / sample_id / "object_images"
    refs_dir.mkdir(parents=True)
    for i in range(n):
        (refs_dir / f"img_{i}.jpg").write_bytes(b"not a real jpeg, never opened by this test")
    return refs_dir


def test_generate_prompt_for_sample_builds_space_joined_phrase(tmp_path, monkeypatch):
    cfg = _FakeCfg(tmp_path)
    _make_sample_with_ref_images(tmp_path, "Sample_0", n=3)

    fake_tags = iter([
        ["backpack", "black", "floor"],
        ["backpack", "black", "table"],
        ["backpack", "red strap", "wall"],
    ])
    monkeypatch.setattr(gp, "tags_for_image", lambda *a, **k: next(fake_tags))

    result = gp.generate_prompt_for_sample(cfg, "Sample_0", model=None, transform=None, inference=None, device="cpu", min_agree_frac=0.5)
    assert result is not None
    prompt_text, per_image_tags, used_fallback = result
    assert prompt_text == "backpack black."
    assert used_fallback is False
    assert len(per_image_tags) == 3


def test_generate_prompt_for_sample_returns_none_when_no_ref_images(tmp_path):
    cfg = _FakeCfg(tmp_path)
    (tmp_path / "Sample_0").mkdir()  # no object_images subdir at all
    result = gp.generate_prompt_for_sample(cfg, "Sample_0", model=None, transform=None, inference=None, device="cpu", min_agree_frac=0.5)
    assert result is None


def test_generate_prompt_for_sample_returns_none_when_all_tags_empty(tmp_path, monkeypatch):
    cfg = _FakeCfg(tmp_path)
    _make_sample_with_ref_images(tmp_path, "Sample_0", n=2)
    monkeypatch.setattr(gp, "tags_for_image", lambda *a, **k: [])

    result = gp.generate_prompt_for_sample(cfg, "Sample_0", model=None, transform=None, inference=None, device="cpu", min_agree_frac=0.5)
    assert result is None


# ---------------------------------------------------------------------------
# _list_sample_ids / _load_ref_image_paths
# ---------------------------------------------------------------------------

def test_list_sample_ids_skips_dotfiles(tmp_path):
    (tmp_path / "Sample_0").mkdir()
    (tmp_path / "Sample_1").mkdir()
    (tmp_path / ".ipynb_checkpoints").mkdir()
    cfg = _FakeCfg(tmp_path)
    assert gp._list_sample_ids(cfg) == ["Sample_0", "Sample_1"]


def test_load_ref_image_paths_caps_at_num_references(tmp_path):
    refs_dir = _make_sample_with_ref_images(tmp_path, "Sample_0", n=5)
    cfg = _FakeCfg(tmp_path)
    cfg.data.num_references = 3
    paths = gp._load_ref_image_paths(cfg, "Sample_0")
    assert len(paths) == 3
    assert all(p.parent == refs_dir for p in paths)
