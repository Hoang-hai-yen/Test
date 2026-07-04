# AERO EYES — Few-Shot Spatio-Temporal Drone Object Localization

Search-and-rescue perception: given **3 close-up reference images** of one
target object + a **drone video**, predict per-frame bounding boxes of that
object. Scored by **ST-IoU** (joint when + where overlap as one space–time
volume).

> Status: SKELETON. Implement per `docs/CLAUDE_CODE_PROMPT.md`.
> **Do not write code before completing the data-format Q&A in that prompt.**

## Pipeline (the spec)

| Stage | Does | Key models | Configurable |
|------|------|-----------|--------------|
| 1 | Reference processing (offline, once) | MobileSAM → DINOv2 ViT-S/14 → multi-view prototype | dinov2 variant, fusion, synth-view aug |
| 2 | Candidate generation per keyframe | **YOLOv11n OR FastSAM-s** + optional **SAHI** | `proposal_model`, `use_sahi`, keyframe interval |
| 3 | Cross-domain matching | cosine sim vs prototype + NMS | `match_threshold`, `nms_iou`, calibration |
| 4 | Tracking between keyframes | **builtin / litetrack / none** | `tracker`, `tracker_conf_threshold` (τ) |
| 5 | Spatio-temporal output | tube build + smoothing + submission | smoothing, gap fill |

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Per-stage execution (core feature — every stage runs standalone)

Each stage reads the previous stage's on-disk artifact and writes its own,
so any stage can be run / debugged / swapped in isolation.

```bash
python -m aero_eyes.stages.stage1   --config configs/config.yaml --sample s001
python -m aero_eyes.stages.stage2   --config configs/config.yaml --sample s001
python -m aero_eyes.stages.stage3   --config configs/config.yaml --sample s001
python -m aero_eyes.stages.stage4   --config configs/config.yaml --sample s001
python -m aero_eyes.stages.stage5   --config configs/config.yaml --sample s001
# or the whole thing:
python -m aero_eyes.stages.run_all  --config configs/config.yaml
# evaluate:
python -m aero_eyes.evaluate --pred runs/exp001/s001/submission.json \
    --gt data/s001/gt.json --config configs/config.yaml
```

Override any config field inline:

```bash
python -m aero_eyes.stages.stage2 --config configs/config.yaml --sample s001 \
    --set stage2.proposal_model=fastsam_s --set stage2.sahi.use_sahi=false
```

## Key config switches (`configs/config.yaml`)

- `stage2.proposal_model`: `yolov11n` | `fastsam_s` (YOLOv8 not allowed)
- `stage2.sahi.use_sahi`: SAHI tiling on/off
- `stage4.tracker`: `builtin` (default, no extra weights) | `litetrack`
  (needs `litetrack.onnx_path`) | `none` (detect every frame)
- `accuracy.mode`: `baseline` | `cheap_boosters` | `max_accuracy`
  (the last adds synthetic viewpoint augmentation + CD-ViTO-style domain
  prompter; each technique individually ablatable)

## Testing offline (no real data needed)

```bash
python -m scripts.make_synthetic_fixture --out tests/fixtures
pytest -q
```

## Data layout

> ⚠️ TO BE CONFIRMED with the user before implementation — see the prompt.
> Placeholder assumption: `data/<sample>/{refs/ref_*.jpg, video.mp4, gt.json}`.

## Scope

CPU / generic GPU only. Jetson Xavier NX + TensorRT deployment is **out of
scope** (documented for reference only — see `docs/`).

## The hard part

The ground-to-aerial domain gap (close-up side-view photo → ~20×20 px
top-down appearance) is the core unsolved difficulty. `accuracy.mode`
features specifically target it; start from `baseline` and ablate upward.
