# Claude Code Instructions — AERO EYES

You are handed a **project skeleton** (this repo). Implement it into a
working system. Read this whole file first.

## Mission

Few-shot spatio-temporal object localization for search-and-rescue.
**Input:** 3 close-up reference images of ONE target object + a drone video
scanning an area from above. **Output:** predicted bounding boxes of that
target in every frame where it appears. **Metric:** 3D Spatio-Temporal IoU
(ST-IoU) — jointly scores *when* (temporal overlap) and *where* (spatial
IoU) as one continuous space–time volume. A detection earns credit only if
BOTH timing and box location align with ground truth. Leaderboard score =
mean ST-IoU over all evaluation videos.

Cross-domain few-shot: references are close-up/multi-angle ground-level
photos; the object appears tiny (often ~20×20 px) and top-down in the
drone footage. No class label — only the 3 images define the target.

## ⚠️ STEP 0 — ASK BEFORE WRITING ANY CODE

Ask the user these and WAIT for answers. Do not guess.

1. **Exact data layout.** Confirm/correct the placeholder in
   `configs/config.yaml` (`data:` block). Where do the 3 reference images
   live per sample? Video format/codec/resolution/FPS? One video per sample
   or many? Where is ground truth and **what is its exact schema**:
   box format (`xyxy`/`xywh`/`cxcywh`, normalized?), frame index base
   (0 or 1), how "object absent" frames are encoded, one object per video?
2. **Submission format.** Exact leaderboard format: file type, columns/keys,
   box convention, how absence is represented.
3. **Hardware.** RAM/VRAM, CUDA GPU present? (affects DINOv2 variant +
   batch sizes). Confirm Jetson/TensorRT is out of scope (it is).

Update `config.yaml` `data:` + `runtime:` to the confirmed reality before
implementing `utils/io.py`.

## Architecture — implement exactly as the skeleton stubs describe

Stages 1→5 as in `aero_eyes/stages/*.py` and the README table. The SVG the
user supplied is the canonical pipeline diagram; the stub docstrings encode
it. Do not deviate from the stage boundaries.

## Hard requirements (non-negotiable)

1. **YOLOv8 is forbidden.** Stage 2 proposal model is `yolov11n` OR
   `fastsam_s`, selected by `stage2.proposal_model`. Exactly one active.
2. **SAHI is a toggle** (`stage2.sahi.use_sahi`). Off → run proposals on
   the full resized frame.
3. **Tracker is a toggle** (`stage4.tracker`): `builtin` (OpenCV, default,
   zero extra weights), `litetrack` (ONNX; REQUIRES `litetrack.onnx_path`,
   else raise a clear actionable error — never an opaque crash), `none`
   (detect every frame).
4. **Every stage runs standalone**, consuming the prior stage's on-disk
   artifact and producing inspectable artifacts (JSON/NPZ + visual
   overlays). `run_all` is a thin wrapper only — no logic the stages lack.
5. **`accuracy.mode`** = `baseline` | `cheap_boosters` | `max_accuracy`.
   Each technique (multi-scale, multi-ref embedding, synthetic viewpoint
   augmentation, CD-ViTO-style domain prompter) is an individually
   ablatable sub-toggle, not a monolith.
6. **ST-IoU implemented exactly** per `aero_eyes/evaluate.py` docstring,
   with the unit tests in `tests/test_st_iou.py` passing.
7. **Offline testability**: `scripts/make_synthetic_fixture.py` produces a
   deterministic tiny sample (object absent for some frames); `pytest`
   passes with NO real data and NO network (mock/skip weight downloads).
8. **CPU/GPU portable**, auto-detect, graceful CPU fallback. No Jetson,
   no TensorRT. Pin all dependency versions in `requirements.txt`.
9. **Graceful failure** with actionable messages whenever optional weights
   (MobileSAM, LiteTrack, model checkpoints) are missing.

## Internal conventions

- All boxes internally: absolute **xyxy** float, pixel space, **0-based**
  frame indices. Convert to/from the user's GT & submission schema ONLY in
  `utils/io.py`.
- Every artifact JSON carries a `schema_version`.
- Per-stage timing + log level from `runtime.log_level`.

## Deliverable order

1. Do STEP 0 (ask the 3 question groups). Wait.
2. Fill `config.py` (Pydantic models mirroring `config.yaml` + validators)
   and `utils/io.py` to the confirmed formats. Show the user, get sign-off.
3. Implement Stage 1→5 + metric. After EACH stage: it must run standalone
   on the synthetic fixture and its tests pass before moving on.
4. End-to-end smoke test on the fixture, then a dry run on the user's real
   data layout. Report mean ST-IoU.

Ask clarifying questions any time data format or metric semantics are
ambiguous rather than guessing. Keep the baseline clean and modular first;
accuracy boosters are opt-in layers on top.
