"""PET-DINO (Fu et al., CVPR 2026, arXiv:2604.00503) -- "Unifying Visual
Cues into Grounding DINO with Prompt-Enriched Training". UNLIKE
grounding_dino_detector.GroundingDinoDetector (loaded through
`transformers`, including its own mm_tiny/mm_base/mm_large MM-Grounding-
DINO variants), PET-DINO has NOT been merged upstream into transformers --
it lives in its own MMDetection-based repo
(https://github.com/fuweifuvtoo/PET_DINO), never vendored/pip-installable
as a single package. See Stage123PetDinoConfig's own docstring
(aero_eyes/config.py) for the full story on why this needs a SEPARATE
venv (mmcv/mmdet only ship wheels for Python<=3.11 + torch<=2.4, older
than this project's own torch pin) and for exactly which pieces of this
module are confirmed against a real checkpoint + real frames vs. assumed.

PetDinoDetector is a thin dispatcher: it builds either an
_InProcessBackend or a _SubprocessBackend (stage123_pet_dino.backend) and
forwards every public call to it. Both backends expose the same two
primitives (raw_boxes_and_scores / raw_boxes_and_scores_visual) so
detect_frame/filter_boxes below never need to know which one is active --
same pattern GroundingDinoDetector/GeCo2Detector already use for their own
config-selected variants.

CONFIRMED this session (real checkpoint, real video frames, not just
doc-reading) -- fixing two concrete bugs an earlier doc-only draft of this
wrapper had:
  - DetInferencer's __call__ needs custom_entities=True for a literal
    category-name prompt like "black box" -- without it, the prompt gets
    NLTK-tokenized as a natural-language sentence instead of a discrete
    category list.
  - The default __call__ returns None and writes results to disk instead
    of handing them back; align_trex_format=True is what returns a plain
    list of {"scores":[...], "labels":[...], "boxes":[...]} dicts (one per
    input image) directly -- NOT the {"predictions": [{"bboxes":...}]}
    shape the old draft assumed (wrong key name too: "boxes", not
    "bboxes").
"""
from __future__ import annotations

import base64
import collections
import json
import logging
import os
import subprocess
import threading
from pathlib import Path

import cv2
import numpy as np

from aero_eyes.types import Box
from aero_eyes.utils.geometry import box_iou, nms

log = logging.getLogger(__name__)


def _result_to_boxes_scores_labels(res: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Common tail for both backends: a single align_trex_format result
    dict ({"scores":[...], "labels":[...], "boxes":[...]}) -> (boxes_xyxy
    [N,4], scores [N], labels [N] int). Labels matter when the prompt sent
    to the model combined MULTIPLE categories in one call (see
    PetDinoDetector.detect_frame's negative_text_prompt path below) --
    each box's label is the index of whichever category it matched, in
    the order categories appeared in the combined prompt string."""
    boxes = np.asarray(res.get('boxes', []), dtype=np.float32).reshape(-1, 4)
    scores = np.asarray(res.get('scores', []), dtype=np.float32)
    labels = np.asarray(res.get('labels', []), dtype=np.int64)
    return boxes, scores, labels


def _result_to_boxes_scores(res: dict) -> tuple[np.ndarray, np.ndarray]:
    """Labels-dropping convenience wrapper over
    _result_to_boxes_scores_labels, for the single-category-prompt-per-
    call contract every caller except the negative_text_prompt path
    uses (mirroring GroundingDinoDetector/GeCo2Detector), where the label
    is always the same index and carries no information."""
    boxes, scores, _ = _result_to_boxes_scores_labels(res)
    return boxes, scores


class _InProcessBackend:
    """Direct, same-process mmdet.apis.DetInferencer call -- only viable
    if mmdet/mmcv/mmengine are importable in THIS SAME Python environment
    aero_eyes itself runs in. Not true as of this project's current
    requirements.txt (see Stage123PetDinoConfig's own docstring); kept for
    whenever that changes. Structurally the same call shape
    _SubprocessBackend's worker script makes, just without the pipe."""

    def __init__(self, p, device: str):
        try:
            from mmdet.apis import DetInferencer
        except ImportError:
            raise RuntimeError(
                "mmdet is not importable in this environment -- pipeline.detector == "
                "'pet_dino' with stage123_pet_dino.backend == 'inprocess' needs mmdet/"
                "mmcv/mmengine installed in THIS SAME environment aero_eyes itself runs "
                "in. That stack only ships prebuilt wheels for Python<=3.11 and "
                "torch<=2.4 -- use stage123_pet_dino.backend == 'subprocess' instead "
                "(the default), which runs PET-DINO in a separate, compatible venv. See "
                "Stage123PetDinoConfig's own docstring for the full story."
            )
        config_path = os.path.join(p.repo_path, p.config_file)
        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"PET-DINO config file not found: {config_path} -- check "
                f"stage123_pet_dino.repo_path (currently '{p.repo_path}') points at your "
                f"actual PET_DINO clone, and config_file (currently '{p.config_file}') "
                f"matches a real file under it."
            )
        if not os.path.exists(p.weights_path):
            raise FileNotFoundError(
                f"PET-DINO checkpoint not found: {p.weights_path} -- download it from "
                f"https://huggingface.co/fuweifu/PET-DINO and point stage123_pet_dino."
                f"weights_path at it."
            )
        self.inferencer = DetInferencer(model=config_path, weights=p.weights_path, device=device)

    def _detect(self, frame_bgr: np.ndarray, prompt_type: str, **call_kwargs) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # frame_bgr passed AS-IS (no channel flip): PET-DINO's own model
        # config sets data_preprocessor.bgr_to_rgb=True, i.e. it already
        # expects raw BGR (OpenCV's native order) and converts internally
        # -- confirmed this session by running real frames straight from
        # cv2.VideoCapture/cv2.imread with no pre-flip and getting correct
        # detections. An earlier doc-only draft of this wrapper pre-
        # flipped to RGB here on a generic "mmdet's own convention"
        # assumption, which would have double-flipped against this
        # specific model's own preprocessor.
        results = self.inferencer(
            inputs=[frame_bgr],
            prompt_type=prompt_type, custom_entities=True, align_trex_format=True,
            batch_size=1, **call_kwargs,
        )
        return _result_to_boxes_scores_labels(results[0])

    def raw_boxes_and_scores(self, frame_bgr: np.ndarray, text_prompt: str) -> tuple[np.ndarray, np.ndarray]:
        boxes, scores, _ = self._detect(frame_bgr, 'Text', texts=text_prompt)
        return boxes, scores

    def raw_boxes_scores_labels(self, frame_bgr: np.ndarray, text_prompt: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self._detect(frame_bgr, 'Text', texts=text_prompt)

    def raw_boxes_and_scores_visual(
        self, frame_bgr: np.ndarray, prompt_bboxes: list[list[float]] | None = None,
        prompt_bboxes_labels: list[int] | None = None,
        prompt_image: str | None = None, prompt_visual_embedding_path: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        kwargs = {}
        if prompt_visual_embedding_path is not None:
            kwargs['prompt_visual_embedding_path'] = prompt_visual_embedding_path
        else:
            kwargs['prompt_bboxes'] = prompt_bboxes
            kwargs['prompt_bboxes_labels'] = prompt_bboxes_labels
            if prompt_image is not None:
                kwargs['prompt_image'] = prompt_image
        boxes, scores, _ = self._detect(frame_bgr, 'Visual', **kwargs)
        return boxes, scores

    def extract_embedding(self, frame_bgr: np.ndarray, box_xyxy: list[float], label_id: int) -> np.ndarray | None:
        """One box's own raw visual-prompt embedding -- see
        PetDinoDynamicPrototypeTracker's own docstring for why this is
        deliberately NOT averaged/saved here (the caller owns that)."""
        import tempfile

        import torch

        with tempfile.TemporaryDirectory() as td:
            self.inferencer(
                inputs=[frame_bgr], prompt_type='Visual',
                prompt_bboxes=[box_xyxy], prompt_bboxes_labels=[label_id],
                extract_visual_embedding=True, custom_entities=True,
                out_dir=td, batch_size=1,
            )
            pt_path = os.path.join(td, f'{label_id}.pt')
            if not os.path.exists(pt_path):
                log.warning("PET-DINO in-process extract_embedding produced no output file")
                return None
            d = torch.load(pt_path, map_location='cpu')
            return d['visual_embedding'].numpy().astype(np.float32)

    def save_embedding(self, embedding: list[float], label_id: int, out_path: str) -> None:
        import torch

        torch.save(
            {'visual_embedding': torch.tensor(embedding, dtype=torch.float32),
            'label': label_id, 'class_name': f'class {label_id}'},
            out_path,
        )

    def close(self) -> None:
        pass


class _SubprocessBackend:
    """Runs PET_DINO/scripts/detector_worker.py as a persistent child
    process under p.subprocess_python (a SEPARATE venv with PET-DINO's own
    mmdet/mmcv/mmengine stack installed -- see Stage123PetDinoConfig's own
    docstring for why aero_eyes' own venv can't just import mmdet
    directly), exchanging one JSON line per detection call over its
    stdin/stdout. The model is loaded ONCE at construction (subprocess
    startup, which can take a while -- see worker_startup_timeout), not
    per frame/call.

    Only carries prompt_visual_embedding_path for Visual mode -- see this
    module's own top-of-file docstring and Stage123PetDinoConfig's, for
    why raw prompt_bboxes/prompt_image (on-the-fly extraction from a
    reference image) aren't supported here yet; nothing currently calls
    that path anyway.
    """

    def __init__(self, p, device: str):
        if not os.path.exists(p.subprocess_python):
            raise FileNotFoundError(
                f"stage123_pet_dino.subprocess_python not found: {p.subprocess_python} -- "
                f"point it at the Python interpreter of a venv with PET-DINO's own mmdet/"
                f"mmcv/mmengine stack installed (see Stage123PetDinoConfig's own docstring)."
            )
        # Absolute from here on: the child is launched with cwd=repo_path
        # (below), so any of these that were relative (e.g. the config
        # default repo_path="./PET_DINO") would otherwise be re-resolved
        # against the CHILD's cwd instead of the one this constructor
        # itself saw, silently doubling the path (".../PET_DINO/PET_DINO/
        # ..."). Confirmed necessary this session.
        repo_path = os.path.abspath(p.repo_path)
        subprocess_python = os.path.abspath(p.subprocess_python)

        worker_script = os.path.join(repo_path, "scripts", "detector_worker.py")
        if not os.path.exists(worker_script):
            raise FileNotFoundError(
                f"PET-DINO worker script not found: {worker_script} -- check "
                f"stage123_pet_dino.repo_path (currently '{p.repo_path}') points at your "
                f"actual PET_DINO clone (it should carry scripts/detector_worker.py)."
            )
        config_path = os.path.join(repo_path, p.config_file)
        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"PET-DINO config file not found: {config_path} -- check "
                f"stage123_pet_dino.repo_path (currently '{p.repo_path}') points at your "
                f"actual PET_DINO clone, and config_file (currently '{p.config_file}') "
                f"matches a real file under it."
            )
        weights_path = os.path.abspath(p.weights_path)
        if not os.path.exists(weights_path):
            raise FileNotFoundError(
                f"PET-DINO checkpoint not found: {weights_path} -- download it from "
                f"https://huggingface.co/fuweifu/PET-DINO and point stage123_pet_dino."
                f"weights_path at it."
            )

        cmd = [subprocess_python, worker_script,
              "--config", config_path, "--weights", weights_path, "--device", device]
        log.info("PET-DINO subprocess backend: starting worker (%s)", " ".join(cmd))
        self.proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1,
            # PET-DINO's own model config carries repo-relative paths
            # (lang_model_name pointing at a local BERT snapshot dir, the
            # Objects365 label-map JSON its MemoryBank reads at init) that
            # only resolve correctly with the PET_DINO clone itself as
            # the working directory -- confirmed necessary this session
            # (the worker's first real load attempt failed on exactly
            # this until cwd was set here).
            cwd=repo_path,
        )
        # Continuously drain the worker's stderr on a background thread for
        # the worker's whole lifetime -- confirmed necessary this session:
        # with NOTHING reading it during normal operation (the old
        # _drain_stderr() below only ever read reactively, after an error
        # was already detected), the worker's own library log/warning
        # noise (several lines per call) fills the OS pipe's kernel buffer
        # after enough successful calls; once full, the worker's next
        # write() to its own stderr blocks, and since that happens before
        # it can write its JSON response to stdout, the parent's
        # _read_line() then blocks forever waiting for a response that can
        # now never arrive -- a classic two-pipe deadlock. Reproduced
        # deterministically at the same call count (~10) in an isolated
        # repro regardless of GPU/VRAM state, ruling out a PET-DINO
        # inference bug or GPU contention as the cause.
        self._stderr_lines: collections.deque[str] = collections.deque(maxlen=4000)
        self._stderr_thread = threading.Thread(target=self._drain_stderr_loop, daemon=True)
        self._stderr_thread.start()
        ready = self._read_line(timeout=p.worker_startup_timeout)
        if not ready.get('ok'):
            raise RuntimeError(
                f"PET-DINO worker failed to start: {ready.get('error')}\n"
                f"worker stderr:\n{self._drain_stderr()}"
            )
        log.info("PET-DINO worker ready (pid=%d)", self.proc.pid)
        self._request_timeout = p.worker_request_timeout

    def _drain_stderr_loop(self) -> None:
        try:
            for line in self.proc.stderr:
                self._stderr_lines.append(line)
        except Exception:
            pass

    def _drain_stderr(self) -> str:
        return "".join(self._stderr_lines)

    def _read_line(self, timeout: float | None = None) -> dict:
        """Blocking readline(), or (if `timeout` is given) readline() run on
        a background thread joined with a timeout -- confirmed necessary
        this session: a plain blocking readline() here (used unconditionally
        by every earlier version of this method) left __init__'s own
        startup handshake able to hang FOREVER if the worker process stalls
        after spawning but before ever printing its "ready" line (observed
        once this session, cause unconfirmed -- disk/AV contention from a
        fresh process spawn and prior GPU-driver/VRAM churn after hours of
        repeated worker spawns in the same session are both plausible,
        neither confirmed). worker_startup_timeout (config) governs the
        startup handshake specifically; _request() (every ordinary
        detect/extract_embedding/save_embedding call) passes
        worker_request_timeout instead -- also confirmed necessary this
        session: without it, a single stalled/crashed worker response left
        the whole stage4 loop blocked indefinitely with no error and no
        visible worker stderr (the worker's own traceback, if any, sits
        unread in its stderr pipe until something calls _drain_stderr()).
        subprocess.Popen's stdout has no native cross-platform readline-
        with-timeout on Windows (no select() on pipes) -- a daemon thread
        is the portable way to bound a blocking call like this."""
        if timeout is None:
            line = self.proc.stdout.readline()
        else:
            import threading
            result: dict = {}
            def _reader():
                result['line'] = self.proc.stdout.readline()
            th = threading.Thread(target=_reader, daemon=True)
            th.start()
            th.join(timeout)
            if th.is_alive():
                self.proc.kill()
                self.proc.wait(timeout=5)
                raise RuntimeError(
                    f"PET-DINO worker produced no output within {timeout}s -- killed it. "
                    f"stderr:\n{self._drain_stderr()}"
                )
            line = result.get('line', '')
        if not line:
            raise RuntimeError(
                f"PET-DINO worker exited unexpectedly (no output). "
                f"stderr:\n{self._drain_stderr()}"
            )
        return json.loads(line)

    def _request(self, req: dict) -> dict:
        if self.proc.poll() is not None:
            raise RuntimeError(
                f"PET-DINO worker process is no longer running (exit code "
                f"{self.proc.returncode}). stderr:\n{self._drain_stderr()}"
            )
        self.proc.stdin.write(json.dumps(req) + '\n')
        self.proc.stdin.flush()
        return self._read_line(timeout=self._request_timeout)

    def _detect(self, frame_bgr: np.ndarray, prompt_type: str, **kwargs) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        ok, buf = cv2.imencode('.jpg', frame_bgr)
        if not ok:
            raise RuntimeError("failed to JPEG-encode frame for the PET-DINO worker")
        req = {
            "cmd": "detect", "prompt_type": prompt_type,
            "frame_jpg_b64": base64.b64encode(buf.tobytes()).decode('ascii'),
            "pred_score_thr": 0.0,
        }
        req.update(kwargs)
        resp = self._request(req)
        if not resp.get('ok'):
            raise RuntimeError(f"PET-DINO worker error: {resp.get('error')}")
        return _result_to_boxes_scores_labels(resp)

    def raw_boxes_and_scores(self, frame_bgr: np.ndarray, text_prompt: str) -> tuple[np.ndarray, np.ndarray]:
        boxes, scores, _ = self._detect(frame_bgr, 'Text', texts=text_prompt)
        return boxes, scores

    def raw_boxes_scores_labels(self, frame_bgr: np.ndarray, text_prompt: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self._detect(frame_bgr, 'Text', texts=text_prompt)

    def raw_boxes_and_scores_visual(
        self, frame_bgr: np.ndarray, prompt_bboxes: list[list[float]] | None = None,
        prompt_bboxes_labels: list[int] | None = None,
        prompt_image: str | None = None, prompt_visual_embedding_path: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if prompt_visual_embedding_path is None:
            raise NotImplementedError(
                "_SubprocessBackend only supports prompt_visual_embedding_path for Visual "
                "mode (a pre-extracted .pt file readable by the worker process) -- raw "
                "prompt_bboxes/prompt_image (on-the-fly extraction from a reference "
                "image) would need the worker protocol extended to carry the reference "
                "image too; not implemented since no caller uses this path yet (see "
                "Stage123PetDinoConfig's own docstring). Use backend='inprocess' if you "
                "need that and mmdet is importable in aero_eyes' own environment."
            )
        boxes, scores, _ = self._detect(frame_bgr, 'Visual', prompt_visual_embedding_path=prompt_visual_embedding_path)
        return boxes, scores

    def extract_embedding(self, frame_bgr: np.ndarray, box_xyxy: list[float], label_id: int) -> np.ndarray | None:
        ok, buf = cv2.imencode('.jpg', frame_bgr)
        if not ok:
            raise RuntimeError("failed to JPEG-encode frame for the PET-DINO worker")
        resp = self._request({
            "cmd": "extract_embedding", "label_id": label_id, "prompt_bboxes": [box_xyxy],
            "frame_jpg_b64": base64.b64encode(buf.tobytes()).decode('ascii'),
        })
        if not resp.get('ok'):
            log.warning("PET-DINO worker extract_embedding failed: %s", resp.get('error'))
            return None
        return np.asarray(resp['embedding'], dtype=np.float32)

    def save_embedding(self, embedding: list[float], label_id: int, out_path: str) -> None:
        resp = self._request({"cmd": "save_embedding", "label_id": label_id,
                              "embedding": embedding, "out_path": out_path})
        if not resp.get('ok'):
            raise RuntimeError(f"PET-DINO worker save_embedding failed: {resp.get('error')}")

    def close(self) -> None:
        if self.proc.poll() is not None:
            return
        try:
            self.proc.stdin.write(json.dumps({"cmd": "shutdown"}) + '\n')
            self.proc.stdin.flush()
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class PetDinoDetector:
    """Wraps one PET-DINO checkpoint (via _InProcessBackend or
    _SubprocessBackend, per stage123_pet_dino.backend). detect_frame()
    takes a BGR frame + a text prompt and returns Box objects in absolute
    pixel xyxy, already NMS'd and top-K'd -- the SAME contract
    aero_eyes.models.grounding_dino_detector.GroundingDinoDetector
    .detect_frame / geco2_detector.GeCo2Detector.detect_frame have."""

    def __init__(self, cfg):
        from aero_eyes.models.features import _resolve_device

        p = cfg.stage123_pet_dino
        self.box_threshold = p.box_threshold
        self.nms_iou = p.nms_iou
        self.topk_per_keyframe = p.topk_per_keyframe
        self.min_box_area_enabled = p.min_box_area_enabled
        self.min_box_area = p.min_box_area
        self.max_box_area_frac_enabled = p.max_box_area_frac_enabled
        self.max_box_area_frac = p.max_box_area_frac
        self.negative_text_prompt = p.negative_text_prompt.strip()
        self.negative_suppress_iou = p.negative_suppress_iou
        self.device = _resolve_device(cfg.device())
        if p.backend == "subprocess":
            self._backend = _SubprocessBackend(p, self.device)
        elif p.backend == "inprocess":
            self._backend = _InProcessBackend(p, self.device)
        else:
            raise ValueError(f"Unknown stage123_pet_dino.backend {p.backend!r}")
        log.info("PET-DINO (%s, backend=%s) on %s", p.config_file, p.backend, self.device)

    def raw_boxes_and_scores(self, frame_bgr: np.ndarray, text_prompt: str) -> tuple[np.ndarray, np.ndarray]:
        """One detection call with a TEXT prompt, BEFORE min_box_area/
        max_box_area_frac/NMS/top-K filtering -- mirrors
        GroundingDinoDetector.raw_boxes_and_scores's own contract. Returns
        (boxes_xyxy [N,4], scores [N])."""
        return self._backend.raw_boxes_and_scores(frame_bgr, text_prompt)

    def raw_boxes_scores_labels(self, frame_bgr: np.ndarray, text_prompt: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Like raw_boxes_and_scores, but keeps each box's label index --
        only meaningful when `text_prompt` combines MULTIPLE categories in
        one call (period-separated), as detect_frame's negative_text_prompt
        path below does."""
        return self._backend.raw_boxes_scores_labels(frame_bgr, text_prompt)

    def raw_boxes_and_scores_visual(
        self, frame_bgr: np.ndarray, prompt_bboxes: list[list[float]] | None = None,
        prompt_bboxes_labels: list[int] | None = None,
        prompt_image: str | None = None, prompt_visual_embedding_path: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """VISUAL-prompt counterpart to raw_boxes_and_scores -- exposed for
        future use (see Stage123PetDinoConfig's own docstring: nothing in
        the real per-keyframe pipeline consumes this for actual detection
        decisions yet, only stage123_pet_dino.dynamic_prototype's own
        bank-building -- see PetDinoDynamicPrototypeTracker below).
        `prompt_image`/`prompt_visual_embedding_path` are mutually
        exclusive alternate ways PET-DINO's own CLI accepts a visual
        prompt (a reference image to crop the prompt boxes from, or a
        pre-extracted embedding .pt file); the subprocess backend only
        supports the latter (see _SubprocessBackend's own docstring)."""
        return self._backend.raw_boxes_and_scores_visual(
            frame_bgr, prompt_bboxes, prompt_bboxes_labels, prompt_image, prompt_visual_embedding_path,
        )

    def extract_embedding(self, frame_bgr: np.ndarray, box_xyxy: list[float], label_id: int) -> np.ndarray | None:
        """One box's own raw AFVPG visual-prompt embedding (256-dim numpy
        vector), or None if extraction failed -- NOT averaged/saved (see
        PetDinoDynamicPrototypeTracker's own docstring for why that's the
        caller's job, not this method's)."""
        return self._backend.extract_embedding(frame_bgr, box_xyxy, label_id)

    def save_embedding(self, embedding: list[float] | np.ndarray, label_id: int, out_path: str) -> None:
        """Writes `embedding` (e.g. a bank mean) as a .pt file in the same
        shape a later raw_boxes_and_scores_visual(prompt_visual_embedding_
        path=out_path) call expects."""
        if isinstance(embedding, np.ndarray):
            embedding = embedding.tolist()
        self._backend.save_embedding(embedding, label_id, out_path)

    def detect_frame(self, frame_bgr: np.ndarray, text_prompt: str) -> list[Box]:
        if self.negative_text_prompt:
            boxes_xyxy, scores = self._detect_with_negative_suppression(frame_bgr, text_prompt)
        else:
            boxes_xyxy, scores = self.raw_boxes_and_scores(frame_bgr, text_prompt)
        h, w = frame_bgr.shape[:2]
        keep = scores >= self.box_threshold
        return self.filter_boxes(boxes_xyxy[keep], scores[keep], (h, w))

    def _detect_with_negative_suppression(
        self, frame_bgr: np.ndarray, text_prompt: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """stage123_pet_dino.negative_text_prompt: ONE combined detection
        call carrying both the positive category/ies (`text_prompt`) and
        the negative/confuser category/ies (self.negative_text_prompt),
        period-separated Grounding-DINO style -- cheaper than two separate
        calls and lets the model's own open-vocab matching directly
        compete the two descriptions against each other for the same
        region, rather than just thresholding the positive score alone
        (which doesn't help when the confuser's score is comparable to
        the real object's, as observed on LifeJacket_1's yellow road-
        marking confuser). Any positive-category box overlapping
        (IoU >= negative_suppress_iou) a CONFIDENT negative-category box
        from this SAME call is dropped before box_threshold/NMS/top-K.

        "CONFIDENT" is load-bearing: Grounding-DINO-style open-vocab
        detectors return a large fixed-size candidate list (num_queries,
        300 here) for EVERY category regardless of whether that category
        is actually present -- the overwhelming majority score near 0.
        Confirmed this session (LifeJacket_1 frame 24): the real object's
        own box, scored 0.664 for "orange object", had an "orange ground"
        box at the EXACT same coordinates (IoU=1.0) scored only 0.054 --
        a low-confidence filler candidate, not a genuine competing
        detection. Comparing IoU alone (the first version of this method)
        let that 0.054 filler suppress the real object, collapsing every
        sample's detections to zero. Requiring the negative box to ALSO
        clear box_threshold before it's allowed to suppress anything
        fixes this: a filler candidate never clears that bar, while a
        real confuser (the model genuinely confident it's looking at the
        negative category) does."""
        n_positive = max(1, len([c for c in text_prompt.split('.') if c.strip()]))
        combined_prompt = f"{text_prompt} . {self.negative_text_prompt}"
        boxes_xyxy, scores, labels = self.raw_boxes_scores_labels(frame_bgr, combined_prompt)

        pos_mask = labels < n_positive
        pos_boxes, pos_scores = boxes_xyxy[pos_mask], scores[pos_mask]
        neg_confident_mask = (~pos_mask) & (scores >= self.box_threshold)
        neg_boxes = boxes_xyxy[neg_confident_mask]
        if len(neg_boxes) == 0 or len(pos_boxes) == 0:
            return pos_boxes, pos_scores

        neg_box_objs = [
            Box(x1=float(b[0]), y1=float(b[1]), x2=float(b[2]), y2=float(b[3])) for b in neg_boxes
        ]
        keep_idx = []
        n_suppressed = 0
        for i, b in enumerate(pos_boxes):
            box_obj = Box(x1=float(b[0]), y1=float(b[1]), x2=float(b[2]), y2=float(b[3]))
            max_iou = max((box_iou(box_obj, nb) for nb in neg_box_objs), default=0.0)
            if max_iou < self.negative_suppress_iou:
                keep_idx.append(i)
            else:
                n_suppressed += 1
        if n_suppressed:
            log.info(
                "[PET-DINO negative_text_prompt] suppressed %d/%d positive box(es) "
                "overlapping a negative-category ('%s') match",
                n_suppressed, len(pos_boxes), self.negative_text_prompt,
            )
        if not keep_idx:
            return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32)
        return pos_boxes[keep_idx], pos_scores[keep_idx]

    def filter_boxes(
        self, boxes_xyxy: np.ndarray, scores: np.ndarray, frame_shape: tuple[int, int],
    ) -> list[Box]:
        """min_box_area/max_box_area_frac/NMS/top-K over already-decoded raw
        boxes -- identical logic to GroundingDinoDetector.filter_boxes,
        duplicated rather than shared (small, self-contained, and the two
        detectors' configs are otherwise fully independent)."""
        h, w = frame_shape
        boxes = [
            Box(x1=float(b[0]), y1=float(b[1]), x2=float(b[2]), y2=float(b[3]), score=float(s))
            for b, s in zip(boxes_xyxy, scores)
        ]
        if self.min_box_area_enabled:
            boxes = [b for b in boxes if b.area() >= self.min_box_area]
        if self.max_box_area_frac_enabled:
            frame_area = float(h * w)
            boxes = [b for b in boxes if b.area() <= self.max_box_area_frac * frame_area]
        if not boxes:
            return []
        keep = nms(boxes, self.nms_iou)
        boxes = [boxes[i] for i in keep][: self.topk_per_keyframe]
        return boxes

    def close(self) -> None:
        """Only meaningful for backend='subprocess' (stops the worker
        process) -- no-op for 'inprocess'. Not called automatically by
        stage123_pet_dino today (the process exits at the end of a run
        regardless); provided for callers that construct a PetDinoDetector
        with a longer lifetime than one script invocation."""
        self._backend.close()


def _calibrated_blur_ksize(scale_factor: float, cal_cfg) -> int:
    """Heavier blur the more aggressively a reference image was shrunk --
    a heuristic, NOT independently validated beyond this session's own
    downscale+blur sweep on ONE sample (BlackBox_1): blur alone did
    nothing, but blur PAIRED with substantial downscale (~1/8-1/16) raised
    Visual-route confidence ~2.5x over the raw reference. Consistent with
    this project's own stage1.ref_degradation_ensemble docstring flagging
    blur as "untested" territory generally -- treat this the same way,
    watch actual results before trusting a specific project/config's
    values. Returns an odd kernel size, or 0 (no blur)."""
    lo, hi = cal_cfg.min_scale_factor, cal_cfg.max_scale_factor
    t = 0.0 if hi <= lo else max(0.0, min(1.0, (scale_factor - lo) / (hi - lo)))
    ksize = round(cal_cfg.blur_max_ksize - t * (cal_cfg.blur_max_ksize - cal_cfg.blur_min_ksize))
    if ksize <= 1:
        return 0
    return ksize if ksize % 2 == 1 else ksize + 1


def calibrate_and_extract_pet_dino_base_embedding(
    cfg, sample_id: str, work_dir, detector: "PetDinoDetector",
    label_id: int, sample_reference_size: float | None,
) -> str | None:
    """Builds stage123_pet_dino.dynamic_prototype's STARTING embedding by
    degrading (scale + blur) this sample's own data.refs_subdir reference
    photos to look more like this specific video's own footage BEFORE
    extracting, instead of using the raw close-up studio photos as-is.

    Why this matters for PET-DINO specifically (more than it would for a
    generic cosine-similarity extractor): this session's own experiments
    found PET-DINO's Visual route notably weaker than its Text route on
    this project's aerial footage, and tracing WHY pointed at a domain
    gap -- the reference photos are crisp, close-up, subject-filling
    product shots; the actual video shows a tiny, compressed, motion-
    blurred object from altitude. Degrading the reference BEFORE
    embedding (rather than after, or not at all) shrinks that gap at the
    input the model actually sees.

    Calibration signal used -- `sample_reference_size`: the SAME value
    stage4.py's own box_refine.adaptive_context_margin already computes
    (median sqrt(area) over every keyframe detection box for this sample,
    straight from detections.json -- deliberately UNFILTERED by
    confirm_new_track or any other trust gate: even a wrong-identity
    confuser is usually still roughly object-scaled, and this only needs
    a size statistic, not correct identity). Per reference image: run
    stage1.segmentation (the same MobileSAM/FastSAM/SAM2 this project
    already uses for its OWN DINOv2 prototype-building) to find the
    object's own bbox, compute scale_factor = sample_reference_size /
    ref_object_size (clamped to calibration.min_scale_factor/
    max_scale_factor), resize the whole reference image + bbox by it, and
    apply a blur scaled to how aggressive that shrink was (see
    _calibrated_blur_ksize). Extracts each calibrated reference
    separately (detector.extract_embedding) and mean-pools into ONE
    vector -- PET-DINO's own inference contract only ever consumes a
    single vector per category (see this module's top-of-file docstring),
    unlike GeCo2's own multi-token exemplar concatenation.

    Runs ONCE per sample, at stage4 startup (sample_reference_size is
    already available then -- Stage123-PETDINO's detections.json was
    written before stage4 ever starts) -- NOT incrementally as stage4's
    own tracking progresses, unlike PetDinoDynamicPrototypeTracker's own
    per-confirmed-track offer() below.

    Returns the saved .pt path, or None if no reference image yielded a
    usable embedding (data.refs_subdir missing/empty, or every extraction
    call failed) -- callers should fall back to disabling the Visual
    route entirely in that case rather than guessing.
    """
    from aero_eyes.models.segmentation import build_segmenter
    from aero_eyes.utils.geometry import mask_bbox

    cal_cfg = cfg.stage123_pet_dino.dynamic_prototype.calibration

    data_root = Path(cfg.data.data_root)
    refs_dir = data_root / sample_id / cfg.data.refs_subdir
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    ref_paths = sorted(
        p for p in (refs_dir.iterdir() if refs_dir.is_dir() else [])
        if p.suffix.lower() in exts
    )
    if not ref_paths:
        log.warning(
            "[PET-DINO dynamic_prototype] %s: no reference images found under %s -- "
            "cannot build a starting Visual-route embedding.",
            sample_id, refs_dir,
        )
        return None

    seg_cfg = cfg.stage1.segmentation
    segmenter = build_segmenter(seg_cfg, cfg) if seg_cfg.enabled else None
    if sample_reference_size is None or sample_reference_size <= 0:
        log.warning(
            "[PET-DINO dynamic_prototype] %s: no keyframe detections to derive "
            "sample_reference_size from -- calibration will use scale_factor=1.0 "
            "(uncalibrated) for every reference image.",
            sample_id,
        )

    embeddings: list[np.ndarray] = []
    for ref_path in ref_paths:
        img = cv2.imread(str(ref_path))
        if img is None:
            log.warning("[PET-DINO dynamic_prototype] %s: could not read %s, skipping.", sample_id, ref_path)
            continue
        h, w = img.shape[:2]

        if segmenter is not None:
            mask = segmenter.segment(img)
            bbox = mask_bbox(mask)
        else:
            bbox = None
        if bbox is None:
            bbox = (0.0, 0.0, float(w), float(h))  # whole-image fallback, same convention as MobileSAMSegmenter's own passthrough

        x1, y1, x2, y2 = bbox
        ref_object_size = max(1e-6, ((x2 - x1) * (y2 - y1)) ** 0.5)
        if sample_reference_size and sample_reference_size > 0:
            scale_factor = sample_reference_size / ref_object_size
        else:
            scale_factor = 1.0
        scale_factor = max(cal_cfg.min_scale_factor, min(cal_cfg.max_scale_factor, scale_factor))

        new_w, new_h = max(1, round(w * scale_factor)), max(1, round(h * scale_factor))
        calibrated_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        ksize = _calibrated_blur_ksize(scale_factor, cal_cfg)
        if ksize > 0:
            k = min(ksize, max(1, (min(new_w, new_h) // 2) * 2 - 1))
            if k >= 3:
                calibrated_img = cv2.GaussianBlur(calibrated_img, (k, k), 0)
        calibrated_box = [x1 * scale_factor, y1 * scale_factor, x2 * scale_factor, y2 * scale_factor]

        log.info(
            "[PET-DINO dynamic_prototype] %s: calibrated %s -- scale_factor=%.3f "
            "(ref_object_size=%.0fpx -> target=%.0fpx), blur_ksize=%d",
            sample_id, ref_path.name, scale_factor, ref_object_size,
            sample_reference_size or ref_object_size, ksize,
        )

        emb = detector.extract_embedding(calibrated_img, calibrated_box, label_id)
        if emb is None:
            log.warning(
                "[PET-DINO dynamic_prototype] %s: embedding extraction failed for %s, skipping.",
                sample_id, ref_path.name,
            )
            continue
        embeddings.append(emb)

    if not embeddings:
        log.warning(
            "[PET-DINO dynamic_prototype] %s: every reference image failed extraction -- "
            "no starting Visual-route embedding built.", sample_id,
        )
        return None

    mean_embedding = np.mean(embeddings, axis=0)
    # Absolute: for backend="subprocess" this path is handed to a worker
    # process running with cwd=PET_DINO/ (see _SubprocessBackend.__init__),
    # so a relative path here would resolve against the WRONG directory
    # once it crosses the pipe -- confirmed necessary this session (this
    # crashed with "parent directory does not exist" using a relative
    # work_dir until this fix).
    out_path = str(Path(work_dir).resolve() / f"pet_dino_calibrated_base_{label_id}.pt")
    detector.save_embedding(mean_embedding, label_id, out_path)
    log.info(
        "[PET-DINO dynamic_prototype] %s: calibrated base embedding (%d/%d reference(s) used) -> %s",
        sample_id, len(embeddings), len(ref_paths), out_path,
    )
    return out_path


class PetDinoDynamicPrototypeTracker:
    """stage123_pet_dino.dynamic_prototype's online/incremental state --
    one instance per sample, built once at stage4 startup (AFTER
    calibrate_and_extract_pet_dino_base_embedding above has produced a
    starting point) and fed the tracker's own box at every CONFIRMED
    track via offer(). Mirrors GeCo2DynamicPrototypeTracker (aero_eyes.
    models.geco2_detector) closely, with one structural difference:
    PET-DINO's own extract_visual_embedding always mean-pools everything
    handed to it into ONE 256-dim vector before saving -- there is no
    per-token concatenation the way GeCo2's own exemplar tokens allow, so
    the "bank" here is a bounded list of raw vectors that gets RE-
    AVERAGED (base + every surviving dynamic token) on every change,
    rather than GeCo2's effective_prototype() concatenation.

    Gating deliberately does NOT duplicate GeCo2's own DetectionConfirmer
    here: stage4's own confirm_new_track (TrackerAgreementGate) already
    requires a track to survive one independent keyframe re-check before
    it's trusted -- offer() is only ever called AT that exact "accept"
    moment (see stage4.py's own handling of TrackerAgreementGate.judge()
    returning "accept"), so a second consecutive-hit gate here would be
    redundant. A cosine cross-check IS still applied here (does this NEW
    crop's own embedding look enough like the CURRENT effective mean?) --
    confirm_new_track only validates spatial/motion consistency (does
    this box agree with where the tracker already is), never visual
    appearance, so this is a genuinely different, non-redundant check.

    Wired into a live decision as of this session: offer()'s own return
    value (accepted, cos_sim) lets stage4.py veto a confirm_new_track
    "accept" that fails this visual check, treating it as a false
    positive rather than silently just skipping the bank update -- see
    Stage123PetDinoDynamicPrototypeConfig's own docstring for the full
    story and stage4.py's own call site for exactly how the veto unwinds
    state (wipe-if-unconfirmed + forced re-detect next keyframe).
    """

    def __init__(self, cfg, detector: "PetDinoDetector", base_embedding_path: str,
                label_id: int, work_dir, sample_id: str):
        dp_cfg = cfg.stage123_pet_dino.dynamic_prototype
        self.dp_cfg = dp_cfg
        self.detector = detector
        self.label_id = label_id
        self.sample_id = sample_id
        self.base_embedding_path = base_embedding_path
        self._base_embedding = self._load_pt_embedding(base_embedding_path)
        self._dynamic_tokens: list[np.ndarray] = []
        # Absolute -- same cwd=PET_DINO/ subprocess-worker reasoning as
        # calibrate_and_extract_pet_dino_base_embedding's own out_path.
        self._effective_path = str(Path(work_dir).resolve() / f"pet_dino_prototype_adapted_{label_id}.pt")
        self._n_offers = 0
        self._n_accepted = 0
        self._n_cross_check_rejected = 0

    @staticmethod
    def _load_pt_embedding(path: str) -> np.ndarray:
        # aero_eyes' own venv has torch independently of PET-DINO's mmcv-
        # constrained stack (already used for this project's own DINOv2
        # etc. models) -- reading a .pt file directly here is simpler
        # than round-tripping through the worker for something this cheap.
        import torch
        d = torch.load(path, map_location="cpu")
        return d["visual_embedding"].numpy().astype(np.float32)

    def effective_embedding_path(self) -> str:
        """Path to hand a FUTURE caller's prompt_visual_embedding_path
        (once something actually consumes this -- see this class's own
        docstring) -- base_embedding_path unchanged (same file, nothing
        new written) until offer() has accepted at least one token."""
        return self._effective_path if self._dynamic_tokens else self.base_embedding_path

    def _effective_mean(self) -> np.ndarray:
        vecs = [self._base_embedding] + self._dynamic_tokens
        return np.mean(vecs, axis=0)

    def offer(self, frame_bgr: np.ndarray, box, frame_idx: int | None = None) -> tuple[bool, float | None]:
        """Call exactly once per CONFIRMED track (stage4.py's
        confirm_new_track "accept" transition), with that same frame +
        the tracker's own box. Returns (accepted, cos_sim):
          - dynamic_prototype disabled -> (True, None) -- no opinion, caller
            must not veto on this.
          - embedding extraction failed -> (True, None) -- inconclusive,
            refused to ADD it to the bank, but also not grounds to veto a
            detection that's otherwise spatially/motion-confirmed.
          - cos_sim < cross_check_cosine_floor -> (False, cos_sim) -- the
            caller (stage4.py) treats this as a visual false positive.
          - else -> (True, cos_sim), and the token is added to the bank."""
        if not self.dp_cfg.enabled:
            return True, None
        self._n_offers += 1
        new_embedding = self.detector.extract_embedding(
            frame_bgr, [box.x1, box.y1, box.x2, box.y2], self.label_id,
        )
        if new_embedding is None:
            return True, None  # extraction failed -- inconclusive, not a veto

        effective = self._effective_mean()
        denom = (np.linalg.norm(new_embedding) * np.linalg.norm(effective)) + 1e-9
        cos_sim = float(new_embedding @ effective / denom)
        if cos_sim < self.dp_cfg.cross_check_cosine_floor:
            self._n_cross_check_rejected += 1
            log.info(
                "[PET-DINO dynamic_prototype] %s: frame %s: candidate REJECTED (cosine=%.3f < %.3f "
                "vs current effective embedding) -- visual veto",
                self.sample_id, frame_idx, cos_sim, self.dp_cfg.cross_check_cosine_floor,
            )
            return False, cos_sim

        self._dynamic_tokens.append(new_embedding)
        if len(self._dynamic_tokens) > self.dp_cfg.max_tokens:
            self._dynamic_tokens.pop(0)  # FIFO -- base embedding itself is never evicted
        self._n_accepted += 1
        self.detector.save_embedding(self._effective_mean(), self.label_id, self._effective_path)
        log.info(
            "[PET-DINO dynamic_prototype] %s: frame %s: accepted a token (cosine=%.3f vs previous "
            "effective) -- %d/%d dynamic token(s) active",
            self.sample_id, frame_idx, cos_sim, len(self._dynamic_tokens), self.dp_cfg.max_tokens,
        )
        return True, cos_sim

    def log_summary(self) -> None:
        log.info(
            "[PET-DINO dynamic_prototype] %s: %d offer(s), %d accepted, %d cross-check-rejected, "
            "%d dynamic token(s) in final bank",
            self.sample_id, self._n_offers, self._n_accepted, self._n_cross_check_rejected,
            len(self._dynamic_tokens),
        )
