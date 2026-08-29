"""End-to-end orchestrator: runs AERO EYES pipeline in sequence.

Supported pipeline modes (configured via cfg.pipeline.detector or auto-detected):
  - "merged" (Mới): Stage 1+2 (stage12.py) -> Stage 3+4 (stage34.py) -> Stage 5 (stage5.py)
  - "legacy":       Stage 1 -> Stage 2 -> Stage 3 -> Stage 4 -> Stage 5
  - "geco2":        Stage 1+2+3 (stage123_geco2.py) -> Stage 4 -> Stage 5

Usage:
    python -m aero_eyes.stages.run_all --config configs/config.yaml
    python -m aero_eyes.stages.run_all --config configs/config.yaml --sample BlackBox_0
    python -m aero_eyes.stages.run_all --config configs/config.yaml --from-stage 3
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

log = logging.getLogger(__name__)


def _resolve_stage_pipeline(cfg) -> list[tuple[int, str, object]]:
    """Xác định danh sách các stage cần chạy dựa trên cấu hình pipeline."""
    detector_mode = getattr(cfg.pipeline, "detector", "merged") if hasattr(cfg, "pipeline") else "merged"

    if detector_mode in ("merged", "fast", "unified"):
        from aero_eyes.stages.stage12 import run_stage12
        from aero_eyes.stages.stage34 import run_stage34
        from aero_eyes.stages.stage5 import run_stage5
        return [
            (1, "Stage 1+2 (Target-Guided Proposal)", run_stage12),
            (3, "Stage 3+4 (Matching & Tracking)", run_stage34),
            (5, "Stage 5 (Spatio-Temporal Tube)", run_stage5),
        ]
    elif detector_mode == "geco2":
        from aero_eyes.stages.stage123_geco2 import run_stage123_geco2
        from aero_eyes.stages.stage4 import run_stage4
        from aero_eyes.stages.stage5 import run_stage5
        return [
            (1, "Stage 1+2+3 (GECO2 Detector)", run_stage123_geco2),
            (4, "Stage 4 (Tracking)", run_stage4),
            (5, "Stage 5 (Spatio-Temporal Tube)", run_stage5),
        ]
    else:  # Legacy 5-stage
        from aero_eyes.stages.stage1 import run_stage1
        from aero_eyes.stages.stage2 import run_stage2
        from aero_eyes.stages.stage3 import run_stage3
        from aero_eyes.stages.stage4 import run_stage4
        from aero_eyes.stages.stage5 import run_stage5
        return [
            (1, "Stage 1 (Prototype)", run_stage1),
            (2, "Stage 2 (Candidate Gen)", run_stage2),
            (3, "Stage 3 (Matching)", run_stage3),
            (4, "Stage 4 (Tracking)", run_stage4),
            (5, "Stage 5 (Spatio-Temporal Tube)", run_stage5),
        ]


def run_all(cfg, sample_id: str | None = None, from_stage: int = 1, merge: bool = True) -> None:
    """Chạy toàn bộ pipeline cho 1 sample hoặc toàn bộ thư mục samples."""
    # Xác định danh sách sample cần chạy
    if sample_id is not None:
        sample_ids = [sample_id]
    else:
        data_root = Path(cfg.data.data_root)
        if not data_root.exists():
            raise FileNotFoundError(f"data_root không tồn tại: {data_root}")
        sample_ids = [d.name for d in sorted(data_root.iterdir()) if d.is_dir()]
        if not sample_ids:
            raise ValueError(f"Không tìm thấy thư mục sample nào tại {data_root}")

    detector_mode = getattr(cfg.pipeline, "detector", "merged") if hasattr(cfg, "pipeline") else "merged"
    stage_pipeline = _resolve_stage_pipeline(cfg)

    failed: list[str] = []
    successful_samples: list[str] = []
    total_start = time.time()

    for sid in sample_ids:
        log.info("==================================================")
        log.info("🚀 [%s] BẮT ĐẦU PIPELINE (Mode=%s, From Stage=%d)", sid, detector_mode, from_stage)
        log.info("==================================================")
        sample_start = time.time()

        try:
            for stage_num, stage_name, fn in stage_pipeline:
                # Xử lý cờ --from-stage (nếu stage_num tiếp theo vượt quá from_stage quy định)
                # Ví dụ: from_stage=3 ở chế độ merged sẽ skip Stage 1+2 (stage_num=1) và chạy từ Stage 3+4 (stage_num=3)
                if (stage_num < from_stage and not (stage_num == 1 and from_stage == 2) 
                    and not (stage_num == 3 and from_stage == 4)):
                    log.debug("Bỏ qua %s (--from-stage %d)", stage_name, from_stage)
                    continue

                log.info("--- [%s] Đang chạy: %s ---", sid, stage_name)
                fn(cfg, sid)

            successful_samples.append(sid)
            log.info("✓ [%s] Hoàn thành trong %.1fs\n", sid, time.time() - sample_start)

        except Exception:
            log.exception("❌ [%s] THẤT BẠI — Chuyển tiếp sang sample tiếp theo", sid)
            failed.append(sid)
            continue

    # Tổng kết
    total_elapsed = time.time() - total_start
    log.info("==================================================")
    log.info("📊 TỔNG KẾT: Thành công %d/%d samples trong %.1fs", len(successful_samples), len(sample_ids), total_elapsed)
    if failed:
        log.warning("Các samples bị lỗi: %s", failed)
    log.info("==================================================")

    # Gom kết quả submission_all.json
    if merge and successful_samples:
        try:
            from aero_eyes.utils.io import merge_submissions
            work_dir = Path(cfg.project.work_dir)
            merged_path = work_dir / "submission_all.json"
            sub_name = getattr(cfg.data.submission, "path_name", "submission.json")
            merge_submissions(str(work_dir), successful_samples, sub_name, str(merged_path))
            log.info("✓ Đã gộp %d submissions vào %s", len(successful_samples), merged_path)
        except Exception:
            log.exception("Không thể gộp file submission_all.json")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    p = argparse.ArgumentParser(description="Chạy Pipeline AERO EYES End-to-End")
    p.add_argument("--config", required=True, help="Đường dẫn file config YAML")
    p.add_argument("--sample", default=None, help="Tên sample cụ thể (bỏ trống để chạy toàn bộ)")
    p.add_argument("--set", action="append", default=[], help="Ghi đè cấu hình: --set k=v")
    p.add_argument("--from-stage", type=int, default=1, dest="from_stage", help="Chạy tiếp từ stage N (1-5)")
    p.add_argument("--no-merge", action="store_false", dest="merge", help="Bỏ qua bước gộp file submission_all.json")
    
    args = p.parse_args()
    from aero_eyes.config import load_config
    cfg = load_config(args.config, args.set)
    run_all(cfg, sample_id=args.sample, from_stage=args.from_stage, merge=args.merge)


if __name__ == "__main__":
    main()