#!/bin/bash
#SBATCH --job-name=geco2_eval
#SBATCH --output=/datastore/%u/AeroEyes-Geco2/runs/slurm-%j.out
#SBATCH --error=/datastore/%u/AeroEyes-Geco2/runs/slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=mps:a100:1
#SBATCH --partition=defq
#SBATCH --time=1-00:00:00

set -euo pipefail

module clear -f
module load slurm/slurm/24.11
module load cuda12.8/toolkit/12.8.1
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:${LD_LIBRARY_PATH:-}

export PYTHONUNBUFFERED=1   # Python mac dinh buffer stdout theo block khi khong phai TTY
                             # (bi redirect ra file .out) -> print() co the da chay nhung
                             # chua flush ra file, gay hieu nham "khong co log = dang treo".

PROJECT_DIR=/datastore/$USER/AeroEyes-Geco2                          # (*) sửa đúng path repo (Test/aero_eyes) thật trên datastore
PYTHON=/datastore/$USER/miniconda3/envs/geco2/bin/python  # (*) đúng path env tạo ở GECO2/install.sh
DATA_ROOT=/datastore/$USER/AeroEyes-Geco2/public/samples             # (*) thư mục chứa video + annotations.json can eval
GT_FILE="$DATA_ROOT/annotations (1).json"
WORK_DIR="$PROJECT_DIR/runs/geco2_eval"                     # thu muc rieng, khong dung chung voi WORK_DIR cua train_aeroeyes.sh
OUT_CHECKPOINT="$PROJECT_DIR/GECO2/CNTQG_aeroeyes_finetuned.pth"  # (*) checkpoint da finetune, dung lam stage123_geco2.weights_path

export TORCH_HOME=/datastore/$USER/.torch_cache
mkdir -p "$TORCH_HOME" "$WORK_DIR" "$PROJECT_DIR/runs"

cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"

# VRAM tối thiểu (MB) trước khi gpu_check.sh cấp GPU cho task này.
REQUIRED_VRAM=20000

srun --unbuffered ./GECO2/slurm_gpu_task.sh "$REQUIRED_VRAM" "$PYTHON" -m aero_eyes.stages.run_all \
--config configs/config.yaml \
--set pipeline.detector=geco2 \
--set stage123_geco2.weights_path="$OUT_CHECKPOINT" \
--set stage123_geco2.use_shape_token=false \
--set stage123_geco2.scale_calibration.enabled=false \
--set stage123_geco2.ref_downscale_factor=1.0 \
--set data.data_root="$DATA_ROOT" \
--set data.gt.global_file="$GT_FILE" \
--set project.work_dir="$WORK_DIR" \
--set project.use_cache=false \
--set accuracy.mode=cheap_boosters \
--set stage3.adaptive_z_score=0.35 \
--set stage3.adaptive_min_floor=0.05 \
--set stage5.fill_short_gaps=15 \
--set stage123_geco2.cosine_rescore.enabled=true \
--set stage123_geco2.cosine_rescore.candidate_score_threshold_ratio=0.0 \
--set stage3.adaptive_threshold=true \
--set stage3.dynamic_prototype.enabled=true \
--set stage123_geco2.keyframe_interval=8 \
--set stage5.temporal_smoothing.enabled=false \
--set accuracy.cheap_boosters.multi_reference_embedding=true \
--set accuracy.cheap_boosters.multi_ref_pooling=max \
--set box_refine.enabled=true \
--set box_refine.method=sam_dense \
--set stage4.verify_interval=0 \
--set stage4.geco2_redetect_cosine_filter=true \
--set runtime.save_visualizations=false \
--set box_refine.min_iou_with_original=0.5

echo "=== check_stage_prf1_progression (moi sample, tung stage) ==="
"$PYTHON" -m scripts.check_stage_prf1_progression \
    --config configs/config.yaml \
    --set data.data_root="$DATA_ROOT" \
    --set data.gt.global_file="$GT_FILE" \
    --set project.work_dir="$WORK_DIR"
