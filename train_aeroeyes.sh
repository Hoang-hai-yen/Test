#!/bin/bash
#SBATCH --job-name=Geco2_finetune
#SBATCH --output=/datastore/%u/AeroEyes-Geco2/runs/slurm-%j.out
#SBATCH --error=/datastore/%u/AeroEyes-Geco2/runs/slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=mps:a100:1
#SBATCH --partition=defq
#SBATCH --time=3-00:00:00

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
DATA_ROOT=/datastore/$USER/AeroEyes-Geco2/train/train/samples                     # (*) thư mục chứa 20 video (14 train + 6 test) + annotations
WORK_DIR="$PROJECT_DIR/runs/geco2_finetune"

TRAIN_GT_FILE="$DATA_ROOT/annotations.json"          # 14 video TRAINING
BASE_CHECKPOINT="$PROJECT_DIR/GECO2/GECO2_FSCD.pth"          # (*) GeCo2 pretrained (tai o muc 7 notebook)
OUT_CHECKPOINT="$PROJECT_DIR/GECO2/CNTQG_aeroeyes_finetuned_fretr.pth"

HOLDOUT_CATEGORIES=(Lifering Person1)  # internal train/val split -- xem GECO2_FINETUNE_PLAN.md muc 6

EPOCHS=40
STEPS_PER_EPOCH=400
BATCH_SIZE=4
LR=1e-4
WEIGHT_DECAY=1e-5
MAX_GRAD_NORM=0.1
AUX_WEIGHT=0.3
# NOTE: da thu 100 truoc do -- gay overfit nghiem trong (best chi o epoch 1, tut doc sau do). Quay
# ve gia tri goc cua train.py (25) cho ket qua on dinh hon nhieu -- KHONG tang lai gia tri nay.
AUX_SIZE_THRESHOLD_PX=10.0
P_PRESENT=0.8
REF_DOWNSCALE_LO=0.02
REF_DOWNSCALE_HI=1.0
# Augmentation do sang/tuong phan cho anh mau -- NO-OP o mac dinh (0,0)/(1,1).
# CHUA duoc kiem chung bang ST-IoU that -- chi bat khi da xac nhan lai duoc
# baseline on dinh (0.1321), tranh them bien moi khi con dang co gang quay ve
# cau hinh da biet hoat dong.
BRIGHTNESS_LO=-25.0
BRIGHTNESS_HI=25.0
CONTRAST_LO=0.8
CONTRAST_HI=1.2
# Augmentation do net/chi tiet cho ANH QUERY -- NO-OP o mac dinh (1,1), cung ly
# do voi BRIGHTNESS/CONTRAST o tren.
QUERY_DOWNSCALE_LO=0.5
QUERY_DOWNSCALE_HI=1.0
# ReduceLROnPlateau -- LR_PATIENCE=3 giam LR qua som; 6 cho ket qua tot nhat tu
# truoc den gio (val_loss=0.6710).
LR_PATIENCE=6
LR_DECAY_FACTOR=0.5
EARLY_STOP_PATIENCE=10
SEED=42

export TORCH_HOME=/datastore/$USER/.torch_cache
mkdir -p "$TORCH_HOME" "$WORK_DIR" "$PROJECT_DIR/runs"

cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"

# VRAM tối thiểu (MB) trước khi gpu_check.sh cấp GPU cho task này.
REQUIRED_VRAM=20000

srun --unbuffered ./GECO2/slurm_gpu_task.sh "$REQUIRED_VRAM" "$PYTHON" -m scripts.train_geco2_aeroeyes \
--config configs/config.yaml \
--set data.data_root="$DATA_ROOT" \
--set data.gt.global_file="$TRAIN_GT_FILE" \
--set project.work_dir="$WORK_DIR" \
--set stage123_geco2.segmentation.enabled=false \
--holdout-categories "${HOLDOUT_CATEGORIES[@]}" \
--epochs $EPOCHS --steps-per-epoch $STEPS_PER_EPOCH --batch-size $BATCH_SIZE \
--lr $LR --weight-decay $WEIGHT_DECAY --max-grad-norm $MAX_GRAD_NORM \
--aux-weight $AUX_WEIGHT --aux-size-threshold-px $AUX_SIZE_THRESHOLD_PX --p-present $P_PRESENT \
--ref-downscale-lo $REF_DOWNSCALE_LO --ref-downscale-hi $REF_DOWNSCALE_HI \
--brightness-lo $BRIGHTNESS_LO --brightness-hi $BRIGHTNESS_HI \
--contrast-lo $CONTRAST_LO --contrast-hi $CONTRAST_HI \
--query-downscale-lo $QUERY_DOWNSCALE_LO --query-downscale-hi $QUERY_DOWNSCALE_HI \
--lr-patience $LR_PATIENCE --lr-decay-factor $LR_DECAY_FACTOR \
--base-checkpoint "$BASE_CHECKPOINT" --out-checkpoint "$OUT_CHECKPOINT" \
--early-stop-patience $EARLY_STOP_PATIENCE --seed $SEED