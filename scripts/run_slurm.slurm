#!/bin/bash
# ============================================================================
# sbatch script để chạy pipeline aero_eyes trên cụm Slurm (slurm.uit.edu.vn).
# Dựa đúng theo template /datastore/templates/job_template.slurm trong
# "HuongDanSuDungSlurm.pdf". Các chỗ cần TỰ SỬA được đánh dấu "<<< SỬA".
#
# Cách dùng:
#   1. SSH vào slurm.uit.edu.vn (trong mạng trường hoặc qua VPN).
#   2. Copy toàn bộ repo aero_eyes_starter + dataset lên /datastore/<username>/.
#   3. Sửa các dòng có "<<< SỬA" bên dưới.
#   4. Nộp job:  sbatch scripts/run_slurm.slurm
#   5. Theo dõi: squeue -u <username>      (xem trạng thái, PD = đang chờ GPU)
#   6. Hủy job:  scancel <jobid>
#   7. Log chạy ra ở logs/aero_eyes_<jobid>.out / .err
# ============================================================================

#SBATCH --job-name=aero_eyes                 # <<< SỬA nếu muốn tên khác
#SBATCH --output=logs/aero_eyes_%j.out
#SBATCH --error=logs/aero_eyes_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4                    # DINOv2/CLIP + YOLO decode video cần vài core
#SBATCH --gres=mps:1                         # dùng GPU bất kỳ; đổi thành mps:l40:1 hoặc mps:a100:1 nếu cần chỉ định card
#SBATCH --mem=16G                            # video + feature buffer; tăng nếu OOM
#SBATCH --time=04:00:00                      # <<< SỬA theo độ dài video / số sample (tối đa 72h)

# REQUIRED_VRAM: quan trọng — VRAM job cần, dùng để gpu_check.sh tìm đúng GPU.
# DINOv2 ViT-B/14 + YOLOv11n chạy CPU/GPU nhẹ (~2-4GB) là đủ cho baseline.
REQUIRED_VRAM=4096                           # <<< SỬA nếu dùng ensemble (DINOv2+CLIP) hoặc batch_size lớn -> tăng lên ~8192

# ----------------------------------------------------------------------------
# CHUẨN BỊ MÔI TRƯỜNG
# ----------------------------------------------------------------------------
module clear -f
module load shared python312             # <<< SỬA version python nếu venv của bạn khác

# Kích hoạt venv đã cài requirements.txt của project (torch, ultralytics, sahi, ...)
source /datastore/<username>/venv/bin/activate      # <<< SỬA đường dẫn venv

# Thư mục project (nơi chứa configs/config.yaml, aero_eyes/, data/)
cd /datastore/<username>/aero_eyes_starter           # <<< SỬA đường dẫn project

mkdir -p logs

# Xóa biến môi trường Slurm để tự chọn GPU qua gpu_check.sh
unset CUDA_VISIBLE_DEVICES

# ----------------------------------------------------------------------------
# TÌM GPU CÒN ĐỦ VRAM (helper có sẵn trên cluster)
# ----------------------------------------------------------------------------
CHECK_OUT=$(/usr/local/bin/gpu_check.sh $REQUIRED_VRAM $SLURM_JOB_ID)
EXIT_CODE=$?
if [ $EXIT_CODE -eq 10 ]; then
    echo "$CHECK_OUT"
    exit 0   # Không tìm được slot -> thoát để Slurm tự requeue job
elif [ $EXIT_CODE -eq 11 ]; then
    echo "$CHECK_OUT"
    exit 1   # Đã requeue 5 lần vẫn không có GPU -> lỗi thật, dừng job
fi
BEST_GPU=$CHECK_OUT
echo "✅ Job $SLURM_JOB_ID bắt đầu trên GPU: $BEST_GPU"

# ----------------------------------------------------------------------------
# KHỞI TẠO PRIVATE MPS SERVER (mỗi job 1 pipe riêng để không đụng job khác)
# ----------------------------------------------------------------------------
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-job$SLURM_JOB_ID
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log-job$SLURM_JOB_ID
rm -rf $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
export CUDA_VISIBLE_DEVICES=$BEST_GPU

# ----------------------------------------------------------------------------
# CHẠY PIPELINE
# ----------------------------------------------------------------------------
# Mặc định: chạy toàn bộ Stage 1->5 cho MỌI sample trong data.data_root.
# Muốn chạy 1 sample cụ thể, thêm --sample <video_id>.
# Muốn chạy riêng Stage B (đang phụ trách) từ Stage 1: --from-stage 1 (tự dùng cache Stage2 nếu đã chạy trước đó với use_cache: true).
python -m aero_eyes.stages.run_all \
    --config configs/config.yaml \
    --set runtime.device=cuda

echo "Job $SLURM_JOB_ID done."
