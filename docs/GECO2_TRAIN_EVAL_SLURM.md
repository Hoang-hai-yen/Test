# Hướng dẫn Train & Evaluate GECO2 (gốc, FSC147) trên SLURM

Hướng dẫn này áp dụng cho **training gốc của GECO2** (`GECO2/train.py`, train từ đầu/tiếp
tục trên FSC147) — không phải script finetune AERO EYES
(`scripts/train_geco2_aeroeyes.py`, xem [GECO2_FINETUNE_PLAN.md](GECO2_FINETUNE_PLAN.md) cho
việc đó).

Toàn bộ pipeline gồm 3 bước tuần tự:
`train.py` (tạo checkpoint) → `inference.py` (sinh dự đoán `geco2_val.json`/`geco2_test.json`
+ in MAE/RMSE) → `eval_bboxes.py` (tính AP/AP50 dùng Detectron2).

---

## 0. Checklist trước khi bắt đầu

- [ ] Đã có quyền truy cập 1 cluster SLURM có GPU (biết tên `--partition`, số GPU/node khả dụng).
- [ ] Đã tải bộ dữ liệu **FSC147** (+ annotation mở rộng FSCD147 cho AP/AP50).
- [ ] Có quyền ghi vào 1 thư mục lưu checkpoint (`model_path`).

---

## 1. Chuẩn bị dữ liệu FSC147

`utils/data.py::FSC147DATASET` ([GECO2/utils/data.py:151-161](../GECO2/utils/data.py#L151-L161))
yêu cầu `--data_path` trỏ tới một thư mục có đúng cấu trúc sau:

```
<data_path>/
├── annotations/
│   ├── Train_Test_Val_FSC_147.json      # danh sách file ảnh theo split train/val/test
│   ├── annotation_FSC147_384.json       # box_examples_coordinates (3 exemplar/ảnh) + điểm đếm
│   ├── instances_train.json             # COCO-format GT boxes (dùng để train + AP eval)
│   ├── instances_val.json
│   └── instances_test.json
├── images_384_VarV2/                    # ảnh gốc FSC147
│   └── *.jpg
└── gt_density_map_adaptive_512_512_object_VarV2/   # density map GT (.npy, dùng tính MAE/RMSE)
    └── *.npy
```

Đây là bộ dữ liệu FSC147 chuẩn (từ paper gốc "Learning To Count Everything") cộng với
annotation dạng bbox theo chuẩn COCO (FSCD147, dùng để evaluate AP/AP50 bằng Detectron2).
Repo này **không đóng gói dữ liệu** — bạn phải tự tải và đặt đúng cấu trúc trên, ở một
đường dẫn có thể truy cập từ node GPU (ví dụ thư mục project trên hệ thống lưu trữ dùng chung
của cluster).

---

## 2. Cài môi trường (chạy 1 lần trên login node hoặc interactive GPU node)

```bash
cd GECO2
bash install.sh
```

`install.sh` sẽ:
1. Tạo conda env `test_geco2` (Python 3.10), cài torch 2.7.1/cu126.
2. **Biên dịch CUDA extension MultiScaleDeformableAttention** từ `Deformable-DETR/models/ops`
   — bước này **phải chạy trên node có GPU cùng kiến trúc với node sẽ train**, nếu không sẽ
   lỗi khi load ops lúc training. Copy kết quả build vào `GECO2/models/ops` (thư mục này
   không có sẵn trong repo — không thể bỏ qua bước này).
3. Cài các thư viện phụ (hydra-core, scikit-image, pycocotools, einops, numpy<2, gradio...).

> Nếu môi trường SLURM của bạn có `module load CUDA/...` riêng, đảm bảo bản CUDA toolkit
> dùng để build ops khớp (hoặc tương thích ABI) với bản mà `module load` nạp lúc training
> chạy (`train.sh` gốc dùng `CUDA/12.3.0` qua `module load`).

### Cài Detectron2 (chỉ cần cho bước evaluate AP/AP50, không cần để train)

`eval_bboxes.py` import trực tiếp `detectron2` — cài theo hướng dẫn chính thức
(https://detectron2.readthedocs.io) khớp với bản torch/CUDA đã cài ở bước trên. Không có
sẵn trong `install.sh`/`req.txt`.

---

## 3. Sửa các đường dẫn hard-code trong code (bắt buộc)

`GECO2/models/counter.py` dòng ~38 có:

```python
# TODO REMOVE!!
torch.hub.set_dir('/d/hpc/projects/FRI/pelhanj/CNT_SAM2/models/')
```

Đây là đường dẫn cache `torch.hub` **của tác giả gốc trên cluster của họ**. Model dùng
`torch.hub` để tải checkpoint backbone SAM2 Hiera-Base+ (`sam2_hiera_base_plus.pt`, tải từ
`dl.fbaipublicfiles.com`, xem [counter.py:60-71](../GECO2/models/counter.py#L60-L71)).
Trên cluster khác, đường dẫn này có thể không tồn tại/không có quyền ghi.

**Việc cần làm**: sửa dòng đó thành một thư mục cache hợp lệ trên cluster của bạn (hoặc xóa
dòng để dùng cache mặc định `~/.cache/torch/hub`), ví dụ:

```python
torch.hub.set_dir(os.environ.get("TORCH_HOME", os.path.expanduser("~/.cache/torch")))
```

Node chạy training job **cần có Internet ra ngoài** để tải checkpoint này lần đầu (hoặc bạn
tự tải trước `sam2_hiera_base_plus.pt` và đặt đúng vào thư mục cache torch hub theo cấu trúc
`checkpoints/<hash>/sam2_hiera_base_plus.pt` để tránh phụ thuộc mạng lúc chạy job).

---

## 4. Sửa `train.sh` cho khớp cluster của bạn

File gốc [GECO2/train.sh](../GECO2/train.sh) chứa toàn bộ cấu hình riêng của cluster HPC gốc
(Slovenia). Cần sửa:

```bash
#SBATCH --job-name=CNTQG
#SBATCH --output=results/GECO2_%j.txt   # (*) tạo thư mục results/ trước — bị .gitignore, không có sẵn
#SBATCH --error=results/GECO2_%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2             # số process = số GPU dùng (DDP 1 process/GPU)
#SBATCH --cpus-per-task=12
#SBATCH --partition=gpu                 # (*) đổi theo tên partition thật trên cluster bạn
#SBATCH --gres=gpu:2                    # (*) số GPU thật sự có/được cấp
#SBATCH --time=4-00:00:00               # (*) đổi theo giới hạn thời gian job cho phép
#SBATCH --exclude=gwn[01-10]            # (*) tên node loại trừ riêng của cluster gốc — xoá dòng này
```

```bash
module load Anaconda3        # (*) đổi tên module đúng với `module avail` trên cluster bạn
module load CUDA/12.3.0
source activate cnt2          # (*) đổi thành env đã tạo ở bước 2 (vd: test_geco2)
conda activate base
conda activate cnt2
```

Và phần lệnh chạy — **bắt buộc sửa `--data_path`/`--model_path`**:

```bash
srun --unbuffered python train.py \
--training \
--model_name GECO2_FSCD \
--model_path /path/to/your/checkpoint/dir \    # (*) thư mục có quyền ghi, tự tạo trước (mkdir -p)
--data_path /path/to/your/fsc147 \             # (*) đúng thư mục đã chuẩn bị ở bước 1
--backbone resnet50 \     # LƯU Ý: flag này thực chất KHÔNG được code đọc — backbone luôn
                          # là SAM2 Hiera-Base+ hard-code trong counter.py, giữ nguyên flag
                          # này không ảnh hưởng gì, nhưng đừng hiểu nhầm là đổi được backbone qua đây
--reduction 16 \
--image_size 1024 \
--emb_dim 256 \
--num_heads 8 \
--kernel_dim 1 \
--num_objects 3 \
--epochs 200 \
--lr 1e-4 \
--backbone_lr 0 \
--lr_drop 50 \
--weight_decay 1e-5 \
--batch_size 4 \
--dropout 0.1 \
--num_workers 8 \
--max_grad_norm 0.1 \
--aux_weight 0.3 \
--tiling_p 0.5 \
--pre_norm
```

`--ntasks-per-node`/`--gres=gpu:N` phải khớp: `train.py` đọc `WORLD_SIZE`/`RANK`/`LOCAL_RANK`
từ biến môi trường do `srun`/SLURM cấp (hoặc `SLURM_PROCID` nếu chạy qua `srun` trực tiếp dưới
SLURM — code đã tự nhận diện `SLURM_PROCID` trong `train.py`, xem
[GECO2/train.py:30-38](../GECO2/train.py#L30-L38)). Không cần thêm launcher DDP riêng, `srun`
với `--ntasks-per-node=N` đã tương đương N process, mỗi process 1 GPU.

Tạo thư mục output trước khi submit:

```bash
mkdir -p results
mkdir -p /path/to/your/checkpoint/dir
```

---

## 5. Submit job training

```bash
cd GECO2
sbatch train.sh
```

Theo dõi:

```bash
squeue -u $USER
tail -f results/GECO2_<jobid>.txt
```

Log mỗi epoch in ra dạng:

```
Epoch: <n> Train loss: ... Val loss: ... Train MAE: ... Val MAE: ... Val RMSE: ... Test MAE: ... Test RMSE: ... Epoch time: ... best
```

Checkpoint tốt nhất (theo `val_rmse` thấp nhất) tự động lưu vào
`<model_path>/<model_name>.pth` (chỉ rank 0 ghi, xem
[GECO2/train.py:317-334](../GECO2/train.py#L317-L334)) — **không** lưu checkpoint mỗi epoch,
chỉ ghi đè khi có cải thiện.

> Không có early stopping hay resume checkpoint tự động theo lịch — `--resume_training` phải
> tự thêm vào lệnh `srun` và cần checkpoint `<model_name_resume_from>.pth` đã tồn tại trong
> `model_path` nếu muốn train tiếp từ checkpoint cũ.

---

## 6. Sinh dự đoán để evaluate (`inference.py`)

Sau khi `train.py` chạy xong (hoặc bất cứ lúc nào có checkpoint muốn đánh giá), chạy trên 1
node/job có GPU (không cần multi-GPU, `inference.py` chỉ dùng 1 GPU đơn — `torch.cuda.set_device(0)`):

```bash
python inference.py \
    --model_name GECO2_FSCD \
    --model_path /path/to/your/checkpoint/dir \
    --data_path /path/to/your/fsc147 \
    --backbone resnet50 --reduction 16 --image_size 1024 \
    --emb_dim 256 --num_heads 8 --kernel_dim 1 --num_objects 3 \
    --batch_size 1 --num_workers 8
```

`--model_name`/`--model_path` phải khớp checkpoint đã lưu ở bước train (nạp từ
`<model_path>/<model_name>.pth`, xem [GECO2/inference.py:34](../GECO2/inference.py#L34)).

Script sẽ:
- Chạy qua cả 2 split `val` và `test`, in ra **MAE/RMSE đếm số lượng** cho từng split.
- Ghi 2 file **`geco2_val.json`** và **`geco2_test.json`** (dự đoán dạng COCO) vào **thư mục
  làm việc hiện tại** ([GECO2/inference.py:189-190](../GECO2/inference.py#L189-L190)) — chạy
  lệnh này từ đúng thư mục bạn muốn 2 file này xuất hiện (thường là `GECO2/`).

---

## 7. Evaluate AP/AP50 (`eval_bboxes.py`, cần Detectron2)

Chạy **cùng thư mục làm việc** đã sinh ra `geco2_val.json`/`geco2_test.json` ở bước 6 (script
đọc 2 file này theo đường dẫn tương đối, xem
[GECO2/eval_bboxes.py:527,543](../GECO2/eval_bboxes.py#L527)):

```bash
python eval_bboxes.py \
    --model_name GECO2_FSCD \
    --model_path /path/to/your/checkpoint/dir \
    --data_path /path/to/your/fsc147
```

(Chỉ `--data_path` thực sự được dùng bên trong script — các flag khác chỉ cần có mặt vì
`eval_bboxes.py` dùng chung `utils.arg_parser.get_argparser()` với `train.py`/`inference.py`,
không dùng riêng flag `--input_folder` như docstring nội bộ của file gợi ý — đó là hàm
`get_args_parser` cục bộ không được gọi tới, có thể bỏ qua.)

Output: MAE/RMSE/NAE/SRE + bảng AP/AP50/AP75/APs/APm/APl cho cả `val` và `test`, in ra
console, đồng thời lưu chi tiết từng ảnh vào `each_img_infor<split>.pkl` trong thư mục hiện tại.

> Script visualize (`vis_res/`) mặc định **tắt** trong `main` (`visualize_res=False`) — bật
> lên nếu muốn xem ảnh có box vẽ đè lên nếu cần debug trực quan.

---

## Tóm tắt trình tự lệnh

```bash
# 1. Cài môi trường (1 lần)
cd GECO2 && bash install.sh
# + cài Detectron2 riêng theo hướng dẫn chính thức

# 2. Sửa torch.hub.set_dir(...) trong models/counter.py
# 3. Sửa train.sh: partition, gres, data_path, model_path, module load, env name
mkdir -p results /path/to/checkpoint/dir

# 4. Train
sbatch train.sh
squeue -u $USER

# 5. Inference (sinh geco2_val.json / geco2_test.json)
python inference.py --model_name GECO2_FSCD --model_path /path/to/checkpoint/dir \
    --data_path /path/to/fsc147 --backbone resnet50 --reduction 16 --image_size 1024 \
    --emb_dim 256 --num_heads 8 --kernel_dim 1 --num_objects 3 --batch_size 1 --num_workers 8

# 6. Evaluate AP/AP50
python eval_bboxes.py --model_name GECO2_FSCD --model_path /path/to/checkpoint/dir \
    --data_path /path/to/fsc147
```

## Vấn đề thường gặp

| Triệu chứng | Nguyên nhân | Cách xử lý |
|---|---|---|
| `ImportError`/lỗi load `models/ops` khi train | Chưa build Deformable-DETR CUDA ops trên đúng node GPU | Chạy lại bước build trong `install.sh` trên node GPU sẽ dùng để train |
| `AttributeError: module 'numpy' has no attribute 'long'` | `scipy` bị nâng version tự động, không tương thích numpy<2 (đã gặp ở finetune script, cùng nguyên nhân `matcher.py` dùng `scipy.optimize.linear_sum_assignment`) | Pin lại `numpy<2` và `scipy<1.13` sau khi cài các gói khác |
| Job treo ở bước tải checkpoint SAM2 | Node compute không có Internet ra ngoài | Tải trước `sam2_hiera_base_plus.pt` và đặt vào thư mục cache torch hub, hoặc chạy trên node có Internet |
| `FileNotFoundError: geco2_val.json` khi chạy `eval_bboxes.py` | Chạy `eval_bboxes.py` khác thư mục với lúc chạy `inference.py` | Chạy cả 2 lệnh từ cùng một thư mục làm việc |
| `torch.hub.set_dir(...)` lỗi quyền ghi | Đường dẫn hard-code của tác giả gốc không tồn tại trên cluster bạn | Sửa theo bước 3 |
