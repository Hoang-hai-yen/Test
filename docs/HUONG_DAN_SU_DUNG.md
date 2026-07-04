# Hướng Dẫn Sử Dụng AERO EYES

## Mục Lục
1. [Cài đặt môi trường](#1-cài-đặt-môi-trường)
2. [Cấu trúc thư mục dự án](#2-cấu-trúc-thư-mục-dự-án)
3. [Chuẩn bị dataset](#3-chuẩn-bị-dataset)
4. [Định dạng annotation (ground truth)](#4-định-dạng-annotation-ground-truth)
5. [Chạy pipeline](#5-chạy-pipeline)
6. [Các tùy chọn cấu hình](#6-các-tùy-chọn-cấu-hình)
7. [Đánh giá kết quả (ST-IoU)](#7-đánh-giá-kết-quả-st-iou)
8. [Test với dữ liệu giả (không cần dataset thật)](#8-test-với-dữ-liệu-giả-không-cần-dataset-thật)
9. [Xử lý lỗi thường gặp](#9-xử-lý-lỗi-thường-gặp)

---

## 1. Cài Đặt Môi Trường

### Yêu cầu hệ thống
- Python 3.10 trở lên
- GPU NVIDIA (khuyến nghị 8 GB VRAM trở lên) — chạy CPU cũng được nhưng chậm hơn
- Windows / Linux / macOS

### Bước 1 — Tạo virtual environment
```powershell
# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\activate

# Linux / macOS
python -m venv .venv
source .venv/bin/activate
```

### Bước 2 — Cài đặt dependencies

**Cài PyTorch trước** (chọn đúng CUDA version của máy):
```powershell
# GPU — CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# GPU — CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# CPU only
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

**Cài các thư viện còn lại:**
```powershell
pip install numpy opencv-python pyyaml pydantic>=2 ultralytics sahi pillow tqdm pytest
```

### Bước 3 — Cài dự án ở chế độ editable (để import được)
```powershell
pip install -e .
```

---

## 2. Cấu Trúc Thư Mục Dự Án

```
aero_eyes_starter/
│
├── configs/
│   └── config.yaml              # File cấu hình chính (sửa ở đây)
│
├── aero_eyes/
│   ├── stages/
│   │   ├── stage1.py            # Xử lý ảnh tham chiếu → prototype.npz
│   │   ├── stage2.py            # Sinh candidate boxes → candidates.json
│   │   ├── stage3.py            # Matching cosine similarity → detections.json
│   │   ├── stage4.py            # Tracking giữa keyframes → tracks.json
│   │   ├── stage5.py            # Xuất kết quả → tube.json + submission.json
│   │   └── run_all.py           # Chạy toàn bộ pipeline 1 lệnh
│   ├── models/                  # DINOv2, YOLOv11n, MobileSAM, trackers
│   ├── utils/                   # geometry, video, io, viz
│   └── evaluate.py              # Tính ST-IoU
│
├── data/                        # ← ĐẶT DATASET VÀO ĐÂY
│   ├── Backpack_0/
│   │   ├── refs/
│   │   │   ├── ref_0.jpg
│   │   │   ├── ref_1.jpg
│   │   │   └── ref_2.jpg
│   │   └── video.mp4
│   ├── Person_1/
│   │   ├── refs/
│   │   └── video.mp4
│   └── ...
│
├── annotations (1).json         # ← FILE GROUND TRUTH (đặt ở root)
│
├── runs/                        # Kết quả pipeline tự động tạo
│   └── exp001/
│       └── Backpack_0/
│           ├── prototype.npz    # Output Stage 1
│           ├── candidates.json  # Output Stage 2
│           ├── detections.json  # Output Stage 3
│           ├── tracks.json      # Output Stage 4
│           ├── tube.json        # Output Stage 5
│           └── submission.json  # File nộp leaderboard
│
├── scripts/
│   └── make_synthetic_fixture.py
│
└── tests/
    ├── fixtures/                # Dữ liệu test giả (tự sinh)
    ├── test_st_iou.py
    └── test_stages.py
```

---

## 3. Chuẩn Bị Dataset

### Cấu trúc thư mục dataset

Mỗi **sample** (1 video + 1 đối tượng cần tìm) cần:

```
data/
└── <video_id>/              # Tên phải khớp với video_id trong annotations
    ├── refs/
    │   ├── ref_0.jpg        # Ảnh tham chiếu 1 (ảnh cận cảnh đối tượng)
    │   ├── ref_1.jpg        # Ảnh tham chiếu 2 (góc khác)
    │   └── ref_2.jpg        # Ảnh tham chiếu 3 (góc khác nữa)
    └── video.mp4            # Video drone quét khu vực từ trên xuống
```

> **Lưu ý:** `video_id` trong tên thư mục phải **khớp chính xác** với trường `video_id` trong file `annotations (1).json`.  
> Ví dụ: thư mục `data/Backpack_0/` → `video_id` trong JSON là `"Backpack_0"`.

### Ảnh tham chiếu (refs/)
- **Số lượng:** đúng 3 ảnh
- **Tên file:** `ref_0.jpg`, `ref_1.jpg`, `ref_2.jpg` (hoặc `.png`)
- **Nội dung:** ảnh cận cảnh đối tượng cần tìm, có thể từ nhiều góc độ khác nhau
- **Kích thước:** bất kỳ (pipeline tự resize về 224×224)

### Video drone
- **Định dạng:** `.mp4` (hoặc bất kỳ codec cv2 đọc được)
- **Nội dung:** video quay từ drone nhìn xuống khu vực tìm kiếm
- **Ghi chú:** đối tượng thường chỉ xuất hiện trong một phần của video (~20×20 px từ trên xuống)

---

## 4. Định Dạng Annotation (Ground Truth)

### File annotation

File `annotations (1).json` đặt **ở thư mục gốc** của dự án (cùng cấp với `configs/`).

### Schema JSON

```json
[
  {
    "video_id": "Backpack_0",
    "annotations": [
      {
        "bboxes": [
          { "frame": 3483, "x1": 321, "y1": 0,  "x2": 381, "y2": 12  },
          { "frame": 3484, "x1": 302, "y1": 0,  "x2": 387, "y2": 21  },
          { "frame": 3485, "x1": 314, "y1": 0,  "x2": 401, "y2": 40  }
        ]
      }
    ]
  },
  {
    "video_id": "Person_1",
    "annotations": [
      {
        "bboxes": [
          { "frame": 120, "x1": 200, "y1": 150, "x2": 220, "y2": 170 },
          { "frame": 121, "x1": 202, "y1": 151, "x2": 222, "y2": 171 }
        ]
      }
    ]
  }
]
```

### Giải thích các trường

| Trường | Kiểu | Ý nghĩa |
|--------|------|---------|
| `video_id` | string | Tên định danh video, **phải khớp** với tên thư mục trong `data/` |
| `annotations` | array | Danh sách annotation (1 phần tử = 1 đối tượng; mỗi video có 1 đối tượng) |
| `bboxes` | array | Danh sách bounding box theo từng frame |
| `frame` | int | Số thứ tự frame, **bắt đầu từ 0** |
| `x1`, `y1` | int | Tọa độ góc trên-trái của bounding box (pixel, không normalize) |
| `x2`, `y2` | int | Tọa độ góc dưới-phải của bounding box (pixel, không normalize) |

### Quy tắc quan trọng

- **Frame vắng mặt (absent):** nếu đối tượng **không xuất hiện** trong frame đó → **không thêm entry** vào `bboxes` (bỏ qua frame đó)
- **Hệ tọa độ:** góc trên-trái của frame là `(0, 0)`; tăng dần sang phải và xuống dưới
- **Chỉ số frame:** bắt đầu từ **0** (frame đầu tiên = frame 0)
- **1 đối tượng / video:** mỗi video chỉ track 1 đối tượng duy nhất

### Ví dụ minh họa tọa độ

```
(0,0) ──────────────────────→ x
  │   ┌─────────────────────┐
  │   │                     │
  │   │   (x1,y1)           │
  │   │     ┌───────┐       │   ← bounding box
  │   │     │  obj  │       │     x1=100, y1=80
  │   │     └───────┘       │     x2=140, y2=110
  │   │           (x2,y2)   │
  │   └─────────────────────┘
  ↓ y
```

---

## 5. Chạy Pipeline

### Cách 1 — Chạy toàn bộ pipeline (khuyến nghị)

```powershell
# Chạy tất cả sample trong data/
.venv\Scripts\python -m aero_eyes.stages.run_all --config configs/config.yaml

# Chạy 1 sample cụ thể
.venv\Scripts\python -m aero_eyes.stages.run_all --config configs/config.yaml --sample Backpack_0
```

### Cách 2 — Chạy từng stage riêng lẻ

Mỗi stage đọc artifact từ stage trước (trên disk) và ghi artifact của mình. Có thể chạy độc lập, debug từng bước:

```powershell
# Stage 1: Xử lý 3 ảnh tham chiếu → prototype.npz
.venv\Scripts\python -m aero_eyes.stages.stage1 --config configs/config.yaml --sample Backpack_0

# Stage 2: Sinh candidate boxes từ video → candidates.json
.venv\Scripts\python -m aero_eyes.stages.stage2 --config configs/config.yaml --sample Backpack_0

# Stage 3: Matching → detections.json
.venv\Scripts\python -m aero_eyes.stages.stage3 --config configs/config.yaml --sample Backpack_0

# Stage 4: Tracking → tracks.json
.venv\Scripts\python -m aero_eyes.stages.stage4 --config configs/config.yaml --sample Backpack_0

# Stage 5: Xuất kết quả → tube.json + submission.json
.venv\Scripts\python -m aero_eyes.stages.stage5 --config configs/config.yaml --sample Backpack_0
```

### Cách 3 — Tiếp tục từ stage bị lỗi

Nếu pipeline chạy được đến stage 3 rồi bị lỗi, không cần chạy lại từ đầu:

```powershell
.venv\Scripts\python -m aero_eyes.stages.run_all \
    --config configs/config.yaml \
    --sample Backpack_0 \
    --from-stage 3
```

### Override cấu hình không cần sửa file YAML

```powershell
# Dùng FastSAM-s thay YOLOv11n
.venv\Scripts\python -m aero_eyes.stages.run_all \
    --config configs/config.yaml \
    --sample Backpack_0 \
    --set stage2.proposal_model=fastsam_s

# Tắt SAHI tiling (nhanh hơn nhưng recall thấp hơn)
    --set stage2.sahi.use_sahi=false

# Bật accuracy boosters
    --set accuracy.mode=cheap_boosters

# Thay đổi ngưỡng matching
    --set stage3.match_threshold=0.45
```

---

## 6. Các Tùy Chọn Cấu Hình

File cấu hình chính: `configs/config.yaml`

### Các switch quan trọng nhất

```yaml
# Thư mục chứa dataset
data:
  data_root: ./data            # ← đổi đường dẫn dataset ở đây

# Model đề xuất candidate (chọn 1 trong 2, không dùng YOLOv8)
stage2:
  proposal_model: yolov11n     # "yolov11n" | "fastsam_s"
  keyframe_interval: 8         # Xử lý 1 frame mỗi N frame
  sahi:
    use_sahi: true             # Bật/tắt SAHI tiling

# Tracker giữa các keyframe
stage4:
  tracker: builtin             # "builtin" | "litetrack" | "none"
  builtin:
    algorithm: csrt            # "csrt" | "kcf" | "mosse"

# Độ chính xác
accuracy:
  mode: baseline               # "baseline" | "cheap_boosters" | "max_accuracy"
```

### Bảng so sánh mode accuracy

| Mode | Tốc độ | Độ chính xác | Khi nào dùng |
|------|--------|--------------|--------------|
| `baseline` | Nhanh nhất | Thấp nhất | Debug, test nhanh |
| `cheap_boosters` | Trung bình | Cao hơn | Multi-scale scan + multi-ref embedding |
| `max_accuracy` | Chậm nhất | Cao nhất | Nộp leaderboard chính thức |

### Bảng so sánh tracker

| Tracker | Yêu cầu | Tốc độ | Ghi chú |
|---------|---------|--------|---------|
| `builtin` | Chỉ OpenCV | Nhanh | Mặc định, luôn hoạt động |
| `litetrack` | File `.onnx` | Trung bình | Cần `stage4.litetrack.onnx_path` |
| `none` | Không | Chậm nhất | Detect lại mỗi frame |

---

## 7. Đánh Giá Kết Quả (ST-IoU)

### Chỉ số ST-IoU là gì?

**Spatio-Temporal IoU** = đánh giá đồng thời:
- **Khi nào** đối tượng xuất hiện (temporal overlap)
- **Ở đâu** đối tượng xuất hiện (spatial bounding box IoU)

Công thức:
```
ST-IoU(video) = mean(IoU mỗi frame) over temporal union
              = Tổng IoU tất cả frame / Số frame trong union
```

Trong đó:
- Frame có trong cả prediction và GT → IoU của 2 bounding box
- Frame chỉ có trong 1 bên (prediction hoặc GT) → IoU = 0

**Điểm leaderboard = mean ST-IoU trên tất cả video đánh giá.**

### Chạy đánh giá

```powershell
# Đánh giá 1 video
.venv\Scripts\python -m aero_eyes.evaluate \
    --pred runs/exp001/Backpack_0/submission.json \
    --gt "annotations (1).json" \
    --config configs/config.yaml

# Output ví dụ:
# Backpack_0    ST-IoU = 0.4231
# Mean ST-IoU: 0.4231
```

### Định dạng file submission (nộp leaderboard)

File `submission.json` do Stage 5 tự động tạo, định dạng giống hệt file annotation:

```json
[
  {
    "video_id": "Backpack_0",
    "annotations": [
      {
        "bboxes": [
          { "frame": 3483, "x1": 318, "y1": 2,  "x2": 379, "y2": 14 },
          { "frame": 3484, "x1": 299, "y1": 1,  "x2": 385, "y2": 22 }
        ]
      }
    ]
  }
]
```

---

## 8. Test Với Dữ Liệu Giả (Không Cần Dataset Thật)

Nếu chưa có dataset thật, có thể test toàn bộ pipeline với dữ liệu giả tổng hợp:

### Tạo synthetic fixture
```powershell
.venv\Scripts\python -m scripts.make_synthetic_fixture --out tests/fixtures
```

Tạo ra:
```
tests/fixtures/synth001/
├── refs/
│   ├── ref_0.jpg   (224×224, hình chữ nhật màu trên nền nhiễu)
│   ├── ref_1.jpg
│   └── ref_2.jpg
├── video.mp4       (30 frame 640×480, đối tượng xuất hiện frame 5-25)
└── gt.json         (annotation theo đúng schema)
```

### Chạy pipeline trên synthetic fixture
```powershell
# Sửa config để trỏ vào fixture
.venv\Scripts\python -m aero_eyes.stages.run_all \
    --config configs/config.yaml \
    --sample synth001 \
    --set data.data_root=tests/fixtures
```

### Chạy unit tests
```powershell
# ST-IoU tests (không cần numpy DLL)
.venv\Scripts\python -m pytest tests/test_st_iou.py -v

# Tất cả tests
.venv\Scripts\python -m pytest tests/ -v
```

---

## 9. Xử Lý Lỗi Thường Gặp

### Lỗi: `FileNotFoundError: prototype.npz not found`
**Nguyên nhân:** Chưa chạy Stage 1.  
**Cách fix:** Chạy `run_all` từ đầu hoặc `stage1` riêng lẻ.

### Lỗi: `LiteTrack ONNX weights not found`
**Nguyên nhân:** Đang dùng `tracker: litetrack` nhưng chưa cung cấp file `.onnx`.  
**Cách fix:** Đổi sang `tracker: builtin`, hoặc set đường dẫn:
```powershell
--set stage4.tracker=builtin
# hoặc
--set stage4.litetrack.onnx_path=path/to/litetrack.onnx
```

### Lỗi: MobileSAM không tải được
**Hành vi:** Tự động fallback sang chế độ passthrough (dùng ảnh gốc không cắt background).  
**Không cần fix gì** — pipeline vẫn chạy được, chỉ kém chính xác hơn một chút.

### Lỗi: `No video matching '*.mp4' found`
**Nguyên nhân:** Video không đúng tên hoặc đường dẫn.  
**Cách fix:** Kiểm tra thư mục `data/<video_id>/` có file `.mp4` không. Nếu dùng `.avi` hoặc định dạng khác, sửa trong config:
```yaml
data:
  video_glob: "*.avi"
```

### Lỗi: `video_id 'X' not found in annotations`
**Nguyên nhân:** Tên thư mục trong `data/` không khớp với `video_id` trong file annotation.  
**Cách fix:** Đảm bảo `data/Backpack_0/` và `"video_id": "Backpack_0"` trong JSON phải giống nhau chính xác (phân biệt hoa thường).

### Lỗi: `DLL load failed` (Windows)
**Nguyên nhân:** Windows Smart App Control chặn DLL của numpy/cv2.  
**Cách fix:**  
1. Vào *Windows Security → App & Browser Control → Smart App Control → Off*
2. Restart máy
3. Chạy lại

### Pipeline chạy nhưng submission.json trống (0 detections)
**Nguyên nhân thường gặp:** `match_threshold` quá cao.  
**Cách thử:**
```powershell
--set stage3.match_threshold=0.30   # Hạ ngưỡng xuống
--set stage2.keyframe_interval=4    # Tăng mật độ keyframe
--set accuracy.mode=cheap_boosters  # Dùng multi-scale scan
```
