# Bài nộp Lab Day 2 — Đào Gia Bảo

- MSSV: **2A202602793**.
- Họ tên: **Đào Gia Bảo**.
- Dữ liệu: DeepWeeds, fold 0 nguyên bản.
- Mã triển khai: `code/`, sao chép từ `starter/` của repo.
- Chọn cấu hình bằng macro-F1 validation; chỉ đánh giá test sau khi chốt cấu hình.

Tài liệu chạy lại và kết quả thực nghiệm sẽ được cập nhật theo từng giai đoạn đã kiểm tra.

## Môi trường và dữ liệu

Môi trường kiểm tra: Python 3.12.7, PyTorch 2.6.0, torchvision 0.21.0,
`timm` 1.0.22; các phiên bản còn lại nằm trong `code/requirements.txt`.
Từ thư mục gốc repo:

```bash
python -m venv .venv
.venv/bin/python -m pip install -r submissions/2A202602793_dao_gia_bao/code/requirements.txt
.venv/bin/python submissions/2A202602793_dao_gia_bao/code/download_data.py
.venv/bin/python submissions/2A202602793_dao_gia_bao/code/check_data.py
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m unittest discover -s submissions/2A202602793_dao_gia_bao/code/tests -v
```

Ảnh được giải nén vào `data/images`, CSV vào `data/labels`. Trình tải kiểm tra
MD5 `b7b30f96d466fba86016aa5a26606e0f` trước khi giải nén. Tất cả dữ liệu nằm
ngoài phần commit. Nếu tải bị ngắt, chạy lại với cùng số `--workers` để tiếp tục.

CSV split chính thức chỉ có `Filename,Label`. Loader giữ nguyên nội dung và thứ tự
CSV, lấy tên/thứ tự lớp từ `labels.csv`. Có một sai khác nhãn nguồn tại
`20170714-110407-3.jpg`: train split ghi 0, `labels.csv` ghi 1. Bài nộp giữ nhãn 0
của train split và ghi sai khác trong bằng chứng; không sửa hay loại ảnh.

## Backbone và quy ước model

Danh sách sàng lọc trong `code/model.py`: ResNet-18 (`a1_in1k`), ConvNeXt-Atto
(`d2_in1k`), DeiT-Tiny (`fb_in1k`), EfficientNet-B0 (`ra_in1k`),
MobileNetV3-Small (`lamb_in1k`). Danh sách đủ ResNet, ConvNeXt, transformer và
mạng nhẹ; ưu tiên mô hình nhỏ phù hợp Apple M4, RAM 16 GiB. Mọi model thay head
thành 9 lớp. DeiT cho phép nội suy vị trí khi đổi độ phân giải.

```bash
.venv/bin/python submissions/2A202602793_dao_gia_bao/code/check_models.py
.venv/bin/python submissions/2A202602793_dao_gia_bao/code/check_models.py --pretrained
```

Nhóm optimizer tách backbone/head và tách decay/no-decay ở cả hai phần:
LR head gấp 10 lần backbone; bias, normalization và embedding được model khai
báo miễn decay có weight decay bằng 0. Khi đóng băng, gọi `model.train()` vẫn
giữ backbone/BatchNorm ở eval và chỉ bật head. GMAC đo bằng
`torch.utils.flop_counter.FlopCounterMode` rồi quy đổi FLOP/2, chỉ tính toán tử
được công cụ hỗ trợ; đây là số đo tính toán, không đại diện trực tiếp cho độ trễ.

## Engine huấn luyện

`train.run(Config(...))` là lối vào duy nhất cho B/T/F. Mặc định giữ công thức
starter (12 epoch, batch 64, AdamW, LR backbone/head 1e-4/1e-3, WD 0,05).
Các lần chạy trên MPS dùng FP32; AMP/GradScaler chỉ bật thực sự trên CUDA và
trạng thái thực tế được ghi vào `config.json`. Warmup + cosine cập nhật theo
iteration. EMA đánh giá bằng bản trọng số EMA, trung bình buffer số thực và
sao chép buffer số nguyên. Val loss luôn là CE không trọng số để dễ đối chiếu;
train loss là loss mục tiêu của thí nghiệm. Với Mixup/CutMix, không báo accuracy train.

Mỗi lần chạy lưu `config.json`, `history.csv`, `summary.json`, `best.pt`,
`latest.pt`, đầu ra val và CSV dự đoán. `latest.pt` giữ optimizer, scheduler,
scaler, EMA, RNG và trạng thái generator của DataLoader để tiếp tục đúng cấu hình.
Chạy lại cùng exp_id/seed và cấu hình sẽ tiếp tục hoặc trả kết quả đã hoàn tất;
đổi cấu hình phải dùng exp_id mới. Checkpoint chọn theo macro-F1 val, hòa lấy
epoch sớm hơn. Thời gian train và val được đo riêng; không gồm ghi biểu đồ/checkpoint.

```bash
.venv/bin/python submissions/2A202602793_dao_gia_bao/code/check_training.py
.venv/bin/python submissions/2A202602793_dao_gia_bao/code/train.py --set \
  exp_id=EXAMPLE backbone=mobilenetv3_small_100.lamb_in1k seed=0 \
  epochs=10 img_size=128 batch_size=32 amp=false \
  out_dir=submissions/2A202602793_dao_gia_bao/logs \
  pred_dir=submissions/2A202602793_dao_gia_bao/predictions \
  curves_dir=submissions/2A202602793_dao_gia_bao/curves
```

Lệnh `EXAMPLE` là ví dụ cú pháp, không phải kết quả đã đo. Thí nghiệm smoke chỉ
kiểm tra kỹ thuật trên tập nhỏ, không dùng để chọn cấu hình. Bằng chứng giai đoạn 4
bao gồm nạp lại checkpoint, backbone đóng băng không đổi, và tiếp tục CPU cho
trọng số khớp chính xác với lần chạy không gián đoạn. Test mặc định bị niêm phong;
phải có cấu hình chốt trước khi xuất dự đoán test.

## EDA và kiểm tra trước khi train

```bash
.venv/bin/python submissions/2A202602793_dao_gia_bao/code/validate_pipeline.py
```

Bằng chứng ở `evidence/stage5.json`, các bảng đếm lớp, ảnh phân bố, 27 ảnh train
mẫu, ảnh augmentation/Mixup/CutMix và lịch sử quá khớp một batch. Việc xem mẫu,
kiểm tra kích thước/kênh chỉ dùng ảnh train; test chỉ được kiểm tra cấu trúc split
và tồn tại file. Toàn bộ 10.501 ảnh train là RGB 256×256. `Negative` chiếm 52,01%
tổng split, gấp khoảng 9,02 lần lớp nhỏ nhất.

Kiểm tra ban đầu phát hiện head MobileNet của timm dùng khởi tạo theo fan-out,
đưa CE lên 7,60 với 9 lớp. CPU và MPS khớp gần nhau, và đầu vào RGB/normalization
đúng. Factory nay dùng **cùng normal(0, 0,01), bias=0 cho mọi head mới**.
Bằng chứng thất bại ban đầu được giữ ở `evidence/stage5_initial_failure.log`;
forward và nhóm tham số của cả năm model đã được kiểm tra lại sau sửa.

Đo thử batch tính toán trên M4 cho thấy 128×128 giảm chi phí so với 224×224.
Công thức sàng lọc sẽ dùng **10 epoch, batch 32, 128×128, FP32**, giữ mọi yếu tố
này giống nhau cho năm backbone. Đây là giảm độ phân giải theo GUIDE mục 7 để
hoàn thành toàn bộ thiết kế trên phần cứng hiện có. Bảng `compute_probe.csv` là
ước lượng tính toán từ batch giả lập, không phải thời gian epoch hay độ trễ
suy luận cuối; các thời gian thật sẽ lấy từ log thí nghiệm.

Để chạy một lần tái lập tách biệt khỏi sản phẩm gốc, đặt `LAB_OUTPUT_DIR` tới
thư mục mới (notebook dùng `reproduction/`); `LAB_DATA_DIR` có thể chỉ tới dữ liệu
đã tải. Các biến này chỉ điều khiển đường dẫn.
