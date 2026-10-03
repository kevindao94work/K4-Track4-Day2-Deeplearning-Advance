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

## So sánh backbone

```bash
.venv/bin/python submissions/2A202602793_dao_gia_bao/code/workflow.py backbones
.venv/bin/python submissions/2A202602793_dao_gia_bao/code/workflow.py audit-backbones
```

Lệnh đầu chạy lần lượt B01–B05 bằng cùng `train.run`; lần chạy đã hoàn thành
được nạp từ log, lần gián đoạn tiếp tục checkpoint. Mỗi ID giữ cấu hình bất biến.
Trong máy đã có đủ trọng số cache, có thể đặt `HF_HUB_OFFLINE=1` để tránh kiểm tra
metadata qua mạng. Máy mới cần mạng để tải đúng tag tiền huấn luyện.

`tables/Backbones.csv` được sinh trực tiếp từ summary. `training_seconds_per_epoch`
chỉ tính huấn luyện, còn `seconds_per_epoch_train_val` gồm huấn luyện và validation;
cả hai chưa tính ghi biểu đồ/checkpoint hoặc tải trọng số. Độ trễ trong bảng này
chỉ là sàng lọc **10 warmup/5 lần đo**, batch 1, FP32, không tiền xử lý. Benchmark
suy luận chính thức sẽ dùng ít nhất 50 lần đo riêng.

`configs/backbone_choice.json` và `evidence/backbone_selection.md` lưu lựa chọn
theo macro-F1 validation; `evidence/stage6.json` đối chiếu bảng, log, CSV và biểu đồ.
Đây là sàng lọc một seed. Các công thức tiền huấn luyện upstream khác nhau, nên
không diễn giải thứ hạng này như tác động thuần túy của kiến trúc.

## Ablation công thức huấn luyện

```bash
.venv/bin/python submissions/2A202602793_dao_gia_bao/code/check_loader_equivalence.py
.venv/bin/python submissions/2A202602793_dao_gia_bao/code/workflow.py training
```

Mốc `T00_screen` và bảy thay đổi dùng cùng backbone chọn trên val, seed 0 và
công thức sàng lọc. `configs/training_plan.json` ghi trước các override: A gồm
frozen/scratch so với finetune; B gồm ColorJitter/CutMix so với basic; C gồm
label smoothing/focal so với CE. T07 kết hợp CutMix và label smoothing. Delta
lấy so với T00_screen; F1 Chinee apple và Snake weed được tính bằng evaluator.

Dataset truy cập cột cache thay cho pandas iloc nhằm tránh sao chép metadata
cho từng ảnh. Ảnh/nhãn/thứ tự/augmentation giữ nguyên, có kiểm tra tensor khớp
chính xác ở `evidence/loader_equivalence.json`. Mọi lượt ablation dùng cùng bản
tối ưu này. Không so thời gian I/O giai đoạn backbone và ablation như tác động
của hyperparameter. `tables/Training.csv` và `evidence/training_selection.md`
được sinh từ log thật; ưu thế nhỏ ở một seed cần kiểm tra lại nhiễu seed.

## So sánh suy luận và benchmark

```bash
.venv/bin/python submissions/2A202602793_dao_gia_bao/code/workflow.py inference
```

I00 là một view trên checkpoint công thức chọn; I01/I02 là hflip gộp prob/logit;
I03 là năm crop; I04/I05 là độ phân giải 160/224; I06 là ba scale. I07 khớp
nhiệt độ cho I00 và I08 khớp cho phương pháp có macro-F1 val cao nhất. Cùng tập
val nguyên bản, không huấn luyện lại. Chọn macro-F1, khi hòa dùng NLL rồi p95.
Nhiệt độ tối ưu NLL val, giữ argmax; ECE 15 bin theo `eval.py`.

`tables/Inference.csv`, `tables/Latency.csv` và JSON trong `logs/inference/` lưu
p50/p95/p99, 10 warmup/50 mẫu đồng bộ, FP32, batch 1 và throughput batch 32.
Các mẫu thời gian thô cũng được giữ. Bao gồm biến đổi view/forward/gộp/softmax
trên GPU; **không tính đọc ảnh/resize/normalize CPU**. Đây chưa phải độ trễ toàn
bộ hệ thống camera/robot. ConvNeXt dùng LayerNorm nên không có BN để gộp;
utility BN fusion được kiểm tra riêng, không gán một kết quả tăng tốc giả.

## Chốt cấu hình trước test

```bash
.venv/bin/python submissions/2A202602793_dao_gia_bao/code/freeze_final.py
```

`configs/frozen_final.json` cố định F01 và mốc T00, các seed 0/1/2, đường dẫn,
hash CSV/evaluator/pipeline và phép suy luận. F01 dùng công thức T05_ls, infer
một view 160 + TS khớp NLL val riêng mỗi seed. Mốc dùng CE, một view 128, T=1.
Dry run đầy đủ val và cấu hình sáu lượt đã qua kiểm tra (`evidence/stage9.json`).
Mốc chốt được commit/push trước huấn luyện chung kết và trước mọi suy luận test.
Khi đã có file chốt, lệnh không tự ghi đè lựa chọn theo kết quả mới.

## Huấn luyện chung kết và validation

```bash
.venv/bin/python submissions/2A202602793_dao_gia_bao/code/workflow.py final-training
```

Chạy mới F01/T00 cho từng seed 0/1/2 bằng cùng engine, mỗi lượt đủ 10 epoch.
Đường cong và checkpoint được chọn bằng val ở độ phân giải train 128. Dự đoán
val đó giữ trong `<ID>_train_seed<k>_val.csv`; sau đó áp dụng policy inference
cố định và lưu `<ID>_seed<k>_val.csv`, cùng bản `<ID>_uncal_seed<k>_val.csv`.
Nhiệt độ/metric/hash ở `logs/<ID>/seed<k>/final_val.json`; không chọn lại
checkpoint hoặc hyperparameter theo các điểm chung kết. Bảng
`tables/FinalValidation.csv` và `evidence/stage10.json` dùng sample std ddof=1.
Benchmark được đo lại trên đúng checkpoint seed 0 chung kết, 50 mẫu batch 1/32.
Giai đoạn này chưa suy luận test.

### Khôi phục cache trọng số chốt

Nếu cache HF toàn cục mất, dùng cache riêng (dữ liệu/trọng số không commit):

```bash
export HF_HUB_CACHE="$PWD/data/hf_cache"
.venv/bin/python submissions/2A202602793_dao_gia_bao/code/restore_frozen_weights.py
export HF_HUB_OFFLINE=1
.venv/bin/python submissions/2A202602793_dao_gia_bao/code/workflow.py final-training
```

Utility tải đúng snapshot trong file chốt, kiểm tra SHA-256 trước khi factory
đọc. Lỗi cache trước T00 seed0 và bằng chứng hash được giữ trong `evidence/`.
Các lượt F01 đã hoàn tất được nạp lại từ log/CSV, không thay công thức hoặc
trọng số nguồn. Máy mới chạy toàn bộ quy trình cần bỏ offline cho giai đoạn tải.
