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
