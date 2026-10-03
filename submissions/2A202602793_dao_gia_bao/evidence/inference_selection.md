# So sánh suy luận trên validation

| ID | Phương pháp | K | Macro-F1 | ECE | p95 (ms) | Ảnh/s batch32 |
|---|---|---:|---:|---:|---:|---:|
| I00 | Một view | 1 | 0.9312 | 0.0869 | 6.91 | 1792.2 |
| I01 | Lật ngang, trung bình xác suất | 2 | 0.9285 | 0.0877 | 12.13 | 918.7 |
| I02 | Lật ngang, trung bình logit | 2 | 0.9288 | 0.0876 | 12.20 | 920.4 |
| I03 | Năm crop 128 từ ảnh 160 | 5 | 0.9380 | 0.0992 | 31.05 | 377.2 |
| I04 | Một view 160 | 1 | 0.9434 | 0.0876 | 7.15 | 1168.1 |
| I05 | Một view 224 | 1 | 0.9283 | 0.0885 | 7.85 | 618.4 |
| I06 | Ba scale 96/128/160 | 3 | 0.9112 | 0.1045 | 24.07 | 560.8 |
| I07 | Một view + TS khớp NLL val | 1 | 0.9312 | 0.0100 | 6.88 | 1801.0 |
| I08 | Một view 160 + TS khớp NLL val | 1 | 0.9434 | 0.0133 | 7.14 | 1164.9 |

Chọn I08. Macro-F1 val cao nhất; khi hòa ưu tiên NLL val rồi p95 thấp. TS fit riêng val mỗi seed trước test.

Mọi phương pháp dùng cùng checkpoint và 3.501 ảnh val đúng thứ tự. Five-crop lấy bốn góc và tâm 128×128 trên ảnh tiền xử lý 160×160. Multi-scale nội suy bilinear không antialias tensor đã chuẩn hóa tới 96/128/160.
Độ trễ có 10 warmup, 50 mẫu, đồng bộ MPS/CUDA trước và sau mỗi mẫu; gồm view/forward/gộp/softmax, chưa gồm đọc ảnh/resize/normalize CPU. Batch 1 và throughput batch 32 được đo độc lập.
Temperature scaling giữ nguyên argmax, tối ưu NLL trên val, không bảo đảm ECE luôn giảm trên dữ liệu mới. Các nhiệt độ và ECE/NLL trước-sau ở stage8.json.
convnext_atto.d2_in1k: không có BN để gộp; utility đã kiểm tra trên Conv/BN CPU.
Test chưa được suy luận. Không chạy ensemble, EMA, FP16 hay AMP như thí nghiệm chất lượng; các utility/engine tương ứng không được coi là kết quả đo.
