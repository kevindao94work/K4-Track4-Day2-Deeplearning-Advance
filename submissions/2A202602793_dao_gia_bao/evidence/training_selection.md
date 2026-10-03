# Ablation công thức huấn luyện

| ID | Trục | Thay đổi | Macro-F1 val | Δ so mốc | F1 Chinee | F1 Snake |
|---|---|---|---:|---:|---:|---:|
| T00_screen | Mốc | Không đổi | 0.9271 | +0.0000 | 0.8547 | 0.8675 |
| T01_frozen | A | Đóng băng backbone | 0.5954 | -0.3317 | 0.5297 | 0.4800 |
| T02_scratch | A | Khởi tạo ngẫu nhiên | 0.2106 | -0.7165 | 0.2278 | 0.0360 |
| T03_color | B | Thêm ColorJitter | 0.9236 | -0.0035 | 0.8786 | 0.8601 |
| T04_cutmix | B | Thêm CutMix alpha=1 | 0.9198 | -0.0073 | 0.8318 | 0.8657 |
| T05_ls | C | Label smoothing epsilon=0,1 | 0.9312 | +0.0042 | 0.8671 | 0.8718 |
| T06_focal | C | Focal gamma=2 | 0.9191 | -0.0080 | 0.8384 | 0.8468 |
| T07_cutmix_ls | B+C | CutMix alpha=1 + label smoothing 0,1 | 0.9218 | -0.0053 | 0.8246 | 0.8486 |

Chọn T05_ls dựa trên toàn bộ validation; test còn niêm phong.
Mỗi so sánh đơn giữ các yếu tố khác như mốc; T07 được khai báo là kết hợp hai trục.
Sàng lọc một seed chưa đo nhiễu; các delta nhỏ không đủ để khẳng định ưu thế tổng quát.
Tối ưu truy cập Dataset từ iloc sang cột cache giữ chính xác tensor/nhãn/augmentation (loader_equivalence.json). Tất cả ablation dùng cùng phiên bản này; thời gian đọc dữ liệu không so trực tiếp với giai đoạn backbone.
