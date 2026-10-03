# Kết quả sàng lọc backbone

| exp_id | Macro-F1 val | Top-1 val | Train/epoch (s) | p50 sơ bộ (ms) |
|---|---:|---:|---:|---:|
| B01_resnet18 | 0.7461 | 0.8112 | 49.94 | 7.06 |
| B02_convnext_atto | 0.9271 | 0.9460 | 48.45 | 9.99 |
| B03_deit_tiny | 0.9102 | 0.9346 | 64.46 | 12.17 |
| B04_efficientnet_b0 | 0.8846 | 0.9129 | 91.00 | 17.71 |
| B05_mobilenetv3_small | 0.8870 | 0.9160 | 47.83 | 11.77 |

Chọn **B02_convnext_atto**, macro-F1 val **0.9271** để tiếp tục ablation.

Chọn macro-F1 val cao nhất; khi hòa chọn latency thấp hơn. Chỉ một seed, thứ hạng còn sơ bộ.

Cùng 10 epoch, batch 32, 128×128, AdamW, CE, crop + flip, warmup + cosine và head normal(0,0,01).
Mọi điểm chọn mô hình lấy từ toàn bộ val. Các tag tiền huấn luyện khác nhau được ghi rõ; thứ hạng này không cô lập riêng ảnh hưởng kiến trúc khỏi công thức tiền huấn luyện.
Test chưa được suy luận. Các thời gian epoch lấy từ history.csv; không gồm tải trọng số hay ghi biểu đồ/checkpoint.
