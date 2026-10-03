# Cấu hình chốt trước test

F01: ConvNeXt-Atto d2_in1k, finetune, 10 epoch, batch 32, train 128×128, crop+flip, label smoothing 0,1, AdamW LR backbone/head 1e-4/1e-3, WD 0,05, warmup 1 epoch + cosine, FP32, không EMA.

Suy luận: resize/center-crop 160, bicubic, crop_pct 0,875, một view; mỗi seed khớp một nhiệt độ trên NLL validation rồi áp dụng nguyên trạng sang test.

Mốc T00: cùng backbone và lịch train, CE, infer một view 128, không hiệu chỉnh. Cả hai dùng các seed 0,1,2, đủ 10 epoch và chọn checkpoint bằng macro-F1 val ở độ phân giải train. Sau đó policy inference cố định; không chọn lại checkpoint bằng val160/test.

Các đường dẫn và hash ở configs/frozen_final.json. Dry run chỉ trên val; test chưa được suy luận. Cấu hình này được commit/push trước khi train chung kết và trước khi mở kết quả test.
