# Kiểm toán cuối theo RUBRIC

| Mục | Kết quả | Bằng chứng |
|---|---|---|
| Giữ nguyên nguồn lớp học | Đạt | git diff fe9791a: eval.py, starter/, tests/, README/ GUIDE/ RUBRIC không đổi |
| Chốt trước test và giữ nguyên toán học | Đạt | 1629d9e; fingerprint train/model/dataset/loss/inference; ngoại lệ cờ runtime 6c07c9f trước test |
| Fold0, EDA và sanity | Đạt | stage5.json; lớp/ảnh mẫu/augmentation/overfit; giữ annotation conflict nguồn |
| 5 backbone, 3 trục, kết hợp và 9 cấu hình suy luận | Đạt | Backbones/Training/Inference CSV khớp evaluator; các gate6/7/8; sàng lọc một seed |
| 19 lần train thật và 19 biểu đồ | Đạt | Mỗi history đủ epoch1–10, mọi summary/curve có thật; không dùng pilot làm chung kết |
| Test một lượt/seed và TS chỉ val | Đạt | Sáu ledger độc quyền complete/pass_count1/n3507; argmax TS giữ nguyên; gọi lại chỉ xác minh cache, mock chặn model |
| Metric mean/sample std và per-class | Đạt | Tính lại từ đủ CSV test; đối chiếu Final/PerClass/stage11 và evaluator ddof=1 |
| Latency thực và giới hạn phạm vi | Đạt | 22 hàng batch1/32; 10 warmup/50 mẫu sync; p50/p95/p99 từ mẫu thô; không tính CPU preprocessing |
| Workbook bảy sheet đúng từng ô | Đạt | Đọc XLSX đã lưu, đối chiếu CSV và mean/std; tiêu đề Việt/đơn vị/filter/freeze; preview mọi sheet/cột đã xem |
| Báo cáo tiếng Việt, giới hạn và lỗi thật | Đạt | 9 mục; số/ảnh từ nguồn có hash; nêu n3/fold0/budget/cache/head/MPS/runtime fix; liên kết nội bộ đủ |
| Notebook và mã hoàn thiện | Đạt | nbformat/AST mọi code cell; không output cũ hoặc stub; cùng engine CLI; chưa chạy Colab toàn bộ |
| Kiểm tra CPU thực thi | Đạt | 38 test gốc + 25 test bài nộp đạt; không thêm lượt test dataset |
| Không commit dữ liệu/trọng số | Đạt | git ls-files không chứa data/.venv/checkpoint/safetensors/NPZ/ZIP; các sản phẩm nhỏ được commit |

## Đối chiếu các mục chấm

- P1–P4: đủ sáu sản phẩm, nguồn thực, fold0, CSV chung kết và mốc đủ ba seed.
- A/B/D/E/F/G/H: các bằng chứng kiểm tra trong bảng trên. Không tự gán điểm toàn phần A–H.
- C: đủ ba trục, kiểm soát và T07 kết hợp LS+CutMix; CutMix đơn lẻ không tốt hơn mốc. Vì rubric nói kết hợp các yếu tố tốt, tiêu chí này cần giảng viên cân nhắc; không tuyên bố thử hai yếu tố đều tốt.
- I: eval.py nguyên bản tự chấm 19/20. Hai recall tăng so T00 nhưng chưa vượt mốc bài báo. Giảng viên quyết định điểm.
- Điểm thưởng: không yêu cầu hoặc tuyên bố có; chưa thực nghiệm thêm fold/DINO/distillation/ONNX/TTA thích ứng.
- Notebook hợp lệ và gọi cùng module; chưa chạy toàn bộ trên Colab. Biểu đồ XLSX kiểm tra bằng preview đọc file đã lưu; không mở ứng dụng Excel native.
- Test đã mở: cấu hình giữ nguyên, không huấn luyện lại hoặc chạy model test lần hai trong audit.
