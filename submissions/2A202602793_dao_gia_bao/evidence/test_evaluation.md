# Đánh giá test chính thức

Sáu lượt F01/T00 × seed 0/1/2 đã hoàn tất, mỗi lượt duyệt đúng một lần toàn bộ 3.507 ảnh. Bản chưa hiệu chuẩn và bản TS xuất từ cùng logits, không chạy test thêm. Ledger và SHA ở test_passes.json.

Evaluator nguyên bản: F01 macro-F1 0,945066 ± 0,003553; T00 0,932536 ± 0,005069 (sample std, ddof=1). Delta ghép seed +0,012530 ± 0,002648. Top-1 F01 95,7514% ± 0,3018 điểm phần trăm. ECE trước/sau TS 0,085573/0,008749; nhiệt độ chỉ khớp NLL val.

Recall Chinee apple/Snake weed F01 là 88,05%/86,93%, tăng so với mốc T00 87,32%/84,64%, nhưng còn dưới mốc tham khảo bài báo 88,5%/88,8%. Tự chấm phần I bằng eval.py là 19/20, giảng viên xác nhận.

Ma trận cộng ba seed cho thấy Chinee→Negative 49/678, Snake→Negative 40/612; Chinee→Snake 20/678 và chiều ngược 26/612. Không coi 10.521 dự đoán là ảnh độc lập hoặc ensemble. Gallery ưu tiên 8 tên ảnh lỗi khác nhau trong cặp 0/7: nhiều lá/cành chồng lấp, vùng sáng tối tương phản và nền dày đặc. Đây là giả thuyết khó nhận dạng, chưa có thí nghiệm can thiệp chứng minh; không chỉnh mô hình sau test.

25 test bản triển khai và 38 test gốc đạt; evaluator/starter/tests gốc giữ nguyên. Benchmark thật checkpoint F01 seed0: p95 14,72 ms, chỉ thành phần trên GPU, chưa gồm đọc/tiền xử lý CPU hay camera.
