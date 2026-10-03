# Sửa cờ runtime trước mọi suy luận test

Phát hiện `run(Config(save_test_predictions=True))` từ một lượt đã train với cờ False bị coi là recipe khác. Sửa đúng điều kiện so sánh cấu hình để bỏ qua cờ runtime này như `resume`. Không sửa vòng lặp học, validation, optimizer, scheduler, EMA, RNG, head, augmentation hoặc công thức đã chốt. File frozen_final.json giữ nguyên từng byte.

Fingerprint các hàm học trước/sau giống nhau; SHA toàn file train.py thay đổi được ghi rõ trong pretest_runtime_fix.json. Guard chỉ cho phép đúng phiên bản tương thích này. Kiểm tra mock chứng minh đổi cờ chỉ gọi exporter, không train lại. Bản sửa được commit/push trước mọi vòng test; 24 kiểm tra phần triển khai đạt.
