# Thiết kế và phạm vi diễn giải ablation

- Mốc: T00_screen, ConvNeXt-Atto được chọn bằng macro-F1 validation, seed 0, fold 0, 10 epoch, 128×128, batch 32, AdamW, LR backbone 1e-4/head 1e-3, WD 0,05, warmup 1 epoch rồi cosine, CE, crop + flip, FP32.
- Trục A có finetune/frozen/scratch. `pretrained=False` trong scratch là thành phần của chế độ khởi tạo, không phải một yếu tố độc lập. Frozen chỉ cập nhật head, backbone ở eval.
- Trục B có basic/ColorJitter/CutMix. CutMix giữ crop + flip, trộn nhãn theo diện tích vùng thực; accuracy train không được diễn giải như nhãn cứng thông thường.
- Trục C có CE/label smoothing epsilon 0,1/focal gamma 2. Gamma=0 được kiểm tra số học tương đương CE, không dùng như một phép cải thiện thực nghiệm.
- T07 kiểm tra tương tác CutMix + label smoothing; được khai báo là kết hợp hai yếu tố, không đưa vào nhóm thay đổi đơn.
- Mọi điểm chọn công thức lấy từ 3.501 ảnh validation nguyên bản. Test chưa chạy.

Ngân sách và LR vốn dùng cho fine-tuning được giữ cố định cả ở scratch/frozen. Vì vậy không suy rộng một kết quả thấp thành kết luận rằng huấn luyện từ đầu hoặc đóng băng không thể đạt kết quả tốt với ngân sách/LR khác. Những thí nghiệm đó không được thực hiện trong thiết kế này.

Đây là sàng lọc seed 0. Chênh lệch nhỏ chưa chứng minh ưu thế ổn định; chung kết sẽ đối chiếu mean và sample std (ddof=1) trên cùng ba seed cho công thức chọn và mốc. Việc chọn trong nhiều cấu hình trên cùng validation có thể làm điểm validation lạc quan.

Tối ưu Dataset chỉ thay thao tác lấy tên/nhãn qua iloc bằng cột cache. Kiểm tra tensor theo seed cho 18 mẫu train/val sai số tối đa 0; sau 10 epoch, CSV validation của T00_screen và B02_convnext_atto giống từng byte. Các mốc thời gian hai giai đoạn dùng phiên bản I/O khác nên không dùng delta thời gian đó để kết luận hiệu quả hyperparameter.
