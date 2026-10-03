# Sửa phép nội suy multi-scale trên MPS

Bicubic với antialias báo thiếu toán tử `aten::_upsample_bicubic2d_aa.out` trong PyTorch 2.6/MPS; traceback giữ ở stage8_initial_failure.log. Utility đổi sang bilinear không antialias, đã kiểm tra native MPS 96/128/160, shape và tensor hữu hạn. Nhóm suy luận/benchmark được chạy lại đầy đủ trên validation; không bật CPU fallback. Test chưa chạy.
