# Tiêu chí nghiệm thu Text và Vision

> [English](ACCEPTANCE.md) | **Tiếng Việt**

## Trạng thái xác minh

GitHub Actions xác minh các bài kiểm thử REST contract và trình duyệt trên CPU. Notebook Kaggle public hiện tại cùng bộ ảnh Vision độ phân giải cao **chưa được nghiệm thu end-to-end bằng một lượt chạy TPU mới**. Endpoint HTTP TPU thử nghiệm cũng **chưa được nghiệm thu**. Không sử dụng notebook hoặc số liệu từ lượt chạy cũ làm chứng cứ cho phiên bản hiện tại.

## Quy trình nghiệm thu có thể tái lập

1. Ghi chính xác Git SHA, model và support mounts, SHA-256 patch, phiên bản dependency, 8 TPU devices và ngày chạy.
2. Kiểm tra đúng một engine BF16/một lần nạp tham số; đo thời gian khởi tạo và load.
3. Chạy bài Text, hai thẻ số 731/482 và toàn bộ sáu ảnh cảnh mới; lưu prompt, output, số trường đúng và latency.
4. Với OCR, công bố số trường khớp và lỗi. Không suy diễn năng lực chung từ ảnh được tuyển chọn.
5. Kiểm thử riêng HTTP TPU thật: readiness, xác thực, Text/Vision, giới hạn input, busy 429, lỗi 503, vòng đời và shutdown.
6. Lưu executed notebook, JSON evidence và checksum mới; ghi rõ bài chưa đạt. Chỉ đánh dấu PASS theo phạm vi thực sự được xác minh.

CI CPU PASS không đồng nghĩa TPU inference thật đã PASS.
