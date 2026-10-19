# Hướng dẫn tái lập

> [English](REPRODUCIBILITY.md) | **Tiếng Việt**

1. Checkout chính xác Git commit được review. Notebook pin riêng commit chứa 11 ảnh ví dụ và manifest SHA-256.
2. Trên Kaggle, attach model Qwen3.8-27B BF16 gốc và support dataset offline; chọn TPU v5e-8, bật Internet chỉ để clone bộ ảnh GitHub công khai đã pin.
3. Xác minh index, 18 model shards, SHA-256 của MaxText archive/patch, JAX/JAXLIB 0.11.0 và Transformers 5.12.1. Trọng số không được đưa vào Git.
4. Chạy Run All, ghi cả kết quả đạt và lỗi. Sáu ảnh minh họa Vision 1280×960 cần nghiệm thu mới; không kế thừa số liệu OCR của bộ 256×192 trước đây.
5. Kiểm thử CPU contracts và Playwright E2E riêng biệt. Chúng không thay thế nghiệm thu HTTP TPU thật.
6. Chỉ công bố executed notebook mới, manifest và latency thật sau khi đã tự xác minh.

Ba ảnh bổ sung 1024×768 chỉ là ảnh đầu vào để quan sát, không phải ảnh tạo bởi model hay bằng chứng inference đã PASS.
