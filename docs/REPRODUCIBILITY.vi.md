# Reproducibility

> 🌐 Ngôn ngữ / Language: [English](REPRODUCIBILITY.md) | **Tiếng Việt**

1. Clone repository theo commit SHA đã rà soát. Notebook mới ghim riêng commit chứa ví dụ; không thay SHA tùy tiện.
2. Attach model Qwen3.8-27B và offline support dataset lên Kaggle; xác minh 18 shard và dung lượng được công bố.
3. Giữ JAX/JAXLIB 0.11.0 và Transformers 5.12.1 của môi trường TPU đã nghiệm thu. Kiểm tra SHA256 tar/patch MaxText; metadata không đủ chứng minh quan hệ tổ tiên Git.
4. Chạy `python -m pytest -q tests` và `npm run test:e2e` trên CPU; xem CI GitHub.
5. Chạy Run All/HTTP acceptance mới trên TPU rồi tạo evidence mới; không ghép kết quả cũ vào notebook thay đổi.
6. Không tuyên bố ảnh preview lớn đã PASS inference hoặc nói OCR tốt khi chỉ đạt 22/55.
