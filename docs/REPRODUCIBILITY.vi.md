# Reproducibility

> 🌐 Ngôn ngữ / Language: [English](REPRODUCIBILITY.md) | **Tiếng Việt**

1. Clone repository theo commit SHA đã rà soát. Notebook mới ghim riêng commit chứa ví dụ; không thay SHA tùy tiện.
2. Attach model Qwen3.8-27B và offline support dataset lên Kaggle; xác minh 18 shard và dung lượng được công bố.
3. Giữ JAX/JAXLIB 0.11.0 và Transformers 5.12.1 của môi trường TPU đã nghiệm thu. Kiểm tra SHA256 tar/patch MaxText; metadata không đủ chứng minh quan hệ tổ tiên Git.
4. Chạy `python -m pytest -q tests` và `npm run test:e2e` trên CPU; xem CI GitHub.
5. Chạy Run All/HTTP acceptance mới trên TPU rồi tạo evidence mới; không ghép kết quả cũ vào notebook thay đổi.
6. Không tuyên bố ảnh preview lớn đã PASS inference hoặc nói OCR tốt khi chỉ đạt 22/55.

## Kiểm tra preflight CPU (19/10/2026)

Bootstrap `api/tpu_factory.py` xác minh checkpoint index, đủ tên của 18 shard không rỗng, SHA256 bản patch, SHA256 archive MaxText offline và metadata của source đã giải nén trước khi import JAX hoặc nạp weights. Nếu source chưa có, có thể giải nén offline mà không xóa hoặc ghi đè thư mục MaxText đang tồn tại; từ chối path traversal, links và device nodes trong archive. Các bước preflight đã PASS với model/support gắn trên máy Kaggle CPU hiện tại; chưa kiểm chứng giải nén trên môi trường sạch hoặc chạy TPU mới.

**Blocker công bố:** notebook yêu cầu Internet OFF nhưng lại `git clone` repository GitHub hiện đang **private**. Bộ support dataset không có đủ 11 ảnh ví dụ (thiếu ba ảnh 1024×768). Vì vậy chưa thể khẳng định Run All offline tái lập được. Cần tạo Kaggle examples dataset có version/checksum, hoặc chủ động đổi repository sang public và cho phép Internet ON khi clone. Không nhúng token vào notebook công khai. Notebook chưa chạy lại sau khi đổi pin commit.
