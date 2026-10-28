# Nhật ký cập nhật dự án

> 🌐 Ngôn ngữ / Language: [English](CHANGELOG.md) | **Tiếng Việt**

Nhật ký này tóm tắt **tính năng hữu ích và kết quả kỹ thuật đã được ghi nhận**, không kể lại quy trình làm việc nội bộ. Ngày tháng phản ánh mốc đã ghi kết quả hoặc tài nguyên.

## 28/10/2026 — Bộ dữ liệu ứng viên có thể tái lập

- Bổ sung ảnh crossing **đèn người đi bộ xanh** vào bộ ví dụ Git, cập nhật SHA256 trong manifest 16 ảnh.
- Tạo **Private Full19 Release Support Dataset v1** riêng với ảnh mới. Tải lại từ Kaggle và kiểm độc lập **48/48 checksum**.
- Làm rõ khác biệt giữa dữ liệu ảnh thay thế đã kiểm checksum và kết quả Full19 đã thực thi trước đây. Output TPU của ảnh mới cần được kiểm riêng; tài liệu chính không phụ thuộc URL notebook thử nghiệm tạm thời.
- Cải thiện tài liệu provenance ảnh và nhãn API Playground để phân biệt CPU mock với TPU inference.

## 27/10/2026 — Bằng chứng Full19 experimental

- Ghi nhận lượt TPU thật hoàn thành **19/19 case tự động có EOS** trên tám thiết bị v5e trong **1.160,844 giây**.
- Tài liệu hóa nhóm Text, OCR, Rich Vision và Photography, SHA256 input và output nguyên văn.
- Công khai lỗi thuế invoice và so sánh dashboard để không đánh đồng chương trình hoàn thành với độ chính xác tổng quát.
- Chuẩn bị bundle offline có manifest 48 file SHA256 để thuận tiện tái lập.

## Đọc tiếp

[Bắt đầu](README.vi.md) · [Kiến trúc](docs/ARCHITECTURE.vi.md) · [Tái lập](docs/REPRODUCIBILITY.vi.md) · [Đánh giá](docs/ACCEPTANCE.vi.md)
