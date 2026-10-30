# Hỗ trợ, khắc phục lỗi và thảo luận

> 🌐 Ngôn ngữ / Language: [English](SUPPORT.md) | **Tiếng Việt**

Để bắt đầu, xem [README](README.vi.md), [hướng dẫn chạy TPU](docs/REPRODUCIBILITY.vi.md) và [CPU Playground](api/README.vi.md). Ngay cả khi chưa có TPU Kaggle, bạn vẫn có thể học kiến trúc API và cách đánh giá model.

## Tìm đúng hướng khắc phục

| Vấn đề | Tài liệu |
|---|---|
| CPU API không khởi động | [FastAPI Playground](api/README.vi.md), cài Python packages |
| Lỗi upload ảnh hoặc UI | [Browser tests](tests/browser/showcase.spec.cjs) |
| Không tìm thấy Kaggle Dataset | [Support offline và cách mount](docs/OFFLINE-SUPPORT.vi.md) |
| Lỗi JAX/XLA shape hoặc sharding | [Kiến trúc](docs/ARCHITECTURE.vi.md), versions runtime |
| Model trả lời sai | [Đánh giá](docs/ACCEPTANCE.vi.md), [bài học lỗi](docs/FAILURE-ANALYSIS.vi.md) |
| Muốn đóng góp | [Hướng dẫn đóng góp](CONTRIBUTING.vi.md) |

## Viết issue để người khác dễ tái hiện

Cung cấp commit SHA, lệnh chạy, phiên bản Python/JAX, môi trường và danh sách GPU/TPU nếu liên quan. Kèm ví dụ tối thiểu, kết quả kỳ vọng/thực tế và log **đã che dữ liệu nhạy cảm**. Nếu báo lỗi output, ghi SHA256 ảnh và prompt đầy đủ; không coi file trùng tên chắc chắn trùng byte.

**Quyền truy cập:** kiểm tra trạng thái tài nguyên Kaggle và quyền của tài khoản trước khi báo lỗi không thể truy cập. Không đăng credentials vào issue.

Trao đổi chung qua GitHub issues; báo cáo lỗ hổng theo [kênh bảo mật riêng](SECURITY.vi.md). Dự án mẫu phục vụ cộng đồng không cam kết SLA production.
