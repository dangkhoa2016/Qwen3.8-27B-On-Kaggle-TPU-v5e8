# Bảo mật và báo cáo lỗ hổng có trách nhiệm

> 🌐 Ngôn ngữ / Language: [English](SECURITY.md) | **Tiếng Việt**

Repository có **CPU mock API phục vụ học tập** và **adapter TPU thử nghiệm**, không phải dịch vụ Qwen đã được bảo vệ đầy đủ để mở Internet. Chạy playground trên **127.0.0.1**, và kiểm authentication, request limits, xử lý lỗi, readiness TPU thực trước khi deploy endpoint.

## Báo cáo lỗ hổng

Gửi thông tin bảo mật riêng tới **i.am@dangkhoa.dev**. Nêu thành phần bị ảnh hưởng, bước tái hiện tối thiểu, mức độ ảnh hưởng và cách liên hệ an toàn. **Không** đăng exploit, credentials hoặc log nhạy cảm trong GitHub issue trước khi người duy trì có cơ hội xử lý.

## Bảo vệ secrets và tài nguyên

- Không commit Kaggle API token, GitHub PAT, private key hoặc cloud credentials.
- Che dữ liệu nhạy cảm trong log, screenshot và notebook outputs trước khi chia sẻ.
- Không mở adapter HTTP TPU ra public tunnel nếu chưa kiểm authentication và tải thật.
- Dùng quyền tối thiểu và giới hạn request/concurrency cho model tiêu thụ bộ nhớ lớn.
- Nếu lộ credentials, phải thu hồi/rotate ngay.

## Giới hạn cần hiểu

Unit test authorization và request bounds có ích nhưng **không thay thế security review đầy đủ của dịch vụ TPU đã deploy**. Xem [tích hợp API](api/TPU-INTEGRATION.vi.md).
