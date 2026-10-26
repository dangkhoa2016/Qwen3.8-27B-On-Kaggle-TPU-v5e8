# Ghi công thư viện và dự án nguồn mở

> 🌐 Ngôn ngữ / Language: [English](THIRD_PARTY_NOTICES.md) | **Tiếng Việt**

Bản demo TPU sử dụng nhiều dự án nguồn mở. Phần mềm và model vẫn chịu giấy phép riêng, độc lập với [MIT của mã nguồn dự án](LICENSE).

| Dự án | Vai trò | Nguồn và giấy phép |
|---|---|---|
| [bản mirror BF16 trên Kaggle](https://www.kaggle.com/models/dangkhoa2016/qwen-qwen3-8-27b) (nguồn Qwen upstream) | Checkpoint BF16 thực sự attach và chạy trên Kaggle TPU | [Bản sao giấy phép Apache-2.0 upstream](THIRD_PARTY_LICENSES/QWEN3.8-27B-APACHE-2.0) |
| [MaxText](https://github.com/AI-Hypercomputer/maxtext) | Runtime transformer cho TPU | Kiểm revision và giấy phép đúng phiên bản |
| [JAX](https://github.com/jax-ml/jax) | Biên dịch và tính toán TPU | Apache-2.0 upstream |
| [FastAPI](https://github.com/fastapi/fastapi) | Demo HTTP API | MIT upstream |
| [Vue](https://github.com/vuejs/core) | Giao diện trình duyệt | MIT upstream |
| [Playwright](https://github.com/microsoft/playwright) | E2E browser tests | Giấy phép theo upstream |

Dataset support còn có Python wheels, helpers, patches và cache biên dịch. Các thành phần này **không phải đều do dự án tự viết**; phải kiểm version, license và quyền phân phối trước khi đóng gói lại.

Xem [nguồn gốc model](MODEL_LICENSE.vi.md) để phân biệt mã dự án và pretrained weights. URL hay hash chỉ chứng minh nguồn tham chiếu, **không mặc nhiên cấp quyền phân phối mọi artifact**.
