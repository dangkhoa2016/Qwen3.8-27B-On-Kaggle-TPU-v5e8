# Qwen3.8-27B trên Kaggle TPU v5e-8 — Text, Vision & API Showcase

[![CPU CI](https://github.com/dangkhoa2016/Qwen3.8-27B-On-Kaggle-TPU-v5e8/actions/workflows/cpu-acceptance.yml/badge.svg)](https://github.com/dangkhoa2016/Qwen3.8-27B-On-Kaggle-TPU-v5e8/actions/workflows/cpu-acceptance.yml)
[![MIT](https://img.shields.io/badge/Source-MIT-green.svg)](LICENSE)
[![Qwen Apache 2.0](https://img.shields.io/badge/Model-Apache--2.0-blue.svg)](MODEL_LICENSE.md)
![Python](https://img.shields.io/badge/Python-3.12-3776AB)
![TPU](https://img.shields.io/badge/TPU-v5e--8-4285F4)
![JAX](https://img.shields.io/badge/Inference-JAX%20%2B%20MaxText-purple)
![TPU HTTP](https://img.shields.io/badge/TPU%20HTTP-Pending-orange)

> 🌐 Ngôn ngữ / Language: [English](README.md) | **Tiếng Việt**

Dự án kỹ thuật độc lập chạy mô hình Qwen3.8-27B BF16 Text + Vision trên Kaggle TPU v5e-8 với JAX/MaxText. Không lượng tử hóa trọng số; không đưa weights lên Git. Không đại diện Qwen.

## Phạm vi đã xác minh

- **Notebook TPU:** đã có bằng chứng chạy Text/Vision thực tế trong [evidence/](evidence/); notebook mới có bước git clone phải chạy Run All mới.
- **REST API và Web UI CPU:** các hợp đồng API, fail-closed và Playwright CI đã PASS.
- **TPU HTTP thật:** **chưa được nghiệm thu**. Chế độ thử nghiệm chỉ chạy loopback khi chủ động bật; không coi phản hồi mock là mô hình thật.
- Sáu ảnh minh họa gốc chỉ 256×192; ba ảnh 1024×768 dùng để xem, chưa được nghiệm thu inference.
- OCR trên bộ cảnh phong phú chỉ đạt **22/55**, không thể kết luận OCR tổng quát là tốt.

## Chạy nhanh CPU mock

```bash
python -m pip install -r requirements-cpu.txt
python -m pytest -q tests
QWEN38_BACKEND=mock python -m uvicorn api.app:app --host 127.0.0.1 --port 8090
```

Mở giao diện tại `http://127.0.0.1:8090`. Chế độ này không nạp mô hình thật.

## Cấu trúc

- `notebooks/`: notebook Kaggle, ví dụ Git-pinned và checksum ảnh
- `api/`: ASGI mock, TPU thử nghiệm được bảo vệ, adapter MaxEngine, vòng đời và bảo mật
- `web/`: giao diện text/vision
- `evidence/`: notebook đã thực thi, JSON, SHA256
- `tests/`: kiểm thử CPU và trình duyệt

## Tài liệu

[Kiến trúc](docs/ARCHITECTURE.vi.md) · [Tái lập](docs/REPRODUCIBILITY.vi.md) · [Nghiệm thu](docs/ACCEPTANCE.vi.md) · [Nhật ký](CHANGELOG.vi.md) · [Đóng góp](.github/CONTRIBUTING.vi.md) · [Bảo mật](.github/SECURITY.vi.md) · [Hỗ trợ](.github/SUPPORT.vi.md).

## Giấy phép

Mã nguồn/tài liệu dự án: **MIT © 2026 Đăng Khoa <i.am@dangkhoa.dev>**. Model và trọng số Qwen3.8-27B: **Apache-2.0**. Xem [giấy phép model](MODEL_LICENSE.vi.md) và [thông báo bên thứ ba](THIRD_PARTY_NOTICES.vi.md). MIT của repository không cấp quyền sở hữu trọng số upstream.
