# Qwen3.8-27B trên Kaggle TPU v5e-8

## Inference Text + Vision BF16 gốc · FastAPI · Giao diện Vue

[![CPU CI](https://github.com/dangkhoa2016/Qwen3.8-27B-On-Kaggle-TPU-v5e8/actions/workflows/cpu-acceptance.yml/badge.svg)](https://github.com/dangkhoa2016/Qwen3.8-27B-On-Kaggle-TPU-v5e8/actions/workflows/cpu-acceptance.yml)
[![Source MIT](https://img.shields.io/badge/Project%20code-MIT-green)](LICENSE)
[![Model Apache-2.0](https://img.shields.io/badge/Model-Apache--2.0-blue)](THIRD_PARTY_LICENSES/QWEN3.8-27B-APACHE-2.0)
![TPU v5e-8](https://img.shields.io/badge/Kaggle%20TPU-v5e--8-blue)
![BF16](https://img.shields.io/badge/Original%20weights-BF16-purple)
![HTTP pending](https://img.shields.io/badge/Real%20TPU%20HTTP-Pending-orange)

**Ngôn ngữ:** [English](README.md) | Tiếng Việt

Dự án trình diễn kỹ thuật độc lập, có khả năng kiểm tra và tái lập, dành cho model Text/Vision gốc [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), chạy trên Kaggle TPU v5e-8 bằng JAX và MaxText đã patch. Bao gồm notebook có thể tái lập, runtime inference thử nghiệm được bảo vệ, REST API mock CPU, giao diện Vue CDN không cần build, kiểm thử Python/trình duyệt và dữ liệu bằng chứng.

**Nguyên tắc:** trọng số BF16 gốc, 8 thiết bị TPU; không GGUF, không lượng tử hóa, không đưa pretrained weights vào Git. Dự án không đại diện Qwen.

## Tổng quan trạng thái kiểm chứng

| Thành phần | Triển khai | Giới hạn nghiệm thu |
|---|---|---|
| Notebook | JAX + MaxText, Text và Vision trên Kaggle | Có evidence TPU 8 thiết bị lịch sử; Run All mới vẫn cần kiểm chứng |
| Nguồn gốc model | Kiểm tra 18 shard, index, patch và archive | Preflight CPU không thay thế inference mới |
| Text | Prompt xác định và kiểm thử ngữ nghĩa | Dữ liệu lịch sử, chưa phải HTTP acceptance |
| Vision | Thẻ số và sáu cảnh | Bằng chứng lịch sử; OCR cảnh phức tạp chỉ **22/55** |
| REST service | FastAPI mock và TPU ASGI thử nghiệm | CPU tests PASS; HTTP TPU thật chưa nghiệm thu |
| Giao diện | Vue 3 CDN, Text Chat và upload ảnh | Playwright browser E2E |
| CI | Python 3.12, Node 22 và Chromium | Chỉ kiểm thử CPU/mock |

Chi tiết tại [nghiệm thu](docs/ACCEPTANCE.vi.md), [tái lập](docs/REPRODUCIBILITY.vi.md) và [evidence](evidence/).

## Kiến trúc kỹ thuật

1. **Tài nguyên đính kèm:** Kaggle Model BF16 gốc và bộ MaxText offline-support riêng. Kiểm tra checksum, checkpoint và source trước khi nạp model.
2. **Inference:** một MaxEngine, cây tham số BF16, kiểm tra JAX TPU, các tình huống Text/Vision xác định và đo thời gian khởi tạo, prefill, decode.
3. **HTTP backend:** giới hạn kích thước request, chế độ TPU bật có điều kiện, chỉ bind loopback, yêu cầu xác thực, giới hạn một request mô hình đang chạy và fail-closed. HTTP TPU thật chưa được nghiệm thu.
4. **Giao diện:** Vue CDN dùng API mock CPU có đánh dấu kết quả mock rõ ràng, không giả nhận là inference Qwen.
5. **Bằng chứng:** Python unit tests, Playwright E2E, kiểm tra nguồn checkpoint và notebook/JSON kết quả TPU lịch sử.

Xem [kiến trúc](docs/ARCHITECTURE.vi.md) và [tích hợp TPU](api/TPU-INTEGRATION.vi.md).

## Chuẩn bị Kaggle TPU Notebook

| Thành phần | Giá trị |
|---|---|
| Accelerator | Kaggle TPU v5e-8, tám thiết bị |
| Model | dangkhoa2016/qwen-qwen3-8-27b (Transformers/default/version 1) |
| Model mount | /kaggle/input/models/dangkhoa2016/qwen-qwen3-8-27b/transformers/default/1 |
| Offline support | dangkhoa2016/qwen3-8-27b-tpu-v5e8-offline-support |
| Support mount | /kaggle/input/datasets/dangkhoa2016/qwen3-8-27b-tpu-v5e8-offline-support |
| Notebook | [Notebook Kaggle TPU](notebooks/kaggle-tpu-v5e8-production-showcase.ipynb) |
| Phiên bản phần mềm | JAX/JAXLIB 0.11.0; Transformers 5.12.1 |

**Trước khi Run All:** attach cả model và support, chọn TPU v5e-8, giữ nguyên bootstrap, bật **Internet ON** chỉ để clone bộ ảnh ví dụ công khai đã pin Git commit. Không tải lại model/offline support vô ích; tránh nạp hai model 27B cùng lúc.

11 ảnh tại [examples/images](examples/images/) có [manifest SHA-256](examples/assets_manifest.json). Notebook pin commit 84a7587a40d930ea83c4cee8cc2aeca22ee59ea9. Ảnh 1024×768 chỉ dùng quan sát, **không phải inference được nghiệm thu riêng**.

## Kiểm thử CPU và giao diện mock

Yêu cầu Python 3.12 và Node/npm.

~~~bash
git clone https://github.com/dangkhoa2016/Qwen3.8-27B-On-Kaggle-TPU-v5e8.git
cd Qwen3.8-27B-On-Kaggle-TPU-v5e8
python -m pip install -r requirements-cpu.txt
python -m pytest -q tests
npm ci
npx playwright install chromium
npm run test:e2e
~~~

Chạy server **CPU mock**:

~~~bash
QWEN38_BACKEND=mock python -m uvicorn api.app:app --host 127.0.0.1 --port 8090
~~~

Mở http://127.0.0.1:8090. Chế độ này **không nạp model Qwen**. Có thể review health, readiness, metrics, Text Chat và upload ảnh, nhưng không được coi kết quả mock là inference thật.

## Hiệu năng và chứng cứ

- [Notebook TPU lịch sử đã thực thi](evidence/qwen38-production-fresh-executed.ipynb)
- [Kết quả đo BF16 Text/Vision](evidence/qwen38_27b_tpu_v5e8_text_vision_fresh_20261018.json)
- [Manifest SHA-256](evidence/SHA256_MANIFEST.json)
- [Tiêu chí nghiệm thu](docs/ACCEPTANCE.vi.md)

Notebook lịch sử ghi nhận tám thiết bị và một lần nạp engine BF16. Chỉ sử dụng số liệu latency đo thật trong evidence, không suy diễn thành throughput phổ quát hoặc khẳng định notebook đã sửa cũng PASS. HTTP TPU thật cần test Text, Vision, concurrent requests, xác thực, trạng thái và shutdown. CI CPU PASS không thay thế các bước đó.

## Cấu trúc repository

| Đường dẫn | Vai trò |
|---|---|
| notebooks/ | Showcase BF16 Text/Vision trên Kaggle |
| api/ | FastAPI, MaxEngine adapter, vòng đời, preflight TPU |
| web/ | Giao diện Vue CDN |
| examples/ | Ảnh đầu vào, manifest và hướng dẫn |
| evidence/ | Output lịch sử, JSON và SHA-256 |
| tests/ | Kiểm thử Python và trình duyệt |
| docs/ | Kiến trúc, tái lập và nghiệm thu |
| .github/ | CI và quản trị |

[Tái lập](docs/REPRODUCIBILITY.vi.md) · [Nhật ký](CHANGELOG.vi.md) · [Đóng góp](.github/CONTRIBUTING.vi.md) · [Bảo mật](.github/SECURITY.vi.md) · [Hỗ trợ](.github/SUPPORT.vi.md).

## Giấy phép, bản quyền và ghi công

**Hai giấy phép độc lập:** mã nguồn và tài liệu do dự án phát triển áp dụng [MIT © 2026 Đăng Khoa](LICENSE). Mô hình và trọng số upstream **Qwen/Qwen3.8-27B áp dụng Apache License 2.0**, không phải MIT.

**[Toàn văn giấy phép model đã có trong Git](THIRD_PARTY_LICENSES/QWEN3.8-27B-APACHE-2.0)**; xem thêm [chính sách giấy phép và ghi công](MODEL_LICENSE.vi.md), [model card upstream](https://huggingface.co/Qwen/Qwen3.8-27B) và [thông báo bên thứ ba](THIRD_PARTY_NOTICES.vi.md). Không đưa pretrained weights vào Git. Kaggle mirror và bộ runtime đã patch phải được đánh giá riêng về các yêu cầu ghi công và phân phối.

**Trạng thái dự án:** đã triển khai mã nguồn và kiểm thử CPU; có evidence TPU lịch sử; fresh notebook/HTTP TPU vẫn chờ nghiệm thu. Đây không phải bản phát hành chính thức của Qwen.
