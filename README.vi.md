# Qwen3.8-27B trên Kaggle TPU v5e-8

**BF16 nguyên gốc · Text + Vision · JAX / MaxText · Ví dụ kỹ thuật có thể kiểm chứng**

> 🌐 Ngôn ngữ / Language: [English](README.md) | **Tiếng Việt**

[![CPU checks](https://github.com/dangkhoa2016/Qwen3.8-27B-On-Kaggle-TPU-v5e8/actions/workflows/cpu-acceptance.yml/badge.svg)](https://github.com/dangkhoa2016/Qwen3.8-27B-On-Kaggle-TPU-v5e8/actions/workflows/cpu-acceptance.yml)
[![Mã dự án: MIT](https://img.shields.io/badge/Project_code-MIT-green)](LICENSE)
[![Model: Apache 2.0](https://img.shields.io/badge/Upstream_model-Apache--2.0-blue)](MODEL_LICENSE.vi.md)
![Kaggle TPU v5e-8](https://img.shields.io/badge/Accelerator-TPU_v5e--8-blue)
![BF16 gốc](https://img.shields.io/badge/Checkpoint-Original_BF16-purple)

Dự án tìm hiểu cách chạy **Qwen3.8-27B BF16 nguyên gốc từ [Kaggle Model mirror](https://www.kaggle.com/models/dangkhoa2016/qwen-qwen3-8-27b)**, trên **tám thiết bị Kaggle TPU v5e** bằng JAX và MaxText đã điều chỉnh. Khi thực thi, notebook nạp checkpoint được attach từ Kaggle, không tải trực tiếp từ đường dẫn Hugging Face. Repository gồm kết quả suy luận TPU đã đo, notebook 19 case Text + Vision, CPU Playground FastAPI/Vue và tài liệu kiểm chứng dữ liệu, đánh giá kết quả.

Mục tiêu là chia sẻ **kiến thức thực tiễn về triển khai mô hình lớn trên TPU**: fusion đa phương thức, AOT prefill, kiểm toàn vẹn input, đo hiệu năng và diễn giải kết quả một cách chính xác.

## Bạn có thể học gì từ dự án?

| Chủ đề | Nội dung hữu ích | Bắt đầu |
|---|---|---|
| Hiểu inference đa phương thức TPU | Từ ảnh tới token, MRoPE và kiểm JAX/XLA | [Kiến trúc](docs/ARCHITECTURE.vi.md) |
| Thử API không cần TPU | FastAPI + Vue CDN mock và tests | [Chạy CPU](#thử-cpu-playground) |
| Khám phá bộ ví dụ | 16 ảnh SHA256 và 19 case | [Bộ ảnh](examples/README.vi.md) |
| Chạy notebook trên TPU | Model BF16, support offline, notebook | [Hướng dẫn](docs/REPRODUCIBILITY.vi.md) |
| Đánh giá chất lượng model | OCR, image reasoning, output thật và lỗi | [Đánh giá](docs/ACCEPTANCE.vi.md) |
| Hiểu số đo hiệu năng | Một lượt đo thật, cold/warm và giới hạn | [Hiệu năng](docs/PERFORMANCE-ANALYSIS.vi.md) |

## Khám phá ảnh nguồn thực tế

Đây là **ảnh đầu vào của mô hình**, không phải kết quả hình ảnh do Qwen sinh ra. Nhóm Photography có ảnh minh họa do AI tạo và được ghi nhận rõ.

| Sang đường | Cảnh 3D minh họa | Bữa ăn Việt Nam |
|---|---|---|
| ![Đèn người đi bộ xanh tại vạch sang đường](examples/images/photo_ai_city_crossing_1280x960.png) | ![Ví dụ trạm robotics](examples/images/vision_3d_robotics_station.jpg) | ![Ví dụ bữa ăn Việt Nam](examples/images/photo_ai_vietnamese_meal_1280x960.png) |
| Đèn giao thông, quan hệ không gian | Vật thể và bố cục cảnh | Nhận diện món ăn và vật thể |

Prompt, kích thước và SHA256 nằm trong [manifest 16 ảnh](examples/assets_manifest.json) và [hướng dẫn bộ ảnh](examples/README.vi.md).

## Số đo suy luận trên tám thiết bị TPU

Một phiên Kaggle TPU v5e-8 thực tế ngày **27/10/2026** đã ghi nhận:

| Chỉ số | Kết quả |
|---|---|
| Checkpoint | Qwen3.8-27B BF16 gốc, **18 safetensors shard** |
| Phần cứng | Tám v5e, một MaxEngine và một lần nạp checkpoint |
| Khối lượng | **3 Text + 2 OCR + 6 Rich Vision + 8 Photography = 19 case** |
| Hoàn tất sinh token | **19/19 case có EOS** |
| Một số phép kiểm | Text 3/3, OCR số **731 / 482**, Rich Vision **55/55** từ khóa |
| Tổng thời gian | **1.160,844 giây (19 phút 20,844 giây)** |

**Cách hiểu:** số đo gắn với SHA256 input và runtime của phiên được ghi nhận, không đại diện cho độ chính xác tổng quát hoặc chứng minh mọi revision notebook đều đã chạy. Một số output có lỗi số liệu (thuế hóa đơn và so sánh dashboard); tài liệu giữ nguyên các lỗi để đánh giá.

Xem [bằng chứng Full19](docs/FULL19-EVIDENCE.vi.md), [phân tích hành vi mô hình](docs/FAILURE-ANALYSIS.vi.md) và [hiệu năng](docs/PERFORMANCE-ANALYSIS.vi.md).

## Thử CPU Playground

Bạn có thể khám phá request/response mà không cần TPU hoặc checkpoint. Từ thư mục gốc:

```bash
python -m pip install -r requirements-cpu.txt
QWEN38_BACKEND=mock python -m uvicorn api.app:app --host 127.0.0.1 --port 8090
```

Mở **http://127.0.0.1:8090**. Giao diện ghi rõ **MOCK BACKEND · NO MODEL LOADED**; câu trả lời không được sinh bởi Qwen3.8-27B.

Chạy kiểm thử:

```bash
python -m pytest -q tests
npm ci
npx playwright install chromium
npm run test:e2e
```

Mã nguồn mới nhất đã được kiểm **38 Python PASS, 1 SKIP** và **hai Playwright PASS**. Xem [tài liệu API](api/README.vi.md) để biết endpoints và điều kiện validate.

## Chạy notebook BF16 Text + Vision trên TPU

[Notebook Full19 trong Git](notebooks/kaggle-tpu-v5e8-production-showcase.ipynb) sử dụng **một MaxEngine**, Vision encoder tách riêng, JAX/XLA prefill, tensor ảnh **512 × 384**, **prefill 320 token** và **target 768 token**.

Trên Kaggle, chọn **TPU v5e-8** và attach cả [BF16 model mirror](https://www.kaggle.com/models/dangkhoa2016/qwen-qwen3-8-27b) lẫn [Full19 support Dataset](https://www.kaggle.com/datasets/dangkhoa2016/qwen38-27b-tpu-v5e8-full19-runtime-support). Notebook kiểm tra manifest 48 file với SHA256 `cf5343f614eb32ff9234c870e4a26093f9f07e62199ee90396d516a804eb8a28` trước khi nạp trọng số.

Làm theo [hướng dẫn thực thi](docs/REPRODUCIBILITY.vi.md) và [tài nguyên offline](docs/OFFLINE-SUPPORT.vi.md). Quyền truy cập tài nguyên và giấy phép thành phần bên thứ ba có thể hạn chế khả năng tải. File notebook trong Git chứa mã nguồn, khi chạy sẽ sinh output mới; số đo TPU ở trên không phải output được nhúng sẵn.

## Kiến trúc và cấu trúc repository

```text
notebooks/   Notebook Full19 BF16 / Split-Vision TPU
examples/    Ảnh đầu vào, manifest SHA256 và prompts
docs/        Kiến trúc, tái lập, đánh giá và hiệu năng
api/         FastAPI CPU playground và source adapter TPU riêng
web/         Vue CDN browser playground
tests/       Python và browser regression tests
support/     Manifest runtime support đã ghim checksum
```

Một điểm kỹ thuật quan trọng là **mã hóa ảnh riêng, fusion embeddings với text qua MRoPE, rồi dùng JAX/XLA prefill đã được kiểm chứng**. Xem sơ đồ trong [Kiến trúc](docs/ARCHITECTURE.vi.md). Adapter HTTP TPU là đường riêng và **chưa có nghiệm thu request TPU thật**.

## Tái sử dụng, ghi công và đóng góp

Repository **không chứa trọng số Qwen pretrained**. Mã nguồn/tài liệu do dự án viết áp dụng [MIT](LICENSE); model upstream chịu [Apache-2.0](MODEL_LICENSE.vi.md). Runtime dependencies và quyền phân phối ảnh cần kiểm riêng; xem [thông báo bên thứ ba](THIRD_PARTY_NOTICES.vi.md).

Dự án hoan nghênh ý tưởng, sửa tài liệu và regression tests tái lập được. Xem [Đóng góp](CONTRIBUTING.vi.md), [Hỗ trợ](SUPPORT.vi.md), [Bảo mật](SECURITY.vi.md). Đây là dự án nghiên cứu kỹ thuật độc lập, **không phải sản phẩm chính thức hoặc sự bảo trợ của Qwen**.

[Changelog](CHANGELOG.vi.md) · [Kiến trúc](docs/ARCHITECTURE.vi.md) · [Tái lập](docs/REPRODUCIBILITY.vi.md) · [Đánh giá](docs/ACCEPTANCE.vi.md)
