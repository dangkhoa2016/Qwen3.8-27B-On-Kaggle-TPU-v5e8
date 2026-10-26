# Trải nghiệm Text + Vision API Playground

> 🌐 Ngôn ngữ / Language: [English](README.md) | **Tiếng Việt**

API Playground giúp tìm hiểu validate request và giao diện trình duyệt bằng **FastAPI + Vue CDN**. Nó chạy trên CPU thông thường, **không cần trọng số Qwen**—một cách bắt đầu thuận tiện trước khi tìm hiểu notebook TPU.

## Chạy trên máy cá nhân

Từ thư mục gốc repository, cài Python 3.12 và Node.js:

```bash
python -m pip install -r requirements-cpu.txt
QWEN38_BACKEND=mock python -m uvicorn api.app:app --host 127.0.0.1 --port 8090
```

Mở **http://127.0.0.1:8090** để thử gửi text và upload ảnh. UI ghi rõ **MOCK BACKEND · NO MODEL LOADED**. Response mock chỉ minh họa hợp đồng API, **không phải kết quả Qwen sinh ra**.

## Các endpoint

| Method | Endpoint | Công dụng |
|---|---|---|
| GET | `/health` | Xem trạng thái health |
| GET | `/metrics` | Thống kê runtime cơ bản |
| POST | `/v1/chat/completions` | Gửi text theo request schema |
| POST | `/v1/vision/analyze` | Gửi ảnh multipart và prompt |

Response ở chế độ mock có `mock: true`. Ví dụ payload và điều kiện validate nằm trong [API tests](../tests/test_api.py) và [browser tests](../tests/browser/showcase.spec.cjs).

## Chạy bộ test

```bash
python -m pytest -q tests
npm ci
npx playwright install chromium
npm run test:e2e
```

Lần kiểm gần nhất đạt **36 Python PASS, 1 SKIP và 2 Playwright PASS**. Các test kiểm input, error, UI và hợp đồng mock—không đánh giá mô hình TPU.

## Khi chuyển sang inference thật

Phục vụ model 27B thật cần checkpoint BF16, tám TPU, runtime có SHA256, một engine, authentication và giới hạn concurrency. Adapter hiện vẫn là thử nghiệm, **chưa nghiệm thu HTTP TPU thực**, nên chỉ để loopback và tắt mặc định.

Xem [kiến trúc TPU API](TPU-INTEGRATION.vi.md) và [pipeline notebook](../docs/ARCHITECTURE.vi.md).
