# API runtime / Môi trường API

> 🌐 Ngôn ngữ / Language: [English](README.md) | **Tiếng Việt**

Server CPU trong `api.app:app` chỉ chạy **mock**. Không có trọng số Qwen3.8-27B được nạp và không được giới thiệu phản hồi này là kết quả suy luận thật.

## Chạy thử

```bash
python -m pip install -r requirements-cpu.txt
QWEN38_BACKEND=mock python -m uvicorn api.app:app --host 127.0.0.1 --port 8090
python -m pytest -q tests
```

Các endpoint: `GET /health`, `GET /readyz`, `GET /metrics`, `POST /v1/chat/completions` và `POST /v1/vision/analyze` (multipart).

Bật `QWEN38_API_KEY` nếu cần yêu cầu Bearer token. Không mở demo mock hay TPU lên Internet khi chưa review ingress.

Các file `tpu_asgi.py`, `tpu_factory.py`, `maxengine_adapter.py`, `runtime.py` là ứng viên thử nghiệm riêng, **chưa được nghiệm thu HTTP TPU**. Đọc [TPU Integration](TPU-INTEGRATION.vi.md).
