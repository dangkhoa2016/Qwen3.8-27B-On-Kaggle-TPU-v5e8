# Tích hợp TPU / TPU integration

> 🌐 Ngôn ngữ / Language: [English](TPU-INTEGRATION.md) | **Tiếng Việt**

**Trạng thái: CPU acceptance PASS, HTTP inference TPU thật CHƯA nghiệm thu.**

Notebook đã thực thi trong `evidence/qwen38-production-fresh-executed.ipynb` là nguồn tham chiếu cho `prefill`, `generate`, xử lý ảnh và Qwen3.8 BF16.

## Kiến trúc

- `maxengine_adapter.py`: tái sử dụng logic Text/Vision từ notebook; chưa chạy API TPU thật.
- `runtime.py`: chỉ tạo một runner, tuần tự hóa inference, trạng thái NOT_STARTED/LOADING/READY/FAILED/CLOSED, chống hàng đợi vô hạn, shutdown có kiểm soát.
- `tpu_factory.py`: xác minh model 18 shard, hash patch/archive MaxText, 8 TPU, phiên bản JAX và nạp BF16 đúng một lần.
- `app.py`: ASGI mặc định chỉ phục vụ mock và fake runner cho tests.
- `tpu_asgi.py`: entrypoint thử nghiệm riêng, không nạp model khi import; chỉ nạp ở FastAPI lifespan.

## Quy tắc an toàn

- Chế độ TPU thử nghiệm yêu cầu `QWEN38_SERVING_MODE=tpu-experimental`, `QWEN38_EXPERIMENTAL_TPU_ENABLE=YES_I_ACCEPT_UNQUALIFIED_API`, `QWEN38_TPU_HTTP_ACCEPTANCE=I_ACCEPT_NOT_YET_QUALIFIED`, Bearer key dài ít nhất 24 ký tự; chỉ bind `127.0.0.1` và một worker.
- Tối đa 160 output tokens, 4.000 ký tự prompt, 8 MiB mỗi ảnh; request đồng thời bị từ chối thay vì tự xếp hàng.
- Chế độ này **không được công bố trên Cloudflare Tunnel** trước khi nghiệm thu bảo mật và thực thi TPU thật.
- Các bài CPU tests chỉ xác minh hợp đồng, không xác nhận suy luận 27B thực.
- Trước bản release: kiểm tra readiness, authentication, text+vision, latency, 429/503, shutdown, một engine/load, checksum và lưu bằng chứng mới từ TPU.
