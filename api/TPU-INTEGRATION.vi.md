# Từ API Playground tới suy luận thực trên TPU

> 🌐 Ngôn ngữ / Language: [English](TPU-INTEGRATION.md) | **Tiếng Việt**

Repository cung cấp **FastAPI + Vue CDN playground** để tìm hiểu hợp đồng request/response Text+Vision, cùng thiết kế **adapter TPU bật theo cấu hình**. Hai đường này độc lập: giao diện mặc định **không nạp Qwen3.8-27B** và không gửi yêu cầu inference thật lên TPU.

## Chạy CPU Playground

```bash
python -m pip install -r requirements-cpu.txt
QWEN38_BACKEND=mock python -m uvicorn api.app:app --host 127.0.0.1 --port 8090
```

Mở `http://127.0.0.1:8090`. UI hiển thị nhãn mock rõ ràng. Có thể dùng để tìm hiểu JSON response, upload ảnh, trạng thái lỗi và hoạt động trình duyệt, đồng thời nghiên cứu mã TPU riêng.

| Route | Mục đích |
|---|---|
| `GET /health` | Sức khỏe runtime |
| `GET /metrics` | Bộ đếm runtime |
| `POST /v1/chat/completions` | Hợp đồng request Text |
| `POST /v1/vision/analyze` | Hợp đồng phân tích ảnh multipart |

Response mock có thông tin nhận diện mock. Không mô tả chúng là output của model 27B.

## Kiến trúc API TPU thực tế cần có

| Module | Chức năng |
|---|---|
| `api/app.py` | Validate input, routes và errors |
| `api/serving_config.py` | Cấu hình và bảo mật |
| `api/runtime.py` | Vòng đời READY/FAILED/CLOSED, hàng đợi tuần tự, shutdown |
| `api/tpu_asgi.py` | Entrypoint ASGI opt-in |
| `api/tpu_factory.py` | Kiểm chứng model và tạo MaxEngine |
| `api/maxengine_adapter.py` | Tiền xử lý Text/Vision, prefill, generation |

Model 27B BF16 chỉ nên được nạp **một lần**, sau khi xác minh checkpoint và tám TPU. Không tạo MaxEngine mới mỗi HTTP request. Readiness phải phản ánh trạng thái model thực, không chỉ cổng mạng đang mở.

## Thiết kế bảo mật và giới hạn tải

Đường TPU thử nghiệm chỉ dành cho **loopback**, **một worker**, cần bật cấu hình rõ ràng và API key tối thiểu **24 ký tự**. Source có giới hạn **4.000 ký tự prompt**, ảnh **8 MiB** và **160 token output**. Trạng thái bận/không sẵn sàng cần trả **429/503** thay vì tạo tải vô hạn.

Đây là **thiết kế và kết quả test trên CPU**, chưa phải hệ thống TPU production đã được xác nhận. Không công khai endpoint TPU khi chưa kiểm end-to-end.

## Điểm khác biệt phải hiểu

Notebook trong Git cấu hình **320/768 token prefill/target** và ảnh suy luận 512 × 384. Adapter HTTP độc lập hiện có cấu hình token và xử lý ảnh khác, vì vậy kết quả và hiệu năng HTTP phải được kiểm chứng riêng; không thể dùng số đo notebook để nghiệm thu adapter này.

## Cần kiểm gì trước khi chạy HTTP TPU thật?

1. Đồng bộ model, manifest, xử lý ảnh, budget token và MRoPE với notebook TPU đã nghiệm thu.
2. Gửi request Text/Vision có xác thực tới **TPU thật**.
3. Đo startup, readiness, latency và hàng đợi có giới hạn.
4. Thử ảnh lỗi, prompt dài, unauthorized, concurrency, 429/503 và shutdown.
5. Lưu token IDs, EOS, SHA256 input và errors; so sánh độc lập với notebook.

[Chạy API CPU](README.vi.md) · [Kiến trúc TPU](../docs/ARCHITECTURE.vi.md) · [Đánh giá](../docs/ACCEPTANCE.vi.md)
