# Tích hợp suy luận trên TPU

> [English](TPU-INTEGRATION.md) | **Tiếng Việt**

## Kiến trúc và phạm vi hỗ trợ

Backend TPU Qwen3.8-27B BF16 sử dụng MaxText bootstrap được kiểm tra nguồn gốc và một MaxEngine runner duy nhất. REST service mặc định chỉ là CPU mock, **không** chạy suy luận model. HTTP inference trên TPU hiện là tính năng thử nghiệm và **chưa nghiệm thu end-to-end**.

| Module | Vai trò |
|---|---|
| maxengine_adapter.py | Adapter xử lý Text/Vision và prefill/decode |
| runtime.py | Khởi tạo/đóng runner; trạng thái READY/FAILED/CLOSED; từ chối request đồng thời |
| tpu_factory.py | Kiểm tra 18 model shards, checksum MaxText archive/patch, tám TPU devices và nạp BF16 |
| app.py | FastAPI mock mặc định, cho phép inject test runner |
| serving_config.py | Kiểm tra chế độ thử nghiệm, giới hạn input và xác thực |
| tpu_asgi.py | ASGI TPU riêng, bật khi chủ động lựa chọn và chỉ nạp model ở lifespan |

Nguồn, model và patch được ghim và kiểm tra tính toàn vẹn. Notebook showcase hiện tại mô tả luồng Text/Vision. Phiên bản mã nguồn và bộ ảnh 1280×960 cần executed notebook cùng HTTP evidence mới; không sử dụng output thử nghiệm cũ làm bằng chứng nghiệm thu hiện tại.

## Giao ước thực thi và bảo mật

Chế độ thử nghiệm yêu cầu QWEN38_SERVING_MODE=tpu-experimental, QWEN38_EXPERIMENTAL_TPU_ENABLE=YES_I_ACCEPT_UNQUALIFIED_API, QWEN38_TPU_HTTP_ACCEPTANCE=I_ACCEPT_NOT_YET_QUALIFIED, cùng QWEN38_API_KEY tối thiểu 24 ký tự. Chỉ bind 127.0.0.1 với đúng một worker; không bật Uvicorn auto-reload.

Giới hạn: tối đa 160 output tokens, 4.000 ký tự prompt và 8 MiB mỗi ảnh. Request đồng thời bị từ chối thay vì xếp hàng vô hạn; lỗi khởi tạo phải fail-closed. Kiểm tra cấu hình không thay thế audit ingress security: **không đưa endpoint chưa nghiệm thu ra Cloudflare Tunnel công khai**.

## Điều kiện nghiệm thu

Kiểm thử HTTP Text/Vision thật trên TPU với bộ ảnh 1280×960 hiện tại; xác minh một engine/một lần nạp BF16, readiness, xác thực, phản hồi 429 khi bận, lỗi 503, shutdown và latency đo trực tiếp. Lưu chứng cứ mới có checksum SHA-256 trước khi đánh dấu PASS. Xem [tiêu chí nghiệm thu](../docs/ACCEPTANCE.vi.md).

Python unit và browser tests trên CPU chỉ chứng minh hợp đồng mock.
