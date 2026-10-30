# Kiến trúc notebook TPU trình diễn

🌐 Ngôn ngữ / Language: [English](NOTEBOOK-ARCHITECTURE.md) | **Tiếng Việt**

## Người dùng sẽ nhìn thấy gì?

Notebook `notebooks/kaggle-tpu-v5e8-production-showcase.ipynb` chỉ đóng vai trò **hướng dẫn và điều phối chạy**, không còn chép hàng nghìn dòng code triển khai. Notebook có 17 cell code: 16 bước inference/Gallery dùng module Git đã xác minh; cell 17 độc lập mở Cloudflare Quick Tunnel tạm thời. Mỗi bước vẫn có giải thích bằng tiếng Anh và tiếng Việt.

Người dùng Kaggle chỉ cần attach **hai nguồn**: model BF16 gốc và **Runtime Assets Dataset phiên bản 1** (48 file). Mã nguồn được clone ẩn danh từ GitHub public. **Phải bật Kaggle Internet để clone Git**; bước 17 tùy chọn mở Tunnel.

## Phân chia mã nguồn

| File/thư mục | Vai trò |
|---|---|
| `qwen38_runtime/workflow.py` | Quản lý thứ tự thực thi và chuyển trạng thái có tên giữa các cell |
| `qwen38_runtime/stages/preflight.py` | Xác minh model, manifest, 48 payload và thư mục kết quả |
| `qwen38_runtime/stages/dependencies.py` | Kiểm tra wheel offline và tương thích môi trường TPU Kaggle |
| `qwen38_runtime/stages/maxtext.py` | Khôi phục MaxText và áp dụng patch Vision đã ghim SHA256 |
| `qwen38_runtime/stages/inputs.py`, `checkpoint.py` | Kiểm tra ảnh, prompt, metadata index BF16 và danh sách shard; chưa xác minh hash nội dung từng shard |
| `qwen38_runtime/stages/tpu.py`, `engine.py` | Khởi tạo tám thiết bị TPU, nạp một inference engine |
| `qwen38_runtime/stages/inference_helpers.py` | Tokenizer, xử lý ảnh, suy luận |
| `qwen38_runtime/stages/ocr_probe.py`, `ocr_evidence.py` | OCR đầu tiên, chẩn đoán AOT, lưu trace |
| `qwen38_runtime/stages/vision_reuse.py`, `image_cases.py` | Tái sử dụng executable cho 15 case ảnh tiếp theo |
| `qwen38_runtime/stages/text_cases.py` | Chạy 3 case văn bản, lưu JSON 19 case và cờ review |
| `qwen38_runtime/stages/vietnamese.py` và `gallery/vi_state.py` | Tùy chọn 17 case tiếng Việt, lưu và tiếp tục tiến độ |
| `qwen38_runtime/stages/gallery.py` và `gallery/render.py` | Tạo trang HTML offline từ kết quả thực tế |

Cell 17 import `gallery/quick_tunnel.py` từ source Git đã xác minh; không còn nhúng mã Tunnel dài. Cell code đầu cho phép sửa hai mặc định Tunnel. Chỉ trong **phiên tương tác Kaggle** (`KAGGLE_KERNEL_RUN_TYPE=Interactive`) mới giữ URL 900 giây mặc định. Batch/API/CLI và môi trường không xác định bỏ qua Tunnel, không giữ thêm TPU. Có thể chỉnh `QWEN38_TUNNEL_HOLD_SECONDS` hoặc tắt chia sẻ tương tác bằng `QWEN38_TUNNEL_ENABLED=0`. Chỉ thư mục Gallery được phục vụ.

Bước `text_schedule` chỉ thông báo lịch chạy case văn bản, chưa thực hiện inference.

## Mã nguồn Git và runtime offline trên Kaggle

Runtime Assets Dataset **phiên bản 1** có bảy ZIP runtime (48 payload ghim SHA256), không chứa Python source của dự án. Runtime Support Dataset v1 cũ chỉ được giữ làm bằng chứng lịch sử. Không có trọng số hay output TPU.

Trước khi import module, cell đầu clone Git và xác minh SHA commit chính xác. Git-first preflight kiểm tra repository và gọi bộ xác minh manifest 48 file gốc trước khi import JAX.

**Điều kiện bắt buộc:** GitHub phải public để clone ẩn danh; attach Runtime Assets Dataset **phiên bản 1** và model BF16. Không cần source archive của dự án hay GitHub token. Kiểm tra Git commit chính xác cùng SHA256 trước khi khởi tạo TPU.

## Đóng gói và kiểm thử


Khi có **Dataset gốc với 48 file đã xác minh** trên máy, chạy:

```bash
python3 support/package_runtime_assets.py \
  --source /path/to/verified-legacy-dataset-v1 \
  --destination /path/to/new-empty-runtime-assets
```

Lệnh chỉ tạo bảy ZIP runtime cục bộ, **không upload hoặc ghi đè** Kaggle. Kiểm manifest và đủ 48 file nguồn trước khi phát hành Runtime Assets Dataset.


```bash
python3 support/sync_notebook_bootstrap.py --check
python3 support/sync_vi_state_notebook.py --check
python3 support/sync_gallery_notebook.py --check
python3 -m pytest -q tests
```

Khi sửa module Git, review diff rồi chạy sync và tests. Giữ mặc định `main` trong notebook; mỗi lần chạy lưu Git commit SHA thực tế vào provenance. Chỉ chọn SHA cụ thể khi cần tái lập một lượt chạy lịch sử. Thay đổi source không đòi hỏi Dataset Runtime Assets mới hay thay đổi trọng số model.
