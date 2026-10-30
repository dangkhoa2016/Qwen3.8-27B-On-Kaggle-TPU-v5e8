# Kiến trúc notebook TPU trình diễn

🌐 Ngôn ngữ / Language: [English](NOTEBOOK-ARCHITECTURE.md) | **Tiếng Việt**

## Người dùng sẽ nhìn thấy gì?

Notebook `notebooks/kaggle-tpu-v5e8-production-showcase.ipynb` chỉ đóng vai trò **hướng dẫn và điều phối chạy**, không còn chép hàng nghìn dòng code triển khai. Notebook có 17 cell code: 16 bước inference/Gallery vẫn dùng module offline đã xác minh; cell 17 độc lập mở Cloudflare Quick Tunnel tạm thời. Mỗi bước vẫn có giải thích bằng tiếng Anh và tiếng Việt.

Người dùng Kaggle chỉ cần attach **hai nguồn**: model BF16 nguyên gốc và **Runtime Support Dataset phiên bản 1**. Không cần `git clone`, chép code thủ công hoặc attach Dataset thứ ba. **Chỉ cần bật Internet nếu muốn mở Tunnel ở bước 17.**

## Phân chia mã nguồn

| File/thư mục | Vai trò |
|---|---|
| `qwen38_runtime/workflow.py` | Quản lý thứ tự thực thi và chuyển trạng thái có tên giữa các cell |
| `qwen38_runtime/stages/preflight.py` | Xác minh model, manifest, 48 payload và thư mục kết quả |
| `qwen38_runtime/stages/dependencies.py` | Kiểm tra wheel offline và tương thích môi trường TPU Kaggle |
| `qwen38_runtime/stages/maxtext.py` | Khôi phục MaxText và áp dụng patch Vision đã ghim SHA256 |
| `qwen38_runtime/stages/inputs.py`, `checkpoint.py` | Kiểm tra ảnh, prompt và shard BF16 |
| `qwen38_runtime/stages/tpu.py`, `engine.py` | Khởi tạo tám thiết bị TPU, nạp một inference engine |
| `qwen38_runtime/stages/inference_helpers.py` | Tokenizer, xử lý ảnh, suy luận |
| `qwen38_runtime/stages/ocr_probe.py`, `ocr_evidence.py` | OCR đầu tiên, chẩn đoán AOT, lưu trace |
| `qwen38_runtime/stages/vision_reuse.py`, `image_cases.py` | Tái sử dụng executable cho 15 case ảnh tiếp theo |
| `qwen38_runtime/stages/text_cases.py` | Chạy 3 case văn bản, lưu JSON 19 case và cờ review |
| `qwen38_runtime/stages/vietnamese.py` và `gallery/vi_state.py` | Tùy chọn 17 case tiếng Việt, lưu và tiếp tục tiến độ |
| `qwen38_runtime/stages/gallery.py` và `gallery/render.py` | Tạo trang HTML offline từ kết quả thực tế |

Cell 17 import `gallery/quick_tunnel.py` từ source bundle đã xác minh trong Dataset v1; không còn nhúng mã Tunnel dài. Cell code đầu cho phép sửa hai mặc định Tunnel. Chỉ trong **phiên tương tác Kaggle** (`KAGGLE_KERNEL_RUN_TYPE=Interactive`) mới giữ URL 900 giây mặc định. Batch/API/CLI và môi trường không xác định bỏ qua Tunnel, không giữ thêm TPU. Có thể chỉnh `QWEN38_TUNNEL_HOLD_SECONDS` hoặc tắt chia sẻ tương tác bằng `QWEN38_TUNNEL_ENABLED=0`. Chỉ thư mục Gallery được phục vụ.

Bước `text_schedule` chỉ thông báo lịch chạy case văn bản, chưa thực hiện inference.

## Bảo đảm chạy offline trên Kaggle

Runtime Support Dataset **phiên bản 1** gồm bảy ZIP dữ liệu (48 payload ghim SHA256) và ZIP thứ tám `notebook_source.zip` (23 module Python), cùng `notebook-source-manifest.json`. Kaggle có thể giải nén source ZIP thành thư mục `notebook_source/`. Không có trọng số, output TPU hay kết quả điền sẵn.

Trước khi import module, cell đầu kiểm SHA256 của manifest mã nguồn, xác minh từng file trong ZIP hoặc thư mục Kaggle đã giải nén, từ chối đường dẫn và kiểu file không an toàn. Sau đó module preflight xác minh manifest dữ liệu và đủ 48 payload trước khi import JAX.

**Điều kiện bắt buộc:** Attach Runtime Support Dataset **phiên bản 1**, có `notebook-source-manifest.json` và `notebook_source.zip` **hoặc** `notebook_source/` gồm 23 module Python đã xác minh. SHA256 phải khớp notebook. Kiểm tra file attach trước khi chạy TPU; 48 payload dữ liệu vẫn độc lập.

## Đóng gói và kiểm thử


Khi đã có đủ **48 file phân phối gốc đã xác minh** trên máy, chạy:

```bash
python3 support/package_dataset.py \
  --source /path/to/verified-flat-support \
  --destination /path/to/new-empty-kaggle-dataset
```

Lệnh chỉ tạo file cục bộ, **không upload hoặc ghi đè** Kaggle Dataset. Kiểm tra hai manifest, tám ZIP và đủ 48 payload trước khi phân phối bộ dữ liệu hỗ trợ tương thích.


```bash
python3 support/sync_notebook_bootstrap.py --check
python3 support/sync_vi_state_notebook.py --check
python3 support/sync_gallery_notebook.py --check
python3 -m pytest -q tests
```

Khi sửa bất kỳ module nào trong gói nguồn, review diff rồi chạy `python3 support/sync_notebook_bootstrap.py --write` để cập nhật SHA256 ghim trong notebook; sau đó chạy lại test. Các phép đồng bộ source không thay đổi Kaggle Dataset đã phát hành hoặc trọng số model.
