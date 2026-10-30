# Tái lập suy luận Qwen3.8-27B BF16 Text + Vision trên Kaggle TPU v5e-8

> 🌐 Ngôn ngữ / Language: [English](REPRODUCIBILITY.md) | **Tiếng Việt**

Hướng dẫn này sử dụng [notebook Full19 trong Git](../notebooks/kaggle-tpu-v5e8-production-showcase.ipynb) cùng các tài nguyên Kaggle tương ứng. Bộ thử gồm 19 case trên tám thiết bị TPU v5e. CPU tests và giao diện mock có thể chạy độc lập nhưng không thực thi mô hình 27B.

## Điều kiện cần

- Kaggle Notebook chọn **TPU v5e-8**, đủ tám TPU.
- [Kaggle Model Qwen3.8-27B BF16 nguyên gốc](https://www.kaggle.com/models/dangkhoa2016/qwen-qwen3-8-27b) — checkpoint thực thi, Transformers/default/v1.
- [Full19 support Dataset](https://www.kaggle.com/datasets/dangkhoa2016/qwen38-27b-tpu-v5e8-full19-runtime-support) — manifest 48 file dữ liệu, ảnh và runtime offline. Attach **phiên bản 1** có `notebook-source-manifest.json` và `notebook_source.zip` **hoặc** thư mục `notebook_source/` Kaggle đã giải nén (23 file Python), khớp SHA256 notebook. **Quyền truy cập:** kiểm tra Dataset, quyền và file trong tài khoản Kaggle trước khi chạy TPU.
- Yêu cầu runtime được ghim: **Python 3.12, JAX/JAXLIB 0.11.0, Flax 0.12.8 và Transformers 5.12.1**, tám thiết bị TPU v5e và manifest 48 file đúng SHA256. Không khẳng định các phiên bản runtime khác tương thích.

SHA256 manifest: `fb2ee48dfe423733499112668d61e965b5d8d7466d52361b403c68dbc5768817`.

## Chạy notebook trên TPU

1. Import [notebook trong Git](../notebooks/kaggle-tpu-v5e8-production-showcase.ipynb) vào Kaggle, chọn TPU v5e-8 và bật Internet nếu muốn chia sẻ Gallery tại cell 17.
2. Attach Kaggle BF16 Model và Full19 support Dataset. Việc tải các tài nguyên phụ thuộc quyền truy cập và trạng thái cung cấp trên Kaggle.
3. Chạy lần lượt các cell, không restart kernel giữa chừng. Preflight kiểm model, manifest và **SHA256 của 48 file**; hỗ trợ cả các thư mục Kaggle tự giải nén và bộ ZIP nguyên gốc.
4. Notebook ghi output vào thư mục riêng theo UTC dưới `/kaggle/working`.
5. Kiểm tra raw text, token IDs, EOS, cảnh báo chất lượng, lỗi và thời gian; phân biệt đủ 19 case với đạt EOS.

Cấu hình gồm một MaxEngine, một lần nạp checkpoint, ảnh xử lý 512 × 384, prefill 320 token, target 768 token và tối đa 448 bước decode cho Rich Vision/Photography. Cache đúng checksum vẫn phải qua kiểm tra native JAX/XLA.

## Suy luận tiếng Việt và xuất HTML tĩnh tùy chọn

19 case tiếng Anh được lưu trước. Notebook **mặc định bật** 17 suy luận tiếng Việt (3 Text, 6 Rich Vision, 8 Photography); mỗi thẻ OCR số tái sử dụng kết quả chữ số của chính case đó để hiển thị VI (không chạy thêm OCR). Đặt QWEN38_ENABLE_VI=0 trước Run All để bỏ qua tiếng Việt. Đánh giá EOS và chất lượng câu trả lời tiếng Việt từ output thật; prompt giữ nguyên câu hỏi tiếng Anh và nối thêm chỉ dẫn chỉ trả lời tiếng Việt, không phải benchmark prompt gốc tiếng Việt. Không dùng keyword checks tiếng Anh để chấm tiếng Việt.

Xuất HTML mặc định bật (QWEN38_RENDER_HTML=1), chạy trên CPU sau suy luận và tạo inference-gallery.zip. **Cell 17** có thể mở URL Cloudflare HTTPS tạm thời ngay từ Kaggle (bật Internet). Chỉ phiên chỉnh sửa tương tác Kaggle mới mở Tunnel, giữ URL tối đa 900 giây theo mặc định. API/CLI và Save & Run All dạng batch tự bỏ qua Tunnel, không giữ thêm TPU để chia sẻ link; tải Gallery ZIP sau khi chạy. **Hoặc**, tải ZIP và host trên máy CPU riêng sau khi dừng TPU. Chỉnh mặc định Tunnel ở cell code đầu; xem [hướng dẫn Gallery](../gallery/README.vi.md).

## Kiểm thử thành phần CPU

Từ thư mục gốc repository:

```bash
python -m pip install -r requirements-cpu.txt
python -m pytest -q tests
npm ci
npx playwright install chromium
npm run test:e2e
```

Các lệnh kiểm API, guard dữ liệu và frontend CPU mock, **không** chạy suy luận Qwen trên TPU.

## Khi Kaggle thay đổi runtime

Chạy kiểm tra manifest và dependencies trước các cell nặng trên TPU. Nếu gặp `BASELINE_RUNTIME_MISMATCH`, dừng và ghi lại phiên bản thực tế. Không pip-replace JAX/JAXLIB trong phiên TPU đang chạy và không bỏ qua SHA256. Cần dựng lại và nghiệm thu riêng bundle tương thích trước khi tuyên bố có thể tái lập.

## Xử lý lỗi thường gặp

| Hiện tượng | Kiểm tra |
|---|---|
| Thiếu model | Kaggle model attachment, index, đủ 18 shard |
| Manifest không khớp | Mount Dataset, thư mục giải nén, SHA256 |
| Không đủ TPU | Accelerator, danh sách thiết bị JAX |
| AOT không biên dịch | Runtime, shape, sharding, dtype và cache |
| Thiếu EOS hoặc trả lời sai | Budget, prompt, SHA256 ảnh và raw output |

[Support Dataset](OFFLINE-SUPPORT.vi.md) · [Đánh giá](ACCEPTANCE.vi.md) · [Hiệu năng](PERFORMANCE-ANALYSIS.vi.md)


## Dựng lại Gallery bằng CPU và kiểm tra nhất quán mã nguồn

Bộ dựng HTML chuẩn là gallery/render.py. Cell cuối notebook gọi
module offline đã xác minh qua workflow.run('gallery') tại cell 16; dùng support/sync_gallery_notebook.py --check
trong CI và --write khi cần đồng bộ lời gọi cùng SHA256.

Dựng lại HTML/CSS/JavaScript/ghi chú có SHA256 bằng JSON EN/VI
và 16 ảnh nguồn thật, không cần TPU hay GPU. Render Gallery trên CPU không thực hiện suy luận TPU. Trước khi chia
sẻ cần xác minh SHA256 EN/VI, đủ 16 ảnh, evidence.json và kiểm thử
trình duyệt.

Ghi chú đánh giá phải khớp SHA256 của kết quả thực thi tương ứng;
case chưa được review phải hiển thị Chưa đánh giá.
