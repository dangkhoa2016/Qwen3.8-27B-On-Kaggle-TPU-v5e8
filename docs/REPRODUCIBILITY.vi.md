# Hướng dẫn tái lập Qwen3.8-27B trên TPU

> 🌐 Ngôn ngữ / Language: [English](REPRODUCIBILITY.md) | **Tiếng Việt**

Tài liệu này hướng dẫn chuẩn bị môi trường để kiểm tra Text+Vision bằng mô hình **BF16 nguyên gốc trên Kaggle TPU v5e-8**. Các thao tác chuẩn bị đã xác minh được phân biệt rõ với kết quả **chỉ có thể xác nhận sau khi chạy TPU thật**.

## Chọn cách kiểm thử

| Cách chạy | Mục đích | Quyền truy cập / kết quả |
|---|---|---|
| CPU tests và mock UI | Kiểm HTTP, nhập liệu, giao diện | Không cần trọng số hoặc TPU |
| Full19 support gốc | Nghiên cứu thiết lập experimental 27/10 | Cần đúng các file support đã ghi; output 19/19 cũ được lưu riêng |
| Full19 với ảnh mới | Kiểm tra ảnh crossing đèn xanh | Cần notebook và support khớp SHA256 ảnh, cùng output TPU mới |

**Kaggle Model và các file support có thể cần quyền truy cập.** Tên và SHA256 bên dưới dùng để nhận diện input đã ghi, không cam kết mọi Dataset lịch sử luôn tải được công khai.

## Điều kiện trước khi chạy

- Kaggle TPU v5e-8 với **đúng tám thiết bị TPU**; một MaxEngine và một lần nạp BF16.
- Model thực sự được attach khi inference: [bản mirror BF16 trên Kaggle](https://www.kaggle.com/models/dangkhoa2016/qwen-qwen3-8-27b) (`dangkhoa2016/qwen-qwen3-8-27b`, Transformers/default/v1). Index BF16 có **18 shard**, **1.199 tensor** và **55.562.855.904 byte**.
- Dataset support khớp SHA256 manifest đã ghim trong notebook. **Notebook nguồn trong Git là tài liệu tham chiếu được duy trì**; nếu đổi ảnh đầu vào phải dùng manifest tương ứng và đo kết quả mới.
- Runtime tương thích. Mốc experimental đã ghi **JAX/JAXLIB 0.11.0** và **Transformers 5.12.1**.

| Dataset | Vai trò | SHA256 manifest |
|---|---|---|
| `Snapshot support experimental gốc` | Input thực nghiệm 27/10 | `1447ee114dc2d2abce6f60f5125927f55311bf530b8cf53ab86a2576ac637ebc` |
| `Snapshot support ảnh thay thế` | Ứng viên với ảnh crossing mới | `87d172b3b3e2fb4b478888665da164e6abe7cdff3f566b4699d396c919a2ec08` |

Bộ input ảnh mới đã được kiểm **48/48 SHA256**, nhưng điều đó **không chứng minh inference**. Tham khảo [notebook Full19 trong Git](../notebooks/kaggle-tpu-v5e8-production-showcase.ipynb) và support đã ghim để tái lập. Nếu thay ảnh đầu vào, cần đồng bộ manifest rồi chạy model lại.

## Bước 1 — Khám phá dự án không cần TPU

Từ repository đã clone:

```bash
python -m pip install -r requirements-cpu.txt
python -m pytest -q tests
npm ci
npx playwright install chromium
npm run test:e2e
```

Bộ test xác minh API, điều kiện nhập liệu và UI mock. Source mới nhất đã đạt **36 Python PASS, 1 SKIP và 2 Playwright PASS**. Các lệnh này **không chạy Qwen inference**.

## Bước 2 — Chuẩn bị notebook Kaggle

1. Bật **TPU v5e-8**; có thể tắt Internet khi đã attach bộ support offline.
2. Attach Kaggle Model BF16 gốc, không thay bằng GGUF/INT8.
3. Attach support khớp notebook và SHA256 manifest. Nếu dùng ảnh crossing mới, phải đồng bộ manifest và chạy lại; không ghép ảnh của snapshot này với hash của snapshot khác.
4. Kiểm tra Kaggle giải nén ZIP. Notebook tìm `portable-bundle-manifest.json` ở mount gốc hoặc thư mục con sau giải nén. Có thể dùng `QWEN38_SUPPORT_DIR` nếu mount khác.
5. Chạy **preflight đầu tiên**. Preflight kiểm 48 SHA256, kích thước file, patches, model index và đường dẫn **trước khi import JAX**.

## Bước 3 — Chạy và lưu số đo

Notebook dùng **prefill 320**, **target 768**, ảnh suy luận **512 × 384**. Chạy tuần tự các cell trong cùng phiên TPU và giữ thư mục evidence riêng theo thời gian dưới `/kaggle/working`.

Với mỗi case cần giữ SHA256 ảnh, prompt gốc, token IDs, câu trả lời nguyên văn, EOS, thời gian và exception. Ghi danh sách thiết bị, checksum model index/patch/manifest, phiên bản runtime và tổng wall time. Cache đúng checksum vẫn phải đi qua native JAX/XLA sharding/dtype validation.

## Bước 4 — Đối chiếu công bằng

Full19 experimental cũ hoàn thành **19/19 kiểm tra tự động với EOS trong 1.160,844 giây**. Tuy nhiên mô hình đọc sai thuế invoice và mô tả sai tỷ lệ dashboard. **Không được dùng output từ ảnh cũ để chứng minh ảnh mới**.

Chỉ ghi *hoàn thành* khi phiên chạy mới thực sự sinh đủ case. Chất lượng ngữ nghĩa phải được đánh giá riêng; lưu cả output sai và những lần chạy lỗi. Xem [đánh giá](ACCEPTANCE.vi.md) và [bằng chứng](FULL19-EVIDENCE.vi.md).

## Khắc phục sự cố

| Triệu chứng | Kiểm tra đầu tiên |
|---|---|
| Không thấy model | Đường dẫn attach, index và đủ 18 shard |
| Sai SHA256 support | Cặp notebook/Dataset, thư mục giải nén, phiên bản |
| Thiếu tám TPU | Thiết lập accelerator và backend JAX |
| Lỗi AOT/sharding | Image-token offset, shape, dtype, layout và patch |
| Không có EOS | Decode budget, token cuối và raw output |
| Nội dung sai | Ảnh gốc, prompt, rubric; tuyệt đối không sửa generated text |

[Kiến trúc](ARCHITECTURE.vi.md) · [Support](OFFLINE-SUPPORT.vi.md) · [Đánh giá](ACCEPTANCE.vi.md) · [Ảnh ví dụ](../examples/README.vi.md)
