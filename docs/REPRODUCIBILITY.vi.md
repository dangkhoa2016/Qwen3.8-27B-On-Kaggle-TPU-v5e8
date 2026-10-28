# Tái lập suy luận Qwen3.8-27B BF16 Text + Vision trên Kaggle TPU v5e-8

> 🌐 Ngôn ngữ / Language: [English](REPRODUCIBILITY.md) | **Tiếng Việt**

Hướng dẫn này sử dụng [notebook Full19 trong Git](../notebooks/kaggle-tpu-v5e8-production-showcase.ipynb) cùng các tài nguyên Kaggle tương ứng. Bộ thử gồm 19 case trên tám thiết bị TPU v5e. CPU tests và giao diện mock có thể chạy độc lập nhưng không thực thi mô hình 27B.

## Điều kiện cần

- Kaggle Notebook chọn **TPU v5e-8**, đủ tám TPU.
- [Kaggle Model Qwen3.8-27B BF16 nguyên gốc](https://www.kaggle.com/models/dangkhoa2016/qwen-qwen3-8-27b) — checkpoint thực thi, Transformers/default/v1.
- [Full19 support Dataset](https://www.kaggle.com/datasets/dangkhoa2016/qwen38-27b-tpu-v5e8-full19-runtime-support) — manifest 48 file, ảnh đầu vào và runtime offline.
- Runtime tương thích, gồm JAX/JAXLIB 0.11.0 và Transformers 5.12.1 theo cấu hình đã ghim.

SHA256 manifest: `cf5343f614eb32ff9234c870e4a26093f9f07e62199ee90396d516a804eb8a28`.

## Chạy notebook trên TPU

1. Import [notebook trong Git](../notebooks/kaggle-tpu-v5e8-production-showcase.ipynb) vào Kaggle, bật TPU v5e-8.
2. Attach Kaggle BF16 Model và Full19 support Dataset. Việc tải các tài nguyên phụ thuộc quyền truy cập và trạng thái cung cấp trên Kaggle.
3. Chạy lần lượt các cell, không restart kernel giữa chừng. Preflight kiểm model, manifest và **SHA256 của 48 file**.
4. Notebook ghi output vào thư mục riêng theo UTC dưới `/kaggle/working`.
5. Đối chiếu toàn bộ câu trả lời, token IDs, EOS, lỗi và thời gian của từng case trước khi kết luận hoàn tất.

Cấu hình gồm một MaxEngine, một lần nạp checkpoint, ảnh xử lý 512 × 384, prefill 320 token, target 768 token và tối đa 448 bước decode cho Rich Vision/Photography. Cache đúng checksum vẫn phải qua kiểm tra native JAX/XLA.

**Bằng chứng thực thi:** Lượt Full19 đo ngày 27/10/2026 đã hoàn thành 19/19 đường sinh token với EOS trong 1.160,844 giây; SHA256 đầu vào và phiên bản mã của lượt đo thuộc hồ sơ tương ứng. Khi chạy notebook hiện tại phải tạo output riêng, không chép số liệu benchmark thành kết quả mới. Sinh token thành công không đồng nghĩa trả lời chính xác.

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

## Xử lý lỗi thường gặp

| Hiện tượng | Kiểm tra |
|---|---|
| Thiếu model | Kaggle model attachment, index, đủ 18 shard |
| Manifest không khớp | Mount Dataset, thư mục giải nén, SHA256 |
| Không đủ TPU | Accelerator, danh sách thiết bị JAX |
| AOT không biên dịch | Runtime, shape, sharding, dtype và cache |
| Thiếu EOS hoặc trả lời sai | Budget, prompt, SHA256 ảnh và raw output |

[Support Dataset](OFFLINE-SUPPORT.vi.md) · [Đánh giá](ACCEPTANCE.vi.md) · [Hiệu năng](PERFORMANCE-ANALYSIS.vi.md)
