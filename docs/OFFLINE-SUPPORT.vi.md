# Bộ dữ liệu hỗ trợ offline và kiểm tra tính toàn vẹn

> 🌐 Ngôn ngữ / Language: [English](OFFLINE-SUPPORT.md) | **Tiếng Việt**

Notebook TPU sử dụng hai tài nguyên Kaggle attach riêng: **trọng số BF16 của mô hình** và **Full19 runtime support Dataset**. Dataset hỗ trợ chứa dependencies, mã tích hợp MaxText, ảnh đầu vào, case definitions và cache biên dịch tùy chọn; **không chứa trọng số mô hình**.

## Tài nguyên cần thiết

| Tài nguyên | Vai trò |
|---|---|
| [Kaggle Qwen3.8-27B BF16 Model](https://www.kaggle.com/models/dangkhoa2016/qwen-qwen3-8-27b) | Checkpoint thực sự được nạp khi chạy TPU |
| [Kaggle Full19 support Dataset](https://www.kaggle.com/datasets/dangkhoa2016/qwen38-27b-tpu-v5e8-full19-runtime-support) | Runtime offline và 16 ảnh đầu vào |
| [Notebook trong Git](../notebooks/kaggle-tpu-v5e8-production-showcase.ipynb) | Mã suy luận và ghi bằng chứng |

Manifest hỗ trợ đã được kiểm gồm **48 file**. SHA256 của `portable-bundle-manifest.json` là:

```text
cf5343f614eb32ff9234c870e4a26093f9f07e62199ee90396d516a804eb8a28
```

Notebook kiểm tra checksum này và tiếp tục xác minh kích thước, SHA256 của từng file **trước khi nạp trọng số**. Bản manifest trong Git nằm tại [support/portable-bundle-manifest.json](../support/portable-bundle-manifest.json).

## Cách sử dụng Dataset

1. Trên Kaggle, chọn **TPU v5e-8** và attach đủ hai tài nguyên nêu trên.
2. Chạy notebook trong Git. Notebook dò thư mục mount và kiểm `portable-bundle-manifest.json`.
3. Nếu mount theo layout khác, đặt `QWEN38_SUPPORT_DIR` đến thư mục chứa manifest.
4. Dừng khi checksum không khớp; không bỏ qua bước kiểm tra toàn vẹn.
5. Thu thập các file output và thời gian mới trong `/kaggle/working`.

Khả năng chạy còn phụ thuộc phiên bản JAX/MaxText, TPU topology và metadata cache. Manifest trùng khớp là điều kiện cần, **không bảo đảm** việc biên dịch hoặc suy luận sẽ thành công ở runtime khác.

## Quyền truy cập và giấy phép

Trọng số, wheels bên thứ ba, mã upstream đã sửa, ảnh và cache có thể chịu những điều kiện giấy phép hoặc truy cập khác nhau. Clone Git không tự tải hai tài nguyên Kaggle. Hãy xác minh tài khoản có thể truy cập cả hai và xem [thông báo bên thứ ba](../THIRD_PARTY_NOTICES.vi.md) trước khi phân phối lại.

[Hướng dẫn chạy](REPRODUCIBILITY.vi.md) · [Kiến trúc](ARCHITECTURE.vi.md) · [Manifest ảnh](../examples/assets_manifest.json)
