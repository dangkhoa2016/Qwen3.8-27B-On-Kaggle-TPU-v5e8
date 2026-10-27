# Bộ Full19 offline support — cách attach và chứng minh nguồn gốc

> [English](OFFLINE-SUPPORT.md) | **Tiếng Việt**

Notebook production cần attach độc lập hai tài nguyên: Kaggle Model với BF16 nguyên gốc và một Dataset offline-support. Gói support **không chứa trọng số model 27B**.

## Bundle chuẩn bị ngày 27/10/2026

- Thư mục trên Kaggle hiện tại: qwen38-tpu-v5e8-full19-offline-support-20261027
- **38 file support gốc** giữ nguyên và được xác minh SHA256: MaxText đã patch, runtime wheels, helper engine, 16 ảnh nguồn cùng manifests.
- **6 helper** gồm patch Split-Vision và năm helper giám sát JAX/XLA native/strict reuse.
- **3 tệp cache biên dịch ứng viên** của phiên trước, cộng một manifest cache.
- **48 mục dữ liệu**, tổng 125.595.148 byte, đều đã xác minh SHA256.
- Manifest gốc: portable-bundle-manifest.json
- **SHA256 manifest:** 1447ee114dc2d2abce6f60f5125927f55311bf530b8cf53ab86a2576ac637ebc
- ZIP portable chứa 49 mục: 48 tệp dữ liệu và manifest gốc, chỉ nằm tại máy Kaggle hiện tại, không phải link public.

Manifest khóa nội dung chính xác của notebook mới. Bộ 38 file vẫn có thêm kiểm tra qua private-rc-sha256-manifest.json.

## Upload, attach và thực thi

1. Upload **nội dung thư mục** thành một Kaggle Dataset mới, giữ **Private** và không tạo thư mục lồng thêm. Slug đề xuất: qwen38-tpu-v5e8-full19-offline-support. **Chưa thực hiện/xác nhận upload** dataset này. Phải kiểm tra lại SHA256 sau khi attach.
2. Attach riêng Kaggle Model dangkhoa2016/qwen-qwen3-8-27b, Transformers/default/v1 với 18 shards safetensors BF16.
3. Attach support dataset; notebook tìm các mount trong /kaggle/input/datasets/dangkhoa2016/ hoặc /kaggle/input/, hoặc đặt QWEN38_SUPPORT_DIR trỏ vào thư mục chính xác.
4. Chạy một lần trên TPU v5e-8 (8 thiết bị). Giai đoạn trước nạp model kiểm checksum toàn bộ, evidence mỗi lần chạy vào thư mục mới có timestamp.
5. Ba cache cũ được copy sang thư mục writable riêng. Chúng có thể bị bỏ qua hoặc không tương thích trên TPU khác; **không cam kết cache hit**, không bỏ native sharding/layout checks.
6. Lưu output/token IDs/EOS/timing/lỗi thật. Đã nghiệm thu experimental Full19; **fresh Run All chính thức vẫn chưa diễn ra**.

**Lưu ý phân phối:** Giấy phép Apache-2.0 của model không tự cấp quyền phát hành mã MaxText đã patch, các wheel bên thứ ba hoặc ảnh AI. Phải review quyền phân phối trước khi công khai Dataset. Không đưa token, thông tin đăng nhập hoặc model weights vào gói.

[Tái lập](REPRODUCIBILITY.vi.md) · [Nghiệm thu](ACCEPTANCE.vi.md) · [Evidence](FULL19-EVIDENCE.vi.md)
