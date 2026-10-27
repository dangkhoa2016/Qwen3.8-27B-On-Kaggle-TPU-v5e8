# Hướng dẫn tái lập và chạy Kaggle mới

> [English](REPRODUCIBILITY.md) | **Tiếng Việt**

## Phạm vi kết quả đã xác minh

Kết quả Full19 **19/19 đạt kiểm tra tự động** ngày 27/10/2026 được đo trên TPU v5e-8 thật trong 1.160,844 giây, sử dụng một **notebook thử nghiệm riêng**. Đây **chưa phải** fresh Run All từ notebook trong Git. Cần phân biệt hai cột mốc. Xem [nghiệm thu](ACCEPTANCE.vi.md) và [evidence](FULL19-EVIDENCE.vi.md).

## Điều kiện trước khi chạy notebook chính thức

1. **Checkpoint:** attach model `dangkhoa2016/qwen-qwen3-8-27b`, đúng Transformers/default/v1; xác minh index, 1.199 tensor, 18 shard BF16 và 55.562.855.904 bytes khai báo.
2. **Support:** attach dataset offline tương thích dưới dạng Kaggle input thực. Lượt PASS dùng 38 tệp được kiểm SHA256 ở `/kaggle/working/qwen38-production-rc-20261025/private-support-v3`. Đây là đường dẫn **chỉ có trong session hiện tại**, không được coi là resource tồn tại trong fresh notebook.
3. **Ảnh đầu vào:** dùng đủ 16 ảnh tại [examples](../examples/README.md), xác minh [SHA256](../examples/assets_manifest.json), 6 case Rich Vision và 8 Photography; lúc inference resize về 512×384. Đây là ảnh đầu vào, **không phải hình do Qwen tạo**.
4. **Nguồn và dependencies:** pin Git commit thực sự có thể truy cập, xác minh source/patch MaxText; dùng JAX/JAXLIB 0.11.0 và Transformers 5.12.1. Không giả định clone GitHub Private ẩn danh thành công. Cần authentication hoặc cách đính kèm dữ liệu độc lập mạng.
5. **TPU:** đúng 8 devices, BF16 gốc (không GGUF/INT8), một MaxEngine, một lần nạp 18 shards. Không Stop/Restart TPU chờ lâu nếu chưa có yêu cầu rõ ràng.
6. **Biên dịch và inference:** giữ kiểm tra JAX native sharding/layout. Dùng Split-Vision prefill 320, target 768, compiled executable được xác minh và greedy decoding. Rich/Photography giới hạn 448 token. Cache lịch sử không bảo đảm dùng được giữa các cấu hình.
7. **Bằng chứng:** ghi Git commit, model/support/ảnh SHA256, token IDs, câu trả lời nguyên văn, EOS, thời gian compile/cold generate/từng case, tổng runtime, seed/RNG provenance. Không gán một seed đã xác minh nếu mới chỉ ghi nhận việc split RNG của engine.
8. **Đánh giá:** giữ mọi output lỗi, xem trực tiếp ảnh gốc, công khai lỗi số học Dashboard và lỗi thuế Invoice; nghiệm thu HTTP TPU là dự án kiểm tra riêng.

## Blocker hiện tại

- Notebook Git cũ còn đường prefill ảnh trực tiếp và 256/384; cần thay mới và kiểm tra CPU.
- Bộ support 38 file đang ở thư mục private /kaggle/working, chưa xác minh có mặt tại mount dataset mới.
- Repository đang Private; lệnh clone Git ẩn danh từ notebook cũ không phù hợp.
- Chưa hợp nhất runner/evidence riêng thành một notebook production có thể fresh Run All.
- HTTP TPU thật chưa nghiệm thu end-to-end.

Kết quả experimental Full19 đã có không bị xóa hoặc gắn nhãn thành fresh run.
