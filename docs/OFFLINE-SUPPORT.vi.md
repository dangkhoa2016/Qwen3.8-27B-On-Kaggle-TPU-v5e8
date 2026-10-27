# Tài nguyên offline và kiểm chứng tính toàn vẹn

> 🌐 Ngôn ngữ / Language: [English](OFFLINE-SUPPORT.md) | **Tiếng Việt**

Thử nghiệm mô hình lớn trên TPU dễ tái lập hơn khi **trọng số, dependencies và ảnh đầu vào được quản lý tách biệt theo phiên bản**. Dự án attach model BF16 bằng Kaggle Model, còn runtime support bằng Dataset có SHA256.

## Bộ support gồm những gì?

| Thành phần | Mục đích |
|---|---|
| MaxText archive đã patch và patch source | Tái lập tích hợp Split-Vision cụ thể |
| Python wheels có phiên bản | Cài trong Kaggle hạn chế Internet |
| JAX helpers | Chuẩn bị và kiểm AOT/prefill |
| Sáu ảnh Rich + tám ảnh Photo + hai thẻ OCR | 16 nguồn ảnh dùng kiểm thử |
| Manifests và case definitions | Liên kết prompt, ảnh và SHA256 |
| Cache biên dịch tùy chọn | Có thể tăng tốc nếu runtime thực sự tương thích |

Manifest có **48 file nội dung**, ngoài ra là file manifest chính. **Không chứa trọng số BF16**.

## Hai phiên bản nguồn dữ liệu

| Snapshot input | Bộ ảnh | SHA256 manifest | Trạng thái |
|---|---|---|---|
| Snapshot Full19 gốc | Ảnh crossing đèn đỏ | `1447ee114dc2d2abce6f60f5125927f55311bf530b8cf53ab86a2576ac637ebc` | Đã kiểm 48/48 trong bộ support ghi nhận |
| Snapshot thay ảnh đầu vào | Ảnh crossing đèn xanh | `87d172b3b3e2fb4b478888665da164e6abe7cdff3f566b4699d396c919a2ec08` | Tải lại độc lập kiểm 48/48; cần evidence TPU riêng |

Ảnh crossing mới có SHA256 `ed532c2ba4e7eace4a81e4adfd6867c68f244fe4d64293f2bbfd1097ad7d0356`. Không đưa ảnh mới vào notebook đang ghim manifest cũ; cơ chế kiểm checksum sẽ phát hiện sự không khớp.

## Cách Kaggle mount Dataset

Kaggle có thể tự giải nén ZIP vào thư mục con. Notebook tìm `portable-bundle-manifest.json` trong các mount chuẩn và thư mục đã giải nén. Nếu dùng layout khác, đặt `QWEN38_SUPPORT_DIR` đến thư mục support thực tế.

ZIP support có thể giải nén vào thư mục riêng. Cần attach đúng tệp mà **notebook Git được duy trì** yêu cầu và kiểm SHA256 manifest trước Run All. Không coi tên Dataset tạm hay thư mục ZIP là API công khai được bảo đảm lâu dài.

Cell đầu tiên kiểm **fail-closed** kích thước, SHA256, đường dẫn tương đối, định dạng manifest và số file **trước khi nạp trọng số**. Nếu sai hoặc thiếu file, không bỏ qua bước kiểm.

## Vì sao Dataset vẫn Private?

Archive gồm wheels bên thứ ba, MaxText sửa đổi, ảnh và cache thuộc nhiều phạm vi giấy phép. Dataset Private không có nghĩa cộng đồng có thể tải hoặc phân phối lại mọi nội dung. Trước khi Public cần review giấy phép và provenance theo từng thành phần.

Các snapshot input **không phải training data** hoặc **kết quả production benchmark**. Full19 cũ phải gắn với SHA256 ảnh ban đầu. Kiểm checksum của ảnh thay thế không thay thế cho việc chạy TPU mới.

## Tạo bundle đã kiểm chứng cho dự án khác

1. Liệt kê file bắt buộc và loại mọi bí mật, dữ liệu cá nhân, file tạm.
2. Quản lý trọng số model riêng, ghi đúng revision.
3. Tính SHA256 từng file; ghi kích thước ảnh.
4. Đóng ZIP bất biến, kiểm lại sau giải nén và sau khi tải từ cloud.
5. Ghim hash manifest trong notebook, dừng nếu không khớp.
6. Chạy notebook trên TPU thật trước khi công bố số đo.
7. Khi được phép phân phối, công khai inventory giấy phép và dependencies.

[Chạy thử](REPRODUCIBILITY.vi.md) · [Giấy phép](../MODEL_LICENSE.vi.md) · [Đánh giá](ACCEPTANCE.vi.md)
