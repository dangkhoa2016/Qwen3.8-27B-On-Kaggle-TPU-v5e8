# Bộ hỗ trợ offline có cấu trúc và kiểm tra SHA256

> 🌐 Ngôn ngữ / Language: [English](OFFLINE-SUPPORT.md) | **Tiếng Việt**

[Runtime Assets Dataset v1](https://www.kaggle.com/datasets/dangkhoa2016/qwen38-27b-tpu-v5e8-full19-runtime-assets) cung cấp 48 file runtime và ảnh offline đã xác minh cho [notebook TPU duy nhất](../notebooks/kaggle-tpu-v5e8-production-showcase.ipynb). Cần attach riêng [mô hình BF16 gốc trên Kaggle](https://www.kaggle.com/models/dangkhoa2016/qwen-qwen3-8-27b); Dataset không chứa trọng số model.

## Cấu trúc Dataset

Bảy gói ZIP tương ứng bảy thư mục chức năng:

| Gói | Thư mục giải nén | Nội dung |
|---|---|---|
| wheels.zip | wheels/ | Thư viện Python đã ghim phiên bản |
| source.zip | source/ | Source MaxText và bản vá |
| helpers.zip | helpers/ | Helper JAX layout/sharding và Split-Vision |
| images.zip | images/ | 16 ảnh nguồn |
| cases.zip | cases/ | Danh mục ảnh, prompt và định nghĩa tác vụ |
| cache.zip | cache/ | Cache compilation JAX tùy chọn và manifest |
| runtime.zip | runtime/ | Tham số engine và metadata wheel |

Thư mục gốc có manifest SHA256 và tài liệu song ngữ. Dataset hiện tại **phiên bản 1** gồm bảy ZIP runtime với **48 file đã xác minh**, không chứa Python source của dự án. GitHub cung cấp mã nguồn qua clone ẩn danh. Runtime Support Dataset v1 lịch sử với source ZIP được giữ làm bằng chứng riêng, không dùng cho notebook Git-first.

## Preflight kiểm tra dữ liệu

SHA256 manifest trung tâm là **fb2ee48dfe423733499112668d61e965b5d8d7466d52361b403c68dbc5768817**. [Manifest trong Git](../support/portable-bundle-manifest.json) ghi checksum bảy ZIP và **48 file dữ liệu**.

Trước khi nạp trọng số, notebook xác minh manifest và 48 file dữ liệu. Kaggle tự giải nén các ZIP thành bảy thư mục; ở chế độ này notebook đối chiếu SHA256 từng file trên mount chỉ đọc. Nếu người dùng attach các ZIP nguyên gốc, notebook kiểm thêm dung lượng, SHA256 và danh sách thành viên bảy archive rồi giải nén an toàn vào thư mục mới dưới /kaggle/working.

Cache JAX tùy chọn được kiểm tra SHA256 nhưng khi tái sử dụng vẫn phải tương thích TPU topology, tensor shape, sharding và dtype.

## Đóng gói và phân phối

[Script đóng gói trong Git](../support/package_dataset.py) nhận một thư mục support dạng phẳng đã kiểm chứng, tạo các ZIP và hướng dẫn theo nhóm. Ví dụ:

    python support/package_dataset.py --source /path/to/flat-support --destination /path/to/organized-dataset

Script từ chối file thiếu/sai SHA256 và không ghi đè thư mục output đã có dữ liệu. Việc phân phối có thể bị giới hạn bởi giấy phép wheel, source, ảnh và quyền truy cập Dataset trên Kaggle. Repository không chứa trọng số checkpoint hoặc các ZIP hỗ trợ.

[Hướng dẫn chạy](REPRODUCIBILITY.vi.md) · [Kiến trúc](ARCHITECTURE.vi.md)
