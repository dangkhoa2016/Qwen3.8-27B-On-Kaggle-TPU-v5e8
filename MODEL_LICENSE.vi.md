# Nguồn gốc mô hình và hướng dẫn giấy phép

> 🌐 Ngôn ngữ / Language: [English](MODEL_LICENSE.md) | **Tiếng Việt**

Dự án minh họa cách chạy **Qwen3.8-27B trên Kaggle TPU v5e-8**. Mỗi thành phần có tác giả và điều khoản upstream riêng; không nên coi tất cả đều thuộc một giấy phép.

## Phạm vi từng giấy phép

| Thành phần | Nguồn | Phạm vi |
|---|---|---|
| Mã nguồn/tài liệu do dự án viết | Dự án này | [MIT](LICENSE) |
| Mô hình pretrained Qwen3.8-27B | Qwen upstream | [Bản sao giấy phép Apache 2.0](THIRD_PARTY_LICENSES/QWEN3.8-27B-APACHE-2.0) |
| MaxText, JAX, wheels, Vue và FastAPI | Các nhóm phát triển tương ứng | Điều khoản và notices riêng |
| Ảnh ví dụ và bộ runtime support | Từng nguồn riêng | Cần kiểm quyền phân phối từng thành phần |

Bản Apache-2.0 kèm theo ghi nhận giấy phép của Qwen upstream, **không** phải giấy phép riêng do dự án cấp. Checkpoint được cấu hình để attach khi chạy TPU là bản mirror Kaggle ở phần dưới; việc ghi công tác giả gốc không có nghĩa notebook tải trọng số trực tiếp từ Hugging Face.

## Model được cung cấp bằng cách nào?

Notebook được cấu hình để attach [bản mirror BF16 trên Kaggle](https://www.kaggle.com/models/dangkhoa2016/qwen-qwen3-8-27b) (`dangkhoa2016/qwen-qwen3-8-27b`, Transformers/default/v1), mount tại `/kaggle/input/models/dangkhoa2016/qwen-qwen3-8-27b/transformers/default/1`. **Git không chứa pretrained weights.** Đây là bản mirror của checkpoint Qwen upstream, không phải model mới huấn luyện hoặc endpoint Hugging Face được gọi trực tiếp.

Khi phân phối runtime đã sửa hoặc model derivative, phải kiểm phiên bản chính xác, giữ file copyright/license/NOTICE áp dụng và đánh dấu sửa đổi khi giấy phép yêu cầu. Xem [thông báo bên thứ ba](THIRD_PARTY_NOTICES.vi.md).

## Tái sử dụng có trách nhiệm

Cộng đồng có thể học từ kiến trúc, API, tests và cách đánh giá mà không nhầm đây là hệ thống Qwen chính thức. Trước khi phân phối Dataset support cần kiểm quyền đối với **mọi** wheel, MaxText đã patch, ảnh tổng hợp và cache.

Đây là tài liệu ghi công kỹ thuật, không thay thế tư vấn pháp lý hay bảo đảm mọi cách phân phối đều hợp lệ.
