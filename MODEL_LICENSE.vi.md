# Giấy phép mô hình, nguồn gốc và ghi công

> [English](MODEL_LICENSE.md) | **Tiếng Việt**

## Phạm vi giấy phép độc lập

| Thành phần | Chủ sở hữu / nguồn | Giấy phép |
|---|---|---|
| Mã nguồn và tài liệu do dự án tạo ra | Đăng Khoa | [MIT](LICENSE) |
| Trọng số và mô hình gốc Qwen/Qwen3.8-27B | Qwen upstream | [Apache License 2.0](THIRD_PARTY_LICENSES/QWEN3.8-27B-APACHE-2.0) |
| MaxText, JAX, FastAPI, Vue và các thư viện khác | Các bên phát hành tương ứng | Giấy phép riêng ([thông báo](THIRD_PARTY_NOTICES.vi.md)) |

**Repository đã lưu toàn văn giấy phép mô hình** tại [THIRD_PARTY_LICENSES/QWEN3.8-27B-APACHE-2.0](THIRD_PARTY_LICENSES/QWEN3.8-27B-APACHE-2.0). Đây là bản sao nguyên văn giấy phép upstream, không phải giấy phép mới do dự án cấp.

Nguồn chính thức: [model card Qwen](https://huggingface.co/Qwen/Qwen3.8-27B), [LICENSE upstream](https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/LICENSE), [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0). Model card công bố apache-2.0 tại thời điểm rà soát. Cần đối chiếu upstream và thông báo bổ sung trước khi phân phối lại.

## Phân phối và giới hạn

Trọng số 27B được đính kèm qua Kaggle Model riêng; **Git repository không chứa pretrained weights**. Giấy phép MIT dành cho mã nguồn dự án không chuyển quyền sở hữu hay thay thế giấy phép model. Khi phân phối lại model hoặc bản sửa đổi, phải bảo toàn các văn bản giấy phép, thông tin bản quyền, ghi công và thông báo sửa đổi thích hợp. Cần rà soát riêng Kaggle mirror và bộ MaxText offline-support có patch. Tài liệu này không bảo đảm pháp lý cho mọi hoạt động phân phối ở phía người dùng.

Đây là dự án trình diễn kỹ thuật độc lập, không phải sản phẩm chính thức hoặc được Qwen bảo trợ.
