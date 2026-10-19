# History rewrite audit

> 🌐 Ngôn ngữ / Language: [English](HISTORY-REWRITE-PLAN.md) | **Tiếng Việt**

Dự án tái tạo **các commit có thật mà không bịa timestamp**. Lịch sử gốc lưu tại `backup/pre-governance-rewrite-20261019`. 11 commit gốc có thời điểm thực tế 18–19/10/2026. Có thể sửa danh tính tác giả thành `Đăng Khoa <i.am@dangkhoa.dev>` mà không giả định tài liệu mới đã tồn tại trong quá khứ.

Commit quản trị mới sử dụng timestamp tạo thật. Mỗi commit code nên có diff khoảng 600 dòng trở xuống (ngoại lệ ảnh/bằng chứng nhị phân hoặc cặp tài liệu dài).

Trước khi thay `main`: so sánh cây nguồn cuối, kiểm checksum evidence, kiểm commit examples mà notebook ghim còn tồn tại, chạy CI/E2E, giữ nhánh backup. Chỉ `--force-with-lease`, không dùng `--force` mù.
