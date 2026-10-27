# Nhật ký thay đổi

> [English](CHANGELOG.md) | **Tiếng Việt**

## Chưa phát hành — chuẩn bị công bố (27/10/2026)
- Đồng bộ bộ ảnh Git thành **16 ảnh** được xác minh SHA256 và đã dùng trong Full19 experimental, thêm manifest Photography và năm ảnh AI.
- Ghi rõ Full19 Split-Vision BF16 đã **PASS 19/19 kiểm tra tự động** trên TPU thật trong 1.160,844 giây; không tuyên bố notebook public fresh Run All hay HTTP TPU đã nghiệm thu.
- Giữ lại lỗi đã xác nhận: Invoice đọc 8%/$69.28 thành 5%/$66.28 trong Full19 và cả hai lượt greedy; Dashboard mô tả sai tỷ lệ 114/31 là gần gấp đôi.
- Đã thay notebook Git cũ bằng nguồn Split-Vision 320/768 và đóng gói 48 mục support có SHA256. Vẫn cần upload/attach dataset và chạy fresh TPU Run All.

## 19/10/2026 — nhập repository
- Snapshot Git ban đầu đưa chung notebook, CPU API mock, Vue playground, tests, license/governance và README hai ngôn ngữ vào **một root commit**. Không coi snapshot này là chuỗi commit tính năng đã kiểm chứng.
