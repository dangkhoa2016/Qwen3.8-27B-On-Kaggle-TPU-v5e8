# Đóng góp cho dự án Text + Vision trên TPU

> 🌐 Ngôn ngữ / Language: [English](CONTRIBUTING.md) | **Tiếng Việt**

Dự án hoan nghênh những đóng góp giúp **dễ hiểu, dễ tái lập và dễ đánh giá suy luận TPU**. Bạn không cần có TPU mới đóng góp được: test API, tài liệu, accessibility và cách đánh giá ảnh đều có thể kiểm trên CPU.

## Những đóng góp hữu ích

- Giải thích JAX/MaxText shapes, MRoPE, sharding hoặc compilation bằng ví dụ chạy được.
- Thêm regression test cho request validation, SHA256 ảnh, lỗi và trải nghiệm trình duyệt.
- Cải thiện case Full19, ground truth và cách kiểm các phép so sánh số học.
- Nâng chất lượng tài liệu tiếng Anh/Việt nhưng không thay đổi bằng chứng nguyên bản được thu thập khi thực thi.
- Làm rõ giấy phép upstream và phiên bản dependencies.

## Trước khi gửi Pull Request

1. Mô tả vấn đề hoặc cải thiện muốn thực hiện.
2. Tạo nhánh từ default branch và sửa từng phần vừa đủ để review.
3. Giải thích **lợi ích cho người sử dụng**, không chỉ liệt kê tên file.
4. Chạy test liên quan:

```bash
python -m pip install -r requirements-cpu.txt
python -m pytest -q tests
npm ci
npx playwright install chromium
npm run test:e2e
```

5. Nếu liên quan inference TPU, ghi rõ **đã chạy TPU thật hay chưa**. Kèm topology, runtime, SHA256 input, revision notebook và evidence gốc. CPU mock PASS không thay thế suy luận TPU.
6. Nếu trang có cặp EN/VI, cập nhật cả hai bản.
7. Không đưa tokens, mật khẩu, dữ liệu cá nhân, trọng số không được phép, file Private khó truy cập hoặc output giả vào Git.

## Nguyên tắc bằng chứng

Lưu prompt và SHA256 ảnh, generated token IDs/text, EOS, thời gian và cả lỗi. Nếu sửa validator, cần nói rõ là **logic chấm điểm** hay **thực thi model**; không sửa nội dung model sinh để tạo PASS.

## Trao đổi

Dùng GitHub issues để chia sẻ ý tưởng và bug tái lập được. Nếu phát hiện lỗ hổng hoặc lộ credentials, xem [chính sách bảo mật](SECURITY.vi.md), không đăng công khai.

Cảm ơn bạn đã giúp việc nghiên cứu mô hình lớn trên TPU dễ tiếp cận hơn.
