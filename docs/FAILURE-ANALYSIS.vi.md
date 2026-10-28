# Hành vi mô hình: nghiên cứu OCR và suy luận hình ảnh

> 🌐 Ngôn ngữ / Language: [English](FAILURE-ANALYSIS.md) | **Tiếng Việt**

Full19 có những trường hợp mô hình hoàn tất sinh token nhưng nội dung vẫn sai. Các quan sát này giúp tách dữ liệu thật trên ảnh, câu trả lời nguyên văn và kết quả kiểm tự động.

## Trường hợp 1 — Thuế hóa đơn

| Trường | Giá trị thấy trên ảnh | Model trả về |
|---|---|---|
| Subtotal | $866.00 | Đúng |
| Thuế | **8% / $69.28** | **5% / $66.28** |
| Total | $935.28 | Đúng |

Hai lượt greedy thêm với cùng đầu vào tiếp tục trả sai giá trị thuế và vẫn đạt EOS. Cần giữ output nguyên gốc thay vì âm thầm sửa thành số đúng.

## Trường hợp 2 — So sánh số liệu

Dashboard có **114** đơn Thứ Bảy và **31** đơn Thứ Hai. Model diễn đạt là “gần gấp đôi”, trong khi `114 / 31 ≈ 3,68`. Đọc đúng từng giá trị không có nghĩa mô tả mối quan hệ giữa chúng đúng.

## Trường hợp 3 — Giới hạn của bộ kiểm tự động

Khớp keyword hay regex có thể đạt dù số liệu sai hoặc giải thích gây hiểu nhầm. Cần lưu riêng ground truth trên ảnh, raw output do model sinh và kết quả scorer. Khi cải thiện validator, không được thay đổi output gốc.

## Trường hợp 4 — Provenance ảnh đầu vào

Kết luận thị giác phụ thuộc đúng ảnh mà mô hình đã nhận. Tên file hoặc case ID không đủ để nhận diện nội dung; cần ghim SHA256 và kích thước ảnh nguồn cho từng kết quả. Khi byte ảnh thay đổi phải tạo bằng chứng mới, không dùng câu trả lời của ảnh khác.

Đây là ví dụ về phương pháp đánh giá, **không** phải kết luận mọi phản hồi của Qwen đều sai hoặc khẳng định một tỷ lệ accuracy tổng quát.

[Đánh giá](ACCEPTANCE.vi.md) · [Full19](FULL19-EVIDENCE.vi.md) · [Ảnh đầu vào](../examples/README.vi.md)
