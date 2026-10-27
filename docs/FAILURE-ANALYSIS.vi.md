# Bài học từ lỗi mô hình và những bẫy đánh giá

> 🌐 Ngôn ngữ / Language: [English](FAILURE-ANALYSIS.md) | **Tiếng Việt**

Một demo inference có giá trị khi người xem kiểm chứng được cả lỗi. Full19 đã cho thấy **chương trình sinh output thành công, khớp keyword và có EOS vẫn chưa đủ chứng minh nội dung chính xác**.

## Trường hợp 1 — Thuế và con số trên hóa đơn

Ảnh hóa đơn văn phòng ghi:

| Trường | Giá trị trên ảnh | Model mô tả |
|---|---|---|
| Subtotal | **$866.00** | Giữ đúng |
| Thuế | **8% / $69.28** | **5% / $66.28** |
| Total | **$935.28** | Giữ đúng |

Hai lượt greedy với cùng ảnh và prompt vẫn sinh sai thuế, đều có EOS và **171 token IDs** giống hệt nhau.

**Bài học:** phải đọc đúng nhãn thuế và con số trên ảnh, không chọn tỷ lệ nghe có vẻ hợp lý. Kiểm toán học có thể phát hiện mâu thuẫn, nhưng không được âm thầm sửa raw output của model.

## Trường hợp 2 — Ngôn ngữ so sánh

Dashboard có **114 đơn Thứ Bảy**, **31 đơn Thứ Hai**. Mô hình nói “gần gấp đôi”, trong khi:

`114 / 31 ≈ 3,68`

Nhận diện từng con số đúng chưa có nghĩa mô tả quan hệ đúng. **Bài học:** đánh giá numerical grounding phải kiểm cả phép so sánh.

## Trường hợp 3 — Bộ chấm điểm cũng có bug

Regex validator trước đây khớp từ `Total` ngay trong `Subtotal`. Phiên bản validator sau đã sửa. Đó là **sửa bài kiểm tra**, không phải model được cải thiện hồi tố.

Cần tách **dữ liệu thật trên ảnh**, **văn bản model sinh** và **kết quả kiểm tự động**. Dù sửa validator, raw model output vẫn phải giữ nguyên.

## Trường hợp 4 — SHA256 ảnh quyết định ý nghĩa bằng chứng

Ảnh crossing cũ có **đèn người đi bộ đỏ** khi nhiều người qua vạch. Qwen đã nhận ra và cảnh báo nguy hiểm. Git hiện dùng **ảnh đèn xanh khác**, SHA256 khác. Không thể dùng câu trả lời cũ để đánh giá ảnh mới.

**Bài học:** filename hay case ID không đủ chứng minh ảnh nào đã được đưa vào model. Cần ghi SHA256 của input.

## Các biện pháp hữu ích

- Ghi ground truth cho từng case, nhất là số liệu và chi tiết giao thông.
- Tách kiểm OCR số, phép tính, quan hệ và độ trung thực của diễn đạt.
- Lưu mọi output lỗi, không chỉ chọn kết quả đẹp để công bố.
- Ghim ảnh/prompt, token IDs, EOS và timing cùng nhau.
- Greedy repeat kiểm tính tái lập, không phải nhiều mẫu ngẫu nhiên độc lập.
- Phân biệt chương trình hoàn thành, nội dung đúng và hệ thống sẵn sàng production.

Các ví dụ giúp thiết kế benchmark OCR, image understanding và LLM regression tốt hơn. Không nên suy rộng rằng mọi output Qwen đều mắc những lỗi này.

[Đánh giá](ACCEPTANCE.vi.md) · [Bằng chứng](FULL19-EVIDENCE.vi.md) · [Ảnh đầu vào](../examples/README.vi.md)
