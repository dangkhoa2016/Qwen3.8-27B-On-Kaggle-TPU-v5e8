# Kiểm tra trích xuất tài liệu ảnh và suy luận số liệu

🌐 Ngôn ngữ / Language: [English](FAILURE-ANALYSIS.md) | **Tiếng Việt**

Đây là **tiêu chí review** dành cho kết quả được sinh sau khi chạy
notebook Full19 chính thức. Các giá trị bên dưới lấy từ ảnh đầu vào,
**không phải khẳng định model đã trả lời như thế nào**.

## Kiểm tra hóa đơn

[Xem hóa đơn văn phòng](../examples/images/northstar_office_supply_invoice.jpg).

| Trường cần kiểm | Giá trị trên ảnh |
|---|---|
| Nhà cung cấp | Northstar Office Supply |
| Tiền hàng | $866.00 |
| Thuế suất | 8% |
| Tiền thuế | $69.28 |
| Tổng thanh toán | $935.28 |

Đối chiếu câu trả lời với các trường trên ảnh và kiểm phép tính
866.00 × 8% = 69.28, 866.00 + 69.28 = 935.28.
Câu trả lời trôi chảy hoặc đúng tổng chưa bảo đảm trích xuất thuế chính xác.

## Kiểm tra dashboard giao hàng

[Xem dashboard](../examples/images/vision_2d_city_bike_dashboard.jpg).
Kiểm tra KPI, số liệu theo ngày và tỷ trọng danh mục.
Tỷ số giữa 114 và 31 xấp xỉ **3,68**; không diễn giải thành
“gần gấp đôi”. Đối chiếu trực tiếp với ảnh nguồn.

## Kiểm tra danh thiếp

[Xem ảnh danh thiếp](../examples/images/warm_desk_still_life_with_business_card.jpg).
Đối chiếu họ tên, chức danh và thông tin liên hệ có thể đọc được.
Kiểm tra độc lập câu trả lời EN và VI; ghi nhận các trường bị thiếu.

## Quy trình đánh giá

Sau khi notebook thực thi, lưu nguyên văn raw text.
Đồng thời lưu metadata đầu vào có SHA256.
Đối chiếu các chi tiết với ảnh, tự tính các tỷ lệ, lưu ghi chú review
tách biệt khỏi văn bản do model sinh. Không sửa output để nâng điểm.

[Danh mục Full19](FULL19-EVIDENCE.vi.md) ·
[Checklist nghiệm thu](ACCEPTANCE.vi.md)
