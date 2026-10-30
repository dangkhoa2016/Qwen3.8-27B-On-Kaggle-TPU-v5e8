# Danh mục Full19 và checklist kiểm chứng kết quả

🌐 Ngôn ngữ / Language: [English](FULL19-EVIDENCE.md) | **Tiếng Việt**

Full19 nhóm các tác vụ Text và Vision cho Qwen3.8-27B BF16
trên Kaggle TPU v5e-8. Dùng danh mục case và checklist
để xem câu trả lời gốc, token IDs, EOS, thời gian và SHA256 ảnh đầu vào
từ lần chạy notebook trên Kaggle.

## Các nhóm tác vụ

**3 Text:** 2+2, thủ đô nước Pháp và ngôn ngữ lập trình Ruby.
Cần đối chiếu nội dung câu trả lời và các tiêu chí cơ bản.

**2 OCR số:** hai ảnh đầu vào chứa
[731](../examples/images/vision_test_card_731_1280x960.png) và
[482](../examples/images/vision_test_card_482_1280x960.png).
Prompt yêu cầu chỉ xuất chữ số.

Các mô tả dưới đây được dịch để người đọc tiện đối chiếu; prompt tiếng Anh
đưa vào model được lưu nguyên văn trong các file JSON của bộ ví dụ.

**6 Rich Vision**

| Mã case | Ảnh nguồn | Mô tả tác vụ bằng tiếng Việt |
|---|---|---|
| vision-2d-dashboard | [Open / Xem](../examples/images/vision_2d_city_bike_dashboard.jpg) | Phân tích bảng điều khiển số liệu giao hàng |
| vision-2d-greenhouse | [Open / Xem](../examples/images/vision_2d_smart_greenhouse.jpg) | Giải thích sơ đồ nhà kính thông minh |
| vision-3d-robotics | [Open / Xem](../examples/images/vision_3d_robotics_station.jpg) | Diễn giải hình minh họa trạm robot có chú thích: xác định cánh tay robot và thùng chứa, đọc nhãn Test Run, nhiệt độ, A1/A2/B1/B2 và giấy ghi chú; giải thích thao tác cần thực hiện trước |
| vision-3d-warehouse | [Open / Xem](../examples/images/vision_3d_warehouse_dock_b2.jpg) | Đọc sơ đồ quy trình vận hành kho hàng có chú thích |
| vision-real-workspace | [Open / Xem](../examples/images/vision_real_sprint_workspace.jpg) | Trích xuất ngày lập kế hoạch sprint, ba mục trong chương trình họp, giờ nhắc việc, giờ trên điện thoại và hai giá trên hóa đơn ở bàn làm việc minh họa; tính tổng tiền và số phút còn lại đến giờ nhắc |
| vision-real-market | [Open / Xem](../examples/images/vision_real_fresh_market.jpg) | Trích xuất đủ năm mặt hàng và giá trên bảng chợ minh họa; xác định món rẻ nhất và đắt nhất, đọc giờ mở cửa và tính chênh lệch giá cao nhất với thấp nhất |

**8 Photography**

| Mã case | Ảnh nguồn | Mô tả tác vụ bằng tiếng Việt |
|---|---|---|
| photo-office-invoice | [Open / Xem](../examples/images/northstar_office_supply_invoice.jpg) | Đọc hóa đơn văn phòng, xác định rõ nhà cung cấp, các dòng hàng hóa đọc được và tổng tiền |
| photo-business-card | [Open / Xem](../examples/images/warm_desk_still_life_with_business_card.jpg) | Mô tả bàn làm việc và chép lại chỉ những thông tin liên hệ nhìn rõ trên danh thiếp |
| photo-daylight-workspace | [Open / Xem](../examples/images/cozy_daylight_workspace_with_city_views.jpg) | Mô tả vật dụng, bố cục, ánh sáng và khung cảnh bên ngoài của không gian làm việc |
| photo-ai-farmers-market | [Open / Xem](../examples/images/photo_ai_farmers_market_1280x960.png) | Mô tả khung cảnh khu chợ |
| photo-ai-city-crossing | [Open / Xem](../examples/images/photo_ai_city_crossing_1280x960.png) | Mô tả hoạt động đang diễn ra tại vạch qua đường trong đô thị |
| photo-ai-vietnamese-meal | [Open / Xem](../examples/images/photo_ai_vietnamese_meal_1280x960.png) | Xác định các món ăn, món ăn kèm, gia vị và đồ uống nhìn thấy trên bàn nhà hàng |
| photo-ai-dog-walking | [Open / Xem](../examples/images/photo_ai_dog_walking_1280x960.png) | Mô tả khung cảnh công viên và mối liên hệ giữa người, chó, dây dắt, lối đi và hồ nước gần đó |
| photo-ai-supermarket | [Open / Xem](../examples/images/photo_ai_supermarket_1280x960.png) | Mô tả lối đi trong siêu thị và hành động nhìn thấy của người mua sắm |

Đúng 16 ảnh có SHA256 trong
[manifest ảnh nguồn](../examples/assets_manifest.json).
Đây là ảnh đầu vào, không phải kết quả sinh từ model.
Một số ảnh là minh họa do AI tạo, không được gọi là ảnh hiện trường.

## Bằng chứng phải lưu sau khi chạy notebook chính thức

Mỗi bản ghi kết quả cần có các trường:
1. Git SHA, model BF16, topology TPU, phiên bản JAX/runtime
   và SHA256 manifest của bộ dữ liệu hỗ trợ;
2. Case ID, prompt thực tế và SHA256 ảnh nếu sử dụng ảnh;
3. Raw text nguyên văn, generated token IDs, EOS và lỗi;
4. Thời gian từng giai đoạn và tổng thời gian, phân biệt cold/warm;
5. JSON có cấu trúc và HTML Gallery tĩnh;
6. Đánh giá thủ công chữ, số, đối tượng và quan hệ trong ảnh,
   tách biệt khỏi câu trả lời model.

Notebook sẽ tạo thư mục theo UTC tại /kaggle/working, với
cases/full19-final.json cho EN, cases/full19-vi.json cho VI nếu bật
và các file Gallery xuất bằng CPU.

## Cách diễn giải

EOS chỉ xác nhận kết thúc sinh token.
**Điều đó không chứng nhận câu trả lời đúng sự thật.**
Cần đối chiếu phép tính, tên, văn bản và quan hệ trực tiếp với ảnh nguồn.
Không sửa output model để tạo kết quả PASS.

[Tiêu chí review](FAILURE-ANALYSIS.vi.md) ·
[Checklist nghiệm thu](ACCEPTANCE.vi.md) ·
[Hướng dẫn thực thi](REPRODUCIBILITY.vi.md)
