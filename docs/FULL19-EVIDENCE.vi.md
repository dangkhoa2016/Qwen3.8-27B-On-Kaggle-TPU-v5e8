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

## Kết quả lượt chạy mẫu và nguồn bằng chứng

Các kết quả dưới đây minh họa cách kiểm tra một lượt chạy Full19 đã hoàn tất.
Mức hoàn tất sinh văn bản, độ đúng nội dung ảnh và hiệu năng là các phép đo khác nhau.

### Notebook Git-first (15/11/2026 UTC)

Notebook Kaggle TPU v5e-8 chạy checkpoint BF16 nguyên gốc bằng mã nguồn Git
công khai lấy từ `main`. File `source-provenance.json` ghi lại chính xác
commit nguồn `f3c440fb78d1c6de0892a6ebc87abb9c9ff9d007`.

### Tải kết quả Full19

Dữ liệu gốc của lượt chạy được công bố qua **[GitHub Release Assets v1.0.0](https://github.com/dangkhoa2016/Qwen3.8-27B-On-Kaggle-TPU-v5e8/releases/tag/v1.0.0)**,
không phụ thuộc vào trang hoặc URL notebook Kaggle.

| Tải về | Nội dung |
|---|---|
| [Source provenance](https://github.com/dangkhoa2016/Qwen3.8-27B-On-Kaggle-TPU-v5e8/releases/download/v1.0.0/source-provenance.json) | Git commit nguồn và provenance của runtime |
| [Full19 results ZIP](https://github.com/dangkhoa2016/Qwen3.8-27B-On-Kaggle-TPU-v5e8/releases/download/v1.0.0/full19-results.zip) | Hai file JSON gốc `cases/full19-final.json` và `cases/full19-vi.json`, kèm SHA256 |
| [HTML Gallery ZIP](https://github.com/dangkhoa2016/Qwen3.8-27B-On-Kaggle-TPU-v5e8/releases/download/v1.0.0/inference-gallery.zip) | Gallery tĩnh, gồm `gallery/evidence.json` và ảnh nguồn |
| [SHA256 checksums](https://github.com/dangkhoa2016/Qwen3.8-27B-On-Kaggle-TPU-v5e8/releases/download/v1.0.0/SHA256SUMS) | Mã băm của provenance và hai file ZIP đã công bố |

Nội dung JSON được giữ nguyên từng byte bên trong ZIP kết quả, thay vì đưa
hàng nghìn dòng dữ liệu sinh ra vào Git diff. Sau khi giải nén, có thể
đối chiếu hai file JSON bằng `SHA256SUMS` bên trong. Gallery ZIP cũng được
giữ nguyên và có thể xem bằng HTTP server tĩnh mà không cần chạy TPU hoặc
truy cập notebook bên ngoài.

| Trường dữ liệu | Kết quả quan sát |
|---|---|
| Thiết bị TPU | 8/8 thiết bị TPU v5e |
| Thực thi notebook | Hoàn tất 17/17 code cell, không có Python error output |
| Full19 tiếng Anh | 19/19 case đạt EOS |
| Câu trả lời tiếng Việt | Sinh 17/17; cả hai thẻ OCR số dùng lại kết quả chữ số |
| Runtime Assets | 48 file được kiểm bằng SHA256 theo support manifest |
| Xuất Gallery | HTML tĩnh, 19 bản ghi case, 16 ảnh nguồn và `inference-gallery.zip` |
| Kiểm chứng Gallery | Đã kiểm toàn vẹn ZIP, SHA256 ảnh và truy cập HTTP |

SHA256 của Gallery ZIP đã lưu là
`ad528622567f8c0e283014aac0091eadf6fe1a43476decbc9b78f71151f14a0b`.
SHA256 của support manifest là
`fb2ee48dfe423733499112668d61e965b5d8d7466d52361b403c68dbc5768817`.
Commit nguồn gốc vẫn truy xuất được khi các cập nhật tài liệu sau này làm `main` thay đổi.

Các kết quả **không phải điểm độ chính xác nội dung**. [Tiêu chí đánh giá](FAILURE-ANALYSIS.vi.md)
bao gồm số thuế hóa đơn, phép tính dashboard giao hàng và các trường
danh thiếp bị thiếu giữa câu trả lời tiếng Anh và tiếng Việt. Giữ nguyên raw output.

### Lượt chạy cũ có nhúng source (12/11/2026 UTC)

Đây là lượt chạy Kaggle lịch sử riêng biệt, không phải một notebook khác trong
repository. Không áp dụng các phép đo khởi tạo nguội này cho lượt chạy Git-first.

| Bằng chứng lịch sử | Kết quả quan sát |
|---|---|
| Thiết bị TPU | 8/8 thiết bị TPU v5e |
| Thực thi notebook | 17/17 code cell, không có Python error output |
| Full19 tiếng Anh | 19/19 case đạt EOS |
| Câu trả lời tiếng Việt | Sinh 17/17; cả hai thẻ OCR số dùng lại kết quả chữ số |
| Kết quả xuất | HTML Gallery tĩnh và `inference-gallery.zip` |
| Thời gian nạp checkpoint lần đầu | 149,339 giây |
| Chẩn đoán OCR đầu tiên | Vision ~33,95 giây; prefill ~259,80 giây; decode ~186,80 giây cho 4 token, gồm chi phí cold-start |

Các thời lượng đầu tiên này không phải throughput ổn định hay HTTP latency.
Cần đánh giá độc lập raw output và độ chính xác của số liệu/hình ảnh.

## Cách diễn giải

EOS chỉ xác nhận kết thúc sinh token.
**Điều đó không chứng nhận câu trả lời đúng sự thật.**
Cần đối chiếu phép tính, tên, văn bản và quan hệ trực tiếp với ảnh nguồn.
Không sửa output model để tạo kết quả PASS.

[Tiêu chí review](FAILURE-ANALYSIS.vi.md) ·
[Checklist nghiệm thu](ACCEPTANCE.vi.md) ·
[Hướng dẫn thực thi](REPRODUCIBILITY.vi.md)
