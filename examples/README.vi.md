# Khám phá 16 ảnh nguồn Text + Vision

> 🌐 Ngôn ngữ / Language: [English](README.md) | **Tiếng Việt**

Đây là **ảnh đưa vào Qwen3.8-27B để kiểm thử**, không phải ảnh Qwen tạo ra. Bộ ảnh giúp khám phá OCR số, suy luận từ sơ đồ, nhận diện vật thể, mô tả đời sống và các chi tiết an toàn hình ảnh.

## Có những nhóm ảnh nào?

| Nhóm | Số ảnh | Nội dung |
|---|---:|---|
| Thẻ OCR số | 2 | Đọc số in **731**, **482** |
| Rich Vision | 6 | Mô tả cảnh 2D/3D có nhiều vật thể và quan hệ |
| Photography | 8 | Món ăn, chợ, góc làm việc, đường phố và các cảnh khác |
| **Tổng** | **16** | Ảnh đầu vào kiểm chứng SHA256 độc lập |

Có năm nguồn Photography là **ảnh minh họa do AI tạo**. Không giới thiệu toàn bộ tập ảnh là ảnh máy ảnh thực.

## Xem ảnh và prompt

- [Tên file, kích thước, dung lượng, SHA256](assets_manifest.json)
- [Prompt Rich Vision](vision_showcase_cases.json)
- [Prompt Photography](photo_showcase_cases.json)
- [Hướng dẫn đọc output](../docs/FULL19-EVIDENCE.vi.md)

Mỗi ảnh được nhận diện bằng **SHA256**, không chỉ bằng tên file. Ảnh nguồn có thể là 1280 × 960 hoặc kích thước khác theo manifest, còn pipeline inference resize về **512 × 384**. SHA256 nguồn thuộc **file gốc**, không phải tensor ảnh sau resize.

## Ví dụ: cảnh qua đường

[Ảnh crossing hiện tại](images/photo_ai_city_crossing_1280x960.png) có **đèn người đi bộ xanh**, người ở vạch zebra và phương tiện phía sau. SHA256 là:

`ed532c2ba4e7eace4a81e4adfd6867c68f244fe4d64293f2bbfd1097ad7d0356`

Xem [provenance ảnh và điều kiện đánh giá](IMAGE-PROVENANCE.vi.md). SHA256 xác định dữ liệu đầu vào; kết quả suy luận cần hồ sơ chạy TPU tương ứng.

## Đánh giá ví dụ công bằng

Đối chiếu pixel gốc và prompt trước khi chấm. Với số liệu, kiểm cả giá trị và quan hệ; với giao thông, kiểm trạng thái và hướng đèn. Giữ nguyên cả các câu trả lời sai; EOS không đồng nghĩa mọi mô tả hình ảnh hoàn hảo.

[Chạy notebook](../docs/REPRODUCIBILITY.vi.md) · [Hướng dẫn đánh giá](../docs/ACCEPTANCE.vi.md)

## Nguồn gốc và quyền phân phối ảnh

SHA256 xác minh đúng dữ liệu ảnh, **không chứng minh quyền sở hữu hay quyền phân phối**. Khi phân phối lại, hãy kiểm nguồn gốc/giấy phép hoặc quyền sử dụng của **cả 16 ảnh**; ví dụ provenance người qua đường chưa thay thế chứng cứ toàn bộ. Ảnh do AI tạo cũng không tự động có quyền sử dụng không giới hạn.
