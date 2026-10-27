# Chất lượng ảnh benchmark: một cảnh sang đường, hai đầu vào

> 🌐 Ngôn ngữ / Language: [English](RELEASE-IMAGE-REVIEW.md) | **Tiếng Việt**

Một ví dụ benchmark chỉ tái lập được khi có **đúng byte của ảnh**. Case crossing minh họa vì sao cải thiện hình ảnh cũng làm thay đổi phạm vi của kết quả model có thể sử dụng.

## So sánh hai ảnh

| Ảnh | Nội dung quan sát | SHA256 |
|---|---|---|
| Ảnh experimental trước | Đèn người đi bộ đỏ khi người đang qua đường | `bba573de952942ec97ea6a32c352bdbd3770082b570cfe74520b1cd03b087763` |
| Ảnh hiện tại trong Git | Đèn người đi bộ xanh và người qua vạch | `ed532c2ba4e7eace4a81e4adfd6867c68f244fe4d64293f2bbfd1097ad7d0356` |

Ảnh Git mới 1280 × 960 được resize bằng **Pillow Lanczos** từ PNG đầu vào 1448 × 1086 (SHA256 `63a61fd4ed5788ca156313b0ee7b2335e898b2e7ed7f683a725bb1d836c5395d`). Không thể xác nhận model tạo ảnh, seed và prompt gốc chỉ bằng file PNG.

## Vì sao phải phân biệt?

Câu trả lời Qwen cho **ảnh đèn đỏ** đã nhận ra xung đột giao thông. Đây không phải lỗi suy luận đã xác nhận. **Ảnh đèn xanh** là input khác, nên không thể dùng câu trả lời và thời gian của case cũ làm bằng chứng cho ảnh mới.

[Manifest ảnh](assets_manifest.json) nhận diện input Git mới. Snapshot hỗ trợ thực nghiệm cũ vẫn gắn với output ban đầu; snapshot ảnh thay thế riêng đã đạt **48/48 SHA256** sau khi tải lại độc lập.

Kiểm SHA256 chứng minh file ảnh toàn vẹn, **không** chứng minh Qwen inference. Hãy dùng [notebook nguồn trong Git](../notebooks/kaggle-tpu-v5e8-production-showcase.ipynb) và manifest tương ứng để chạy lại; tuyệt đối không dùng output ảnh cũ cho ảnh mới.

## Checklist đánh giá ảnh có thể tái sử dụng

1. Xem ảnh nguyên độ phân giải, không chỉ ảnh thumbnail.
2. Kiểm đèn, hình thể người, hình học đường và quan hệ vật thể.
3. Lưu SHA256 và phép biến đổi từ ảnh đầu vào.
4. Gắn prompt, raw response và token IDs với hash ảnh.
5. Khi đổi ảnh, chạy inference lại thay vì dùng output cũ.
6. Giữ evidence trước đó để phép so sánh trước/sau minh bạch.

Cách làm này hữu ích cho benchmark ảnh AI, OCR và mô hình đa phương thức nói chung.

[Photography cases](photo_showcase_cases.json) · [Tái lập](../docs/REPRODUCIBILITY.vi.md) · [Đánh giá](../docs/ACCEPTANCE.vi.md)
