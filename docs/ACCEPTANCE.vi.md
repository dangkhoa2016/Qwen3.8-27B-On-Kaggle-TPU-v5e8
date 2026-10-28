# Đánh giá bộ thử Full19 Text + Vision

> 🌐 Ngôn ngữ / Language: [English](ACCEPTANCE.md) | **Tiếng Việt**

Hướng dẫn này đánh giá riêng **mức độ hoàn tất chương trình** và **chất lượng câu trả lời**. Có end-of-sequence token không chứng minh câu trả lời đúng sự thật.

## Phạm vi 19 case

| Nhóm | Số case | Nội dung cần kiểm |
|---|---:|---|
| Text | 3 | Câu trả lời hợp lý, đúng yêu cầu, EOS |
| OCR số | 2 | Đọc 731 và 482, EOS |
| Rich Vision | 6 | Vật thể, quan hệ và 55 phép kiểm từ khóa |
| Photography | 8 | Mô tả dựa trên chi tiết thực sự thấy trên ảnh |
| **Tổng** | **19** | Prompt, SHA256 ảnh, raw output, token IDs và thời gian |

Một số ảnh Photography là minh họa do AI tạo để đưa vào Qwen. Kiểm từ khóa chỉ đo những khái niệm cụ thể, không đo toàn bộ tính đúng của câu trả lời.

## Checklist bằng chứng

1. Ghi Git SHA của notebook, revision checkpoint, SHA256 manifest 48 file và SHA256 ảnh.
2. Xác minh tám TPU, một engine và các kiểm tra tương thích native JAX/XLA.
3. Lưu prompt, token IDs, raw text, EOS và thời gian từng case, kể cả trường hợp lỗi.
4. Kiểm 19 ID duy nhất; công bố kết quả hoàn thành chương trình độc lập với độ chính xác.
5. Đối chiếu OCR, số liệu, phép tính, đèn giao thông, vật thể và quan hệ trong ảnh.
6. Khi so sánh giữa hai lượt chạy, cần cùng bộ input và cấu hình hoặc giải thích rõ sự khác biệt.

## Diễn giải kết quả

Lượt TPU đã ghi ngày 27/10/2026 hoàn thành **19/19 case với EOS** trong **1.160,844 giây**, nhưng vẫn có lỗi về thuế hóa đơn và phép so sánh tỷ lệ trên dashboard. Vì vậy cần tách hai tiêu chí: chương trình sinh output thành công và nội dung chính xác.

CPU Playground FastAPI/Vue chỉ minh họa hợp đồng request; không nạp trọng số BF16 và không chứng minh throughput TPU hoặc chất lượng sinh văn bản.

[Bằng chứng Full19](FULL19-EVIDENCE.vi.md) · [Hiệu năng](PERFORMANCE-ANALYSIS.vi.md) · [Chạy thử](REPRODUCIBILITY.vi.md)
