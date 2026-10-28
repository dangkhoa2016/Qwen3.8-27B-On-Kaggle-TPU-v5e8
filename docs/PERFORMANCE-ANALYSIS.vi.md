# Phân tích hiệu năng suy luận Qwen3.8-27B BF16 trên Kaggle TPU v5e-8

> 🌐 Ngôn ngữ / Language: [English](PERFORMANCE-ANALYSIS.md) | **Tiếng Việt**

Tài liệu trình bày kết quả suy luận Qwen3.8-27B BF16 Text + Vision đo trên Kaggle TPU v5e-8 bằng JAX, MaxText và Split-Vision. Số đo tổng thời gian được phân biệt với độ trễ mỗi yêu cầu; hoàn tất sinh token được phân biệt với độ chính xác câu trả lời.

## Kết quả Full19 đã đo

| Chỉ số | Giá trị ghi nhận |
|---|---|
| Phần cứng | Kaggle TPU v5e-8, tám thiết bị TPU |
| Checkpoint | Qwen3.8-27B BF16 nguyên gốc, 18 safetensors shard |
| Runtime | JAX / MaxText, Vision encoder riêng và multimodal prefill đã biên dịch |
| Khối lượng | 19 case: 3 Text, 2 OCR số, 6 Rich Vision, 8 Photography |
| Tổng thời gian | **1.160,844 giây (19 phút 20,844 giây)** |
| Hoàn tất sinh token | **19/19 case đạt EOS** |
| Một số phép kiểm | Text 3/3; OCR 731 và 482; Rich Vision 55/55 kiểm tra từ khóa |

Kết quả được đo trong **một phiên TPU ngày 27/10/2026**, với đúng phiên bản mã nguồn và tập đầu vào của lượt đo đó. Đây không phải bằng chứng thực thi cho mọi lần cập nhật repository về sau, cũng không phải cam kết 19/19 câu trả lời đều đúng.

## Cấu hình và các pha thực thi

| Thành phần | Thiết lập |
|---|---|
| Nguồn checkpoint thực thi | [Kaggle BF16 model mirror](https://www.kaggle.com/models/dangkhoa2016/qwen-qwen3-8-27b) |
| Engine | Một MaxEngine, nạp checkpoint một lần |
| Độ phân giải ảnh khi suy luận | 512 × 384 pixel |
| Prefill | 320 token |
| Target | 768 token |
| Rich Vision / Photography decode | Tối đa 448 bước |
| Provenance đầu vào | Prompt, SHA256 ảnh và manifest runtime |

Tổng thời gian gồm chuẩn bị, nạp trọng số, compilation, Vision encoder, prefill, decode và ghi kết quả. Không được chia trực tiếp cho 19 để suy ra độ trễ trung bình của HTTP request hoặc throughput của hệ thống phục vụ.

Suy luận đa phương thức cần xử lý ảnh và tạo embeddings trước khi decode văn bản. Cache JAX chỉ có thể tái sử dụng khi tương thích topology, compiler, shape, sharding và dtype; trùng SHA256 của cache không tự chứng minh tính tương thích khi thực thi.

## Phương pháp benchmark có thể áp dụng lại

1. Ghi revision model, Git SHA, SHA256 manifest, prompt, ảnh, phiên bản runtime và thiết bị TPU.
2. Đo riêng cold start, nạp checkpoint, mã hóa ảnh, prefill, decode và end-to-end.
3. Lưu số token thực sinh, EOS và raw output theo từng case.
4. Giữ cả các lần chạy lỗi và sai lệch ngữ nghĩa để phân tích cùng thời gian.
5. Chỉ công bố median, p95 hoặc concurrency khi có đủ lượt đo tương đương.

## Phạm vi diễn giải

Sinh token hoàn tất không bảo đảm nội dung đúng. Output đã đo có lỗi thuế hóa đơn (8% / $69.28 được đọc thành 5% / $66.28) và so sánh không chính xác giữa 114 và 31. Kiểm tra từ khóa cũng không thay thế đánh giá hiểu ảnh đầy đủ.

**Chỉ gán số đo cho đúng SHA256 đầu vào và hồ sơ thực thi tương ứng.** [Notebook trong Git](../notebooks/kaggle-tpu-v5e8-production-showcase.ipynb) cung cấp mã để tạo những lượt chạy mới; bằng chứng của mỗi lượt cần được thu thập riêng. CPU mock không đo tốc độ mô hình Qwen hoặc HTTP TPU.

[Kiến trúc](ARCHITECTURE.vi.md) · [Bằng chứng Full19](FULL19-EVIDENCE.vi.md) · [Tái lập](REPRODUCIBILITY.vi.md)
