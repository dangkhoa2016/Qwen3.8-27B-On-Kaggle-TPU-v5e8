# Phương pháp đo và diễn giải hiệu năng TPU

> 🌐 Ngôn ngữ / Language: [English](PERFORMANCE-ANALYSIS.md) | **Tiếng Việt**

Benchmark mô hình đáng tin cậy phải tách khởi tạo, nạp checkpoint, mã hóa Vision, compilation, prefill, decode token và lưu bằng chứng. [Checklist thực thi Full19](FULL19-EVIDENCE.vi.md) liệt kê tác vụ và dữ liệu cần thu thập từ lần chạy notebook chính thức. Tài liệu này mô tả phương pháp đo, không công bố kết quả hiệu năng.

## Các mốc thời gian

| Pha | Dữ liệu cần ghi | Cách hiểu |
|---|---|---|
| Preflight | Tính toàn vẹn input và dependency | Tránh lượt chạy thiếu dữ liệu hoặc không tương đương |
| Tạo engine | Thời gian khởi tạo | Phân biệt startup framework và I/O model |
| Nạp checkpoint | Thời gian nạp trọng số | Phụ thuộc storage và topology thiết bị |
| Xử lý ảnh | Resize, Vision embedding và thời gian | Chi phí bổ sung so với text-only |
| Compilation và prefill | Thời gian compile/execute JAX/XLA | Có thể chiếm nhiều thời gian khi cold |
| Decode | Token thực sinh, EOS và thời lượng | Dùng đánh giá tốc độ sinh token |
| Lưu bằng chứng | Ghi JSON và thời gian | Thuộc tổng thời gian wall-clock |

## Cache và phần cứng

Tái sử dụng executable đã biên dịch cần tương thích compiler, shape, dtype, sharding và TPU topology. Cache có byte giống nhau **không** tự chứng minh cache hit. Cần lưu phiên bản JAX, device IDs, Git SHA, revision model, SHA256 support manifest và SHA256 prompt/ảnh.

Không lấy thời gian tổng của notebook chia số tác vụ để gọi là HTTP latency. Cold, warm, chạy tuần tự và phục vụ đồng thời có giới hạn đo khác nhau.

## Công bố kết quả có trách nhiệm

Giữ raw output, token IDs, EOS, thời lượng từng tác vụ, lỗi và những sai lệch số liệu/ảnh. Cần nhiều lượt chạy tương đương trước khi công bố median hoặc p95. Độ chính xác mô hình và mức hoàn tất hạ tầng là hai tiêu chí riêng.

[Bằng chứng suy luận](FULL19-EVIDENCE.vi.md) · [Hướng dẫn chạy](REPRODUCIBILITY.vi.md)
