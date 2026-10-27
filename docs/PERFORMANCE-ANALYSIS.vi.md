# Báo cáo hiệu năng Qwen3.8-27B BF16 trên Kaggle TPU v5e-8

> 🌐 Ngôn ngữ / Language: [English](PERFORMANCE-ANALYSIS.md) | **Tiếng Việt**

**Qwen3.8-27B BF16 có thể thực thi tác vụ Text + Vision trên tám thiết bị TPU v5e thông qua JAX và MaxText.** Trang này trình bày số đo từ một lượt chạy Full19 thực tế, giải thích chi phí suy luận đa phương thức và chia sẻ phương pháp benchmark hữu ích cho những dự án mô hình lớn khác.

## Kết quả nổi bật

| Chỉ số | Kết quả ghi nhận |
|---|---|
| Phần cứng | Kaggle TPU v5e-8 — 8 thiết bị TPU |
| Checkpoint | Qwen3.8-27B **BF16 nguyên gốc**, 18 shard safetensors |
| Runtime | JAX / MaxText với tích hợp Split-Vision |
| Khối lượng kiểm thử | **19 tác vụ**: 3 Text, 2 OCR số, 6 Rich Vision, 8 Photography |
| Tổng thời gian Full19 | **1.160,844 giây (19 phút 20,844 giây)** |
| Hoàn tất sinh văn bản | **19/19 tác vụ có EOS** |
| Kiểm tra kết quả có cấu trúc | Text 3/3; OCR số **731** và **482**; Rich Vision **55/55** kiểm tra từ khóa |

Các số đo thuộc **một lượt chạy Full19 trên TPU thật ngày 27/10/2026**, sử dụng đúng bộ ảnh đầu vào tại thời điểm đó. Chỉ số 19/19 nói về khả năng hoàn tất sinh token; tính chính xác của câu trả lời được đánh giá riêng. Tổng thời gian thể hiện chi phí của cả phiên chạy, không phải latency của một API request.

## Cấu hình chạy

| Thành phần | Thiết lập |
|---|---|
| Model | [Kaggle mirror — Qwen3.8-27B BF16](https://www.kaggle.com/models/dangkhoa2016/qwen-qwen3-8-27b) |
| Engine | Một `MaxEngine` trên 8 TPU; checkpoint nạp một lần |
| Ảnh ở đầu vào suy luận | **512 × 384 pixel** |
| Prefill | **320 token** |
| Target budget | **768 token** |
| Decode Rich Vision / Photography | Tối đa **448 bước** |
| Nhận diện dữ liệu | Prompt, manifest và SHA256 ảnh |

Nguồn tham khảo để kiểm chứng cấu hình: [notebook Full19](../notebooks/kaggle-tpu-v5e8-production-showcase.ipynb), [kiến trúc TPU](ARCHITECTURE.vi.md) và [kết quả Full19](FULL19-EVIDENCE.vi.md). Model được nạp từ Kaggle mirror đã attach, không tải trực tiếp từ Hugging Face.

## Các công đoạn quyết định thời gian chạy

| Pha | Công việc chính | Ảnh hưởng tới hiệu năng |
|---|---|---|
| Chuẩn bị | Kiểm phiên bản dependencies, dữ liệu và SHA256 | Tránh lãng phí TPU khi đầu vào sai |
| Khởi tạo engine | Thiết lập JAX/XLA, topology, compilation | Chi phí đáng kể ở cold start |
| Nạp checkpoint | Đọc và phân phối BF16 parameters | I/O và đặt tham số trên thiết bị |
| Vision encoder | Chuyển ảnh thành image embeddings | Tăng chi phí đối với tác vụ có ảnh |
| Multimodal prefill | Ghép text/image tokens và MRoPE | Phụ thuộc sequence length và shape |
| Decode | Sinh token tới EOS hoặc giới hạn | Phụ thuộc độ dài câu trả lời |
| Ghi kết quả | Lưu output, thời gian và evidence | Nằm trong phép đo end-to-end |

Trong một lần đo kỹ thuật trước đó, **khởi tạo engine khoảng 49,8 giây** và **nạp model khoảng 147,0 giây**. Hai số đo này giúp hình dung chi phí khởi động, nhưng được ghi ở bối cảnh đo khác; chúng không phải phép phân rã đã xác minh của tổng thời gian Full19 nêu trên.

Tương tự, phép chia `1.160,844 / 19` không tương đương latency trung bình mỗi request: 19 tác vụ có độ dài và chi phí khác nhau, trong khi tổng thời gian còn bao gồm chuẩn bị.

## Bốn bài học kỹ thuật

### 1. Cold start khác warm inference

Lần chạy đầu có thể phải nạp checkpoint, dựng engine và biên dịch. Sau khi trạng thái đã sẵn sàng, phần chi phí này không nhất thiết xuất hiện lại ở mỗi câu hỏi. Benchmark rõ ràng cần phân biệt **startup**, **first inference** và **warm inference**.

### 2. Vision encoder là một pha độc lập

Text + Vision yêu cầu tiền xử lý ảnh, tạo embeddings và hợp nhất với text trước decode. Vì vậy chỉ số token/giây của decoder không đủ mô tả thời gian phản hồi cho một prompt chứa hình ảnh.

### 3. Độ dài câu trả lời ảnh hưởng tốc độ

Hai prompt sử dụng cùng một ảnh vẫn có thể sinh số token rất khác nhau. Khi so sánh cần ghi rõ kích thước ảnh, prompt tokens, token budget, số token thực sinh và điều kiện dừng EOS.

### 4. Kết quả đúng quan trọng không kém thời gian nhanh

Full19 hoàn tất 19 luồng sinh token và đạt các bài kiểm tra từ khóa đã ghi, nhưng vẫn xuất hiện lỗi OCR thuế hóa đơn và suy luận tỷ lệ dashboard. Một phép đo hữu ích cần lưu **thời gian, văn bản nguyên gốc và đánh giá ngữ nghĩa** cạnh nhau.

## Phương pháp benchmark có thể tái lập

1. **Cố định môi trường:** Git SHA, revision Kaggle Model, SHA256 manifest, phiên bản JAX/MaxText, TPU topology và compile-cache policy.
2. **Giữ nguyên workload:** prompt, SHA256 ảnh, kích thước ảnh đã xử lý, prefill length và decode budget cho từng case.
3. **Đo thời gian từng pha:** engine, checkpoint load, vision encode, prefill, decode và end-to-end.
4. **Phân biệt cold/warm:** đo các lượt độc lập và ghi số mẫu; không bỏ chi phí khởi tạo khỏi số đo cold start.
5. **Báo cáo thống kê đúng:** median/p95 cần nhiều mẫu có điều kiện tương đương, kèm số lượng mẫu.
6. **Giữ toàn bộ evidence:** token IDs, output text, EOS, timing, exception và rubric chất lượng, bao gồm cả các kết quả không đạt.

Khi có đủ log, các chỉ số sau giúp so sánh hiệu năng giữa nhiều cấu hình:

| Chỉ số | Phương pháp tính |
|---|---|
| End-to-end latency | Thời điểm hoàn tất trừ thời điểm bắt đầu một request đã định nghĩa |
| Decode throughput | Số token thực sinh / thời gian decode tương ứng |
| Throughput toàn hệ thống | Số request hoặc token hoàn tất / tổng thời gian quan sát, kèm concurrency |
| p50/p95 latency | Phân vị từ nhiều lượt đo có cùng điều kiện |

## Phạm vi sử dụng kết quả

Bộ số liệu trên mô tả **một lượt chạy Full19 trên phần cứng TPU thật**. Benchmark cho dịch vụ phục vụ người dùng còn cần các phép đo HTTP nhiều request, concurrency, phân phối latency và TPU utilization. CPU Playground trong repository là mock API, không dùng để đo hiệu năng BF16.

Một ảnh crossing đèn xanh mới đã thay ảnh đèn đỏ ở bộ ví dụ Git. Vì SHA256 của hai ảnh khác nhau, thời gian và câu trả lời của lượt Full19 trước được gắn với ảnh cũ; khi chạy ảnh mới cần dùng manifest và kết quả tương ứng.

**Khám phá thêm:** [Kiến trúc](ARCHITECTURE.vi.md) · [Kết quả Full19](FULL19-EVIDENCE.vi.md) · [Đánh giá](ACCEPTANCE.vi.md) · [Tái lập](REPRODUCIBILITY.vi.md)
