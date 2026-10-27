# Cách đánh giá bộ Full19 Text + Vision

> 🌐 Ngôn ngữ / Language: [English](ACCEPTANCE.md) | **Tiếng Việt**

Một bản demo hữu ích cần cho người xem biết **đã thử gì**, **thành công ở đâu** và **mô hình sai ở đâu**. Tài liệu này đưa ra phương pháp đánh giá có thể tái sử dụng, thay vì coi một nhãn PASS là thước đo độ chính xác tổng quát.

## Full19 gồm những gì?

| Nhóm | Số case | Cần kiểm tra |
|---|---:|---|
| Text | 3 | Ý nghĩa của câu trả lời ngắn, EOS |
| Thẻ OCR số | 2 | Đúng số **731**, **482** và EOS |
| Rich Vision | 6 | Các thuộc tính cảnh; 55 lượt kiểm keyword |
| Photography | 8 | Mô tả đúng ảnh, vật thể, quan hệ, giao thông |
| **Tổng** | **19** | ID duy nhất, raw text, token IDs, EOS và thời gian |

Một số ảnh đầu vào do AI tạo. Đây **không phải ảnh Qwen sinh ra**. Kiểm tra keyword chỉ chứng minh khái niệm cụ thể xuất hiện, chưa chứng minh mọi con số hoặc quan hệ trong câu trả lời là đúng.

## Các cấp bằng chứng

| Câu hỏi | Cách kiểm | Tình trạng đã quan sát |
|---|---|---|
| Input có toàn vẹn? | Kiểm SHA256 ảnh và support | 48/48 file đúng sau khi tải lại Kaggle |
| HTTP contract hoạt động? | pytest và Playwright CPU | 36 Python PASS, 1 SKIP; 2 browser PASS |
| Qwen sinh output thật trên TPU? | Token IDs, log tám TPU, runtime | Full19 experimental ngày **27/10/2026** |
| Nội dung có đúng ảnh? | So với source image và số liệu | Có cả kết quả đúng lẫn lỗi invoice/dashboard |
| Ảnh crossing mới đã được thử TPU? | Đối chiếu SHA256 và bằng chứng từ lượt chạy mới | Evidence experimental cũ chưa chứng minh |
| HTTP TPU thật đã chạy? | Request Text/Vision có auth và timing | Chưa nghiệm thu |

## Kết quả từ lượt thực nghiệm

Một lượt experimental đã hoàn thành **19/19 case tự động có EOS** trong **1.160,844 giây** trên TPU v5e-8. Bằng chứng gồm ba case Text, hai OCR số chính xác và 55/55 keyword Rich Vision. Phải giữ các output đó gắn với runner và SHA256 ảnh nguồn ban đầu.

### Vì sao chương trình PASS vẫn có thể trả lời sai?

- **Invoice:** ảnh ghi subtotal **$866.00**, tax **8% / $69.28**, total **$935.28**. Mô hình lại sinh **5% / $66.28**. Hai lần greedy thử lại vẫn lặp đúng lỗi.
- **Dashboard:** 114 lượt giao hàng Thứ Bảy so với 31 lượt Thứ Hai. Mô tả “gần gấp đôi” sai; tỷ lệ khoảng **3,68**.
- **Crossing:** ảnh cũ có đèn người đỏ và mô hình nhận ra tình huống nguy hiểm. Output đó **không phải kết quả** cho ảnh mới đèn xanh trong Git.

Các ví dụ nhắc rằng cần kiểm tra từng nhận định khi kết quả quan trọng. Xem [bài học từ lỗi mô hình](FAILURE-ANALYSIS.vi.md).

## Checklist đánh giá có thể áp dụng lại

1. Ghim commit notebook, revision model, manifest support và SHA256 từng ảnh.
2. Xác minh đúng tám TPU, một engine và JAX/XLA native validation.
3. Lưu mọi prompt cùng **raw output nguyên văn**, token IDs, EOS và duration.
4. Kiểm **19 ID duy nhất** và từng điều kiện kiểm tra tự động.
5. Đánh giá thủ công số liệu, quan hệ hình ảnh, đèn giao thông, độ chắc chắn và nội dung bỏ sót.
6. Ghi cả lỗi. Không có output không phải PASS; output hoàn thành nhưng sai là **completion PASS / semantic FAIL**.
7. Phân biệt tổng wall time với latency từng request, throughput và chỉ số chưa đo.

Bảng kết quả tốt nên tách các cột **đã chạy**, **EOS**, **kiểm tự động**, **đánh giá ngữ nghĩa** và **SHA256 input** thay vì dùng một phần trăm chính xác duy nhất.

## Trạng thái của bộ ảnh mới

[Manifest ảnh](../examples/assets_manifest.json) nhận diện ảnh crossing đèn xanh mới. Đúng SHA256 chỉ chứng minh dữ liệu toàn vẹn, không chứng minh inference thành công; chỉ công bố kết quả Full19 mới sau khi kiểm output thật, EOS, SHA256 ảnh và độ chính xác ngữ nghĩa.

[Bằng chứng](FULL19-EVIDENCE.vi.md) · [Tái lập](REPRODUCIBILITY.vi.md) · [Hiệu năng](PERFORMANCE-ANALYSIS.vi.md)
