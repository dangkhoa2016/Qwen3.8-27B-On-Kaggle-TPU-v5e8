# Tiêu chí nghiệm thu và kết quả đã xác minh

> [English](ACCEPTANCE.md) | **Tiếng Việt**

## Full19 đã nghiệm thu tự động trên TPU thật — 27/10/2026

Mô hình **Qwen3.8-27B BF16 gốc**, gồm 18 safetensors shards và không lượng tử hóa, đã chạy đủ Full19 trên **8 thiết bị Kaggle TPU v5e**, dùng một MaxEngine và đường Split-Vision JAX/MaxText AOT. Notebook đã thực thi và evidence được lưu riêng; đây **chưa phải** kết quả Run All từ notebook hiện được Git theo dõi.

| Hạng mục | Kết quả |
|---|---|
| Runner và notebook | COMPLETE, 14/14 code cells |
| Tổng thời gian | **1.160,844 giây (19 phút 20,844 giây)** |
| Text | **3/3**, ngữ nghĩa và EOS đạt |
| Hai ảnh số OCR | **2/2**, số 731 và 482 chính xác, có EOS |
| Rich Vision | **6/6** EOS; 55/55 kiểm tra từ khóa định trước |
| Photography | **8/8**, có EOS |
| Toàn bộ | **19/19**, output riêng biệt và EOS |
| Model/hardware | BF16 gốc 27B, 18 shards, 8 TPU v5e, một engine |
| Fresh Run All từ notebook public Git | **CHƯA XÁC MINH** |
| HTTP TPU thật | **CHƯA NGHIỆM THU** |
| Độ chính xác VLM tổng quát | **CHƯA CÓ BẰNG CHỨNG** |

Nghiệm thu trên xác nhận mô hình thực sự suy luận thành công, **không chứng minh** OCR, số học hay mọi mô tả ảnh đều hoàn hảo. Người dùng đánh giá tổng thể các ví dụ khác là ổn, nhưng phải giữ lại lỗi chất lượng. Dashboard diễn đạt 114 đơn so với 31 đơn là “gần gấp đôi”, dù tỷ lệ khoảng 3,68. Ảnh hóa đơn ghi **Tax (8%): $69.28**, tạm tính $866.00 và tổng $935.28; mô hình lại ghi **Tax (5%): $66.28**. Hai lượt thử hóa đơn riêng bằng cùng prompt và greedy đều tái hiện lỗi đó, có EOS và cùng token IDs. Greedy mang tính xác định; đây không phải hai lượt lấy mẫu ngẫu nhiên độc lập.

Xem [danh mục evidence Full19](FULL19-EVIDENCE.vi.md) để biết checksum, giới hạn và file lưu. Giữ nguyên câu trả lời do mô hình sinh; nội dung tiếng Việt trên web chỉ là bản dịch hỗ trợ đánh giá.

## Các bước bắt buộc trước khi chạy notebook production chính thức

1. Thay notebook Git cũ đang dùng prefill/target **256/384** bằng một notebook **320/768 Split-Vision AOT** đã kiểm tra, Rich Vision/Photography tối đa 448 token, đúng đường xử lý OCR và Text.
2. Bảo đảm checkpoint, **38 tệp support** và **16 ảnh** đều có thể attach và đọc được trong phiên Kaggle mới. Lượt nghiệm thu trước sử dụng file private trong /kaggle/working, không bảo đảm còn nguyên trong fresh session. Không phụ thuộc clone ẩn danh từ GitHub Private.
3. Xác minh SHA256, phiên bản JAX/JAXLIB/Transformers, duy nhất một lần nạp BF16/MaxEngine, các kiểm tra sharding/layout thật của JAX và tính tương thích cache.
4. Chạy notebook mới, lưu token IDs, output, EOS, thời gian từng case/toàn bộ, số TPU devices và trạng thái. Giữ lại bằng chứng lỗi nếu có, không ghi PASS trước khi đủ dữ liệu.
5. Kiểm thử CPU API contract và Playwright bằng môi trường test độc lập. CPU PASS không thay thế nghiệm thu HTTP TPU.
6. Đánh giá nội dung và các lỗi tồn tại trước khi công bố. Không đưa HTTP TPU thử nghiệm ra tunnel công khai trước khi nghiệm thu bảo mật/end-to-end riêng.

Xem [hướng dẫn tái lập](REPRODUCIBILITY.vi.md). Bản phát hành production public chưa hoàn thành.
