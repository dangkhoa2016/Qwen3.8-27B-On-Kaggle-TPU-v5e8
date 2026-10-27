# Danh mục bằng chứng Full19 trên TPU và các lỗi chất lượng — 27/10/2026

> [English](FULL19-EVIDENCE.md) | **Tiếng Việt**

## Nguồn gốc kết quả nghiệm thu tự động thử nghiệm

| Mục | Kết quả đã xác minh |
|---|---|
| Model | Qwen3.8-27B BF16 nguyên gốc, 18 shards safetensors; 1.199 tensor trong index |
| Thiết bị | Kaggle TPU v5e-8 × 8; JAX/JAXLIB 0.11.0 |
| Engine | Một MaxEngine; giữ nguyên kiểm tra JAX/XLA sharding và layout |
| Notebook | Ứng viên Split-Vision AOT riêng; **khác notebook Git cũ** |
| Dữ liệu | 16 ảnh, 3 Text; resize ảnh inference 512×384 |
| Prefill/target/sinh token | 320/768/448 cho Rich Vision và Photography |
| Kết quả tự động | COMPLETE, 14/14 code cells, 19/19 câu trả lời riêng và EOS |
| Phân bố | 3 Text, 2 OCR số, 6 Rich Vision, 8 Photography |
| Kiểm tra | Text/OCR 5/5 ngữ nghĩa; Rich Vision 55/55 từ khóa |
| Tổng thời gian | **1.160,844 giây** (19 phút 20,844 giây) |
| Đánh giá người dùng | Các ví dụ khác nhìn chung ổn; lỗi còn tồn tại được công khai |
| HTTP TPU thật | Chưa nghiệm thu riêng |

Các gói evidence có SHA256 được lưu trong session Kaggle hiện tại hoặc kho lưu trữ khác khi sao lưu; **đường dẫn filesystem bên dưới không phải URL tải public**.

### Evidence của Full19 đã chạy

`/kaggle/working/Qwen38-Full19-SplitVision-Extended448-TPU-AutomatedPass-QA-Pending-Evidence-20261027.zip`

SHA256 `3449356a47a6c72f43d5279649e3d35af3585f8c10044588563b44be1ed2c06a`

Bao gồm notebook nguồn/đã chạy, output và token IDs thật, 16 ảnh gốc, checksum, timing, logs, trạng thái từng case và trang HTML đối chiếu. Không được xem đây là benchmark độ chính xác tổng quát.

### Hai lượt kiểm thử riêng hóa đơn

`/kaggle/working/Qwen38-TPU-Invoice-Tax-Two-Reruns-Evidence-20261027.zip`

SHA256 `781d9f3060380da8d5bd2fc1514389018cbd81bdf10ef4387ded240d241d15d4`

Một engine BF16 và một lần nạp mô hình; hai lượt Vision/AOT prefill và greedy decode mới trên **cùng ảnh và prompt gốc**. Mỗi lượt 171 token, có EOS và token IDs giống nhau. Cả hai đều sinh `Tax (5%): $66.28` thay vì `Tax (8%): $69.28`; tạm tính `$866.00` và tổng `$935.28` vẫn đúng. Validator v2 độc lập đã sửa lỗi regex của bộ đánh giá cũ vốn nhầm “Total” với “Subtotal”; không sửa output mô hình. Greedy mang tính xác định; **không phải hai lượt lấy mẫu ngẫu nhiên độc lập**.

### Giới hạn khác

Dashboard so sánh 114 đơn Thứ Bảy với 31 đơn Thứ Hai là “gần gấp đôi”, nhưng tỷ lệ đúng khoảng 3,68 lần. Giữ nguyên câu sai trong evidence, không âm thầm chỉnh câu trả lời.

## Giới hạn công bố

**PASS kiểm tra tự động** không có nghĩa ảnh và phép tính đều đúng hoàn hảo; cũng chưa phải fresh Run All của notebook Git public. Trước khi phát hành cần chuẩn hóa bộ support 38 file để attach, đường dẫn nguồn ảnh có thể truy cập, notebook Split-Vision 320/768, chạy thật notebook mới, chứng minh provenance/RNG và nghiệm thu HTTP TPU riêng nếu có tuyên bố triển khai dịch vụ.
