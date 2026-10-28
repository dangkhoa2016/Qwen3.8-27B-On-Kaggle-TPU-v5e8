# Full19: Kết quả suy luận TPU và bằng chứng

> 🌐 Ngôn ngữ / Language: [English](FULL19-EVIDENCE.md) | **Tiếng Việt**

Full19 là bộ thử 19 case Text + Vision cho Qwen3.8-27B BF16 trên Kaggle TPU v5e-8. Bộ thử theo dõi việc hoàn tất suy luận, minh họa nhận diện hình ảnh và OCR số; đây không phải benchmark độ chính xác đại diện cho mọi tác vụ.

## Số đo TPU đã ghi — 27/10/2026

| Thuộc tính | Quan sát |
|---|---|
| Checkpoint | Qwen3.8-27B BF16 nguyên gốc, 18 safetensors shard |
| Phần cứng | 8 thiết bị TPU v5e, một MaxEngine |
| Runtime | JAX / MaxText đã điều chỉnh, Split-Vision |
| Số case | 3 Text, 2 OCR số, 6 Rich Vision, 8 Photography |
| Tiền xử lý | Tensor ảnh 512 × 384 |
| Cấu hình token | Prefill 320, target 768, Rich/Photo tối đa 448 bước decode |
| Hoàn thành tự động | **19/19 case đạt EOS** |
| Một số phép kiểm | 3/3 Text, OCR 731 và 482, Rich Vision 55/55 từ khóa |
| Tổng thời gian | **1.160,844 giây** |

Các số đo gắn với SHA256 ảnh và phiên bản chương trình của **lượt đo thực tế**. Notebook và ảnh ví dụ trong Git cung cấp đầu vào để tạo hồ sơ thực thi mới; không gán số benchmark cho SHA256 ảnh hoặc revision khác khi chưa chạy tương ứng.

## Thông tin cần lưu khi suy luận

Một hồ sơ đầy đủ cần có Git SHA, revision model, topology TPU, phiên bản thư viện, SHA256 support manifest, SHA256 ảnh, prompt, token IDs, câu trả lời nguyên văn, EOS, thời gian từng case và lỗi. Với Vision cần giữ kích thước ảnh sau xử lý và thông tin vị trí token đa phương thức.

## Quan sát về chất lượng

- **OCR số:** hai thẻ số đơn giản trả về **731** và **482**.
- **Rich Vision:** 55 lần kiểm từ khóa khớp ở sáu cảnh; không thể từ đó kết luận mọi mô tả đều đúng.
- **Hóa đơn:** thuế trên ảnh là **8% / $69.28**, nhưng model trả **5% / $66.28**.
- **Dashboard:** giá trị **114** so với **31** bị mô tả là “gần gấp đôi”, trong khi tỷ lệ xấp xỉ **3,68**.

Các lỗi được lưu nguyên văn là kết quả đánh giá hữu ích. Cần báo cáo riêng khả năng hoàn tất sinh token và độ chính xác ngữ nghĩa.

[Notebook](../notebooks/kaggle-tpu-v5e8-production-showcase.ipynb) · [Tái lập](REPRODUCIBILITY.vi.md) · [Đánh giá](ACCEPTANCE.vi.md)
