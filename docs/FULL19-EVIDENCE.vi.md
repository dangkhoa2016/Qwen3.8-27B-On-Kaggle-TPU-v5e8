# Full19: Bằng chứng và kết quả suy luận TPU

> 🌐 Ngôn ngữ / Language: [English](FULL19-EVIDENCE.md) | **Tiếng Việt**

Full19 là **bộ thử 19 prompt Text + Vision**, giúp quan sát một phiên chạy mô hình đa phương thức lớn từ input đến output. Đây là một tập case được chọn để minh họa, **không phải benchmark độ chính xác đại diện cho mọi tác vụ**.

## Lượt thực nghiệm đã ghi ngày 27/10/2026

| Thuộc tính | Quan sát có bằng chứng |
|---|---|
| Checkpoint | Qwen3.8-27B BF16; 18 shard safetensors, 1.199 tensor |
| Phần cứng | TPU v5e-8, tám thiết bị, một MaxEngine |
| Runtime | Split-Vision JAX/MaxText; JAX/JAXLIB 0.11.0 |
| Tiền xử lý ảnh | Ảnh nguồn chuyển về 512 × 384 |
| Budget | Prefill 320 / target 768 / Rich+Photo tối đa 448 decode |
| Case | 3 Text, 2 OCR số, 6 Rich Vision, 8 Photography |
| Hoàn thành chương trình | **19/19 case riêng biệt có EOS** |
| Kiểm tự động | 3/3 Text, OCR **731** và **482**, 55/55 keyword Rich |
| Tổng wall time | **1.160,844 giây (19 phút 20,844 giây)** |

Kết quả thuộc **runner experimental đã chạy thật** với hash ảnh đầu vào ban đầu; đây **không** phải kết quả Run All fresh của notebook Git khi dùng ảnh crossing mới. Một số câu trả lời vẫn sai nội dung dù hoàn thành kiểm thử tự động.

## Cách đọc evidence

Mỗi case cần có prompt gốc, SHA256 ảnh, full generated text, token IDs, EOS, code revision, topology và thời gian. Với Vision còn cần kích thước tiền xử lý và image-token offset.

Các archive evidence gốc trên Kaggle có checksum:

| Archive | SHA256 |
|---|---|
| Notebook đã chạy, kết quả Full19 và tài liệu review | `3449356a47a6c72f43d5279649e3d35af3585f8c10044588563b44be1ed2c06a` |
| Hai lượt greedy kiểm invoice | `781d9f3060380da8d5bd2fc1514389018cbd81bdf10ef4387ded240d241d15d4` |

Đây là **artifact lưu trữ, không phải link tải công khai**. Hash cho phép đối chiếu khi archive được cung cấp hợp lệ. Trong Git hiện có manifest, mã notebook, case definitions và bản mô tả kết quả—không chứa các ZIP evidence experimental đã chạy.

## Các quan sát đáng chú ý

**OCR số:** hai thẻ đơn giản cho kết quả **731**, **482**. Không thể suy ra độ chính xác OCR tài liệu tổng quát.

**Rich Vision:** sáu cảnh đạt 55 keyword, nhưng có keyword không bảo đảm câu trả lời hoàn toàn đúng.

**Invoice:** ảnh ghi thuế **8% / $69.28**, model trả **5% / $66.28**; hai lần greedy kiểm lại vẫn sai giống nhau.

**Dashboard:** model mô tả 114 so với 31 là “gần gấp đôi”, trong khi khoảng **3,68 lần**.

**Ảnh sang đường:** output experimental dùng ảnh cũ đèn đỏ; ảnh đèn xanh trong Git và Dataset Release Candidate Private có SHA256 khác. Không được gán output cũ cho ảnh mới.

## Tái lập và mở rộng bộ thử

Đường dẫn ổn định là [notebook Full19 được quản lý trong Git](../notebooks/kaggle-tpu-v5e8-production-showcase.ipynb), gồm **19 cell / 14 code cell** và pipeline Split-Vision AOT. Khi chạy cần attach đúng Kaggle BF16 mirror và support tương ứng với SHA256 manifest trong notebook. Evidence experimental cũ chưa chứng minh Full19 trên ảnh crossing mới.

Dùng [hướng dẫn tái lập](REPRODUCIBILITY.vi.md) để chạy, [hướng dẫn đánh giá](ACCEPTANCE.vi.md) để chấm và [bài học lỗi](FAILURE-ANALYSIS.vi.md) để kiểm ngữ nghĩa. Không sử dụng kết quả cũ làm số đo cho Dataset mới.
