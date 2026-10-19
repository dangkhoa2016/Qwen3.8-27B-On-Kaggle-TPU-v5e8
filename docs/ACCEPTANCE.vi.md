# Acceptance evidence and limitations

> 🌐 Ngôn ngữ / Language: [English](ACCEPTANCE.md) | **Tiếng Việt**

Notebook TPU thật trước đây ghi nhận một engine/BF16 load, 8 thiết bị TPU, câu hỏi text số học/thủ đô Pháp, thẻ số vision xác định và 6 bài cảnh. Xem `evidence/qwen38-production-fresh-executed.ipynb`, `evidence/qwen38_27b_tpu_v5e8_text_vision_fresh_20261018.json` và `evidence/SHA256_MANIFEST.json`.

**Kết quả lịch sử:** notebook TPU PASS, nhưng OCR cảnh phong phú chỉ **22/55**. Notebook Git-first đã chỉnh lại và **chưa Run All mới**. API HTTP TPU **chưa nghiệm thu**. CI CPU chỉ chứng minh hợp đồng mock/test-double.

Nghiệm thu cuối phải chứng minh readiness, một engine/load, HTTP Text+Vision thật, 429 khi bận, lỗi, xác thực, vòng đời, SHA, latency đo trực tiếp và lưu evidence; phát hành v1.0.0 là công việc riêng.
