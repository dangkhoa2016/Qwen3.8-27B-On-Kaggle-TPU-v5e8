# Kiến trúc hệ thống và ranh giới nghiệm thu

> [English](ARCHITECTURE.md) | **Tiếng Việt**

Có **ba phạm vi nghiệm thu độc lập**:

1. **Notebook thử nghiệm Full19 trên TPU thật:** Qwen3.8-27B BF16 gốc, 18 shards, 8 TPU v5e, một MaxEngine. Notebook Split-Vision MaxText/JAX riêng đã thực thi Vision encoder → đưa image embeddings vào decoder → AOT prefill → greedy generate. Giữ nguyên các kiểm tra native sharding/layout của JAX/XLA. Full19 **PASS 19/19 kiểm tra tự động** gồm 2 OCR, 6 Rich Vision, 8 Photography và 3 Text. **Chưa phải** fresh Run All của notebook trong Git.
2. **FastAPI + Vue:** mock CPU qua loopback và các test HTTP contract. Không thực thi Qwen 27B thật.
3. **HTTP TPU thử nghiệm:** ASGI được bảo vệ và bật có điều kiện; chưa nghiệm thu request Text/Vision thật, xác thực, concurrency, ingress và các trạng thái lỗi. Không đưa endpoint inference này ra tunnel công khai.

```text
Kaggle Model (18 BF16 shards) + offline support (38 tệp kiểm tra SHA256)
          │
          ▼
TPU v5e-8 ×8, MaxText/JAX 0.11.0, một MaxEngine
          ├── ảnh → resize 512×384 → Qwen Vision encoder → image embeddings
          └── text token fusion + MRoPE + AOT prefill có kiểm tra
                                         └── insert / greedy generate → EOS thật
                                                     └── Full19 outputs + provenance
Độc lập: Vue CDN → FastAPI mock CPU → tests (không chạy Qwen thật)
Tùy chọn: TPU ASGI phải nghiệm thu riêng trước khi public
```

Notebook Git hiện còn đường direct-prefill 256/384 cũ, không tương đương với candidate Split-Vision đã PASS. Phải thay và nghiệm thu mới trước official Run All.

[Nghiệm thu](ACCEPTANCE.vi.md) · [Tái lập](REPRODUCIBILITY.vi.md) · [Bằng chứng](FULL19-EVIDENCE.vi.md).
