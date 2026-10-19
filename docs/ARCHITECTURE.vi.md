# System architecture

> 🌐 Ngôn ngữ / Language: [English](ARCHITECTURE.md) | **Tiếng Việt**

Hệ thống có **ba cấp độ bằng chứng**: notebook từng PASS suy luận TPU BF16 Text/Vision thật; REST API/UI CPU được kiểm thử ở chế độ mock; TPU HTTP thử nghiệm **chưa nghiệm thu**.

```text
Trình duyệt (Vue CDN)
     │
FastAPI /api/app.py (mock) ──> CPU mock/test-double
     │
[entrypoint thử nghiệm: api/tpu_asgi.py]
     └──> InferenceRuntime (một runner, tuần tự request)
              └──> TPU factory -> BF16 MaxEngine -> prefill / greedy generate
```

Mô hình thật cần TPU v5e-8 (8 devices). Không dùng nhiều worker TPU hay mở ingress công khai trước khi nghiệm thu HTTP thực. Bootstrap kiểm tra model/support dataset, source đã ghim và các phiên bản JAX. Không đưa trọng số lên Git.
