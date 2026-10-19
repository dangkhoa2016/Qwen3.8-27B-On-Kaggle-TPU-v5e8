# System architecture

> 🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](ARCHITECTURE.vi.md)

The stack has **three evidence levels**. The notebook contains previous real BF16 TPU Text/Vision acceptance, the CPU REST/UI is mock-only with CI coverage, and the experimental TPU server is **not qualified**.

```text
Browser (Vue CDN)
     │
FastAPI /api/app.py (mock only) ──> CPU fake/mock contract
     │
[separate experimental opt-in: api/tpu_asgi.py]
     └──> InferenceRuntime (single runner / serialized requests)
              └──> TPU startup factory -> BF16 MaxEngine -> prefill / greedy generate
```

A single TPU v5e-8 (8 devices) is required for real weights. Do not run two worker processes, do not permit public ingress before a live TPU HTTP acceptance. Startup is gated on attached model and support dataset, pinned source and JAX versions. No weight transfer through Git.
