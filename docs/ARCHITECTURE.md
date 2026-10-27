# System architecture and acceptance boundary

> **English** | [Tiếng Việt](ARCHITECTURE.vi.md)

There are **three separate qualification levels**:

1. **Real TPU Full19 experimental notebook:** original Qwen3.8-27B BF16 weights, 18 shards, Kaggle TPU v5e-8 ×8, one MaxEngine. A separately preserved Split-Vision MaxText/JAX candidate performed Vision encoding → image embedding injection → compiled decoder prefill → greedy token generation. Native JAX/XLA sharding/layout verification remained enabled. The verified 19/19 automated case result includes 2 OCR, 6 rich scenes, 8 photography cases and 3 text prompts. This is **not yet** the public Git-tracked notebook running freshly.
2. **FastAPI + Vue demo:** loopback CPU mock responses and HTTP contract tests. The mock is not the 27B model.
3. **Opt-in experimental TPU HTTP:** separate guarded ASGI lifecycle; concurrency, ingress security, real Text/Vision requests and failure paths are still **unqualified**. Do not expose this inference endpoint via public tunnel.

```text
Model mirror (18 BF16 shards) + offline support (38 verified files)
         │
         ▼
Kaggle TPU v5e-8 ×8, MaxText/JAX 0.11.0, one MaxEngine
         ├── image → 512×384 preprocessing → Qwen Vision encoder
         │                                    └── image_embeddings
         └── text token fusion + MRoPE + verified AOT prefill
                                         └── insert / greedy generate → real EOS
                                                     └── Full19 outputs + provenance
CPU-only: Vue CDN → FastAPI mock → contract tests [separate, not real Qwen inference]
Optional: TPU ASGI entrypoint → unqualified until independent live acceptance
```

The currently checked-in older notebook uses a different direct-prefill 256/384 code path. It must be replaced and verified before official Run All, not silently treated as equivalent to the accepted private candidate.

[Acceptance](ACCEPTANCE.md) · [Reproducibility](REPRODUCIBILITY.md) · [Evidence](FULL19-EVIDENCE.md).
