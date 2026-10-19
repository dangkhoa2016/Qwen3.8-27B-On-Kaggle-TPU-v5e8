# Acceptance evidence and limitations

> 🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](ACCEPTANCE.vi.md)

The earlier real TPU notebook execution recorded one engine/BF16 load, 8 TPU devices, text cases including simple arithmetic and France's capital, deterministic numeric vision cards, and six scene-grounding cases. See `evidence/qwen38-production-fresh-executed.ipynb`, `evidence/qwen38_27b_tpu_v5e8_text_vision_fresh_20261018.json`, and `evidence/SHA256_MANIFEST.json`.

**Historical result:** TPU notebook acceptance PASS with richer-scene OCR **22/55**, which is not a broad OCR pass. The Git-first notebook was subsequently rewritten and has **not** been freshly executed. The HTTP TPU endpoint has **not** been accepted. CPU API unit tests, browser E2E and GitHub CI verify only the mock/test-double contract.

Qualification must include live TPU readiness, 1 engine/1 load, text + vision through HTTP, concurrency 429, error paths, authentication, lifecycle, source hashes, actual measured latency and saved evidence; publishing a v1.0.0 release is separate.
