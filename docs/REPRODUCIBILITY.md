# Reproducibility

> 🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](REPRODUCIBILITY.vi.md)

1. Clone the repository at an exact reviewed commit. The new notebook itself pins the **examples-only** source commit; do not replace its SHA casually.
2. Attach the specified Qwen3.8-27B Kaggle model and offline support dataset; preflight audits 18 shards and the declared checkpoint size.
3. Preserve JAX/JAXLIB 0.11.0 and Transformers 5.12.1 in the qualified TPU environment. Verify the patched MaxText tar/patch SHA256. Archive metadata is not independent Git ancestry proof.
4. Run `python -m pytest -q tests` and `npm run test:e2e` on CPU; confirm GitHub CI.
5. Fresh TPU Run All/HTTP acceptance must produce new evidence. Never silently attach an earlier executed output to a changed notebook.
6. Do not claim the larger preview images were inference-qualified; do not overstate 22/55 OCR.
