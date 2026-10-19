# Reproducibility

> 🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](REPRODUCIBILITY.vi.md)

1. Clone the repository at an exact reviewed commit. The new notebook itself pins the **examples-only** source commit; do not replace its SHA casually.
2. Attach the specified Qwen3.8-27B Kaggle model and offline support dataset; preflight audits 18 shards and the declared checkpoint size.
3. Preserve JAX/JAXLIB 0.11.0 and Transformers 5.12.1 in the qualified TPU environment. Verify the patched MaxText tar/patch SHA256. Archive metadata is not independent Git ancestry proof.
4. Run `python -m pytest -q tests` and `npm run test:e2e` on CPU; confirm GitHub CI.
5. Fresh TPU Run All/HTTP acceptance must produce new evidence. Never silently attach an earlier executed output to a changed notebook.
6. Do not claim the larger preview images were inference-qualified; do not overstate 22/55 OCR.

## CPU preflight audit (2026-10-19)

The pinned `api/tpu_factory.py` bootstrap now verifies the attached checkpoint index, all 18 named nonempty shards, patch SHA256, offline MaxText archive SHA256, and previously extracted source metadata before touching JAX or loading weights. The archive can be safely extracted offline when missing, without deleting or overwriting an existing MaxText source directory. The extraction rejects unsafe archive paths, links and device nodes. These checks passed on the attached Kaggle CPU model/support mount; a fresh disposable extraction and TPU execution remain unqualified.

**Public Git assets:** all 11 images are in `examples/images/` with SHA256 manifest. The repository is public, and the notebook requires **Internet ON** only for its commit-pinned anonymous Git clone (no credentials). Model weights, pinned support artifacts and MaxText remain offline-first. A fresh Kaggle TPU Run All is **still pending** after changing the notebook and Git history.
