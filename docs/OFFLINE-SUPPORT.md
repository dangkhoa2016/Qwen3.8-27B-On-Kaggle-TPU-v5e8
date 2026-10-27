# Offline Full19 support — attach and provenance

> **English** | [Tiếng Việt](OFFLINE-SUPPORT.vi.md)

The production notebook expects two separately attached resources: the original BF16 Kaggle Model and one offline-support Dataset. The support archive contains **no pretrained 27B model weights**.

## Prepared bundle (27 October 2026)

- Local source folder: qwen38-tpu-v5e8-full19-offline-support-20261027
- **38 unmodified, SHA256-verified original support files**: patched MaxText archive, runtime wheels, engine helper, all 16 source images and case manifests.
- **6 verified helpers**: Split-Vision patch and five native JAX/XLA observation and strict reuse helpers.
- **3 existing JAX compilation-cache candidate binaries** plus one cache manifest.
- **48 content entries**, totaling 125,595,148 bytes, checked individually using SHA256.
- Root manifest: portable-bundle-manifest.json
- **Pinned manifest SHA256:** 1447ee114dc2d2abce6f60f5125927f55311bf530b8cf53ab86a2576ac637ebc
- Portable ZIP contains 49 entries: the 48 content files plus root manifest. It is stored on the current Kaggle machine, not a public URL.

The manifest pins the exact bytes used by the candidate notebook. The original 38-file set also retains its existing private-rc-sha256-manifest.json check.

## Upload, attachment and execution

1. Upload the **contents** of the prepared folder as a new private Kaggle Dataset, with no extra directory level. Suggested slug: qwen38-tpu-v5e8-full19-offline-support. Dataset upload has **not** been performed or verified yet. Validate the mounted files and SHA256 after upload.
2. Attach Kaggle Model dangkhoa2016/qwen-qwen3-8-27b (Transformers/default/v1), original 18 safetensors shards.
3. Attach the support dataset; notebook searches /kaggle/input/datasets/dangkhoa2016/qwen38-tpu-v5e8-full19-offline-support or /kaggle/input/qwen38-tpu-v5e8-full19-offline-support. Alternatively set QWEN38_SUPPORT_DIR to the exact attached folder.
4. Run the notebook once with TPU v5e-8 (8 devices). SHA256 is verified **before model load** and output evidence is written to a unique time-stamped directory.
5. The three historical cache files are copied into a writable run-specific cache folder. They may be ignored or incompatible in another session; a cache hit is not guaranteed and never permits skipping native JAX sharding/layout checks.
6. Preserve genuine output, all generated token IDs, EOS, durations and errors. Full19 experimental acceptance already exists, but fresh production notebook execution is still pending.

**Distribution caution:** model Apache-2.0 does not automatically cover patched MaxText code, third-party Python wheels or AI-generated source images. Verify separate license and source redistribution conditions before making this support dataset public. Do not include tokens, credentials or model weights.

[Reproducibility](REPRODUCIBILITY.md) · [Acceptance](ACCEPTANCE.md) · [Full19 evidence](FULL19-EVIDENCE.md)
