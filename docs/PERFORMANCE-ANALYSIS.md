# TPU performance measurement and interpretation

> 🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](PERFORMANCE-ANALYSIS.vi.md)

A trustworthy model benchmark separates initialization, checkpoint load, Vision encoding, compilation, prefill, token decoding and evidence capture. The [Full19 execution checklist](FULL19-EVIDENCE.md) defines the case inventory and fields to collect from the final notebook run. This guide defines a measurement method, not performance results.

## Timing boundaries

| Stage | Record | Interpretation |
|---|---|---|
| Preflight | Input and dependency integrity | Prevents incomplete or incomparable runs |
| Engine construction | Setup elapsed seconds | Separates framework initialization from model I/O |
| Checkpoint load | Weight-loading elapsed seconds | Sensitive to storage and device topology |
| Image processing | Resize, vision embeddings and time | Extra cost relative to text-only tasks |
| Compilation and prefill | JAX/XLA compile and execute timings | Can dominate a cold execution |
| Decoding | Generated tokens, EOS and duration | Needed for token generation speed |
| Evidence capture | JSON writing and elapsed time | Included in total wall-clock time |

## Cache and hardware

Compiled executable reuse requires compatible compiler, shapes, dtypes, sharding and TPU topology. Identical cache bytes do **not** independently establish a cache hit. Preserve JAX runtime versions, device IDs, source Git SHA, model revision, support manifest SHA256 and prompt/image hashes.

Do not divide the total time of a notebook run by its number of cases and label the result HTTP latency. Cold, warm, sequential and concurrent serving have different measurement boundaries.

## Report responsibly

Preserve raw responses, token IDs, EOS, per-case duration, exceptions and numerical/visual errors. Repeat comparable runs before publishing a median or p95. Model correctness and infrastructure completion are separate criteria.

[Inference evidence](FULL19-EVIDENCE.md) · [Execution](REPRODUCIBILITY.md)
