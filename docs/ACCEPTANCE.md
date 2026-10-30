# Evaluating the Full19 Text + Vision exercise

> 🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](ACCEPTANCE.vi.md)

This guide evaluates **execution completeness** and **answer quality** independently. Reaching EOS does not prove factual correctness; content checks must not discard raw outputs.

## Case coverage

| Group | Cases | What to examine |
|---|---:|---|
| Text | 3 | Coherent response, expected content and EOS |
| Numeric OCR | 2 | Digits 731 and 482, EOS |
| Rich Vision | 6 | Visible objects, relationships and 55 keyword checks |
| Photography | 8 | Grounded descriptions of images, objects and relations |
| **Total** | **19** | Prompt, input SHA256, raw answer, token IDs and timings |

Some Photography images are AI-created illustrations used as inputs to Qwen. Keyword checks test coverage of selected terms, not the full factual content.

## Evidence checklist

1. Identify the exact notebook Git SHA, checkpoint revision, 48-file support manifest SHA256 and image SHA256.
2. Verify eight TPU devices, one engine and native JAX/XLA compatibility checks.
3. Preserve the full prompt, generated token IDs, raw text, EOS and time for every case, including failures.
4. Check there are 19 unique case IDs and report automated completion separately from correctness.
5. Review OCR labels, numbers and arithmetic as well as traffic signals, objects and scene relationships.
6. For comparison with another run, require equivalent inputs and configuration or clearly describe the differences.

## Interpretation

The CPU FastAPI/Vue playground demonstrates request contracts but does not load BF16 model weights. It cannot serve as evidence of TPU throughput or generation accuracy.

[Full19 input inventory and evidence checklist](FULL19-EVIDENCE.md) · [Performance](PERFORMANCE-ANALYSIS.md) · [Run guide](REPRODUCIBILITY.md)
