# Lessons from model errors and evaluation pitfalls

> 🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](FAILURE-ANALYSIS.vi.md)

A useful inference example should make failures inspectable. The Full19 exercise produced several observations that help explain **why successful generation, matching keywords and even EOS are not enough to guarantee correctness**.

## Case study 1 — Invoice arithmetic and labels

The office-invoice source image contains:

| Field | Source image | Reported model output |
|---|---|---|
| Subtotal | **$866.00** | Preserved |
| Tax | **8% / $69.28** | **5% / $66.28** |
| Total | **$935.28** | Preserved |

Two additional greedy calls with the same image and prompt reproduced the incorrect tax value, each reaching EOS with **171 generated token IDs** and identical token sequences.

**Lesson:** identify tax rates and values from the image rather than assuming a plausible percentage. Arithmetic validation can detect contradictions, but should never silently repair or relabel the raw output.

## Case study 2 — Comparative language

A delivery dashboard shows **114 Saturday deliveries** and **31 Monday deliveries**. The model called the former “nearly double.” In fact,

`114 / 31 ≈ 3.68`

The entities were identified, yet their relationship was poorly described. **Lesson:** numerical grounding requires checking the relationship, not only the individual figures.

## Case study 3 — Tests can make mistakes, too

An earlier validator regex accidentally matched `Total` within `Subtotal`. A later validator corrected that parsing bug. This was a **test-harness correction**, not a retroactive model improvement.

Separate three things when auditing a run: **source truth**, **generated model text**, and **scorer interpretation**. Preserve the original model text even after fixing a scorer.

## Case study 4 — Image provenance affects claims

The earlier crossing source showed a **red do-not-walk pedestrian signal** while people occupied the zebra crossing. Qwen's archived response noticed and warned about that conflict. The Git repository now uses a **different green-signal image** with a different SHA256. The old answer cannot be reused to score the new image.

**Lesson:** a filename or case ID is not enough to identify the input. Include its SHA256 whenever reporting an output.

## Practical safeguards

- Create per-case truth annotations for exact numbers and safety-critical visual details.
- Evaluate OCR digits, arithmetic comparisons and natural-language fidelity separately.
- Record every output and failure; do not publish a cherry-picked set as the full run.
- Pin image and prompt hashes alongside generated token IDs, EOS and timings.
- Run deterministic repeats to test reproducibility, but do not portray them as statistically independent samples.
- Keep an execution result distinct from factual correctness and from production readiness.

These observations are useful for anyone building image-understanding demos, document OCR evaluations or LLM regression tests. They are **not** a claim that all Qwen outputs exhibit the same errors.

[Evaluation matrix](ACCEPTANCE.md) · [Experimental evidence](FULL19-EVIDENCE.md) · [Source images](../examples/README.md)
