# Model behavior: OCR and visual reasoning case studies

> 🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](FAILURE-ANALYSIS.vi.md)

Full19 contains examples where the model completed generation yet produced inaccurate content. These findings are documented so that readers can distinguish the source image, the model's raw answer and automated validation.

## Case study 1 — Invoice tax

| Field | Value visible in image | Reported model answer |
|---|---|---|
| Subtotal | $866.00 | Correct |
| Tax | **8% / $69.28** | **5% / $66.28** |
| Total | $935.28 | Correct |

Two additional greedy runs with the same input returned the same incorrect tax value and reached EOS. The model output should be preserved rather than silently corrected.

## Case study 2 — Numeric comparison

A dashboard shows **114** Saturday deliveries and **31** Monday deliveries. The model described the difference as “nearly double,” although `114 / 31 ≈ 3.68`. Recognizing individual values is not sufficient to validate a claim about their relationship.

## Case study 3 — Validation boundaries

An automated keyword or pattern match can pass despite a wrong number or a misleading explanation. Evaluate source-image truth, generated raw text and scorer output as separate records. When a validator is improved, keep the original model response for auditability.

## Case study 4 — Input provenance

Visual conclusions depend on the exact image presented to the model. File names and case IDs alone cannot establish input identity; pin image SHA256 and source dimensions for each result. If image bytes change, generate new evidence instead of transferring another image's answer.

These examples illustrate evaluation methodology, **not** a claim that every Qwen response is incorrect or that a particular accuracy percentage has been measured.

[Evaluation guide](ACCEPTANCE.md) · [Full19 results](FULL19-EVIDENCE.md) · [Image inputs](../examples/README.md)
