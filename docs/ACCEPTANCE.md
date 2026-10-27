# How to evaluate the Full19 Text + Vision showcase

> 🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](ACCEPTANCE.vi.md)

A useful model demo should show **what was tested**, **what succeeded** and **what the model got wrong**. This guide defines an observable evaluation method that readers can reuse rather than treating a single PASS badge as a general accuracy score.

## What is Full19?

| Case family | Cases | What to inspect |
|---|---:|---|
| Text | 3 | Meaning of short requested answers, EOS |
| Numeric OCR cards | 2 | Exact digits **731** and **482**, EOS |
| Rich Vision | 6 | Recognition of scene attributes; 55 expected keyword checks |
| Photography | 8 | Description grounded in actual images; traffic/objects/relationships |
| **Total** | **19** | Unique case IDs, raw responses, token IDs, EOS and timing |

The photographic set includes AI-generated source scenes and other images; those inputs are **not images produced by Qwen**. A keyword match establishes only that a specified phrase or concept occurred, not that every numerical or spatial claim was correct.

## Levels of evidence

| Question | How to answer it | Observed status |
|---|---|---|
| Are inputs intact? | Verify source image and support SHA256 | 48/48 file checks on independent Kaggle download |
| Does the API contract work? | CPU pytest and browser Playwright | 36 Python PASS, 1 SKIP; 2 browser PASS |
| Did Qwen really generate outputs on TPU? | Full raw tokens, eight-device runtime and logs | Experimental Full19 on **27 October 2026** |
| Is a specific response factually correct? | Compare text with original source image or exact value | Mixed: known invoice and dashboard errors |
| Has the replacement crossing image been tested on TPU? | Compare input SHA256 and a complete new run | Not established by the archived experimental evidence |
| Does TPU HTTP serving work? | Authenticated real Text/Vision requests with timing | Not qualified |

## What the experimental run showed

The preserved experimental run completed **19/19 automated cases with EOS** in **1,160.844 seconds** on TPU v5e-8. It recorded 3 Text checks, two exact OCR card results and 55/55 Rich Vision keyword checks. The runner and original image SHA256 values must stay associated with those recorded answers.

### Why an automated pass is not a perfect answer

- **Invoice:** source subtotal **$866.00**, tax **8% / $69.28**, total **$935.28**. Model output stated **5% / $66.28**. Two targeted greedy repeats produced the same mistake and matching token IDs.
- **Delivery chart:** source values **114** on Saturday versus **31** on Monday. Describing them as “nearly double” was incorrect; the ratio is about **3.68**.
- **Crossing:** an older red-signal image was correctly recognized as an unsafe pedestrian situation; the newer green-signal Git image has **not** been covered by that earlier output.

These are instructive examples of why every generated sentence needs inspection when accuracy matters. See [model-behavior lessons](FAILURE-ANALYSIS.md).

## Reusable review checklist

1. Freeze notebook commit, original model revision, support manifest and every source image SHA256.
2. Confirm eight TPU devices, one engine and native JAX/XLA validation.
3. Preserve every prompt and its **full unmodified** output, token IDs, EOS status and duration.
4. Verify **19 unique IDs** and all defined automated checks.
5. Manually review quantities, image relations, traffic signals, uncertainty and missing details.
6. Record failures explicitly. A missing answer is not a PASS; a completed but wrong answer is **completion PASS / semantic FAIL**.
7. Report measured wall-clock time separately from per-request latency, throughput and unmeasured values.

A complete evaluation table should have independent columns for **executed**, **EOS**, **automated checks**, **human factual review** and **source SHA256**. It is more informative than a single accuracy percentage.

## Current public-claim boundary

The [image manifest](../examples/assets_manifest.json) identifies the replacement green-signal crossing input. Input integrity alone does not establish successful inference: publish its new Full19 outcomes only after checking real model outputs, EOS, image SHA256 and factual correctness.

[Case evidence](FULL19-EVIDENCE.md) · [Reproduce the experiment](REPRODUCIBILITY.md) · [Performance](PERFORMANCE-ANALYSIS.md)
