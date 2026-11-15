# Full19 task inventory and verification checklist

🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](FULL19-EVIDENCE.vi.md)

Full19 groups Text and Vision tasks for Qwen3.8-27B BF16
on Kaggle TPU v5e-8. Use the task inventory and evidence checklist
to inspect raw model responses, token IDs, EOS, durations and input hashes
from a Kaggle notebook run.

## Case coverage

**3 Text cases:** 2+2, France capital, Ruby programming language.
Check the actual generated wording and simple correctness criteria.

**2 numeric OCR cases:** input cards showing
[731](../examples/images/vision_test_card_731_1280x960.png) and
[482](../examples/images/vision_test_card_482_1280x960.png).
The task asks for the digits only.

The task descriptions below summarize the original English prompts stored
verbatim in the example JSON files.

**6 Rich Vision tasks**

| Case ID | Source image | Task description |
|---|---|---|
| vision-2d-dashboard | [Open / Xem](../examples/images/vision_2d_city_bike_dashboard.jpg) | Analyze this delivery dashboard |
| vision-2d-greenhouse | [Open / Xem](../examples/images/vision_2d_smart_greenhouse.jpg) | Explain this smart greenhouse diagram |
| vision-3d-robotics | [Open / Xem](../examples/images/vision_3d_robotics_station.jpg) | Interpret this annotated robotics workstation illustration: identify the robot arm and storage bins, read the Test Run, temperature, A1/A2/B1/B2 labels and sticky note; explain the action that should happen first |
| vision-3d-warehouse | [Open / Xem](../examples/images/vision_3d_warehouse_dock_b2.jpg) | Read this annotated warehouse workflow graphic |
| vision-real-workspace | [Open / Xem](../examples/images/vision_real_sprint_workspace.jpg) | Extract the sprint-planning date, three agenda items, reminder time, phone time, and two receipt prices shown on this illustrated planning desk; calculate the total and minutes until reminder |
| vision-real-market | [Open / Xem](../examples/images/vision_real_fresh_market.jpg) | Extract all five products and prices on this illustrated market board, determine cheapest and most expensive, read opening time, and compute the price difference between highest and lowest |

**8 Photography tasks**

| Case ID | Source image | Task description |
|---|---|---|
| photo-office-invoice | [Open / Xem](../examples/images/northstar_office_supply_invoice.jpg) | Read this office invoice, clearly identifying the supplier, readable line items and total |
| photo-business-card | [Open / Xem](../examples/images/warm_desk_still_life_with_business_card.jpg) | Describe the desk and transcribe only clearly visible business card contact information |
| photo-daylight-workspace | [Open / Xem](../examples/images/cozy_daylight_workspace_with_city_views.jpg) | Describe the objects, layout, lighting and outside view of this workspace |
| photo-ai-farmers-market | [Open / Xem](../examples/images/photo_ai_farmers_market_1280x960.png) | Describe the market scene |
| photo-ai-city-crossing | [Open / Xem](../examples/images/photo_ai_city_crossing_1280x960.png) | Describe what is happening at this urban crosswalk |
| photo-ai-vietnamese-meal | [Open / Xem](../examples/images/photo_ai_vietnamese_meal_1280x960.png) | Identify the dishes, accompaniments, condiments and drinks visible on the restaurant table |
| photo-ai-dog-walking | [Open / Xem](../examples/images/photo_ai_dog_walking_1280x960.png) | Describe this park scene and the relationship between the person, dog, leash, path and nearby lake |
| photo-ai-supermarket | [Open / Xem](../examples/images/photo_ai_supermarket_1280x960.png) | Describe the supermarket aisle and the shopper's visible action |

All 16 image files have SHA256 entries in
[the input manifest](../examples/assets_manifest.json).
Images are inference inputs, not inference responses. Some scenes are
AI-generated illustrations; do not represent them as documentary photos.

## Evidence required after executing the final notebook

Collect a result record containing:

1. Git SHA, BF16 model reference, TPU device topology, JAX/runtime
   versions and input-support manifest SHA256;
2. Case ID, exact input prompt and image SHA256 where applicable;
3. Original model response, generated token IDs, EOS status and errors;
4. End-to-end and phase-level duration with cold/warm distinctions;
5. Machine-readable JSON and the self-contained static HTML Gallery;
6. Separate manual assessment of visual facts and numerical reasoning.

The template writes a UTC-stamped folder under /kaggle/working, with
cases/full19-final.json for English, cases/full19-vi.json when Vietnamese
inference is enabled, and the exported Gallery.

## Example run results and provenance

These recorded outputs illustrate how to inspect a completed Full19 run.
Generation completion, visual correctness and model performance are different measurements.

### Git-first notebook (15 November 2026 UTC)

The Kaggle TPU v5e-8 notebook ran the original BF16 checkpoint with public Git
source resolved from `main`. Its `source-provenance.json` records the exact
source commit `f3c440fb78d1c6de0892a6ebc87abb9c9ff9d007`.

### Download the Full19 outputs

The original run outputs are distributed as **[v1.0.0 GitHub Release Assets](https://github.com/dangkhoa2016/Qwen3.8-27B-On-Kaggle-TPU-v5e8/releases/tag/v1.0.0)**.
They remain available independently of the Kaggle notebook page or its URL.

| Download | Contents |
|---|---|
| [Source provenance](https://github.com/dangkhoa2016/Qwen3.8-27B-On-Kaggle-TPU-v5e8/releases/download/v1.0.0/source-provenance.json) | Git source commit and runtime provenance |
| [Full19 results ZIP](https://github.com/dangkhoa2016/Qwen3.8-27B-On-Kaggle-TPU-v5e8/releases/download/v1.0.0/full19-results.zip) | Original `cases/full19-final.json` and `cases/full19-vi.json`, plus their SHA256 checksums |
| [HTML Gallery ZIP](https://github.com/dangkhoa2016/Qwen3.8-27B-On-Kaggle-TPU-v5e8/releases/download/v1.0.0/inference-gallery.zip) | Static Gallery, including `gallery/evidence.json` and source images |
| [SHA256 checksums](https://github.com/dangkhoa2016/Qwen3.8-27B-On-Kaggle-TPU-v5e8/releases/download/v1.0.0/SHA256SUMS) | Hashes of the published provenance and ZIP files |

The JSON contents are preserved byte-for-byte inside the results ZIP, rather than
committed as thousands of lines of generated data. After extracting it, verify
the JSON files against the included `SHA256SUMS`. The Gallery ZIP also remains
unchanged and can be viewed locally using a static HTTP server; no TPU run or
external notebook is needed.

| Recorded field | Observed result |
|---|---|
| TPU devices | 8/8 TPU v5e devices |
| Notebook execution | 17/17 code cells completed without Python error outputs |
| English Full19 | 19/19 cases reached EOS |
| Vietnamese responses | 17/17 generated; the two numeric OCR cards reuse their digit outputs |
| Runtime Assets | 48 files checked against the support SHA256 manifest |
| Gallery export | Static HTML, 19 case entries, 16 source images and `inference-gallery.zip` |
| Gallery verification | ZIP integrity, image SHA256 and HTTP access checks completed |

The saved Gallery ZIP has SHA256
`ad528622567f8c0e283014aac0091eadf6fe1a43476decbc9b78f71151f14a0b`.
The support manifest SHA256 is
`fb2ee48dfe423733499112668d61e965b5d8d7466d52361b403c68dbc5768817`.
The original source commit remains traceable if later documentation changes advance `main`.

The results are **not a factual-accuracy score**. The [review criteria](FAILURE-ANALYSIS.md)
cover invoice tax values, delivery-dashboard arithmetic and missing business-card
details across English and Vietnamese responses. Keep raw model outputs unchanged.

### Earlier embedded-source run (12 November 2026 UTC)

This is a separate historical Kaggle execution, not another notebook in this
repository. Its cold-start measurements must not be attributed to the Git-first run.

| Historical observation | Observed result |
|---|---|
| TPU devices | 8/8 TPU v5e devices |
| Notebook execution | 17/17 code cells without Python error outputs |
| English Full19 | 19/19 cases reached EOS |
| Vietnamese responses | 17/17 generated; the two numeric OCR cards reused their digit outputs |
| Export | Static HTML Gallery and `inference-gallery.zip` |
| First checkpoint load | 149.339 s |
| First OCR diagnostic | Vision ~33.95 s; prefill ~259.80 s; decode ~186.80 s for 4 tokens, including cold-start overhead |

These first-pass timings are not steady-state throughput or HTTP latency.
The raw responses and numerical/visual accuracy require independent review.

## Interpretation

EOS indicates that token generation ended. **It is not a certificate of
factual accuracy.** Review arithmetic, names, relationships and visible
labels against the input images, and keep the unmodified model response
separate from human commentary.

[Review criteria](FAILURE-ANALYSIS.md) ·
[Acceptance checklist](ACCEPTANCE.md) ·
[Execution guide](REPRODUCIBILITY.md)
