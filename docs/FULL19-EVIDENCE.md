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

## Interpretation

EOS indicates that token generation ended. **It is not a certificate of
factual accuracy.** Review arithmetic, names, relationships and visible
labels against the input images, and keep the unmodified model response
separate from human commentary.

[Review criteria](FAILURE-ANALYSIS.md) ·
[Acceptance checklist](ACCEPTANCE.md) ·
[Execution guide](REPRODUCIBILITY.md)
