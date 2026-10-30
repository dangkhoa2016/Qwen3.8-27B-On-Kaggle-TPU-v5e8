# Visual document extraction and numerical reasoning review

🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](FAILURE-ANALYSIS.vi.md)

This is a set of **evaluation criteria** for use after executing the final
Full19 notebook. The values below are from the input images. They are
**not assertions about model-generated responses**.

## Invoice verification

[View the office invoice](../examples/images/northstar_office_supply_invoice.jpg).

| Field | Value on the image |
|---|---|
| Supplier | Northstar Office Supply |
| Subtotal | $866.00 |
| Tax rate | 8% |
| Tax amount | $69.28 |
| Total | $935.28 |

Compare the generated answer against all visible fields.
Verify that 866.00 × 8% = 69.28 and 866.00 + 69.28 = 935.28.
A fluent answer or correct total does not guarantee correct tax extraction.

## Delivery dashboard verification

[View the delivery dashboard](../examples/images/vision_2d_city_bike_dashboard.jpg).
Verify the visible KPI cards, daily values and category proportions.
When comparing 114 and 31, the ratio is approximately **3.68**, not
“nearly double”. Compare any generated explanation with the image.

## Business-card verification

[View the desk and business card](../examples/images/warm_desk_still_life_with_business_card.jpg).
Compare readable name, professional title and contact details with each
generated EN/VI answer. Track omissions independently of EOS.

## Review protocol

After the notebook executes, save output text and its SHA256-bound
input metadata. Verify each value against source pixels, calculate
numerical comparisons independently, and attach reviewer notes separately.
Do not rewrite model text to improve a score.

[Full19 task inventory](FULL19-EVIDENCE.md) ·
[Acceptance criteria](ACCEPTANCE.md)
