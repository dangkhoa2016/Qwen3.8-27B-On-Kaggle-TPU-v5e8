"""Build a portable, CPU-only EN/VI results gallery from actual TPU result JSON.

No model inference, network access, external frontend dependencies, or fabricated outputs.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import shutil
from pathlib import Path

EXPECTED = {"text": 3, "ocr": 2, "rich": 6, "photo": 8}
GROUP_ORDER = {"text": 0, "ocr": 1, "rich": 2, "photo": 3}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data


def safe_json(obj: object) -> str:
    """Script-safe JSON: source data cannot terminate an inline <script> element."""
    return (json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
            .replace("&", "\\u0026").replace("<", "\\u003c")
            .replace(">", "\\u003e").replace("\u2028", "\\u2028")
            .replace("\u2029", "\\u2029"))


def build_gallery(en_results: Path, assets_dir: Path, destination: Path,
                  vi_results: Path | None = None, manifest: Path | None = None,
                  review: Path | None = None) -> Path:
    en = load_json(en_results)
    rows = en.get("cases")
    if not isinstance(rows, list) or len(rows) != 19:
        raise ValueError("Gallery needs the original complete 19-case English result JSON")
    accepted = {"AUTOMATED_19_CASES_EOS_COMPLETED",
                "AUTOMATED_19_CASES_RECORDED_NEEDS_REVIEW"}
    if en.get("status") not in accepted:
        raise ValueError("Full19 English case collection has not completed")
    if en.get("status") == "AUTOMATED_19_CASES_EOS_COMPLETED" and not all(
            r.get("eos_reached") for r in rows):
        raise ValueError("Full19 EOS status contradicts the recorded cases")
    ids = [row.get("id") for row in rows]
    if any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != 19:
        raise ValueError("Missing or duplicate case IDs")
    groups = {g: sum(row.get("group") == g for row in rows) for g in EXPECTED}
    if groups != EXPECTED or any(not isinstance(r.get("generated_text"), str) for r in rows):
        raise ValueError(f"Unexpected Full19 case groups or missing real outputs: {groups}")

    image_specs = {}
    if manifest is not None:
        for item in load_json(manifest).get("assets", []):
            image_specs[Path(item["path"]).name] = item["sha256"]
    if len(image_specs) != 16:
        raise ValueError("Verified 16-image source manifest is required")
    if vi_results is not None and not Path(vi_results).is_file():
        raise FileNotFoundError(f"Requested VI evidence not found: {vi_results}")
    vi = load_json(vi_results) if vi_results is not None else {}
    vi_rows = vi.get("cases", [])
    if not isinstance(vi_rows, list):
        raise ValueError("Invalid Vietnamese case data")
    if len(vi_rows) > 17:
        raise ValueError("Expected at most 17 Vietnamese inference cases")
    if vi_rows:
        if vi.get("source_full19") != en_results.name or vi.get("source_full19_sha256") != sha256(en_results):
            raise ValueError("VI evidence is not bound to this exact English output SHA256")
        en_by_id = {r["id"]: r for r in rows}
        for rec in vi_rows:
            source = en_by_id.get(rec.get("id"))
            if source is None or source.get("group") != rec.get("group"):
                raise ValueError("VI case ID or group does not match English source")
            if source["group"] != "text" and rec.get("image_sha256") != source.get("image_sha256"):
                raise ValueError("VI image SHA256 does not match English source")
    vi_by_id = {r.get("id"): r for r in vi_rows}
    if len(vi_by_id) != len(vi_rows) or any(key not in ids for key in vi_by_id):
        raise ValueError("Duplicate or unknown Vietnamese case ID")
    if any(next(r for r in rows if r["id"] == k)["group"] == "ocr" for k in vi_by_id):
        raise ValueError("OCR cards share digits and should not claim Vietnamese inference")

    annotation = None
    if review is not None:
        if not vi_results or not vi_results.is_file():
            raise ValueError("Semantic review requires recorded VI evidence")
        if len(vi_rows) != 17 or vi.get("status") != "COMPLETED":
            raise ValueError("Semantic review requires all 17 completed VI cases")
        if any(not r.get("eos_reached") or not r.get("generated_text") or r.get("error") for r in vi_rows):
            raise ValueError("Semantic review cannot annotate incomplete VI results")
        annotation = load_json(review)
        selected_notes = annotation.get("schema") == "qwen38-selected-review-v1"
        if annotation.get("schema") not in ("qwen38-pairwise-review-v1", "qwen38-selected-review-v1"):
            raise ValueError("Unsupported semantic-review schema")
        expected_status = "SELECTIVE_MANUAL_NOTES" if selected_notes else "PRELIMINARY_MANUAL_REVIEW"
        if annotation.get("status") != expected_status:
            raise ValueError("Semantic review status does not match schema")
        provenance = annotation.get("source", {})
        if (provenance.get("en_sha256") != sha256(en_results)
            or provenance.get("vi_sha256") != sha256(vi_results)):
            raise ValueError("Semantic review source SHA256 mismatch")
        findings = annotation.get("records", [])
        eligible = {r["id"] for r in rows if r["group"] != "ocr"}
        if not isinstance(findings, list):
            raise ValueError("Semantic-review records must be a list")
        ids_seen = [r.get("id") for r in findings if isinstance(r, dict)]
        if len(ids_seen) != len(findings) or len(set(ids_seen)) != len(ids_seen):
            raise ValueError("Duplicate or invalid semantic-review case IDs")
        if selected_notes:
            if not (1 <= len(findings) <= 17) or not set(ids_seen).issubset(eligible):
                raise ValueError("Selective review must contain 1–17 known non-OCR cases")
        elif len(findings) != 17 or set(ids_seen) != eligible:
            raise ValueError("Pairwise semantic review requires 17 unique non-OCR case IDs")
        counts = {key: 0 for key in ("core_equivalent", "partial_equivalent", "material_divergence")}
        for r in findings:
            if r.get("verdict") not in counts:
                raise ValueError("Unknown semantic verdict")
            counts[r["verdict"]] += 1
            for field in ("common", "reference", "difference"):
                tr = r.get(field)
                if not isinstance(tr, dict) or not all(isinstance(tr.get(k), str) and tr[k].strip() for k in ("vi", "en")):
                    raise ValueError(f"Missing bilingual semantic-review notes: {r['id']} {field}")
        if annotation.get("counts") != counts:
            raise ValueError("Semantic review tally mismatch")
        methodology = annotation.get("methodology")
        if not isinstance(methodology, dict) or not all(
            isinstance(methodology.get(k), str) and methodology[k].strip()
            for k in ("en", "vi")
        ):
            raise ValueError("Semantic review must describe its methodology in both languages")

    dest = Path(destination)
    if dest.is_symlink():
        raise ValueError(f"Refusing symlink gallery destination: {dest}")
    if dest.exists() and (not dest.is_dir() or any(dest.iterdir())):
        raise FileExistsError(f"Refusing to overwrite a nonempty or non-directory gallery: {dest}")
    # Validate all source images and optional editorial evidence before writing files.
    verified_images = {}
    display = []
    for row in sorted(rows, key=lambda r: (GROUP_ORDER[r["group"]], r["id"])):
        cid = row["id"]
        record = vi_by_id.get(cid, {})
        group = row["group"]
        image = ""
        if group != "text":
            raw = row.get("image")
            if not raw:
                if cid == "ocr-731":
                    filename = "vision_test_card_731_1280x960.png"
                else:
                    raise ValueError(f"Missing source-image filename: {cid}")
            else:
                filename = Path(raw).name
            if filename not in image_specs:
                raise ValueError(f"Image absent from manifest: {filename}")
            source = assets_dir / filename
            if not source.is_file() or source.is_symlink():
                raise FileNotFoundError(f"Original image missing or symlink: {source}")
            actual = sha256(source)
            if actual != image_specs[filename]:
                raise ValueError(f"Source-image SHA256 mismatch: {filename}")
            recorded = row.get("image_sha256")
            if not recorded or recorded != actual:
                raise ValueError(f"Run/source image SHA256 mismatch: {cid}")
            verified_images[filename] = source
            image = "assets/" + filename

        vi_status = ("shared_numeric" if group == "ocr" else
                     "completed" if record.get("generated_text") and record.get("eos_reached") and not record.get("error") and record.get("generated_token_ids") else
                     "incomplete" if record else "not_run")
        display.append({
            "id": cid, "group": group, "image": image,
            "image_sha256": row.get("image_sha256"),
            "en": {
                "prompt": row.get("prompt", ""),
                "output": row["generated_text"],
                "eos": bool(row.get("eos_reached")),
                "seconds": row.get("case_seconds"),
                "token_count": len(row.get("generated_token_ids", [])),
                "keyword_hits": row.get("check_hits"),
                "keyword_total": row.get("check_count"),
                "review_flags": [f["reason"] for f in en.get("review_flags", [])
                                 if f.get("id") == cid],
            },
            "vi": {
                "prompt": record.get("prompt", "") if group != "ocr" else row.get("prompt", ""),
                "output": record.get("generated_text", "") if group != "ocr" else row["generated_text"],
                "eos": bool(record.get("eos_reached")) if group != "ocr" else bool(row.get("eos_reached")),
                "seconds": record.get("case_seconds") if group != "ocr" else None,
                "token_count": len(record.get("generated_token_ids", [])) if group != "ocr" else None,
                "status": vi_status,
                "error": record.get("error", "") if isinstance(record, dict) else "",
            },
        })
    completed_vi = sum(r["vi"]["status"] == "completed" for r in display)
    summary = {
        "english_cases": len(display),
        "vi_generated": completed_vi,
        "vi_eligible": 17,
        "vi_shared_ocr": 2,
        "source_image_count": len(image_specs),
        "run_started_utc": en.get("run_started_utc", en.get("started_utc")),
        "en_status": en.get("status"),
        "en_eos_count": sum(bool(r.get("eos_reached")) for r in rows),
        "en_review_flag_count": len(en.get("review_flags", [])),
        "en_review_flags": en.get("review_flags", []),
        "vi_status": vi.get("status", "NOT_RUN"),
        "en_wall_seconds": en.get("total_wall_seconds"),
        "model": en.get("model", "Qwen3.8-27B BF16"),
        "device_count": en.get("device_count"),
        "support_manifest_sha256": en.get("portable_manifest_sha256"),
        "checkpoint_index_sha256": en.get("checkpoint_index_sha256"),
    }
    data = {"summary": summary, "cases": display}
    # Validate the review SHA before any destination mutation; do not leave an
    # incomplete Gallery when a valid-looking review belongs to another run.
    evidence_payload = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    if annotation is not None:
        evidence_sha = hashlib.sha256(evidence_payload.encode("utf-8")).hexdigest()
        if annotation["source"].get("evidence_sha256") != evidence_sha:
            raise ValueError("Semantic review does not match generated gallery evidence SHA256")

    # These are actual inputs and outputs, never synthetic/demo inference responses.
    dest.mkdir(parents=True, exist_ok=True)
    images_out = dest / "assets"
    images_out.mkdir()
    for filename, source in verified_images.items():
        shutil.copyfile(source, images_out / filename)
    (dest / "evidence.json").write_text(evidence_payload, encoding="utf-8")
    (dest / "source-full19.sha256").write_text(sha256(en_results) + "\n", encoding="utf-8")
    if annotation is not None:
        shutil.copyfile(review, dest / "semantic-review.v1.json")
        # Derived editorial Markdown is separate from raw inference; bilingual outputs never mix.
        selective = annotation.get("schema") == "qwen38-selected-review-v1"
        for code, title, columns in (
            ("vi", "# Ghi chú đánh giá các case đã xác minh — không bao phủ toàn bộ 17 case" if selective else "# Đối chiếu ngữ nghĩa 17 cặp EN/VI — review nội bộ",
             ["Case", "Mức độ", "Thông tin giống nhau", "Dữ kiện tham chiếu", "Khác biệt"]),
            ("en", "# Verified findings for selected cases — not a complete 17-case review" if selective else "# Pairwise review of 17 EN/VI cases — internal",
             ["Case", "Verdict", "Common ground", "Reference facts", "Differences"]),
        ):
            lines = [title, "", annotation["methodology"][code], "",
                     "| " + " | ".join(columns) + " |",
                     "|" + "|".join("---" for _ in columns) + "|"]
            for record in annotation["records"]:
                fields = [record["id"], record["verdict"]] + [record[k][code] for k in ("common", "reference", "difference")]
                lines.append("| " + " | ".join(str(v).replace("|", r"\\|").replace("\n", " ") for v in fields) + " |")
            (dest / f"manual-comparison.{code}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    page = (HTML_PAGE.replace("__GALLERY_DATA__", safe_json(data))
            .replace("__SEMANTIC_REVIEW_DATA__", safe_json(annotation) if annotation else "null"))
    (dest / "index.html").write_text(page, encoding="utf-8")
    return dest / "index.html"


HTML_PAGE = r'''<!doctype html>
<html lang="vi"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<link rel="icon" href="data:,">
<meta name="referrer" content="no-referrer">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; img-src 'self' data:; style-src 'unsafe-inline'; script-src 'unsafe-inline'; base-uri 'none'; form-action 'none'; connect-src 'none'; object-src 'none'">
<title>Qwen3.8-27B · Task-based EN/VI Gallery</title>
<style>
:root{color-scheme:dark;--bg:#0c111a;--surface:#151e2b;--panel:#1b2635;--panel2:#202d3d;--border:#344458;--fg:#edf3fb;--muted:#a7b8cb;--accent:#8bd3bd;--warm:#ebc58d;--attention:#f0aea2;--radius:15px}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif}
button,input,select{font:inherit}button{cursor:pointer}a{color:var(--accent)}button:focus-visible,input:focus-visible,select:focus-visible,a:focus-visible,summary:focus-visible{outline:2px solid var(--accent);outline-offset:3px}
.container{max-width:1510px;margin:0 auto;padding:0 28px}
.header{padding:33px 0 22px;border-bottom:1px solid var(--border);background:#101825}.topbar{display:flex;justify-content:space-between;gap:24px;align-items:start}
.eyebrow{text-transform:uppercase;font-size:11px;font-weight:750;letter-spacing:.14em;color:var(--accent)}
h1{font-size:clamp(28px,3vw,40px);line-height:1.17;margin:8px 0 9px;letter-spacing:-.025em}h2,h3,h4,p{margin-top:0}
.subtitle{color:var(--muted);max-width:850px;margin:0}
.lang-button,.small-button{background:var(--panel);border:1px solid var(--border);color:var(--fg);border-radius:10px;padding:9px 14px;white-space:nowrap}
.counters{display:flex;flex-wrap:wrap;gap:10px;margin:22px 0 0}
.counter{min-width:145px;padding:11px 16px;background:#192331;border:1px solid var(--border);border-radius:12px}
.counter strong{font-size:21px;line-height:1.3;display:block}.counter span{color:var(--muted);font-size:12px}
.notice{padding:12px 15px;border-left:3px solid var(--accent);color:#cbdcde;background:#122627;border-radius:5px;margin:16px 0 0;font-size:13px}
main{padding:24px 0 65px}
.controls{display:flex;flex-wrap:wrap;align-items:center;gap:11px;margin:0 0 18px}
.search{flex:1;min-width:230px;max-width:560px}
.search,select{color:var(--fg);background:var(--panel);border:1px solid var(--border);border-radius:10px;padding:10px 12px}
.control-note{font-size:12px;color:var(--muted)}
.group-strip{display:flex;gap:9px;overflow-x:auto;padding:0 1px 13px;margin-bottom:5px;scrollbar-width:thin}
.group-tab{border:1px solid var(--border);border-radius:12px;padding:11px 15px;background:var(--surface);color:var(--fg);white-space:nowrap;display:flex;gap:10px;align-items:center;flex-shrink:0}
.group-tab .count{font-size:12px;background:#2b394b;border-radius:20px;padding:2px 8px;color:var(--muted)}
.group-tab[aria-selected=true]{background:#19372f;border-color:var(--accent);color:#f1fff9}.group-tab[aria-selected=true] .count{color:#e8faf3;background:#2a594b}
.group-tab:disabled{opacity:.35;cursor:not-allowed}
.section-intro{display:flex;justify-content:space-between;gap:16px;align-items:end;margin:16px 0}
.section-intro h2{font-size:22px;margin:0 0 4px}.section-intro p{font-size:13px;color:var(--muted);margin:0}.section-count{font-size:12px;color:var(--muted);white-space:nowrap}
.workspace{display:grid;grid-template-columns:minmax(270px,330px) minmax(0,1fr);align-items:start;gap:18px}
.case-list{border:1px solid var(--border);border-radius:var(--radius);background:var(--surface);padding:13px;position:sticky;top:14px;max-height:calc(100vh - 30px);overflow-y:auto;scrollbar-width:thin}
.case-list-title{font-size:11px;letter-spacing:.07em;text-transform:uppercase;color:var(--muted);margin:3px 7px 13px}
.case-tile{display:flex;width:100%;text-align:left;align-items:center;gap:12px;border:1px solid transparent;border-radius:12px;background:transparent;color:var(--fg);padding:9px;margin:4px 0}
.case-tile:hover{background:var(--panel2)}
.case-tile[aria-current=true]{background:#233b3c;border-color:#6fb4a7}
.tile-thumb{width:78px;height:63px;flex-shrink:0;border:1px solid var(--border);background:#101824;border-radius:9px;object-fit:cover}
.tile-none{display:grid;place-items:center;font-size:14px;font-weight:750;color:#c8d8eb}
.tile-text{min-width:0;display:block;flex:1}.tile-name{display:block;font-size:13px;font-weight:650;line-height:1.35}.tile-id{font-size:11px;color:var(--muted);overflow-wrap:anywhere;display:block;margin-top:4px}
.tiny-status{display:inline-block;font-size:10px;line-height:1.4;margin-top:5px;color:var(--muted)}
.tiny-status.material_divergence{color:var(--attention)}.tiny-status.partial_equivalent{color:var(--warm)}.tiny-status.core_equivalent{color:var(--accent)}
.case-detail{min-width:0;border:1px solid var(--border);border-radius:var(--radius);background:var(--surface);overflow:hidden}
.detail-heading{display:flex;align-items:start;justify-content:space-between;gap:12px;padding:21px 25px;border-bottom:1px solid var(--border)}
.detail-heading h2{font-size:24px;margin:5px 0 4px}.detail-sub{font-size:12px;color:var(--muted);overflow-wrap:anywhere}
.badge{font-size:12px;border:1px solid var(--border);border-radius:999px;padding:5px 10px;white-space:nowrap}
.badge.material_divergence{color:var(--attention);border-color:#93534e}.badge.partial_equivalent{color:var(--warm);border-color:#827047}
.badge.core_equivalent{color:var(--accent);border-color:#4a8475}
.detail-media{margin:0;background:#0d1721;border-bottom:1px solid var(--border);text-align:center;padding:15px}
.detail-media img{display:block;max-width:100%;max-height:460px;width:auto;height:auto;margin:auto;border-radius:8px;object-fit:contain}
.detail-media figcaption{font-size:12px;color:var(--muted);padding:7px 0 0}
.detail-text-only{display:flex;align-items:center;justify-content:center;height:126px;color:var(--muted);font-size:14px}
.detail-body{padding:20px 24px}
.review-panel{background:#14292a;border:1px solid #456969;border-radius:12px;padding:16px 18px;margin-bottom:20px}
.review-panel h3{font-size:16px;margin:0 0 10px}.review-panel p{font-size:14px;color:#d6e7e5;line-height:1.6;margin:8px 0}
.review-panel strong{color:#f2fff8}.review-reference{border-top:1px solid #385a5b;margin-top:10px;padding-top:11px}
.review-reference summary{cursor:pointer;color:#d9efea;font-weight:600;font-size:13px}
.small-note{font-size:12px;color:var(--muted)}
.outputs-head{display:flex;align-items:center;justify-content:space-between;gap:10px;margin:19px 0 10px}
.outputs-head h3{font-size:17px;margin:0}.outputs-head span{font-size:12px;color:var(--muted)}
.responses{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:13px}
.response{background:#101925;border:1px solid var(--border);border-radius:12px;padding:16px;min-width:0}
.response-title{display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:8px;margin-bottom:13px}
.response-title strong{font-size:14px}.response-title small{font-size:11px;color:var(--muted)}
.response-body{font-size:13.5px;line-height:1.7;overflow-wrap:anywhere;min-height:90px}
.response-body p{margin:0 0 10px}.response-body li{margin:0 0 6px}.response-body ul,.response-body ol{padding-left:22px;margin:8px 0 12px}
.response-body b{color:#f0f8fb}.response-body .raw-inline{white-space:pre-wrap}
.prompt-disclosure{border-top:1px solid var(--border);padding-top:10px;margin-top:12px}
.prompt-disclosure summary{color:var(--accent);font-size:12px;cursor:pointer}
.prompt-disclosure pre{white-space:pre-wrap;overflow-wrap:anywhere;font:12px/1.6 ui-monospace,monospace;color:#c9d6e5;margin:9px 0 0}
.meta-footer{border-top:1px solid var(--border);margin-top:15px;padding-top:12px;font-size:11px;color:var(--muted);overflow-wrap:anywhere}
.empty{padding:50px 18px;text-align:center;border:1px dashed var(--border);border-radius:14px;color:var(--muted)}
.footer{border-top:1px solid var(--border);padding:23px 0 45px;color:var(--muted);font-size:12px}
.download-links{display:flex;gap:16px;flex-wrap:wrap;margin-bottom:10px}
@media(max-width:1050px){.workspace{grid-template-columns:minmax(225px,270px) minmax(0,1fr)}.responses{grid-template-columns:1fr}.detail-media img{max-height:350px}}
@media(max-width:740px){.container{padding:0 16px}.header{padding:23px 0 18px}.workspace{grid-template-columns:1fr}.case-list{position:static;max-height:none;display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:7px;padding:9px}.case-list-title{grid-column:1/-1;margin:6px}.case-tile{margin:0;display:block;padding:8px}.tile-thumb{width:100%;height:105px;display:block}.tile-name{font-size:12px;margin-top:8px}.tile-id{font-size:10px}.section-intro{align-items:start}.detail-heading{padding:18px}.detail-heading h2{font-size:20px}.detail-body{padding:17px}.detail-media img{max-height:400px}.topbar{gap:12px}.controls{gap:9px}.search{min-width:100%;max-width:none}}
@media(max-width:430px){.counter{min-width:0;flex:1 1 calc(50% - 10px)}.counter strong{font-size:20px}.subtitle{font-size:13px}.lang-button{padding:8px}.tile-thumb{height:88px}.detail-heading{display:block}.detail-heading .badge{display:inline-block;margin-top:9px}.detail-media img{max-height:280px}}
@media(prefers-reduced-motion:reduce){html{scroll-behavior:auto}}

.detail-overview{display:grid;grid-template-columns:minmax(0,1.17fr) minmax(0,1fr);align-items:stretch;border-bottom:1px solid var(--border);background:#0d1721}
.detail-overview:has(> :only-child){display:block}
.detail-overview .detail-media{border-bottom:0;margin:0;height:100%;display:flex;align-items:center;justify-content:center;flex-direction:column}
.detail-overview .detail-media img{max-height:360px}
.detail-overview .review-panel{border-radius:0;border:0;border-left:1px solid #456969;margin:0;background:#132729;padding:18px 21px;align-self:stretch}
.detail-overview .review-panel p{font-size:13px}
@media(max-width:1050px){.detail-overview{grid-template-columns:1fr}.detail-overview .review-panel{border-left:0;border-top:1px solid #456969}.detail-overview .detail-media img{max-height:340px}}
</style></head>
<body>
<header class="header"><div class="container">
<div class="topbar"><div>
<div class="eyebrow">Qwen3.8-27B · Kaggle TPU v5e-8 · BF16 · Internal review</div>
<h1 id="title">Thư viện suy luận theo nhiệm vụ</h1>
<p class="subtitle" id="subtitle"></p></div>
<button class="lang-button" type="button" id="lang" aria-label="Change interface language">English</button>
</div>
<div class="counters" aria-label="Inference counts">
<div class="counter"><strong>19</strong><span id="statEn"></span></div>
<div class="counter"><strong>17/17</strong><span id="statVi"></span></div>
<div class="counter"><strong>16</strong><span id="statImages"></span></div>
<div class="counter"><strong>5</strong><span id="statGroups"></span></div>
</div>
<p class="notice" id="notice"></p>
</div></header>
<main class="container">
<div class="controls">
<input class="search" id="search" type="search" aria-label="Search task content">
<select id="severity" aria-label="Semantic comparison status"></select>
</div>
<div class="eyebrow" id="scopeLabel"></div>
<nav class="group-strip" role="tablist" aria-label="Task groups" id="groups"></nav>
<div class="section-intro">
<div><h2 id="sectionTitle"></h2><p id="sectionDesc"></p></div>
<span class="section-count" id="sectionCount"></span>
</div>
<div class="workspace">
<aside id="caseList" class="case-list" aria-label="Example list"></aside>
<article class="case-detail" id="detail" aria-live="polite"></article>
</div>
</main>
<footer class="footer"><div class="container">
<div class="download-links">
<a href="evidence.json" id="rawLink" download>Raw evidence JSON</a>
<a href="semantic-review.v1.json" id="reviewLink" download>Pairwise review JSON</a>
<a href="manual-comparison.vi.md" id="manualVi" download>Vietnamese report</a>
<a href="manual-comparison.en.md" id="manualEn" download>English report</a>
</div><div id="footerNote"></div></div></footer>
<script>
const DATA=__GALLERY_DATA__;
const AUDIT=__SEMANTIC_REVIEW_DATA__;
const GROUPS=[
 {id:"visual",vi:"Biểu đồ & sơ đồ",en:"Charts & diagrams",descVi:"Đọc KPI, cảm biến, sơ đồ robot và luồng kho vận.",descEn:"Read KPIs, sensors, robot diagrams and warehouse flows.",ids:["vision-2d-dashboard","vision-2d-greenhouse","vision-3d-robotics","vision-3d-warehouse"]},
 {id:"extract",vi:"Trích xuất dữ liệu ảnh",en:"Extract data from images",descVi:"Đọc bảng giá, lịch họp, hóa đơn và danh thiếp.",descEn:"Read price boards, meeting notes, invoices and business cards.",ids:["vision-real-market","vision-real-workspace","photo-office-invoice","photo-business-card"]},
 {id:"scene",vi:"Hiểu nội dung ảnh",en:"Understand scenes",descVi:"Nhận diện người, hoạt động, bối cảnh và vị trí các đối tượng.",descEn:"Identify people, activities, surroundings and spatial relationships.",ids:["photo-ai-farmers-market","photo-ai-city-crossing","photo-ai-vietnamese-meal","photo-ai-dog-walking","photo-ai-supermarket","photo-daylight-workspace"]},
 {id:"ocr",vi:"Đọc chữ số (OCR)",en:"Numeric OCR",descVi:"Hai tác vụ đọc số: 482 và 731; không có lượt VI riêng.",descEn:"Read 482 and 731; numeric OCR is shared, not rerun in VI.",ids:["ocr-482","ocr-731"]},
 {id:"text",vi:"Hỏi đáp văn bản",en:"Text questions",descVi:"Phép toán, kiến thức địa lý và ngôn ngữ lập trình Ruby.",descEn:"Arithmetic, geography and Ruby programming-language questions.",ids:["text-2plus2","text-france-capital","text-ruby-language"]}
];
const TITLES={
"vision-2d-dashboard":["Bảng điều khiển giao hàng","Delivery dashboard"],
"vision-2d-greenhouse":["Hệ thống nhà kính thông minh","Smart greenhouse"],
"vision-3d-robotics":["Trạm robot","Robotics station"],
"vision-3d-warehouse":["Kho vận & điều phối","Warehouse logistics"],
"vision-real-market":["Bảng giá nông sản","Produce price board"],
"vision-real-workspace":["Lịch họp & hóa đơn nhỏ","Meeting notes & receipts"],
"photo-office-invoice":["Đọc hóa đơn văn phòng","Office invoice"],
"photo-business-card":["Đọc danh thiếp","Business card"],
"photo-ai-farmers-market":["Giao dịch tại chợ","Farmers market"],
"photo-ai-city-crossing":["Người đi bộ qua đường","City crosswalk"],
"photo-ai-vietnamese-meal":["Bữa ăn Việt Nam","Vietnamese meal"],
"photo-ai-dog-walking":["Dắt chó trong công viên","Walking a dog"],
"photo-ai-supermarket":["Mua sắm siêu thị","Supermarket shopping"],
"photo-daylight-workspace":["Góc làm việc ban ngày","Daylight workspace"],
"ocr-482":["Nhận dạng số 482","Recognize 482"],
"ocr-731":["Nhận dạng số 731","Recognize 731"],
"text-2plus2":["Phép tính 2 + 2","2 + 2 arithmetic"],
"text-france-capital":["Thủ đô nước Pháp","Capital of France"],
"text-ruby-language":["Giới thiệu Ruby","About Ruby"]
};
const LABEL={
vi:{title:"Thư viện suy luận theo nhiệm vụ",subtitle:"Khám phá kết quả TPU theo từng nhóm chức năng. Hai cột EN và VI là hai lượt suy luận độc lập, không phải bản dịch của nhau.",toggle:"English",search:"Tìm tác vụ, nội dung hoặc nhận xét…",all:"Mọi mức đối chiếu",material_divergence:"Khác biệt đáng kể",not_reviewed:"Chưa đánh giá",not_run:"Chưa sinh VI",incomplete:"VI chưa hoàn tất",partial_equivalent:"Tương đồng một phần",core_equivalent:"Tương đồng ý chính",shared:"OCR số dùng chung",scope:"Nhóm tác vụ",examples:"Chọn ví dụ",selected:"Tác vụ đang xem",image:"Ảnh đầu vào gốc · bấm để xem kích thước đầy đủ",noImage:"Câu hỏi chỉ có văn bản, không dùng ảnh",noteTitle:"Nhận xét đối chiếu ngữ nghĩa",common:"Thông tin giống nhau",difference:"Điểm khác biệt",reference:"Xem dữ kiện tham chiếu",outputs:"Câu trả lời của mô hình",comparison:"Đặt cạnh nhau để so sánh, không phải bản dịch",prompt:"Xem prompt gốc",inference:"Suy luận độc lập",recorded:"Có kết quả",eos:"Kết thúc sinh token",flagged:"Cần kiểm tra kết quả",time:"Thời gian",source:"SHA256 ảnh nguồn",noReview:"OCR chữ số dùng chung: không thực hiện suy luận VI thứ hai.",notice:"Kết quả TPU thực tế · EOS chỉ cho biết mô hình dừng sinh token. Chỉ các case có ghi chú mới được đối chiếu thủ công; các case khác chưa được đánh giá toàn diện.",empty:"Không có tác vụ phù hợp. Hãy đổi từ khóa hoặc bộ lọc.",cases:"tác vụ",en:"19 case EN",vi:"17 case VI",images:"16 ảnh",groups:"5 nhóm",raw:"JSON kết quả gốc",review:"JSON đối chiếu",manualVi:"Báo cáo tiếng Việt",manualEn:"Báo cáo tiếng Anh",foot:"Dữ liệu thô không bị sửa; phân nhóm và ghi chú chỉ là lớp giao diện review.",sharedTitle:"Dùng chung OCR số",textStatus:"Không cần ảnh"},
en:{title:"Task-based inference gallery",subtitle:"Explore the TPU results by task. EN and VI are independent model inferences—not translations of each other.",toggle:"Tiếng Việt",search:"Search tasks, responses or review notes…",all:"All comparison results",material_divergence:"Material divergence",not_reviewed:"Not reviewed",not_run:"VI not generated",incomplete:"VI incomplete",partial_equivalent:"Partial overlap",core_equivalent:"Core meaning aligned",shared:"Shared numeric OCR",scope:"Task categories",examples:"Choose an example",selected:"Selected task",image:"Original input image · open full size",noImage:"Text-only question; no image input",noteTitle:"Pairwise semantic review",common:"Shared information",difference:"Differences and concerns",reference:"Show reference facts",outputs:"Model responses",comparison:"Side-by-side independent inferences, not translation",prompt:"Show original prompt",inference:"Independent inference",recorded:"Recorded",eos:"End-of-sequence",flagged:"Review required",time:"Time",source:"Source-image SHA256",noReview:"Numeric OCR is shared: no second VI inference was run.",notice:"Actual TPU inference outputs · EOS confirms only token generation ended. Notes cover selected verified cases; unannotated cases have not been comprehensively reviewed.",empty:"No matching tasks. Adjust the search or filter.",cases:"cases",en:"19 EN cases",vi:"17 VI cases",images:"16 images",groups:"5 groups",raw:"Raw evidence JSON",review:"Pairwise review JSON",manualVi:"Vietnamese report",manualEn:"English report",foot:"Raw outputs remain unchanged; categorization and commentary are a review-only UI layer.",sharedTitle:"Shared numeric OCR",textStatus:"No image"}
};
const byId=Object.fromEntries(DATA.cases.map(r=>[r.id,r]));
const reviews=Object.fromEntries((AUDIT?.records||[]).map(r=>[r.id,r]));
const groupById=Object.fromEntries(GROUPS.flatMap(g=>g.ids.map(id=>[id,g.id])));
let lang="vi",group="visual",selected="vision-2d-dashboard",search="",severity="";
const el=(id)=>document.getElementById(id);
function make(tag,cls,value){const x=document.createElement(tag);if(cls)x.className=cls;if(value!==undefined&&value!==null)x.textContent=String(value);return x}
function add(parent,...children){for(const x of children)parent.append(x);return parent}
function textTitle(id){const a=TITLES[id];return a?a[lang==="vi"?0:1]:id}
function resultStatus(id){return reviews[id]?.verdict||(byId[id]?.group==="ocr"?"shared":"not_reviewed")}
function cleanOutput(s){return String(s||"").replace(/<\|im_end\|>\s*$/,"").trim()}
function textParts(parent,value){
  // Only render bold spans, never user-provided HTML.
  const parts=String(value).split(/(\*\*[^*]+\*\*)/g);
  for(const part of parts)if(part.startsWith("**")&&part.endsWith("**"))parent.append(make("strong","",part.slice(2,-2)));else parent.append(document.createTextNode(part));
}
function richOutput(raw){
 const box=make("div","response-body");const lines=cleanOutput(raw).split("\n");let list=null,kind="";
 function endList(){list=null;kind=""}
 for(const line of lines){
   const t=line.trim();
   if(!t){endList();continue}
   const m=t.match(/^((?:[-*]\s+)|(?:\d+[.)]\s+))(.+)$/);
   if(m){
      const want=/^\d/.test(m[1])?"ol":"ul";
      if(!list||want!==kind){kind=want;list=make(want);box.append(list)}
      const li=make("li");textParts(li,m[2]);list.append(li);
   } else {endList();const p=make("p");textParts(p,t);box.append(p)}
 }
 return box
}
function relevant(x){
 if(severity&&resultStatus(x.id)!==severity)return false;
 if(!search)return true;
 const a=reviews[x.id];
 const all=[x.id,x.group,...(TITLES[x.id]||[]),x.en.prompt,x.en.output,x.vi.prompt,x.vi.output,
   ...(a?Object.values(a.common).concat(Object.values(a.reference),Object.values(a.difference)):[])].join(" ").toLowerCase();
 return all.includes(search.toLowerCase())
}
function visibleIds(g){return g.ids.filter(id=>relevant(byId[id]))}
function chooseCompatible(){
 const current=GROUPS.find(g=>g.id===group);
 if(!current||visibleIds(current).length===0){
   group=GROUPS.find(g=>visibleIds(g).length>0)?.id||GROUPS[0].id;
   selected=visibleIds(GROUPS.find(g=>g.id===group))[0]||"";
 } else if(!visibleIds(current).includes(selected)){selected=visibleIds(current)[0]||""}
}
function severityLabel(cid){return LABEL[lang][resultStatus(cid)]}
function render(){
 chooseCompatible();
 const t=LABEL[lang],active=GROUPS.find(g=>g.id===group);
 el("severity").hidden=!AUDIT;
 for(const link of ["reviewLink","manualVi","manualEn"])el(link).hidden=!AUDIT;
 document.documentElement.lang=lang;
 el("title").textContent=t.title;el("subtitle").textContent=t.subtitle;
 el("lang").textContent=t.toggle;el("search").placeholder=t.search;
 el("notice").textContent=t.notice;el("scopeLabel").textContent=t.scope;
 el("groups").replaceChildren(...GROUPS.map(g=>{
   const count=visibleIds(g).length,b=make("button","group-tab");
   b.type="button";b.disabled=count===0;b.dataset.group=g.id;b.setAttribute("role","tab");b.setAttribute("aria-selected",String(group===g.id));
   add(b,make("span","",g[lang]),make("span","count",count));
   b.onclick=()=>{group=g.id;selected=visibleIds(g)[0]||"";render()};
   return b;
 }));
 el("sectionTitle").textContent=active[lang];
 el("sectionDesc").textContent=active[lang==="vi"?"descVi":"descEn"];
 el("sectionCount").textContent=visibleIds(active).length+" "+t.cases;
 const filter=el("severity"),current=severity;filter.replaceChildren();
 for(const [value,label] of [["",t.all],["material_divergence",t.material_divergence],["partial_equivalent",t.partial_equivalent],["core_equivalent",t.core_equivalent],["shared",t.shared]]){
    const opt=make("option","",label);opt.value=value;filter.append(opt);
 }
 filter.value=current;
 for(const [id,value] of [["statEn",t.en],["statVi",t.vi],["statImages",t.images],["statGroups",t.groups]])el(id).textContent=value;
 const tileList=el("caseList");tileList.replaceChildren(make("div","case-list-title",t.examples));
 for(const id of visibleIds(active)){
  const x=byId[id],btn=make("button","case-tile");btn.type="button";btn.dataset.caseId=id;
  btn.setAttribute("aria-current",String(id===selected));
  let thumb;
  if(x.image){thumb=make("img","tile-thumb");thumb.src=x.image;thumb.alt="";thumb.loading="lazy"}
  else{thumb=make("span","tile-thumb tile-none",x.group==="ocr"?"123":"TXT")}
  const info=make("span","tile-text");
  add(info,make("span","tile-name",textTitle(id)),make("span","tile-id",id),make("span","tiny-status "+resultStatus(id),severityLabel(id)));
  add(btn,thumb,info);
  btn.onclick=()=>{selected=id;renderDetail();for(const b of tileList.querySelectorAll(".case-tile"))b.setAttribute("aria-current",String(b.dataset.caseId===id));if(window.innerWidth<=740)el("detail").scrollIntoView({behavior:"smooth",block:"start"})};
  tileList.append(btn)
 }
 renderDetail();
 el("rawLink").textContent=t.raw;el("reviewLink").textContent=t.review;
 el("manualVi").textContent=t.manualVi;el("manualEn").textContent=t.manualEn;
 el("footerNote").textContent=t.foot
}
function renderDetail(){
 const t=LABEL[lang],root=el("detail");root.replaceChildren();
 if(!selected){root.append(make("div","empty",t.empty));return}
 const x=byId[selected],r=reviews[selected];
 const head=make("div","detail-heading");
 const headText=make("div");add(headText,make("div","eyebrow",t.selected),make("h2","",textTitle(selected)),make("div","detail-sub",selected+" · "+x.group.toUpperCase()));
 const badge=make("span","badge "+resultStatus(selected),severityLabel(selected));
 add(head,headText,badge);root.append(head);
 const overview=make("div","detail-overview");
 if(x.image){
   const media=make("figure","detail-media"),a=make("a");a.href=x.image;a.target="_blank";a.rel="noopener noreferrer";
   const im=make("img");im.src=x.image;im.alt=textTitle(selected);a.append(im);
   add(media,a,make("figcaption","",t.image));overview.append(media);
 } else {overview.append(make("div","detail-text-only",t.noImage))}
 const body=make("div","detail-body");
 if(r){
  const panel=make("section","review-panel");
  add(panel,make("h3","",t.noteTitle));
  const p1=make("p");add(p1,make("strong","",t.common+": "),document.createTextNode(r.common[lang]));
  const p2=make("p");add(p2,make("strong","",t.difference+": "),document.createTextNode(r.difference[lang]));
  const details=make("details","review-reference");
  add(details,make("summary","",t.reference),make("p","",r.reference[lang]));
  add(panel,p1,p2,details);overview.append(panel)
 } else if(x.group==="ocr"){body.append(make("p","small-note",t.noReview))}
 root.append(overview);
 const outputHead=make("div","outputs-head");
 add(outputHead,make("h3","",t.outputs),make("span","",t.comparison));body.append(outputHead);
 const responseGrid=make("div","responses");
 for(const [code,item] of [["EN",x.en],["VI",x.vi]]){
  const panel=make("section","response");
  const heading=make("div","response-title");
  add(heading,make("strong","",code+" · "+(item.status==="shared_numeric"?t.sharedTitle:t.inference)),make("small","",item.eos?"EOS ✓":"EOS —"));
  panel.append(heading);
  if(code==="EN" && item.review_flags?.length){
    panel.append(make("p","small-note",t.flagged+": "+item.review_flags.join(", ")));
  }
  panel.append(richOutput(code==="VI"&&item.status==="not_run"?t.not_run:
    code==="VI"&&item.status==="incomplete"?(item.error||t.incomplete):item.output));
  const prompt=make("details","prompt-disclosure");
  add(prompt,make("summary","",t.prompt),make("pre","",item.prompt||"—"));
  panel.append(prompt);
  panel.append(make("div","meta-footer",t.time+": "+(item.seconds==null?"—":item.seconds+"s")+" · "+t.eos+": "+String(item.eos)));
  responseGrid.append(panel)
 }
 body.append(responseGrid);
 if(x.image_sha256)body.append(make("div","meta-footer",t.source+": "+x.image_sha256));
 root.append(body)
}
function init(){
 const ids=GROUPS.flatMap(g=>g.ids);
 if(ids.length!==19||new Set(ids).size!==19||ids.some(id=>!byId[id])||Object.keys(byId).length!==19)throw Error("Task inventory does not match raw 19 cases");
 if(AUDIT&&(AUDIT.records.length<1||AUDIT.records.length>17||Object.keys(reviews).length!==AUDIT.records.length))throw Error("Review inventory mismatch");
 el("lang").onclick=()=>{lang=lang==="vi"?"en":"vi";render()};
 el("search").oninput=e=>{search=e.target.value.trim();render()};
 el("severity").onchange=e=>{severity=e.target.value;render()};
 render()
}
init();

</script>
</body></html>
'''


def refresh_gallery(en_results, assets_dir, destination, vi_results=None, manifest=None, review=None):
    """Replace only a recognized gallery for this exact English run."""
    import tempfile
    import uuid
    dest = Path(destination)
    if not dest.exists() or (dest.is_dir() and not any(dest.iterdir())):
        return build_gallery(en_results, assets_dir, dest, vi_results, manifest, review)
    if dest.is_symlink() or not dest.is_dir():
        raise ValueError("Refusing gallery refresh on a symlink or non-directory")
    old_index, old_data = dest/"index.html", dest/"evidence.json"
    if not old_index.is_file() or not old_data.is_file() or not (dest/"assets").is_dir():
        raise ValueError("Existing folder is not a gallery")
    old = load_json(old_data)
    en = load_json(en_results)
    prior = old.get("cases", [])
    rows = en.get("cases", [])
    if len(prior) != 19 or len(rows) != 19:
        raise ValueError("Gallery or source lacks 19 cases")
    by_id = {r["id"]: r for r in prior}
    source_binding = dest / "source-full19.sha256"
    # Verify an exact input-file digest when available. Other galleries
    # predate the sidecar are checked against all recorded EN case metadata.
    if source_binding.is_symlink():
        raise ValueError("Gallery source binding is a symlink")
    if source_binding.exists() and source_binding.read_text(encoding="utf-8").strip() != sha256(en_results):
        raise ValueError("Gallery does not belong to this exact English source SHA256")
    details = (
        ("run_started_utc", en.get("run_started_utc", en.get("started_utc"))),
        ("en_status", en.get("status")),
        ("en_wall_seconds", en.get("total_wall_seconds")),
        ("model", en.get("model", "Qwen3.8-27B BF16")),
        ("device_count", en.get("device_count")),
        ("support_manifest_sha256", en.get("portable_manifest_sha256")),
        ("checkpoint_index_sha256", en.get("checkpoint_index_sha256")),
    )
    summary = old.get("summary", {})
    if len(by_id) != 19 or any(summary.get(k) != v for k, v in details):
        raise ValueError("Gallery does not belong to this exact English run")
    for r in rows:
        previous = by_id.get(r["id"], {})
        recorded = previous.get("en", {})
        expected = {
            "prompt": r.get("prompt", ""),
            "output": r.get("generated_text"),
            "eos": bool(r.get("eos_reached")),
            "seconds": r.get("case_seconds"),
            "token_count": len(r.get("generated_token_ids", [])),
            "keyword_hits": r.get("check_hits"),
            "keyword_total": r.get("check_count"),
        }
        if (previous.get("group") != r.get("group")
            or previous.get("image_sha256") != r.get("image_sha256")
            or any(recorded.get(k) != v for k, v in expected.items())):
            raise ValueError("Gallery does not belong to this exact English run")
    staging = Path(tempfile.mkdtemp(prefix="."+dest.name+".build-",dir=dest.parent))
    try:
        build_gallery(en_results, assets_dir, staging, vi_results, manifest, review)
    except BaseException:
        shutil.rmtree(staging)
        raise
    previous = dest.parent/("."+dest.name+".backup-"+uuid.uuid4().hex)
    dest.rename(previous)
    try:
        staging.rename(dest)
    except BaseException:
        previous.rename(dest)
        shutil.rmtree(staging)
        raise
    shutil.rmtree(previous)
    return dest/"index.html"



def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--en", required=True, type=Path, help="19-case full19-final.json from a real TPU run")
    parser.add_argument("--vi", type=Path, help="Optional recorded Vietnamese inference JSON")
    parser.add_argument("--images", required=True, type=Path, help="Exact original source-image directory")
    parser.add_argument("--manifest", required=True, type=Path, help="16-image manifest containing SHA256")
    parser.add_argument("--output", required=True, type=Path, help="New or empty output directory")
    parser.add_argument("--review", type=Path, help="Optional SHA256-bound, preliminary semantic-review JSON")
    parser.add_argument("--refresh", action="store_true", help="Refresh only a verified Gallery from this same EN run")
    args = parser.parse_args()
    builder = refresh_gallery if args.refresh else build_gallery
    print("GALLERY_READY", builder(args.en, args.images, args.output, args.vi, args.manifest, args.review))


if __name__ == "__main__":
    main()
