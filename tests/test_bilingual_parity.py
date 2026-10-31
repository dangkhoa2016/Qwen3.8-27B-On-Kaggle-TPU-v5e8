"""Bilingual documentation parity guard.

Uses only public-source structure and synthetic validation patterns.
No account credentials or non-public resource identifiers are embedded here.
"""
import collections
import re
import pytest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(
    not (ROOT / "README.md").is_file() or not (ROOT / "README.vi.md").is_file(),
    reason="Documentation-pair audit applies after publication of the root READMEs",
)
FENCE=chr(96)*3

def document_pairs():
    return [(en,en.with_name(en.stem+".vi.md"))
            for en in ROOT.rglob("*.md")
            if ".git" not in en.parts and "node_modules" not in en.parts
            and not en.name.endswith(".vi.md")
            and en.with_name(en.stem+".vi.md").is_file()]

def section_lengths(s):
    lines=s.splitlines()
    starts=[i for i,line in enumerate(lines) if re.match(r"^#{1,6} ",line)]
    return [b-a for a,b in zip(starts,starts[1:]+[len(lines)])]

def link_targets(s):
    return collections.Counter(
        re.sub(r"\.vi\.md(?=$|[#\s])",".md",target)
        for target in re.findall(r"\]\(([^)]+)\)",s))

def code_blocks(s):
    pattern=r"(?ms)^("+re.escape(FENCE)+r"[^\n]*)\n(.*?)^\s*"+re.escape(FENCE)
    return [(lang.strip(),body.strip()) for lang,body in re.findall(pattern,s)
            if "mermaid" not in lang.lower()]

def environment_flags(s):
    return collections.Counter(re.findall(r"QWEN38_[A-Z0-9_]+(?:=\d+)?",s))

def numeric_facts(s):
    s=re.sub(r"^#{1,6} .*$","",s,flags=re.M)
    s=re.sub(r"^\s*\d+\. ","",s,flags=re.M)
    s=re.sub(r"(?<=\d)[,.](?=\d{3}(?:\D|$))","",s)
    s=re.sub(r"(?<=\d),(?=\d{1,2}\b)",".",s)
    return collections.Counter(re.findall(r"(?<![\w])\d+(?:[.,]\d+)*",s))

def structural_counts(s):
    return {
        "headings":len(re.findall(r"^#{1,6} ",s,re.M)),
        "bullets":len(re.findall(r"^\s*[-*] ",s,re.M)),
        "numbered":len(re.findall(r"^\d+\. ",s,re.M)),
        "table_rows":len(re.findall(r"^\|",s,re.M)),
        "code_fences":sum(line.startswith(FENCE) for line in s.splitlines()),
    }

def test_all_document_pairs_match_lines_and_section_structure():
    pairs=document_pairs()
    assert len(pairs)>=27  # Includes the new Git-first migration guide pair.
    for en,vi in pairs:
        a,b=en.read_text(encoding="utf-8"),vi.read_text(encoding="utf-8")
        assert len(a.splitlines())==len(b.splitlines()),en.as_posix()
        assert section_lengths(a)==section_lengths(b),en.as_posix()
        assert structural_counts(a)==structural_counts(b),en.as_posix()

def test_links_commands_flags_and_numbers_have_identical_values():
    for en,vi in document_pairs():
        a,b=en.read_text(encoding="utf-8"),vi.read_text(encoding="utf-8")
        assert link_targets(a)==link_targets(b),en.as_posix()
        assert code_blocks(a)==code_blocks(b),en.as_posix()
        assert environment_flags(a)==environment_flags(b),en.as_posix()
        assert numeric_facts(a)==numeric_facts(b),en.as_posix()

def test_all_fourteen_image_tasks_have_translated_descriptions():
    en=(ROOT/"docs/FULL19-EVIDENCE.md").read_text(encoding="utf-8")
    vi=(ROOT/"docs/FULL19-EVIDENCE.vi.md").read_text(encoding="utf-8")
    ids=("vision-2d-dashboard","vision-2d-greenhouse","vision-3d-robotics",
         "vision-3d-warehouse","vision-real-workspace","vision-real-market",
         "photo-office-invoice","photo-business-card","photo-daylight-workspace",
         "photo-ai-farmers-market","photo-ai-city-crossing",
         "photo-ai-vietnamese-meal","photo-ai-dog-walking","photo-ai-supermarket")
    assert len(ids)==14
    for source in (en,vi):
        for case in ids:assert source.count("| "+case+" |")==1
    descriptions=[line.split("|")[-2].strip() for line in vi.splitlines()
                  if line.startswith("| ") and any("| "+case+" |" in line for case in ids)]
    assert len(descriptions)==14
    for desc in descriptions:
        assert re.match(r"^(?:Analyze|Explain|Interpret|Read|Extract|Describe|Identify)\s",desc) is None
    assert "prompt tiếng Anh" in vi and "example JSON" in en

def test_critical_release_statements_have_matching_meaning():
    en=(ROOT/"README.md").read_text(encoding="utf-8")
    vi=(ROOT/"README.vi.md").read_text(encoding="utf-8")
    assert "not endorsed by Qwen" in en
    assert "không được Qwen chứng thực" in vi
    assert "**Run and review:**" in en
    assert "**Chạy và đánh giá:**" in vi
    performance_en=(ROOT/"docs/PERFORMANCE-ANALYSIS.md").read_text(encoding="utf-8")
    performance_vi=(ROOT/"docs/PERFORMANCE-ANALYSIS.vi.md").read_text(encoding="utf-8")
    assert "not performance results" in performance_en
    assert "không công bố kết quả hiệu năng" in performance_vi
    for path in ("api/README.md","api/README.vi.md"):
        text=(ROOT/path).read_text(encoding="utf-8")
        assert "Playwright PASS" not in text
        assert "36 Python PASS" not in text
        assert "TPU" in text

def test_execution_guides_explain_how_to_run_and_review_outputs():
    # Verify explicit execution and evidence requirements in both languages.
    requirements = {
        "README.md": (
            "**Run and review:**",
            "run the cells on TPU v5e-8",
            "CPU checks verify source and UI",
        ),
        "README.vi.md": (
            "**Chạy và đánh giá:**",
            "chạy các cell trên TPU v5e-8",
            "Kiểm thử CPU",
        ),
        "docs/FULL19-EVIDENCE.md": (
            "Full19 groups Text and Vision tasks",
            "Generated token IDs",
        ),
        "docs/FULL19-EVIDENCE.vi.md": (
            "Full19 nhóm các tác vụ Text và Vision",
            "generated token IDs",
        ),
    }
    for name, fragments in requirements.items():
        source = (ROOT/name).read_text(encoding="utf-8")
        for fragment in fragments:
            assert fragment.casefold() in source.casefold(), (name, fragment)


def test_bilingual_markdown_has_no_literal_newline_escape_or_duplicate_heading_marker():
    for en,vi in document_pairs():
        for path in (en,vi):
            source=path.read_text(encoding="utf-8")
            assert chr(92)+"n" not in source,path.as_posix()
            assert "## ###" not in source,path.as_posix()

def test_localized_mermaid_preserves_graph_topology():
    en=(ROOT/"docs/ARCHITECTURE.md").read_text(encoding="utf-8")
    vi=(ROOT/"docs/ARCHITECTURE.vi.md").read_text(encoding="utf-8")
    edges=lambda s:re.findall(r"(?m)\b([A-L])\s*-->\s*([A-L])\b",s)
    assert edges(en)==edges(vi)
    assert len(edges(en))>=10


def test_readme_badges_and_preview_images_are_renderable():
    # The README is also a visual entry point, not just a text document.
    source_images = (
        "examples/images/photo_ai_city_crossing_1280x960.png",
        "examples/images/vision_3d_robotics_station.jpg",
        "examples/images/photo_ai_vietnamese_meal_1280x960.png",
    )
    preview_images = (
        "examples/previews/photo_ai_city_crossing_preview.jpg",
        "examples/previews/vision_3d_robotics_station_preview.jpg",
        "examples/previews/photo_ai_vietnamese_meal_preview.jpg",
    )
    assert (ROOT / ".github/workflows/cpu-acceptance.yml").is_file()
    for original, preview in zip(source_images, preview_images):
        assert (ROOT / original).is_file(), original
        raw = (ROOT / preview).read_bytes()
        assert raw.startswith(bytes.fromhex("ffd8ff")) and len(raw) < 75_000, preview

    badge_sets = []
    image_sets = []
    for filename in ("README.md", "README.vi.md"):
        source = (ROOT / filename).read_text(encoding="utf-8")
        images = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", source)
        badges = [url for url in images if url.startswith("https://")]
        previews = [url for url in images if url.startswith("examples/previews/")]
        assert len(badges) == 5, filename
        assert len(previews) == 3, filename
        assert tuple(previews) == preview_images, filename
        assert all(f"]({url})" in source for url in source_images), filename
        assert any("/actions/workflows/cpu-acceptance.yml/badge.svg" in url
                   for url in badges), filename
        assert sum("img.shields.io/badge/" in url for url in badges) == 4
        assert "[LICENSE]" not in source
        badge_sets.append(tuple(badges))
        image_sets.append(tuple(previews))
    assert badge_sets[0] == badge_sets[1]
    assert image_sets[0] == image_sets[1]


def test_markdown_guides_do_not_promise_missing_gallery_zip_backup():
    for name in ("gallery/README.md", "gallery/README.vi.md"):
        doc=(ROOT/name).read_text(encoding="utf-8")
        assert "inference-gallery-before-refresh.zip" not in doc
        assert "inference-gallery.zip" in doc
        assert "gallery/install_cloudflared.sh" in doc
        assert "731" not in doc and "482" not in doc
    for name in ("docs/NOTEBOOK-ARCHITECTURE.md", "docs/NOTEBOOK-ARCHITECTURE.vi.md"):
        doc=(ROOT/name).read_text(encoding="utf-8")
        assert "source-only refactor" not in doc
        assert "chưa thực thi trên TPU" not in doc
