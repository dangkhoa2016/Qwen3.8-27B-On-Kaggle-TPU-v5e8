"""Release input alignment and offline notebook preflight invariants."""
import ast
import hashlib
import json
import pytest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/kaggle-tpu-v5e8-production-showcase.ipynb"
SUPPORT = ROOT / "support/portable-bundle-manifest.json"


def test_notebook_support_manifest_matches_version_controlled_examples():
    nb = json.loads(NOTEBOOK.read_text())
    code = (ROOT / "qwen38_runtime/stages/preflight.py").read_text()
    source_code = "".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    manifest = json.loads(SUPPORT.read_text())
    digest = hashlib.sha256(SUPPORT.read_bytes()).hexdigest()
    assert f"PORTABLE_MANIFEST_SHA256='{digest}'" in code
    assert "qwen38-27b-tpu-v5e8-full19-runtime-support" in code and "notebook-source-manifest.json" in source_code
    assets = json.loads((ROOT / "examples/assets_manifest.json").read_text())["assets"]
    assert len(assets) == 16 and len(manifest["files"]) == 48
    for item in assets:
        entry = manifest["files"]["images/" + Path(item["path"]).name]
        assert item["sha256"] == entry["sha256"] and item["bytes"] == entry["bytes"]


def test_code_cells_parse_cleanly():
    notebook = json.loads(NOTEBOOK.read_text())
    assert set(notebook["metadata"]) == {"kernelspec", "language_info"}
    assert not any(k.startswith("qwen38_") for k in notebook["metadata"])
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))
            assert cell["execution_count"] is None and not cell["outputs"]


def test_markdown_cells_are_fully_bilingual_with_matching_structure():
    notebook = json.loads(NOTEBOOK.read_text())
    md = ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"]=="markdown"]
    code_count = sum(cell["cell_type"]=="code" for cell in notebook["cells"])
    assert code_count==17
    assert len(md)==code_count+1==18
    assert len(notebook["cells"])==1+2*code_count
    for idx, value in enumerate(md):
        assert value.count("### English")==value.count("### Tiếng Việt")==1, idx
        english, vietnamese = value.split("### English",1)[1].split("### Tiếng Việt",1)
        en_items=[s for s in english.splitlines() if s.startswith("- ")]
        vi_items=[s for s in vietnamese.splitlines() if s.startswith("- ")]
        assert len(en_items)==len(vi_items)==(4 if idx in (0,15,16,17) else 2),idx
        assert all(len(x)>10 for x in en_items+vi_items)
    for idx in range(1,len(notebook["cells"]),2):
        assert notebook["cells"][idx]["cell_type"]=="markdown"
        assert notebook["cells"][idx+1]["cell_type"]=="code"


def test_archive_integrity_is_verified_before_model_loading():
    notebook=json.loads(NOTEBOOK.read_text())
    first_code=(ROOT / "qwen38_runtime/stages/preflight.py").read_text()
    bootstrap="".join(notebook["cells"][2]["source"])
    assert "SOURCE_MANIFEST_SHA256" in bootstrap and "workflow.run(\"preflight\")" in bootstrap
    for required in ("ZipFile(pack)", "PACKAGE_SHA256_MISMATCH",
                     "PAYLOAD_SHA256_MISMATCH", "file_sha256(BUNDLE_MANIFEST)",
                     "RUN_ROOT.mkdir(exist_ok=False)", "total_verified==48"):
        assert required in first_code


def test_release_notebook_has_clean_code_cells_and_execution_guidance():
    """Verify the notebook structure without inspecting editorial vocabulary."""
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert len(notebook["cells"]) == 35
    code_cells = [cell for cell in notebook["cells"]
                  if cell["cell_type"] == "code"]
    assert len(code_cells) == 17
    assert all(cell.get("execution_count") is None
               and not cell.get("outputs")
               for cell in code_cells)

    # Reader-facing documentation describes running and reviewing results.
    if not (ROOT/"README.md").is_file() or not (ROOT/"README.vi.md").is_file():
        pytest.skip("Release documentation is published in the final README commit")
    english = (ROOT/"README.md").read_text(encoding="utf-8")
    vietnamese = (ROOT/"README.vi.md").read_text(encoding="utf-8")
    requirements = {
        english: ("**Run and review:**",
                  "run the cells on TPU v5e-8",
                  "CPU checks verify source and UI"),
        vietnamese: ("**Chạy và đánh giá:**",
                     "chạy các cell trên TPU v5e-8",
                     "Kiểm thử CPU kiểm mã nguồn và UI"),
    }
    for source, clauses in requirements.items():
        for clause in clauses:
            assert clause.casefold() in source.casefold()


def test_nonfatal_output_review_preserves_all_case_collection():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    code = {i: (ROOT / ("qwen38_runtime/stages/" + name + ".py")).read_text()
            for i, name in ((24,"vision_reuse"),(26,"image_cases"),
                            (28,"text_cases"),(30,"vietnamese"))}
    # Engine/data corruption remains fatal; content and token-budget outcomes
    # become explicit review flags instead of stopping the remaining cases.
    assert "assert TRACE['status']=='COMPLETE'" in code[24]
    assert "ocr_result['format_check']" in code[24]
    assert "OCR731_REVIEW_REQUIRED" in code[24]
    assert "CASE_REVIEW_REQUIRED" in code[26]
    assert "TEXT_REVIEW_REQUIRED" in code[28]
    assert "raise AssertionError(('OCR731_STRICT_FORMAT_MISMATCH'" not in code[24]
    assert "FAILED_OCR482_SEMANTIC" not in code[26]
    assert "FAILED_TEXT_SEMANTIC_OR_EOS" not in code[28]
    assert "cases_state['review_flags']=review_flags" in code[28]
    assert "cases_state['eos_count']" in code[28]
    assert "EN_RECORD_STATUSES" in code[30]


def test_single_official_support_dataset_slug():
    """Official notebook must attach only the stable Runtime Support Dataset v1."""
    from pathlib import Path
    import json
    root = Path(__file__).resolve().parents[1]
    notebook = json.loads((root / "notebooks/kaggle-tpu-v5e8-production-showcase.ipynb").read_text())
    source = "\n".join("".join(c["source"]) for c in notebook["cells"])
    assert "qwen38-27b-tpu-v5e8-full19-runtime-support" + "-v2" not in source
    assert "qwen38-27b-tpu-v5e8-full19-runtime-support" in source
    from support.package_dataset import DATASET_ID
    assert DATASET_ID == "dangkhoa2016/qwen38-27b-tpu-v5e8-full19-runtime-support"
