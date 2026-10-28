"""Release input alignment and offline notebook preflight invariants."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/kaggle-tpu-v5e8-production-showcase.ipynb"
SUPPORT = ROOT / "support/portable-bundle-manifest.json"


def test_notebook_support_manifest_matches_version_controlled_examples():
    nb = json.loads(NOTEBOOK.read_text())
    code = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    markdown = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "markdown")
    manifest = json.loads(SUPPORT.read_text())
    digest = hashlib.sha256(SUPPORT.read_bytes()).hexdigest()
    assert f"PORTABLE_MANIFEST_SHA256='{digest}'" in code
    assert "qwen38-27b-tpu-v5e8-full19-runtime-support" in code
    assert "private-rc-sha256-manifest.json" not in code
    assert "UNEXECUTED source notebook" not in markdown
    assert "CHƯA CHẠY" not in markdown
    assets = json.loads((ROOT / "examples/assets_manifest.json").read_text())["assets"]
    assert len(assets) == 16 and len(manifest["files"]) == 48
    for item in assets:
        entry = manifest["files"][Path(item["path"]).name]
        assert item["sha256"] == entry["sha256"] and item["bytes"] == entry["bytes"]


def test_code_cells_parse_cleanly():
    notebook = json.loads(NOTEBOOK.read_text())
    assert set(notebook["metadata"]) == {"kernelspec", "language_info"}
    assert not any(k.startswith("qwen38_") for k in notebook["metadata"])
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))
            assert cell["execution_count"] is None and not cell["outputs"]
