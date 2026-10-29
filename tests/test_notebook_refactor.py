"""The public TPU notebook is readable, offline, and cryptographically pinned."""
from __future__ import annotations
import ast
import hashlib
import io
import json
import stat
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from qwen38_runtime.workflow import STEPS, Workflow
from support.notebook_source_bundle import build_source_bundle, manifest_bytes
from support.sync_notebook_bootstrap import expected as expected_bootstrap
from support.git_first_bootstrap import BOOTSTRAP as GIT_FIRST_BOOTSTRAP

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/kaggle-tpu-v5e8-production-showcase.ipynb"


def test_user_facing_code_is_only_bootstrap_and_one_stage_call_per_cell():
    notebook = json.loads(NOTEBOOK.read_text())
    code = ["".join(c["source"]) for c in notebook["cells"]
            if c["cell_type"] == "code"]
    assert len(notebook["cells"]) == 35 and len(code) == len(STEPS) + 1 == 17
    assert len(code[0].splitlines()) <= 110
    assert code[0] == GIT_FIRST_BOOTSTRAP
    assert "urllib" not in code[0] and "requests.get" not in code[0]
    assert "PUBLIC_GIT_CLONE_FAILED" in code[0]
    assert "GITHUB_READ_TOKEN" not in code[0]
    assert "from qwen38_runtime import Workflow" in code[0]
    assert 'workflow.run("preflight")' in code[0]
    for name, cell in zip(STEPS[1:], code[1:-1]):
        assert cell == "# " + name.replace("_", " ").capitalize() + "\n" + "workflow.run(" + repr(name) + ")\n"
        ast.parse(cell)
    assert sum(len(c.splitlines()) for c in code[:-1]) <= 150
    assert "KAGGLE_KERNEL_RUN_TYPE" in code[-1] and "interactive" in code[-1]
    assert "GALLERY_TUNNEL_SKIPPED_NON_INTERACTIVE" in code[-1]
    assert len(code[-1].splitlines()) <= 10
    ast.parse(code[-1])
    assert all(c["execution_count"] is None and not c["outputs"]
               for c in notebook["cells"] if c["cell_type"] == "code")



def test_bilingual_notebook_markdown_has_valid_inline_code_delimiters():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    for index, cell in enumerate(notebook["cells"], start=1):
        if cell["cell_type"] != "markdown":
            continue
        source = "".join(cell["source"])
        assert "‹" not in source and "›" not in source, index
        if index in (30, 32, 34):  # EN/VI guidance for steps 15, 16, 17
            assert "### English" in source and "### Tiếng Việt" in source
            if index == 34:
                assert "Save & Run All" not in source
                assert "GALLERY_PUBLIC_URL" in source
            for line in source.splitlines():
                assert line.count(chr(96)) % 2 == 0, (index, line)


def test_offline_notebook_introduction_and_bundled_preview():
    import base64
    notebook = json.loads(NOTEBOOK.read_text())
    cell = notebook["cells"][0]
    intro = "".join(cell["source"])
    assert intro.index("### English") < intro.index("### Tiếng Việt")
    assert "Runtime Assets Dataset" in intro
    assert "19 Evaluation Cases" in intro
    assert "GITHUB_READ_TOKEN" not in intro
    assert "QWEN38_GIT_REF" not in intro
    assert "Git-first" not in intro
    assert "main chỉ dành cho phát triển" not in intro
    assert "HTML Gallery" in intro
    preview = "qwen38-robotics-preview.jpg"
    assert "attachment:" + preview in intro
    image = base64.b64decode(cell["attachments"][preview]["image/jpeg"], validate=True)
    assert image.startswith(bytes.fromhex("ffd8ff"))
    assert len(image) < 16_000


def test_offline_python_archive_is_deterministic_and_matches_all_sources():
    first, m = build_source_bundle()
    second, m2 = build_source_bundle()
    assert first == second and m == m2
    assert m["file_count"] == len(m["files"]) == 23
    assert hashlib.sha256(first).hexdigest() == m["sha256"]
    assert "gallery/render.py" in m["files"]
    assert "gallery/vi_state.py" in m["files"]
    assert "gallery/quick_tunnel.py" in m["files"]
    assert "qwen38_runtime/stages/ocr_probe.py" in m["files"]
    assert m["files"]["qwen38_runtime/workflow.py"]["bytes"] > 0
    with zipfile.ZipFile(io.BytesIO(first)) as z:
        assert set(z.namelist()) == set(m["files"])
        for name, info in sorted(m["files"].items()):
            source = z.read(name)
            assert source == (ROOT / name).read_bytes()
            assert len(source) == info["bytes"]
            assert hashlib.sha256(source).hexdigest() == info["sha256"]
            assert ".." not in Path(name).parts
            assert stat.S_ISREG(z.getinfo(name).external_attr >> 16)
            ast.parse(source.decode("utf-8"))


def test_bootstrap_manifest_pin_follows_exact_python_sources(tmp_path):
    _, m = build_source_bundle()
    manifest_digest = hashlib.sha256(manifest_bytes(m)).hexdigest()
    assert manifest_digest in expected_bootstrap()
    assert "NOTEBOOK_SOURCE_MANIFEST_MISMATCH" in expected_bootstrap()
    assert "NOTEBOOK_SOURCE_ZIP_SHA256_MISMATCH" in expected_bootstrap()
    assert "notebook-source-manifest.json" in expected_bootstrap()
    from support.sync_notebook_bootstrap import sync
    sync(write=False)


def test_stages_import_without_importing_jax_or_loading_weights():
    from importlib import import_module
    for stage in STEPS:
        obj = import_module("qwen38_runtime.stages." + stage)
        assert callable(obj.execute)
        source = (ROOT / "qwen38_runtime" / "stages" / (stage + ".py")).read_text()
        assert "def execute(context):" in source
        # There is no hidden exec/eval bridge: dependencies are named and explicit.
        assert "exec(" not in source
        assert "context.require(" in source or stage in ("preflight", "text_schedule")


def test_workflow_rejects_out_of_order_cells_and_tracks_named_results(monkeypatch):
    import qwen38_runtime.workflow as module
    seen = []
    def fake_import(name):
        stage = name.rsplit(".", 1)[-1]
        def execute(context):
            seen.append(stage)
            if stage == "preflight":
                return {"RUN_ROOT": Path("/tmp/synthetic-run")}
            assert context.require("RUN_ROOT") == Path("/tmp/synthetic-run")
            return {"stage": stage}
        return SimpleNamespace(execute=execute)
    monkeypatch.setattr(module, "import_module", fake_import)
    workflow = Workflow()
    with pytest.raises(RuntimeError, match="Run cells in order"):
        workflow.run("engine")
    workflow.run("preflight")
    workflow.run("dependencies")
    with pytest.raises(RuntimeError, match="Run cells in order"):
        workflow.run("preflight")
    assert seen == ["preflight", "dependencies"]
    assert workflow.completed == ["preflight", "dependencies"]
    assert workflow.require("stage") == "dependencies"
    with pytest.raises(RuntimeError, match="dependency missing"):
        workflow.require("FAKE_KEY")
