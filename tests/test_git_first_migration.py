"""Git-first migration regression tests; the legacy offline release remains intact."""
from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from support.git_first_bootstrap import BOOTSTRAP, NOTEBOOK, notebook_content
from support.package_dataset import build as build_legacy
from support.package_runtime_assets import DATASET_ID, ARCHIVES, build as build_runtime

ROOT = Path(__file__).resolve().parents[1]


def test_git_first_notebook_maintains_17_stages_and_no_source_bundle():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert notebook == notebook_content()
    assert len(notebook["cells"]) == 35
    code = ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert len(code) == 17
    assert code[0] == BOOTSTRAP
    assert "git" in code[0] and "GIT_SOURCE_READY" in code[0]
    assert "notebook_source.zip" not in code[0]
    assert "notebook-source-manifest.json" not in code[0]
    assert "SOURCE_PAYLOAD" not in code[0]
    assert "SOURCE_ARCHIVE_SHA256" not in code[0]
    assert "SOURCE_REVISION" not in code[0]
    assert "PUBLIC_GIT_CLONE_FAILED" in code[0]
    assert "GITHUB_READ_TOKEN" not in code[0]
    assert "kaggle_secrets" not in code[0]
    assert 'DEFAULT_GIT_REF = "main"' in code[0]
    assert "GIT_SOURCE_REUSED_VERIFIED" in code[0]
    assert "GIT_SOURCE_STALE" in code[0]
    assert not re.search(r'^DEFAULT_GIT_REF\s*=\s*["\x27][0-9a-f]{40}', code[0], re.M)
    assert all(c["execution_count"] is None and not c["outputs"]
               for c in notebook["cells"] if c["cell_type"] == "code")
    for cell in code:
        ast.parse(cell)
    assert len(list((ROOT / "notebooks").glob("*.ipynb"))) == 1
    assert NOTEBOOK.name == "kaggle-tpu-v5e8-production-showcase.ipynb"


def _git(*args, cwd: Path) -> str:
    p = subprocess.run(["git", *args], cwd=cwd, check=True, text=True,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return p.stdout.strip()


def test_gallery_instructions_distinguish_interactive_tunnel_and_saved_outputs():
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    gallery = "".join(nb["cells"][31]["source"])
    tunnel = "".join(nb["cells"][33]["source"])
    assert "only in the interactive Kaggle editor" in gallery
    assert "trình chỉnh sửa Kaggle tương tác" in gallery
    assert "API/CLI and batch executions skip Tunnel" in gallery
    assert "output của lượt chạy Kaggle" in gallery
    assert "not permanent storage" in tunnel
    assert "không phải nơi lưu trữ vĩnh viễn" in tunnel
    assert "Step 17 automatically provides" not in gallery
    assert "Bước 17 tự tạo link" not in gallery
    assert "ZIP remains available after the notebook finishes" not in gallery


def test_bootstrap_clones_exact_commit_and_rejects_wrong_expected_sha(tmp_path, monkeypatch, capsys):
    origin = tmp_path / "source"
    origin.mkdir()
    _git("init", "-b", "main", cwd=origin)
    _git("config", "user.name", "Test", cwd=origin)
    _git("config", "user.email", "test@example.invalid", cwd=origin)
    package = origin / "qwen38_runtime"
    package.mkdir()
    (package / "__init__.py").write_text(
        "class Workflow:\n"
        "    def run(self, step):\n"
        "        from pathlib import Path\n"
        "        assert step == 'preflight'\n"
        "        Path(__file__).parents[1].joinpath('workflow-called').write_text(step)\n"
    )
    _git("add", ".", cwd=origin)
    _git("commit", "-m", "Fixture", cwd=origin)
    expected = _git("rev-parse", "HEAD", cwd=origin)
    # Only the test substitutes a local file URL; the distributed notebook
    # uses the fixed GitHub URL and never accepts a URL override.
    code = BOOTSTRAP.replace(
        "https://github.com/dangkhoa2016/Qwen3.8-27B-On-Kaggle-TPU-v5e8.git",
        origin.as_uri(),
    ).replace("from support.git_first_runtime import install_git_first_preflight\ninstall_git_first_preflight()\n", "")
    monkeypatch.setenv("QWEN38_WORKING_ROOT", str(tmp_path / "run"))
    monkeypatch.setenv("QWEN38_GIT_REF", expected)
    monkeypatch.setenv("QWEN38_GITHUB_TOKEN", "unused-test-credential")
    before_path = list(sys.path)
    previous_runtime = sys.modules.pop("qwen38_runtime", None)
    try:
        exec(compile(code, "<git-first-bootstrap>", "exec"), {})
        captured = capsys.readouterr().out
        assert expected in captured
        checkout = tmp_path / "run" / "qwen38-git-source"
        assert _git("rev-parse", "HEAD", cwd=checkout) == expected
        assert (checkout / "workflow-called").read_text() == "preflight"
    finally:
        sys.path[:] = before_path
        sys.modules.pop("qwen38_runtime", None)
        if previous_runtime is not None:
            sys.modules["qwen38_runtime"] = previous_runtime

    bad = "0" * 40 if expected != "0" * 40 else "1" * 40
    monkeypatch.setenv("QWEN38_GIT_REF", "main")
    monkeypatch.setenv("QWEN38_EXPECTED_COMMIT", bad)
    monkeypatch.setenv("QWEN38_WORKING_ROOT", str(tmp_path / "second-run"))
    with pytest.raises(RuntimeError, match="GIT_COMMIT_MISMATCH"):
        exec(compile(code, "<git-first-bootstrap>", "exec"), {})
    assert not (tmp_path / "second-run" / "qwen38-git-source" / "workflow-called").exists()


def test_git_bootstrap_retry_checks_clean_ref_and_remote_without_tpu(tmp_path, monkeypatch, capsys):
    origin = tmp_path / "source"
    origin.mkdir()
    _git("init", "-b", "main", cwd=origin)
    _git("config", "user.name", "Test", cwd=origin)
    _git("config", "user.email", "test@example.invalid", cwd=origin)
    (origin / "README.md").write_text("original")
    _git("add", ".", cwd=origin)
    _git("commit", "-m", "Fixture", cwd=origin)
    initial_sha = _git("rev-parse", "HEAD", cwd=origin)
    code = BOOTSTRAP.split('os.environ["QWEN38_SOURCE_DIR"]')[0]
    code = code.replace(
        "https://github.com/dangkhoa2016/Qwen3.8-27B-On-Kaggle-TPU-v5e8.git",
        origin.as_uri(),
    )
    monkeypatch.setenv("QWEN38_WORKING_ROOT", str(tmp_path / "run"))
    monkeypatch.setenv("QWEN38_GIT_REF", "main")
    monkeypatch.delenv("QWEN38_EXPECTED_COMMIT", raising=False)
    clone = tmp_path / "run" / "qwen38-git-source"
    exec(compile(code, "<clone-only>", "exec"), {})
    assert _git("rev-parse", "HEAD", cwd=clone) == initial_sha
    exec(compile(code, "<safe-retry>", "exec"), {})
    assert "GIT_SOURCE_REUSED_VERIFIED" in capsys.readouterr().out
    (clone / "untracked-file").write_text("unsafe")
    with pytest.raises(RuntimeError, match="GIT_SOURCE_NOT_CLEAN"):
        exec(compile(code, "<dirty-retry>", "exec"), {})
    (clone / "untracked-file").unlink()
    (origin / "README.md").write_text("updated")
    _git("add", ".", cwd=origin)
    _git("commit", "-m", "New remote revision", cwd=origin)
    with pytest.raises(RuntimeError, match="GIT_SOURCE_STALE"):
        exec(compile(code, "<stale-retry>", "exec"), {})
    _git("remote", "set-url", "origin", "https://example.invalid/untrusted", cwd=clone)
    with pytest.raises(RuntimeError, match="GIT_SOURCE_REMOTE_MISMATCH"):
        exec(compile(code, "<untrusted-retry>", "exec"), {})


def test_git_bootstrap_rejects_incomplete_checkout(tmp_path, monkeypatch):
    root = tmp_path / "run"
    (root / "qwen38-git-source").mkdir(parents=True)
    monkeypatch.setenv("QWEN38_WORKING_ROOT", str(root))
    monkeypatch.setenv("QWEN38_GIT_REF", "main")
    monkeypatch.delenv("QWEN38_EXPECTED_COMMIT", raising=False)
    code = BOOTSTRAP.split('os.environ["QWEN38_SOURCE_DIR"]')[0]
    with pytest.raises(RuntimeError, match="GIT_SOURCE_INCOMPLETE"):
        exec(compile(code, "<incomplete-clone>", "exec"), {})


def test_runtime_only_dataset_retains_original_manifest_and_all_48_hashes(tmp_path):
    source = tmp_path / "flat"
    source.mkdir()
    old = json.loads((ROOT / "support/portable-bundle-manifest.json").read_text())
    files = {}
    for index, logical in enumerate(old["files"]):
        filename = Path(logical).name
        content = ("fixture " + str(index) + " " + filename).encode()
        (source / filename).write_bytes(content)
        files[filename] = {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
    (source / "portable-bundle-manifest.json").write_text(
        json.dumps({"required_file_count": 48, "files": files})
    )
    legacy = tmp_path / "legacy-distribution"
    build_legacy(source, legacy)
    migrated = tmp_path / "runtime-only"
    manifest = build_runtime(legacy, migrated)
    assert len(manifest["files"]) == 48
    assert {p.name for p in migrated.glob("*.zip")} == set(ARCHIVES)
    assert not list(migrated.glob("notebook_source*"))
    assert (migrated / "portable-bundle-manifest.json").read_bytes() == (
        legacy / "portable-bundle-manifest.json").read_bytes()
    assert json.loads((migrated / "dataset-metadata.json").read_text())["id"] == DATASET_ID
    assert all((migrated / f).is_file() for f in ARCHIVES)
    for filename in ("THIRD_PARTY_NOTICES.md", "THIRD_PARTY_NOTICES.vi.md"):
        assert (migrated / filename).read_bytes() == (legacy / filename).read_bytes()


def test_provenance_is_bound_to_repository_and_dataset_manifest():
    extension = (ROOT / "support/git_first_runtime.py").read_text()
    for phrase in ("SOURCE_SHA_CHANGED", "SOURCE_REMOTE_CHANGED", "source-provenance.json",
                   "PORTABLE_MANIFEST_SHA256", "QWEN38_SOURCE_COMMIT",
                   "qwen38-27b-tpu-v5e8-full19-runtime-assets"):
        assert phrase in extension
    assert "install_git_first_preflight()" in BOOTSTRAP
