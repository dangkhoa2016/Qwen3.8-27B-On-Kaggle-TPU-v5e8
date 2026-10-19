"""CPU-only acceptance gates for pinned MaxText support and notebook Git inputs."""
import json
import tarfile
from pathlib import Path
import pytest
from api import tpu_factory as t

def test_real_attached_preflight_if_present():
    if not (t.MODEL_DIR.is_dir() and t.SUPPORT_DIR.is_dir() and t.MAXTEXT_DIR.is_dir()):
        pytest.skip("Only runnable on authorized Kaggle machine with attached datasets")
    assert t.preflight()["maxtext_commit"] == t.MAXTEXT_COMMIT

def test_missing_source_rejected_before_loading(tmp_path):
    with pytest.raises((FileNotFoundError, RuntimeError)):
        t.preflight(model_dir=tmp_path / "missing", support_dir=tmp_path / "none",
                    source_dir=tmp_path / "none")

def test_source_archive_rejects_untrusted_member(tmp_path, monkeypatch):
    import io
    root = tmp_path / "support"
    root.mkdir()
    archive = root / "maxtext-f0b2abe-qwen38-text-vision-patched.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        info = tarfile.TarInfo("../escaped.txt")
        content = b"unsafe"
        info.size = len(content)
        tar.addfile(info, io.BytesIO(content))
    monkeypatch.setattr(t, "ARCHIVE_SHA256", t.digest(archive))
    with pytest.raises(RuntimeError, match="Unsafe"):
        t.ensure_pinned_source(support_dir=root, source_dir=tmp_path / "extract")

def test_missing_or_mismatched_archive_rejected(tmp_path):
    with pytest.raises(RuntimeError):
        t._archive_path(tmp_path)

def test_notebook_git_commit_pin_resolves_locally():
    import subprocess
    import nbformat
    repo = Path(__file__).resolve().parents[1]
    notebook = nbformat.read(repo / "notebooks/kaggle-tpu-v5e8-production-showcase.ipynb", as_version=4)
    snippets = [cell.source for cell in notebook.cells if "EXAMPLES_COMMIT =" in cell.source]
    assert len(snippets) == 1
    import re
    match = re.search(r"EXAMPLES_COMMIT = '([0-9a-f]{40})'", snippets[0])
    assert match
    commit = match.group(1)
    subprocess.check_call(["git", "-C", str(repo), "cat-file", "-e", commit + "^{commit}"])
    actual = subprocess.check_output(
        ["git", "-C", str(repo), "show", commit + ":examples/assets_manifest.json"])
    manifest = json.loads(actual)
    assert manifest["assets"]
