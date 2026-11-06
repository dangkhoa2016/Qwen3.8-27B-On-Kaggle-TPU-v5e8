"""CPU-only checks for the optional Cloudflare Gallery preview cell."""
import json
import hashlib
import zipfile
from pathlib import Path

import pytest

from gallery import quick_tunnel as tunnel

ROOT = Path(__file__).resolve().parents[1]


def test_notebook_cell_matches_readable_source():
    notebook = json.loads((ROOT / "notebooks/kaggle-tpu-v5e8-production-showcase.ipynb").read_text())
    assert len(notebook["cells"]) == 35
    assert "KAGGLE_KERNEL_RUN_TYPE" in "".join(notebook["cells"][-1]["source"])
    assert "GALLERY_TUNNEL_SKIPPED_NON_INTERACTIVE" in "".join(notebook["cells"][-1]["source"])
    assert "### English" in "".join(notebook["cells"][-2]["source"])
    assert "### Tiếng Việt" in "".join(notebook["cells"][-2]["source"])


def test_tunnel_options_and_offline_fallback(monkeypatch, tmp_path):
    monkeypatch.delenv("QWEN38_TUNNEL_HOLD_SECONDS", raising=False)
    assert tunnel._hold_seconds() == 900
    for seconds in (60, 900, 3600):
        monkeypatch.setenv("QWEN38_TUNNEL_HOLD_SECONDS", str(seconds))
        assert tunnel._hold_seconds() == seconds
    for invalid in ("0", "3601", "invalid"):
        monkeypatch.setenv("QWEN38_TUNNEL_HOLD_SECONDS", invalid)
        with pytest.raises(ValueError):
            tunnel._hold_seconds()
    monkeypatch.setenv("QWEN38_TUNNEL_ENABLED", "0")
    assert tunnel.launch_gallery_tunnel() is None
    monkeypatch.setenv("QWEN38_TUNNEL_ENABLED", "1")
    assert tunnel.launch_gallery_tunnel(tmp_path / "missing") is None
    assert tunnel._LINK.search("https://friendly-blue-owl.trycloudflare.com")


def test_url_and_process_lifecycle_without_real_network(monkeypatch, tmp_path):
    gallery = tmp_path / "gallery"
    gallery.mkdir()
    (gallery / "index.html").write_text("<h1>synthetic test fixture</h1>")
    fake = tmp_path / "cloudflared-stub"
    fake.write_text("#!/bin/sh\nprintf 'https://test-gallery-url.trycloudflare.com\\n'\nsleep 10\n")
    fake.chmod(0o755)
    monkeypatch.setattr(tunnel, "_verified_cloudflared", lambda: fake)
    monkeypatch.setattr(tunnel, "_hold_seconds", lambda: 0.1)
    assert tunnel.launch_gallery_tunnel(gallery) == "https://test-gallery-url.trycloudflare.com"
    assert not (tmp_path / "gallery-public-url.txt").exists(), "Never save an expired Quick Tunnel URL as an artifact"
    assert not (tmp_path / "gallery-public-url.txt").exists(), "Never save an expired Quick Tunnel URL as an artifact"

def test_public_gallery_presentation_and_zip_only_change_html(tmp_path):
    gallery = tmp_path / "gallery"
    gallery.mkdir()
    page = gallery / "index.html"
    page.write_text("<html>BF16 · Internal review</html>", encoding="utf-8")
    evidence = gallery / "evidence.json"
    evidence.write_text('{"cases":[]}', encoding="utf-8")
    before = hashlib.sha256(evidence.read_bytes()).hexdigest()
    with zipfile.ZipFile(tmp_path / "inference-gallery.zip", "w") as archive:
        archive.write(page, "gallery/index.html")
    assert tunnel._prepare_public_gallery(gallery) is True
    assert tunnel._prepare_public_gallery(gallery) is False
    assert "Inference showcase" in page.read_text()
    assert hashlib.sha256(evidence.read_bytes()).hexdigest() == before
    with zipfile.ZipFile(tmp_path / "inference-gallery.zip") as archive:
        assert archive.read("gallery/index.html") == page.read_bytes()
        assert archive.read("gallery/evidence.json") == evidence.read_bytes()

def test_notebook_gallery_only_opens_in_interactive_kaggle(monkeypatch, tmp_path, capsys):
    """Batch/API, missing and unexpected run types must not open or hold Tunnel."""
    from types import SimpleNamespace

    notebook = json.loads((ROOT / "notebooks/kaggle-tpu-v5e8-production-showcase.ipynb").read_text())
    cell = "".join(notebook["cells"][-1]["source"])
    gallery_dir = tmp_path / "gallery"
    gallery_dir.mkdir()
    calls = []
    monkeypatch.setattr(tunnel, "launch_gallery_tunnel", lambda path: calls.append(path))
    fake_workflow = SimpleNamespace(require=lambda key: tmp_path if key == "RUN_ROOT" else None)
    monkeypatch.setenv("QWEN38_TUNNEL_ENABLED", "1")

    for mode in ("Batch", "batch", "", "Unknown"):
        if mode:
            monkeypatch.setenv("KAGGLE_KERNEL_RUN_TYPE", mode)
        else:
            monkeypatch.delenv("KAGGLE_KERNEL_RUN_TYPE", raising=False)
        exec(compile(cell, "<notebook-cell-17>", "exec"), {"workflow": fake_workflow})
        assert not calls, mode
        assert "GALLERY_TUNNEL_SKIPPED_NON_INTERACTIVE" in capsys.readouterr().out

    monkeypatch.setenv("KAGGLE_KERNEL_RUN_TYPE", "Interactive")
    exec(compile(cell, "<notebook-cell-17>", "exec"), {"workflow": fake_workflow})
    assert calls == [gallery_dir]
