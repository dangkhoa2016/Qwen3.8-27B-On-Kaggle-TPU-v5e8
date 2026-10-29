#!/usr/bin/env python3
"""Synchronize the small offline bootstrap cell with the audited source bundle.

The Kaggle notebook verifies an immutable manifest hash before importing any
project Python code, and the original 48-file integrity gate runs immediately
after the module is loaded. Never fetch unpinned code over the network.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
try:
    from .notebook_source_bundle import build_source_bundle, manifest_bytes
except ImportError:  # direct CLI execution
    from notebook_source_bundle import build_source_bundle, manifest_bytes

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/kaggle-tpu-v5e8-production-showcase.ipynb"
CELL_INDEX = 2

BOOTSTRAP = '''# 01 · Verified offline source + runtime support (no network required).
# Options: 1=enabled, 0=disabled. Run All executes the original 19 EN cases first.
import hashlib
import json
import os
import stat
import sys
import tempfile
import zipfile
from pathlib import Path

DEFAULT_ENABLE_VI = "1"
DEFAULT_RETRY_VI_INCOMPLETE = "0"
DEFAULT_RENDER_HTML = "1"
# Edit these two defaults before Run All for optional public Gallery sharing.
DEFAULT_TUNNEL_ENABLED = "1"
DEFAULT_TUNNEL_HOLD_SECONDS = "900"
os.environ.setdefault("QWEN38_ENABLE_VI", DEFAULT_ENABLE_VI)
os.environ.setdefault("QWEN38_RETRY_VI_INCOMPLETE", DEFAULT_RETRY_VI_INCOMPLETE)
os.environ.setdefault("QWEN38_RENDER_HTML", DEFAULT_RENDER_HTML)
os.environ.setdefault("QWEN38_TUNNEL_ENABLED", DEFAULT_TUNNEL_ENABLED)
os.environ.setdefault("QWEN38_TUNNEL_HOLD_SECONDS", DEFAULT_TUNNEL_HOLD_SECONDS)

# Git source is bundled with the same Kaggle Runtime Support Dataset.
# Verify this exact code archive before importing any project modules.
SOURCE_MANIFEST_SHA256 = "__MANIFEST_SHA256__"
candidates = [
    Path(os.environ["QWEN38_SUPPORT_DIR"]) if os.environ.get("QWEN38_SUPPORT_DIR") else None,
    Path("/kaggle/input/datasets/dangkhoa2016/qwen38-27b-tpu-v5e8-full19-runtime-support"),
    Path("/kaggle/input/qwen38-27b-tpu-v5e8-full19-runtime-support"),
]
mount = next((p for p in candidates if p is not None
              and (p / "notebook-source-manifest.json").is_file()), None)
if mount is None:
    raise FileNotFoundError("NOTEBOOK_SOURCE_NOT_ATTACHED: attach the updated Runtime Support Dataset")

meta_bytes = (mount / "notebook-source-manifest.json").read_bytes()
assert hashlib.sha256(meta_bytes).hexdigest() == SOURCE_MANIFEST_SHA256, "NOTEBOOK_SOURCE_MANIFEST_MISMATCH"
meta = json.loads(meta_bytes)
assert meta["format"] == "QWEN38_NOTEBOOK_SOURCE_V1"
assert meta["file_count"] == len(meta["files"]) == 23
assert meta["archive"] == "notebook_source.zip"
archive = mount / meta["archive"]
if archive.is_file():
    assert archive.stat().st_size == meta["bytes"]
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == meta["sha256"], "NOTEBOOK_SOURCE_ZIP_SHA256_MISMATCH"
    source_dir = Path(tempfile.mkdtemp(prefix="qwen38-source-", dir="/kaggle/working"))
    with zipfile.ZipFile(archive) as source_zip:
        assert set(source_zip.namelist()) == set(meta["files"]), "NOTEBOOK_SOURCE_FILE_LIST_MISMATCH"
        for item in source_zip.infolist():
            relative = Path(item.filename)
            assert not relative.is_absolute() and ".." not in relative.parts
            assert stat.S_ISREG(item.external_attr >> 16), "Unsafe source ZIP member type"
            payload = source_zip.read(item)
            expected = meta["files"][item.filename]
            assert len(payload) == expected["bytes"]
            assert hashlib.sha256(payload).hexdigest() == expected["sha256"]
            out = source_dir / relative
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(payload)
else:
    source_dir = next((p for p in (
        mount / "notebook_source", mount,
        Path(os.environ["QWEN38_SOURCE_DIR"]) if os.environ.get("QWEN38_SOURCE_DIR") else None)
        if p is not None and (p / "qwen38_runtime" / "workflow.py").is_file()), None)
    assert source_dir is not None, "NOTEBOOK_SOURCE_FILES_NOT_FOUND"
    for relative, expected in meta["files"].items():
        data = (source_dir / relative).read_bytes()
        assert len(data) == expected["bytes"]
        assert hashlib.sha256(data).hexdigest() == expected["sha256"]

sys.path.insert(0, str(source_dir))
from qwen38_runtime import Workflow
workflow = Workflow()
workflow.run("preflight")
'''

def expected() -> str:
    _, manifest = build_source_bundle()
    digest = hashlib.sha256(manifest_bytes(manifest)).hexdigest()
    assert len(manifest["files"]) == 23, "Review bundled source paths before changing bootstrap"
    return BOOTSTRAP.replace("__MANIFEST_SHA256__", digest)


def sync(write: bool) -> None:
    """Synchronize the sole Git-first notebook; offline bundle is archival only."""
    try:
        from .git_first_bootstrap import sync as git_first_sync
    except ImportError:
        from git_first_bootstrap import sync as git_first_sync
    git_first_sync(write=write)
    print("NOTEBOOK_SOURCE_PIN_PASS")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--check", action="store_true")
    action.add_argument("--write", action="store_true")
    args = parser.parse_args()
    sync(write=args.write)
