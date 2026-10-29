"""Build a deterministic, offline-verifiable source-only Kaggle code bundle.

The code bundle contains ONLY audited Python modules (no model weights and no
executed outputs). Zip members have a stable order and fixed timestamp.
"""
from __future__ import annotations
import hashlib
import io
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUNDLE_NAME = "notebook_source.zip"
MANIFEST_NAME = "notebook-source-manifest.json"


def digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def source_paths(root: Path = ROOT) -> list[str]:
    names = [p.relative_to(root).as_posix()
             for p in (root / "qwen38_runtime").rglob("*.py")]
    names.extend(["gallery/__init__.py", "gallery/render.py", "gallery/vi_state.py", "gallery/quick_tunnel.py"])
    return sorted(set(names))


def build_source_bundle(root: Path = ROOT) -> tuple[bytes, dict]:
    files = {}
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as bundle:
        for name in source_paths(root):
            contents = (root / name).read_bytes()
            files[name] = {"bytes": len(contents), "sha256": digest_bytes(contents)}
            entry = zipfile.ZipInfo(name, date_time=(2026, 10, 28, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_STORED
            entry.external_attr = 0o100644 << 16
            bundle.writestr(entry, contents)
    archive = stream.getvalue()
    manifest = {
        "format": "QWEN38_NOTEBOOK_SOURCE_V1",
        "archive": BUNDLE_NAME,
        "sha256": digest_bytes(archive),
        "bytes": len(archive),
        "file_count": len(files),
        "files": files,
    }
    return archive, manifest


def manifest_bytes(manifest: dict) -> bytes:
    return (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def install_bundle(destination: Path, root: Path = ROOT) -> dict:
    archive, manifest = build_source_bundle(root)
    for name, contents in (
        (BUNDLE_NAME, archive),
        (MANIFEST_NAME, manifest_bytes(manifest)),
    ):
        path = destination / name
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite source bundle: {path}")
        path.write_bytes(contents)
    return manifest
