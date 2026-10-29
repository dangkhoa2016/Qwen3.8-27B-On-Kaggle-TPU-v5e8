#!/usr/bin/env python3
"""Build a runtime/data-only Kaggle Dataset from the verified legacy distribution.

The 48 original payloads and their manifest are byte-identical. The project
Python source bundle is intentionally NOT copied. Never change Dataset v1.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

DATASET_ID = "dangkhoa2016/qwen38-27b-tpu-v5e8-full19-runtime-assets"
ARCHIVES = ("wheels.zip", "source.zip", "helpers.zip", "images.zip",
            "cases.zip", "cache.zip", "runtime.zip")
NOTICE_FILES = ("THIRD_PARTY_NOTICES.md", "THIRD_PARTY_NOTICES.vi.md")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build(source: Path, dest: Path) -> dict:
    """Accept existing 7 ZIPs or Kaggle's extracted category directories."""
    source, dest = source.resolve(), dest.resolve()
    if not source.is_dir():
        raise FileNotFoundError(source)
    if dest.exists() and any(dest.iterdir()):
        raise FileExistsError("Refusing to overwrite populated output: " + str(dest))
    manifest_path = source / "portable-bundle-manifest.json"
    original_bytes = manifest_path.read_bytes()
    manifest = json.loads(original_bytes)
    if (manifest.get("format") != "QWEN38_FULL19_ORGANIZED_SUPPORT_V1"
            or manifest.get("required_file_count") != 48
            or len(manifest.get("files", {})) != 48
            or manifest.get("archive_count") != 7
            or set(manifest.get("archives", {})) != set(ARCHIVES)):
        raise ValueError("Unexpected runtime manifest; do not build")
    # Complete validation before any output is written.
    packages = {}
    for name, info in sorted(manifest["archives"].items()):
        category = name.removesuffix(".zip")
        expected_paths = {p for p in manifest["files"] if p.startswith(category + "/")}
        if len(expected_paths) != info["file_count"]:
            raise ValueError("Archive member count mismatch: " + name)
        path = source / name
        if path.exists():
            raw = path.read_bytes()
        else:
            # Kaggle expands ZIP attachments into category folders.
            import io
            stream = io.BytesIO()
            with zipfile.ZipFile(stream, "w") as bundle:
                for logical in sorted(expected_paths):
                    p = source / logical
                    if not p.is_file():
                        raise FileNotFoundError(p)
                    member = zipfile.ZipInfo(Path(logical).name, date_time=(2026, 10, 28, 0, 0, 0))
                    member.compress_type = zipfile.ZIP_STORED
                    member.external_attr = 0o644 << 16
                    bundle.writestr(member, p.read_bytes())
            raw = stream.getvalue()
        if len(raw) != info["bytes"] or sha256(raw) != info["sha256"]:
            raise ValueError("Source package checksum mismatch: " + name)
        with zipfile.ZipFile(__import__("io").BytesIO(raw)) as bundle:
            names = bundle.namelist()
            expected_names = {Path(p).name for p in expected_paths}
            if set(names) != expected_names or len(names) != len(expected_names):
                raise ValueError("Archive layout mismatch: " + name)
            for logical in sorted(expected_paths):
                data = bundle.read(Path(logical).name)
                item = manifest["files"][logical]
                if len(data) != item["bytes"] or sha256(data) != item["sha256"]:
                    raise ValueError("Runtime payload checksum mismatch: " + logical)
        packages[name] = raw
    for filename in NOTICE_FILES:
        if not (source / filename).is_file():
            raise FileNotFoundError("Missing redistribution notice: " + filename)
    license_source = source / "THIRD_PARTY_LICENSES"
    if not license_source.is_dir() or not any(license_source.iterdir()):
        raise FileNotFoundError("Missing third-party license texts")
    dest.mkdir(parents=True, exist_ok=True)
    for name, raw in packages.items():
        (dest / name).write_bytes(raw)
    (dest / "portable-bundle-manifest.json").write_bytes(original_bytes)
    for filename in NOTICE_FILES:
        shutil.copyfile(source / filename, dest / filename)
    shutil.copytree(license_source, dest / "THIRD_PARTY_LICENSES")
    # Deliberately do not copy notebook_source.zip, its manifest or extracted modules.
    (dest / "README.md").write_text(
        "# Qwen3.8-27B TPU v5e-8 Full19 Runtime Assets\n\n"
        "This Dataset contains **48 SHA256-verified offline runtime/data payloads** "
        "in seven category archives, plus the original manifest, attribution notices "
        "and third-party licenses. It does **not** contain the BF16 model or "
        "the project's Python implementation. The Git-first Kaggle notebook "
        "clones implementation code from GitHub and records the exact commit SHA. "
        "Enable Kaggle Internet during bootstrap; the repository may require "
        "a read-only GitHub credential while private.\n\n"
        "Categories: wheels, source (upstream MaxText distribution), helpers, "
        "images, cases, cache and runtime. The MaxText source archive is a "
        "pinned third-party runtime dependency, **not the project's Python modules**.\n"
    )
    (dest / "README.vi.md").write_text(
        "# Tài nguyên runtime Full19 cho Qwen3.8-27B TPU v5e-8\n\n"
        "Dataset chỉ chứa **48 payload runtime/dữ liệu đã xác minh SHA256** "
        "trong bảy gói, manifest gốc, thông báo bản quyền và giấy phép bên thứ ba. "
        "Không chứa model BF16 hoặc Python source code của dự án. Notebook Git-first "
        "clone code từ GitHub và ghi nhận chính xác commit SHA; cần bật Kaggle Internet "
        "và token chỉ đọc khi repository còn private.\n\n"
        "Bảy nhóm: wheels, source (MaxText upstream), helpers, images, cases, cache "
        "và runtime. MaxText là dependency bên thứ ba được ghim checksum, "
        "**không phải mã Python của dự án**.\n"
    )
    (dest / "dataset-metadata.json").write_text(json.dumps({
        "title": "Qwen3.8-27B TPU v5e-8 Full19 Runtime Assets",
        "id": DATASET_ID,
        "licenses": [{"name": "other"}],
        "isPrivate": True,
        "description": "48 checksum-verified runtime/data payloads, no project Python source or model weights.",
    }, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True,
                        help="Verified original Dataset folder (7 ZIPs or extracted directories)")
    parser.add_argument("--destination", type=Path, required=True,
                        help="New, empty output folder; never original Dataset v1")
    args = parser.parse_args()
    manifest = build(args.source, args.destination)
    print("RUNTIME_ASSETS_PACKAGE_PASS", len(manifest["files"]), "payloads", len(manifest["archives"]), "archives")
    print("DATASET_ID", DATASET_ID)
    print("MANIFEST_SHA256", sha256((args.destination / "portable-bundle-manifest.json").read_bytes()))


if __name__ == "__main__":
    main()
