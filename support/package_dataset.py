#!/usr/bin/env python3
"""Build an integrity-pinned, directory-organized Kaggle support dataset.

Input: local, already-verified flat support directory. No network, no weights.
Output: 7 category archives, central manifest, README EN/VI and Kaggle metadata.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

try:
    from .notebook_source_bundle import install_bundle
except ImportError:  # direct CLI execution
    from notebook_source_bundle import install_bundle


MODEL_MIRROR = "https://www.kaggle.com/models/dangkhoa2016/qwen-qwen3-8-27b"
DATASET_ID = "dangkhoa2016/qwen38-27b-tpu-v5e8-full19-runtime-support"
GROUPS = (
    "wheels",
    "source",
    "helpers",
    "images",
    "cases",
    "cache",
    "runtime",
)


def digest(file: Path) -> str:
    sha = hashlib.sha256()
    with file.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def group(filename: str) -> str:
    if filename.endswith(".whl"):
        return "wheels"
    if filename.startswith("maxtext-"):
        return "source"
    if filename.startswith("aot_") or filename == "split_vision_maxtext.patch":
        return "helpers"
    if filename.endswith((".png", ".jpg", ".jpeg")):
        return "images"
    if filename in ("assets_manifest.json", "photo_showcase_cases.json", "vision_showcase_cases.json"):
        return "cases"
    if filename.startswith("jit__") or filename == "pinned-cache-manifest.json":
        return "cache"
    if filename in ("engine_args.py", "manifest.json"):
        return "runtime"
    raise ValueError(f"Unclassified file: {filename}")


def build(source: Path, destination: Path) -> dict:
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Refusing to overwrite output: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    source_manifest = source / "portable-bundle-manifest.json"
    old = json.loads(source_manifest.read_text())
    assert old["required_file_count"] == len(old["files"]) == 48
    assert all("/" not in p for p in old["files"]), "Expected the verified flat input distribution"
    data: dict[str, dict] = {}
    for filename, info in old["files"].items():
        file = source / filename
        if not file.is_file() or file.stat().st_size != info["bytes"] or digest(file) != info["sha256"]:
            raise ValueError(f"Unverified source: {filename}")
        relative = f"{group(filename)}/{filename}"
        data[relative] = {"bytes": info["bytes"], "sha256": info["sha256"]}
    assert len(data) == 48
    archives = {}
    for name in GROUPS:
        members = sorted(k for k in data if k.startswith(name + "/"))
        if not members:
            raise AssertionError("Empty archive: " + name)
        package = destination / f"{name}.zip"
        with zipfile.ZipFile(package, "w") as bundle:
            for relative in members:
                # ZIP_STORED: archives, wheels, images and cache are already compressed.
                info = zipfile.ZipInfo(Path(relative).name, date_time=(2026, 10, 28, 0, 0, 0))
                info.compress_type = zipfile.ZIP_STORED
                info.external_attr = 0o644 << 16
                bundle.writestr(info, (source / Path(relative).name).read_bytes())
        archives[package.name] = {
            "sha256": digest(package),
            "bytes": package.stat().st_size,
            "file_count": len(members),
        }
    manifest = {
        "format": "QWEN38_FULL19_ORGANIZED_SUPPORT_V1",
        "checkpoint": "Kaggle Qwen3.8-27B original BF16 model, attached separately",
        "model_url": MODEL_MIRROR,
        "required_file_count": len(data),
        "archive_count": len(archives),
        "image_count": len([p for p in data if p.startswith("images/")]),
        "source_manifest_sha256": digest(source_manifest),
        "files": dict(sorted(data.items())),
        "archives": dict(sorted(archives.items())),
        "notice": "Support files are not model weights. Component licenses and prominent modification notices accompany redistributed source.",
    }
    # Separate, SHA256-verified offline Python code bundle (not in 48 payloads).
    install_bundle(destination)
    (destination / "portable-bundle-manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    (destination / "README.md").write_text(
        "# Qwen3.8-27B TPU Offline Support\n\n"
        "This dataset contains **support files only**, not BF16 model weights. "
        "Attach the [Kaggle BF16 model](" + MODEL_MIRROR + ") separately.\n\n"
        "The packager produces seven data ZIPs and one separate SHA256-verified "
        "notebook_source.zip with its own manifest. Kaggle may expose their contents "
        "as seven data directories plus notebook_source/ (23 Python modules). Both layouts contain "
        "the same 48 pinned files:\n\n"
        "| Package | Contents |\n|---|---|\n"
        "| `wheels.zip` | Pinned offline Python wheel files |\n"
        "| `source.zip` | Patched MaxText source archive and patch |\n"
        "| `helpers.zip` | JAX sharding/layout helpers and Split-Vision patch |\n"
        "| `images.zip` | Original image inputs (not Qwen-generated output) |\n"
        "| `cases.zip` | Source image inventory, prompts and case definitions |\n"
        "| `cache.zip` | Optional JAX executable cache and its manifest |\n"
        "| `runtime.zip` | Engine arguments and wheel metadata |\n\n"
        "Use the [Git notebook](https://github.com/dangkhoa2016/Qwen3.8-27B-On-Kaggle-TPU-v5e8/blob/main/notebooks/kaggle-tpu-v5e8-production-showcase.ipynb) "
        "with a Kaggle TPU v5e-8 and the BF16 model attachment. "
        "The notebook validates the central manifest and all 48 mounted payloads; "
        "for ZIP input it also checks archive SHA256 before extraction. "
        "Compilation cache reuse is optional and requires native TPU compatibility checks. "
        "Preserve component licenses and image provenance: SHA256 verifies identity, not redistribution rights.\n\nSee [third-party notices](THIRD_PARTY_NOTICES.md).\n"
    )
    (destination / "README.vi.md").write_text(
        "# Bộ dữ liệu hỗ trợ offline cho Qwen3.8-27B trên TPU\n\n"
        "Dataset này **chỉ chứa dữ liệu hỗ trợ**, không kèm trọng số BF16. "
        "Cần attach riêng [mô hình BF16 trên Kaggle](" + MODEL_MIRROR + ").\n\n"
        "Bộ đóng gói tạo bảy ZIP dữ liệu và notebook_source.zip chứa mã Python "
        "có manifest SHA256 riêng. Kaggle có thể giải nén bảy thư mục dữ liệu "
        "và notebook_source/ gồm 23 module; hai dạng giữ nguyên 48 payload:\n\n"
        "| Gói | Nội dung |\n|---|---|\n"
        "| `wheels.zip` | Thư viện Python dạng wheel đã ghim |\n"
        "| `source.zip` | Source MaxText và patch đã xác minh |\n"
        "| `helpers.zip` | Helper JAX sharding/layout và patch Split-Vision |\n"
        "| `images.zip` | Ảnh đầu vào (không phải ảnh Qwen sinh ra) |\n"
        "| `cases.zip` | Danh mục ảnh nguồn, prompt và định nghĩa tác vụ |\n"
        "| `cache.zip` | Cache executable JAX tùy chọn và manifest |\n"
        "| `runtime.zip` | Tham số engine và thông tin wheel |\n\n"
        "Dùng [notebook trong Git](https://github.com/dangkhoa2016/Qwen3.8-27B-On-Kaggle-TPU-v5e8/blob/main/notebooks/kaggle-tpu-v5e8-production-showcase.ipynb) "
        "trên Kaggle TPU v5e-8 và attach mô hình BF16. Notebook kiểm tra "
        "manifest trung tâm và SHA256 của 48 payload; nếu cấp ZIP thì kiểm checksum ZIP "
        "trước khi giải nén. Cache chỉ là tùy chọn; vẫn phải kiểm tra "
        "tính tương thích native TPU. Giữ giấy phép gốc và nguồn gốc ảnh; "
        "SHA256 chỉ xác minh file, không chứng minh quyền phân phối lại.\n\nXem [ghi nhận giấy phép](THIRD_PARTY_NOTICES.vi.md).\n"
    )

    # Include standard Apache-2.0 terms without vendor-specific example copyrights.
    license_src = Path(__file__).resolve().parents[1] / "THIRD_PARTY_LICENSES/QWEN3.8-27B-APACHE-2.0"
    raw_terms = license_src.read_text()
    example = "Copyright 2026 Alibaba Cloud"
    assert raw_terms.count(example) == 1, "Unexpected upstream license layout"
    neutral_terms = raw_terms.replace(example, "Copyright [yyyy] [name of copyright owner]").strip() + "\n"
    licenses_dir = destination / "THIRD_PARTY_LICENSES"
    licenses_dir.mkdir(exist_ok=False)
    (licenses_dir / "Apache-2.0.txt").write_text(neutral_terms)
    # Exact upstream xprof-v2.23.1 LICENSE: Apache-2.0 + TensorFlow Authors
    # copyright and bundled Polymer/Web Components BSD attribution clauses.
    xprof_license = Path(__file__).resolve().parents[1] / "THIRD_PARTY_LICENSES/XPROF-2.23.1-LICENSE"
    assert digest(xprof_license) == "399217a79d35c3d4be00b07bf34753b6b4f6c588fd34e4a624a474cbf3d1ed3c"
    shutil.copyfile(xprof_license, licenses_dir / "XPROF-2.23.1-LICENSE")

    # Pin the full license terms for frontend dependencies detected in XProf assets.
    # These sidecar files are outside the 48 runtime payloads and seven ZIPs.
    frontend_licenses = {
        "ANGULAR-15.2.9-MIT-LICENSE": "3b7a34b78350f58838dc24122f3594922d89ac7713e54a3511afb3d3a180bb46",
        "CHROMIUM-BSD-3-CLAUSE-LICENSE": "368cca1106be99d39ecd32a38d8305585d802a475effb66380b91ffc9bcf709b",
        "LIT-BSD-3-CLAUSE-LICENSE": "c94b94b40b85a5083dea825ae844b1cedb40dd425685362c135e143265dd6cd4",
        "TSLIB-0BSD-LICENSE": "210b19e543130388c68654b7497e967119ce17145f66ab7d85688fbd70f08751",
    }
    for name, expected_sha in sorted(frontend_licenses.items()):
        license_file = Path(__file__).resolve().parents[1] / "THIRD_PARTY_LICENSES" / name
        assert digest(license_file) == expected_sha, ("FRONTEND_LICENSE_SHA_MISMATCH", name)
        shutil.copyfile(license_file, licenses_dir / name)

    # Build-time WASM dependencies pinned by the XProf 2.23.1 WORKSPACE.
    wasm_build_license_files = {
        "DAWN-20250713-BSD-LICENSE": "e2908f7576fb12be5bdb8480cddb0be62badb498cadb821d043aeaf01b0b0899",
        "EMSCRIPTEN-4.0.6-LICENSE": "620a78084fc7ca97c0b5dea9abf891f3ffcadfdbf305276f099c9c4e12fc1d86",
        "IMGUI-1.88-MIT-LICENSE": "1b64b23b817e547982a86984685f5e9337695caa542753fb6f08d502d5208e38",
        "PERFETTO-53.0-LICENSE": "df3384bec476fac3525c723d0c10da9ac9820ef56cbb065ec035d32e9b55bd7a",
    }
    for name, expected_sha in sorted(wasm_build_license_files.items()):
        license_file = Path(__file__).resolve().parents[1] / "THIRD_PARTY_LICENSES" / name
        assert digest(license_file) == expected_sha, ("WASM_LICENSE_SHA_MISMATCH", name)
        shutil.copyfile(license_file, licenses_dir / name)
    (destination / "THIRD_PARTY_NOTICES.md").write_text(
        "# Third-party notices for the Kaggle runtime support Dataset\n\n"
        "The support Dataset does not include BF16 Qwen weights.\n"
        "Patched MaxText retains the upstream Apache-2.0 license and includes an in-tar"
        " modification record plus notices on affected source files.\n"
        "xprof 2.23.1 declares Apache-2.0; its 116-member wheel has no standalone LICENSE/NOTICE."
        " The exact upstream xprof-v2.23.1 LICENSE is provided as"
        " THIRD_PARTY_LICENSES/XPROF-2.23.1-LICENSE (TensorFlow copyright and Polymer BSD"
        " attribution included).\n"
        "Embedded xprof browser assets also contain Angular MIT and Chromium copyright"
        " notices; preserve those notices and separately verify any additional asset terms.\n"
        "Other wheels retain their own upstream license terms.\n"
        "The 16 evaluation inputs are not Qwen-generated outputs.\n"
        "Frontend attribution: Angular 15.2.9 (MIT), Lit (BSD-3-Clause), Chromium"
        " (BSD-3-Clause), Microsoft tslib (0BSD), Material Icons (Apache-2.0)."
        " The four additional full-text licenses are in THIRD_PARTY_LICENSES/."
        " Angular and Lit source licenses are upstream snapshots; not all compiled"
        " asset dependency versions can be established solely from the wheel.\n"
        "Material Icons WOFF2 exactly matches the upstream xprof-v2.23.1 asset."
        " WASM BUILD provenance in XProf WORKSPACE: Dear ImGui 1.88 MIT, Perfetto"
        " 53.0 Apache-2.0 with extra bundled notices, Dawn WebGPU snapshot BSD,"
        " Emscripten 4.0.6 MIT/UIUC-NCSA. Full source license copies accompany"
        " the Dataset. This is build-input attribution, not binary closure proof.\n"
        "The project's MIT license does not replace third-party terms.\n"
    )
    (destination / "THIRD_PARTY_NOTICES.vi.md").write_text(
        "# Giấy phép bên thứ ba cho Kaggle Runtime Support Dataset\n\n"
        "Dataset không chứa trọng số BF16 của Qwen.\n"
        "MaxText đã điều chỉnh giữ giấy phép Apache-2.0 upstream và bổ sung thông báo"
        " thay đổi trong source archive và các file liên quan.\n"
        "xprof 2.23.1 khai báo Apache-2.0; wheel gồm 116 mục nhưng không có LICENSE/NOTICE"
        " độc lập. Dataset kèm nguyên bản LICENSE đúng tag xprof-v2.23.1 tại"
        " THIRD_PARTY_LICENSES/XPROF-2.23.1-LICENSE, có copyright TensorFlow và"
        " điều khoản BSD cho Polymer.\n"
        "Các tệp giao diện trong wheel cũng chứa thông báo Angular MIT và copyright"
        " Chromium; giữ nguyên các thông báo này và kiểm thêm điều khoản tài nguyên nếu phân phối.\n"
        "Các wheel khác tuân theo điều khoản giấy phép upstream riêng.\n"
        "16 ảnh đầu vào chỉ dùng phục vụ kiểm thử; không phải ảnh do Qwen tạo.\n"
        "Bản quyền frontend: Angular 15.2.9 (MIT), Lit (BSD-3-Clause), Chromium"
        " (BSD-3-Clause), Microsoft tslib (0BSD), Material Icons (Apache-2.0)."
        " Bốn giấy phép nguyên văn được kèm theo tại THIRD_PARTY_LICENSES/."
        " Các license Angular và Lit là snapshot upstream; wheel không cho phép"
        " xác định chính xác version của mọi thư viện đã biên dịch.\n"
        "Font Material Icons WOFF2 khớp byte với asset nguồn tag xprof-v2.23.1."
        " WORKSPACE XProf ghim nguồn build WASM: Dear ImGui 1.88 MIT, Perfetto"
        " 53.0 Apache-2.0 và notice kèm theo, Dawn WebGPU BSD, Emscripten"
        " 4.0.6 MIT/UIUC-NCSA. Dataset kèm các license nguồn gốc."
        " Đây là kiểm tra dependency build, chưa chứng minh toàn bộ binary closure.\n"
        "Giấy phép MIT của dự án không thay thế giấy phép các thành phần bên thứ ba.\n"
    )


    (destination / "dataset-metadata.json").write_text(
        json.dumps({
            "title": "Qwen3.8-27B TPU v5e-8 Full19 Runtime Support",
            "id": DATASET_ID,
            "licenses": [{"name": "other"}],
            "isPrivate": True,
            "description": "Organized, SHA256-verified BF16 Text + Vision support. Seven purpose-specific ZIPs; model weights attached separately. Third-party licenses apply.",
        }, ensure_ascii=False, indent=2) + "\n"
    )
    return manifest


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--destination", type=Path, required=True)
    args = p.parse_args()
    manifest = build(args.source, args.destination)
    print("PACKAGE_OK", manifest["required_file_count"], "files", manifest["archive_count"], "archives")
    print("MANIFEST_SHA256", digest(args.destination / "portable-bundle-manifest.json"))
    for name, item in manifest["archives"].items():
        print("ARCHIVE", name, item["file_count"], "files", item["bytes"], "bytes", item["sha256"])


if __name__ == "__main__":
    main()
