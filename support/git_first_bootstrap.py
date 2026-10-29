#!/usr/bin/env python3
"""Build the Git-first Kaggle notebook. Original offline release is immutable."""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/kaggle-tpu-v5e8-production-showcase.ipynb"

BOOTSTRAP = r'''# 01 · Initialize verified source and runtime. Enable Kaggle Internet.
import os
import re
import sys
import subprocess
from pathlib import Path

DEFAULT_GIT_REF = "main"
DEFAULT_ENABLE_VI = "1"
DEFAULT_RETRY_VI_INCOMPLETE = "0"
DEFAULT_RENDER_HTML = "1"
DEFAULT_TUNNEL_ENABLED = "1"
DEFAULT_TUNNEL_HOLD_SECONDS = "900"
for key, default in (
    ("QWEN38_ENABLE_VI", DEFAULT_ENABLE_VI),
    ("QWEN38_RETRY_VI_INCOMPLETE", DEFAULT_RETRY_VI_INCOMPLETE),
    ("QWEN38_RENDER_HTML", DEFAULT_RENDER_HTML),
    ("QWEN38_TUNNEL_ENABLED", DEFAULT_TUNNEL_ENABLED),
    ("QWEN38_TUNNEL_HOLD_SECONDS", DEFAULT_TUNNEL_HOLD_SECONDS),
):
    os.environ.setdefault(key, default)

GIT_URL = "https://github.com/dangkhoa2016/Qwen3.8-27B-On-Kaggle-TPU-v5e8.git"
GIT_REF = os.getenv("QWEN38_GIT_REF", DEFAULT_GIT_REF).strip()
EXPECTED_COMMIT = os.getenv("QWEN38_EXPECTED_COMMIT", "").strip().lower()
assert re.fullmatch(r"[A-Za-z0-9._/-]{1,128}", GIT_REF) and ".." not in GIT_REF and not GIT_REF.startswith((".", "-")), "INVALID_GIT_REF"
assert not EXPECTED_COMMIT or re.fullmatch(r"[0-9a-f]{40}", EXPECTED_COMMIT), "INVALID_EXPECTED_COMMIT"
ref_is_sha = bool(re.fullmatch(r"[0-9a-f]{40}", GIT_REF))
if ref_is_sha:
    if EXPECTED_COMMIT and EXPECTED_COMMIT != GIT_REF:
        raise RuntimeError("GIT_REF_AND_EXPECTED_COMMIT_DISAGREE")
    EXPECTED_COMMIT = GIT_REF

work_root = Path(os.getenv("QWEN38_WORKING_ROOT", "/kaggle/working")).resolve()
work_root.mkdir(parents=True, exist_ok=True)
source_dir = work_root / "qwen38-git-source"

# Public source: intentionally no credentials or Kaggle Secrets.
# The notebook downloads this maintained public repository without embedded source code.
git_env = os.environ.copy()
for key in tuple(git_env):
    if (key.startswith("GIT_CONFIG_") or
        key in ("GIT_CONFIG_PARAMETERS", "GIT_ASKPASS", "SSH_ASKPASS",
                "GIT_HTTP_EXTRAHEADER", "GITHUB_TOKEN", "GH_TOKEN",
                "QWEN38_GITHUB_TOKEN")):
        git_env.pop(key, None)
git_env["GIT_TERMINAL_PROMPT"] = "0"
git_env["GIT_CONFIG_GLOBAL"] = "/dev/null"
git_env["GIT_CONFIG_NOSYSTEM"] = "1"
git_env["GIT_ASKPASS"] = "/bin/false"
git_env["GIT_CONFIG_COUNT"] = "1"
git_env["GIT_CONFIG_KEY_0"] = "credential.helper"
git_env["GIT_CONFIG_VALUE_0"] = ""

def git(*args):
    proc = subprocess.run(["git", *args], env=git_env, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if proc.returncode:
        raise RuntimeError(
            "PUBLIC_GIT_CLONE_FAILED: verify Kaggle Internet and "
            "that the project repository is accessible"
        )
    return proc.stdout.strip()

if source_dir.exists() or source_dir.is_symlink():
    # Rerun safely: reuse only the same verified, clean checkout. Never auto-delete it.
    if source_dir.is_symlink() or not source_dir.is_dir() or not (source_dir / ".git").is_dir():
        raise RuntimeError("GIT_SOURCE_INCOMPLETE: inspect the source folder or start a fresh Kaggle session")
    commit = git("-C", str(source_dir), "rev-parse", "HEAD").lower()
    if git("-C", str(source_dir), "remote", "get-url", "origin") != GIT_URL:
        raise RuntimeError("GIT_SOURCE_REMOTE_MISMATCH: refusing existing checkout")
    if git("-C", str(source_dir), "status", "--porcelain"):
        raise RuntimeError("GIT_SOURCE_NOT_CLEAN: refusing existing checkout")
    if EXPECTED_COMMIT and commit != EXPECTED_COMMIT:
        raise RuntimeError("GIT_COMMIT_MISMATCH: existing checkout differs from requested commit")
    if not ref_is_sha:
        # Compare with the public remote; a locally clean branch can still be stale.
        refs = git("ls-remote", GIT_URL, "refs/heads/" + GIT_REF,
                   "refs/tags/" + GIT_REF, "refs/tags/" + GIT_REF + "^{}")
        allowed = {line.split()[0].lower() for line in refs.splitlines() if line.strip()}
        if commit not in allowed:
            raise RuntimeError("GIT_SOURCE_STALE: remote ref changed; start a fresh Kaggle session")
    print("GIT_SOURCE_REUSED_VERIFIED", {"commit": commit, "ref": GIT_REF})
else:
    if ref_is_sha:
        git("clone", "--depth", "1", "--no-checkout", "--branch", "main", GIT_URL, str(source_dir))
        git("-C", str(source_dir), "fetch", "--depth", "1", "origin", GIT_REF)
        git("-C", str(source_dir), "checkout", "--detach", GIT_REF)
    else:
        git("clone", "--depth", "1", "--branch", GIT_REF, GIT_URL, str(source_dir))
    commit = git("-C", str(source_dir), "rev-parse", "HEAD").lower()
assert re.fullmatch(r"[0-9a-f]{40}", commit), "INVALID_GIT_COMMIT"
if EXPECTED_COMMIT and commit != EXPECTED_COMMIT:
    raise RuntimeError("GIT_COMMIT_MISMATCH: no TPU inference started")
if git("-C", str(source_dir), "status", "--porcelain"):
    raise RuntimeError("GIT_WORKTREE_NOT_CLEAN")

os.environ["QWEN38_SOURCE_DIR"] = str(source_dir)
os.environ["QWEN38_SOURCE_COMMIT"] = commit
os.environ["QWEN38_SOURCE_REF"] = GIT_REF
os.environ["QWEN38_SOURCE_URL"] = GIT_URL
os.environ["QWEN38_SOURCE_PINNED"] = "1" if EXPECTED_COMMIT else "0"
sys.path.insert(0, str(source_dir))
print("GIT_SOURCE_READY", {"commit": commit, "ref": GIT_REF, "pinned": bool(EXPECTED_COMMIT)})
from support.git_first_runtime import install_git_first_preflight
install_git_first_preflight()
from qwen38_runtime import Workflow
workflow = Workflow()
workflow.run("preflight")
'''


def notebook_content() -> dict:
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert len(nb["cells"]) == 35 and nb["cells"][2]["cell_type"] == "code"
    intro = """# Qwen3.8-27B BF16 on Kaggle TPU v5e-8
## Text + Vision · 19 Evaluation Cases · English/Vietnamese · HTML Gallery

### English
- **Evaluation:** Run 19 text and vision cases covering OCR, image understanding, photography and text reasoning.
- **Inputs:** Attach the original BF16 Kaggle model and the Runtime Assets Dataset with 48 checksum-verified files.
- **Environment:** Select TPU v5e-8 and enable Internet. Required resources are checked before model initialization.
- **Results:** Save raw responses and execution evidence, optionally generate Vietnamese answers, and build an HTML Gallery.

### Tiếng Việt
- **Thử nghiệm:** Chạy 19 trường hợp văn bản và hình ảnh gồm OCR, hiểu nội dung ảnh, ảnh chụp và suy luận văn bản.
- **Dữ liệu đầu vào:** Attach model BF16 gốc trên Kaggle và Runtime Assets Dataset gồm 48 file đã xác minh SHA256.
- **Môi trường:** Chọn TPU v5e-8 và bật Internet. Notebook kiểm tra các tài nguyên cần thiết trước khi khởi tạo model.
- **Kết quả:** Lưu câu trả lời gốc và bằng chứng thực thi, tùy chọn sinh câu trả lời tiếng Việt và tạo HTML Gallery.

![Input example / Ví dụ đầu vào](attachment:qwen38-robotics-preview.jpg)
"""
    nb["cells"][0]["source"] = intro.splitlines(keepends=True)
    step1 = """## 01 · Prepare the environment and verify input files / Chuẩn bị môi trường và kiểm tra dữ liệu

### English
- Prepare project components and check their integrity before loading the model.
- Verify the attached BF16 model and all 48 Runtime Assets payloads against their manifests.

### Tiếng Việt
- Chuẩn bị các thành phần cần thiết và kiểm tra tính toàn vẹn trước khi nạp model.
- Xác minh model BF16 và đủ 48 file Runtime Assets theo manifest tương ứng.
"""
    nb["cells"][1]["source"] = step1.splitlines(keepends=True)
    nb["cells"][2]["source"] = BOOTSTRAP.splitlines(keepends=True)
    return nb


def sync(write: bool) -> None:
    actual = notebook_content()
    if write:
        NOTEBOOK.write_text(json.dumps(actual, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    elif not NOTEBOOK.exists() or json.loads(NOTEBOOK.read_text(encoding="utf-8")) != actual:
        raise SystemExit("GIT_FIRST_NOTEBOOK_OUT_OF_SYNC")
    print("GIT_FIRST_NOTEBOOK_SYNC_PASS")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--write", action="store_true")
    g.add_argument("--check", action="store_true")
    args = p.parse_args()
    sync(args.write)
