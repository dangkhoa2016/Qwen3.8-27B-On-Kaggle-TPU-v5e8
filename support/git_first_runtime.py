"""Git-first preflight extension; never changes the historical offline stage."""
from __future__ import annotations

import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

GIT_URL = "https://github.com/dangkhoa2016/Qwen3.8-27B-On-Kaggle-TPU-v5e8.git"
NEW_ASSETS = (
    Path("/kaggle/input/datasets/dangkhoa2016/qwen38-27b-tpu-v5e8-full19-runtime-assets"),
    Path("/kaggle/input/qwen38-27b-tpu-v5e8-full19-runtime-assets"),
)


def validate_source() -> dict:
    sha = os.environ.get("QWEN38_SOURCE_COMMIT", "").strip().lower()
    ref = os.environ.get("QWEN38_SOURCE_REF", "").strip()
    src = Path(os.environ["QWEN38_SOURCE_DIR"]).resolve()
    if not re.fullmatch(r"[a-f0-9]{40}", sha) or not src.is_dir():
        raise RuntimeError("INVALID_GIT_PROVENANCE")
    if os.environ.get("QWEN38_SOURCE_URL") != GIT_URL:
        raise RuntimeError("WRONG_GIT_SOURCE_URL")

    def git(*args: str) -> str:
        proc = subprocess.run(
            ["git", "-C", str(src), *args], capture_output=True,
            text=True, check=True,
        )
        return proc.stdout.strip()

    if git("rev-parse", "HEAD").lower() != sha:
        raise RuntimeError("SOURCE_SHA_CHANGED")
    if git("remote", "get-url", "origin") != GIT_URL:
        raise RuntimeError("SOURCE_REMOTE_CHANGED")
    if git("status", "--porcelain"):
        raise RuntimeError("SOURCE_WORKTREE_NOT_CLEAN")
    return {
        "format": "QWEN38_GIT_SOURCE_PROVENANCE_V1",
        "repository": GIT_URL,
        "requested_ref": ref,
        "commit_sha": sha,
        "commit_pinned": os.environ.get("QWEN38_SOURCE_PINNED") == "1",
    }


def install_git_first_preflight() -> None:
    """Install once in the new notebook; the bundled legacy source is untouched."""
    from qwen38_runtime.stages import preflight

    if getattr(preflight.execute, "_git_first", False):
        return
    original = preflight.execute

    def execute(context):
        # Always verify Git provenance before the stage touches model or TPU.
        record = validate_source()
        if "QWEN38_SUPPORT_DIR" not in os.environ:
            mount = next((p for p in NEW_ASSETS
                          if (p / "portable-bundle-manifest.json").is_file()), None)
            if mount is not None:
                os.environ["QWEN38_SUPPORT_DIR"] = str(mount)
        result = original(context)
        # The original stage verifies the 48 SHA256-pinned assets, unchanged.
        record["support_manifest_sha256"] = result["PORTABLE_MANIFEST_SHA256"]
        record["run_started_utc"] = result["RUN_STARTED_UTC"]
        record["captured_utc"] = datetime.now(timezone.utc).isoformat()
        out = Path(result["RUN_ROOT"]) / "source-provenance.json"
        if out.exists():
            raise FileExistsError("Refusing to overwrite source provenance")
        out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print("GIT_SOURCE_PROVENANCE_PASS", record["commit_sha"])
        return result

    execute._git_first = True
    preflight.execute = execute
