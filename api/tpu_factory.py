"""TPU-only MaxEngine bootstrap, derived from executed acceptance notebook.

This module is import-safe on CPU; call build_tpu_runner() only after
the exact supported Kaggle model/support datasets are attached.
Not yet validated by a live TPU API acceptance.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys

MODEL_DIR = Path("/kaggle/input/models/dangkhoa2016/qwen-qwen3-8-27b/transformers/default/1")
SUPPORT_DIR = Path("/kaggle/input/datasets/dangkhoa2016/qwen3-8-27b-tpu-v5e8-offline-support")
MAXTEXT_DIR = Path("/kaggle/working/maxtext-qwen38-src")
MAXTEXT_COMMIT = "f0b2abe3f6f3736c5579673f42b5cbc17b327fd3"
PATCH_SHA256 = "2ec8241c93b3a3e1d54e97e85eace644d27f90ba2147c1bc8059bf67998c38cf"
ARCHIVE_SHA256 = "45aef84519fbd21698a6528bf6a58b4a7f66d8605c60371cf2a0e45fc3b613ee"
PREFILL_LEN = 256
TARGET_LEN = 384

def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def _archive_path(support_dir: Path) -> Path:
    names = ["src/maxtext-f0b2abe-qwen38-text-vision-patched.tar.gz",
             "maxtext-f0b2abe-qwen38-text-vision-patched.tar.gz",
             "maxtext-f0b2abe-qwen38-text-vision-patched.tar.gz.blob"]
    matches = [support_dir / name for name in names if (support_dir / name).is_file()]
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one matching MaxText archive: {len(matches)}")
    archive = matches[0]
    if digest(archive) != ARCHIVE_SHA256:
        raise RuntimeError("MaxText archive SHA256 mismatch")
    return archive


def ensure_pinned_source(*, support_dir: Path = SUPPORT_DIR,
                         source_dir: Path = MAXTEXT_DIR) -> Path:
    """Restore a trusted archive without network access or destructive replacement."""
    import tarfile
    archive = _archive_path(support_dir)
    if source_dir.exists():
        # Existing directory must already have valid provenance; never overwrite it.
        _check_source_metadata(source_dir)
        return source_dir
    source_dir.parent.mkdir(parents=True, exist_ok=True)
    import tempfile
    with tempfile.TemporaryDirectory(prefix="qwen38-source-", dir=source_dir.parent) as tmp:
        stage = Path(tmp) / "staging"
        stage.mkdir()
        with tarfile.open(archive, "r:gz") as tf:
            for member in tf.getmembers():
                rel = Path(member.name)
                if member.issym() or member.islnk() or member.isdev() or rel.is_absolute() or ".." in rel.parts:
                    raise RuntimeError("Unsafe MaxText archive member")
            tf.extractall(stage, filter="data")
        _check_source_metadata(stage)
        stage.rename(source_dir)
    return source_dir


def _check_source_metadata(source_dir: Path) -> None:
    meta = json.loads((source_dir / "QWEN38_OFFLINE_BUILD.json").read_text())
    if (meta.get("maxtext_commit") != MAXTEXT_COMMIT or
        meta.get("patch_sha256") != PATCH_SHA256 or
        meta.get("scope") != "Qwen3.8-27B text+vision combined production candidate"):
        raise RuntimeError("Patched MaxText provenance mismatch")
    if not (source_dir / "src/maxtext/configs/models/qwen3.8-27b.yml").is_file():
        raise RuntimeError("Missing patched Qwen3.8 MaxText model config")


def preflight(*, model_dir: Path = MODEL_DIR, support_dir: Path = SUPPORT_DIR,
              source_dir: Path = MAXTEXT_DIR) -> dict:
    """CPU-only integrity gate, including real local model/checkpoint assets."""
    if not model_dir.is_dir():
        raise RuntimeError("Qwen3.8 model dataset not mounted")
    config = json.loads((model_dir / "config.json").read_text())
    if config.get("model_type") != "qwen3_5":
        raise RuntimeError("Unexpected checkpoint model_type")
    index = json.loads((model_dir / "model.safetensors.index.json").read_text())
    shards = sorted(model_dir.glob("model-*.safetensors"))
    if len(shards) != 18 or index.get("metadata", {}).get("total_size") != 55_562_855_904:
        raise RuntimeError("Incorrect model shard count or index size")
    declared = set(index.get("weight_map", {}).values())
    if {p.name for p in shards} != declared or any(p.stat().st_size == 0 for p in shards):
        raise RuntimeError("Model shard index or file integrity mismatch")
    patch = support_dir / "maxtext-qwen38-production-text-vision.patch"
    if not patch.is_file() or digest(patch) != PATCH_SHA256:
        raise RuntimeError("MaxText patch SHA256 mismatch")
    _archive_path(support_dir)
    _check_source_metadata(source_dir)
    if not (support_dir / "vision_test_card_731_256x192.png").is_file():
        raise RuntimeError("Vision seed card missing")
    return {"model_dir": str(model_dir), "support_dir": str(support_dir),
            "source_dir": str(source_dir), "maxtext_commit": MAXTEXT_COMMIT}

def _set_tpu_env() -> None:
    os.environ.update({
        "PJRT_DEVICE": "TPU",
        "TPU_SKIP_MDS_QUERY": "1",
        "TPU_HOST_BOUNDS": "1,1,1",
        "TPU_CHIPS_PER_HOST_BOUNDS": "2,4,1",
        "TPU_WORKER_HOSTNAMES": os.uname().nodename,
        "TPU_WORKER_ID": "0",
        "TPU_ACCELERATOR_TYPE": "v5e-8",
        "JAX_PLATFORMS": "tpu,cpu",
    })

def build_tpu_runner():
    """One BF16 load. Fail closed unless explicitly enabled and on TPU."""
    if os.environ.get("QWEN38_EXPERIMENTAL_TPU_ENABLE") != "YES_I_ACCEPT_UNQUALIFIED_API":
        raise RuntimeError("TPU adapter disabled: explicit experimental opt-in required")
    ensure_pinned_source()
    paths = preflight()
    # Never import JAX until the TPU process environment has been configured.
    _set_tpu_env()
    from importlib.metadata import version
    expected = {"jax": "0.11.0", "jaxlib": "0.11.0", "transformers": "5.12.1"}
    for name, want in expected.items():
        if version(name) != want:
            raise RuntimeError(f"Runtime dependency mismatch: {name}")
    os.environ["MAXTEXT_REPO_ROOT"] = paths["source_dir"]
    sys.path.insert(0, str(Path(paths["source_dir"]) / "src"))
    import jax
    if jax.default_backend() != "tpu" or jax.device_count() != 8:
        raise RuntimeError("Expected full TPU v5e-8; refusing fallback")
    from transformers import AutoTokenizer
    from maxtext.configs import pyconfig
    from maxtext.inference.maxengine.maxengine import MaxEngine
    from api.maxengine_adapter import MaxEngineRunner

    args = [
        "", f'{paths["source_dir"]}/src/maxtext/configs/base.yml',
        "model_name=qwen3.8-27b",
        f'tokenizer_path={paths["model_dir"]}',
        f'load_parameters_path={paths["model_dir"]}',
        "source_checkpoint_layout=safetensors_dynamic",
        "override_model_config=true", "scan_layers=false", "use_multimodal=true",
        f'image_path={paths["support_dir"]}/vision_test_card_731_256x192.png',
        "prompt=Code?", f"max_prefill_predict_length={PREFILL_LEN}",
        f"max_target_length={TARGET_LEN}",
        "per_device_batch_size=0.125", "attention=dot_product",
        "dtype=bfloat16", "weight_dtype=bfloat16",
        "ici_data_parallelism=1", "ici_fsdp_parallelism=-1",
        "ici_tensor_parallelism=1", "ici_autoregressive_parallelism=1",
        "async_checkpointing=false", "enable_checkpointing=false",
        "base_output_directory=/tmp/maxtext_output",
        "run_name=qwen38-kaggle-text-vision-api-experimental",
    ]
    config = pyconfig.initialize(args)
    engine = MaxEngine(config)
    params = engine.load_params()
    if not getattr(config, "use_multimodal", False):
        raise RuntimeError("Multimodal configuration was disabled")
    tokenizer = AutoTokenizer.from_pretrained(
        paths["model_dir"], local_files_only=True, trust_remote_code=False)
    return MaxEngineRunner(engine, params, config, tokenizer,
                           prefill_length=PREFILL_LEN, max_target_length=TARGET_LEN)
