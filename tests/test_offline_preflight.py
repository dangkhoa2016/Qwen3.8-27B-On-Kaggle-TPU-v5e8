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

def test_notebook_portable_bundle_and_split_vision_invariants():
    """Do not mask the old broken Git commit pin by skipping a test."""
    import ast
    import hashlib
    import nbformat
    repo=Path(__file__).resolve().parents[1]
    n=nbformat.read(repo/'notebooks/kaggle-tpu-v5e8-production-showcase.ipynb',as_version=4)
    nbformat.validate(n)
    cells=[c.source for c in n.cells if c.cell_type=='code']
    assert len(cells)==14
    for c in cells:ast.parse(c)
    source='\n'.join(cells)
    for required in ('PORTABLE_SUPPORT_BUNDLE_SHA256_GATE_PASS',
                     "bundle['required_file_count']==48",
                     'install_native_layout_timing','install_native_metadata_hooks',
                     'install_metadata_reuse','lowered.compile()',
                     'compiled(**dynamic_kwargs)','engine.encode_vision(',
                     'e_args.PREFILL_LEN=320','e_args.MAX_TARGET_LEN=768',
                     "case.get('max_decode_steps',448)"):
        assert required in source,required
    for forbidden in ('EXAMPLES_COMMIT','git clone','70f89f34633169757d62a1daeddfb29e4adb87c6',
                      'qwen38-cpu-recovery-20261026','qwen38-production-rc-20261025'):
        assert forbidden not in source,forbidden
    items=json.loads((repo/'examples/assets_manifest.json').read_text())['assets']
    assert len(items)==16
    for a in items:
        f=repo/a['path']
        assert f.is_file() and f.stat().st_size==a['bytes']
        assert hashlib.sha256(f.read_bytes()).hexdigest()==a['sha256']
    names={Path(a['path']).name for a in items}
    rich=json.loads((repo/'examples/vision_showcase_cases.json').read_text())['cases']
    photo=json.loads((repo/'examples/photo_showcase_cases.json').read_text())['cases']
    assert len(rich)==6 and len(photo)==8
    assert all(x['image'] in names for x in rich+photo)


def test_portable_bundle_manifest_integrity_detects_tampering(tmp_path):
    """The gate must reject a mutated input, rather than silently passing."""
    import hashlib
    p=tmp_path/'image.jpg'
    p.write_bytes(b'good-image')
    checksum=hashlib.sha256(p.read_bytes()).hexdigest()
    manifest={'image.jpg':{'sha256':checksum,'bytes':p.stat().st_size}}
    for name,item in manifest.items():
        current=tmp_path/name
        assert current.stat().st_size==item['bytes']
        assert hashlib.sha256(current.read_bytes()).hexdigest()==item['sha256']
    p.write_bytes(b'bad-image')
    for name,item in manifest.items():
        current=tmp_path/name
        assert hashlib.sha256(current.read_bytes()).hexdigest()!=item['sha256']


def test_repository_contains_exact_pinned_support_provenance_manifest():
    import hashlib
    repo=Path(__file__).resolve().parents[1]
    p=repo/'support/portable-bundle-manifest.json'
    data=p.read_bytes()
    assert hashlib.sha256(data).hexdigest()=='1447ee114dc2d2abce6f60f5125927f55311bf530b8cf53ab86a2576ac637ebc'
    meta=json.loads(data)
    assert meta['required_file_count']==len(meta['files'])==48
    assert meta['source_support_original_files']==38
    assert meta['source_input_images']==16
    assert sum(x['category']=='original_verified_support' for x in meta['files'].values())==38
    assert sum(x['category']=='verified_jax_helper' for x in meta['files'].values())==6
    assert sum(x['category']=='optional_cache_candidate' for x in meta['files'].values())==3
    assert not any('.safetensors' in x for x in meta['files'])


def test_portable_support_mount_discovery_accepts_kaggle_extracted_subfolder(tmp_path):
    """Kaggle expands the uploaded ZIP into a named directory, not a flat mount root."""
    import ast
    import nbformat

    source = Path(__file__).resolve().parents[1] / "notebooks/kaggle-tpu-v5e8-production-showcase.ipynb"
    notebook = nbformat.read(source, as_version=4)
    code = notebook.cells[2].source
    module = ast.parse(code)
    fn = next(
        n for n in module.body
        if isinstance(n, ast.FunctionDef) and n.name == "find_portable_support_folder"
    )
    namespace = {}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "<support-location-test>", "exec"), namespace)
    locate = namespace["find_portable_support_folder"]
    name = "Qwen38-Full19-Portable-Offline-Support-20261027"
    mount = tmp_path / "qwen38-tpu-v5e8-full19-offline-support"
    inner = mount / name
    inner.mkdir(parents=True)
    (inner / "portable-bundle-manifest.json").write_text("{}")
    assert locate([None, mount], name) == inner
    assert locate([inner], name) == inner
    assert locate([tmp_path / "unattached"], name) is None
    (inner / "portable-bundle-manifest.json").unlink()
    assert locate([mount], name) is None
