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
    """Verify source notebook invariants independently of TPU availability."""
    import ast
    import hashlib
    import nbformat
    repo=Path(__file__).resolve().parents[1]
    n=nbformat.read(repo/'notebooks/kaggle-tpu-v5e8-production-showcase.ipynb',as_version=4)
    nbformat.validate(n)
    cells=[c.source for c in n.cells if c.cell_type=='code']
    assert len(cells)==17
    for c in cells:ast.parse(c)
    source='\n'.join((repo/'qwen38_runtime/stages'/ (name+'.py')).read_text()
                     for name in (
                         'preflight','dependencies','maxtext','inputs','checkpoint',
                         'tpu','engine','inference_helpers','text_schedule','ocr_probe',
                         'ocr_evidence','vision_reuse','image_cases','text_cases',
                         'vietnamese','gallery'))
    for required in ('ORGANIZED_SUPPORT_SHA256_GATE_PASS',
                     "bundle['required_file_count']==len(bundle['files'])==48",
                     'install_native_layout_timing','install_native_metadata_hooks',
                     'install_metadata_reuse','lowered.compile()',
                     'compiled(**dynamic_kwargs)','engine.encode_vision(',
                     'e_args.PREFILL_LEN=320','e_args.MAX_TARGET_LEN=768',
                     "case.get('max_decode_steps',448)"):
        assert required in source,required
    assert "file_sha256(BUNDLE_MANIFEST)" in source
    assert "MODEL_MOUNT_MISSING" in source
    assert "PORTABLE_MANIFEST_SHA256_MISMATCH" in source
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
    assert hashlib.sha256(data).hexdigest()=='fb2ee48dfe423733499112668d61e965b5d8d7466d52361b403c68dbc5768817'
    meta=json.loads(data)
    assert meta['required_file_count']==len(meta['files'])==48
    assert meta['archive_count']==len(meta['archives'])==7
    assert meta['image_count']==16
    assert sorted(meta['archives'])==[
        'cache.zip','cases.zip','helpers.zip','images.zip',
        'runtime.zip','source.zip','wheels.zip']
    assert sum(entry['file_count'] for entry in meta['archives'].values())==48
    assert all(Path(rel).parts[0]+'.zip' in meta['archives'] for rel in meta['files'])
    assert not any('.safetensors' in x for x in meta['files'])


def test_portable_support_mount_discovery_accepts_kaggle_extracted_subfolder(tmp_path):
    """Mount discovery supports a Kaggle mount root or extracted subfolder."""
    import ast
    import nbformat

    source = Path(__file__).resolve().parents[1] / "qwen38_runtime/stages/preflight.py"
    module = ast.parse(source.read_text())
    wrapper = next(n for n in module.body
                   if isinstance(n, ast.FunctionDef) and n.name == "execute")
    fn = next(n for n in wrapper.body
              if isinstance(n, ast.FunctionDef) and n.name == "find_portable_support_folder")
    namespace = {}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "<support-location-test>", "exec"), namespace)
    locate = namespace["find_portable_support_folder"]
    name = "support-content"
    mount = tmp_path / "qwen38-tpu-v5e8-full19-offline-support"
    inner = mount / name
    inner.mkdir(parents=True)
    (inner / "portable-bundle-manifest.json").write_text("{}")
    assert locate([None, mount], name) == inner
    assert locate([inner], name) == inner
    assert locate([tmp_path / "unattached"], name) is None
    (inner / "portable-bundle-manifest.json").unlink()
    assert locate([mount], name) is None
