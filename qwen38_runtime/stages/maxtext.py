"""Maxtext stage; runs only when explicitly invoked.

No TPU work occurs at import time.
"""
from __future__ import annotations

def execute(context):
    # Inputs from earlier stages
    RUN_ROOT = context.require("RUN_ROOT")
    SUPPORT_DIR = context.require("SUPPORT_DIR")
    # The original audited stage implementation
    # 3. Offline MaxText bootstrap with exact source/patch provenance.
    import hashlib, json, shutil, tarfile
    from pathlib import Path

    MAXTEXT_COMMIT = 'f0b2abe3f6f3736c5579673f42b5cbc17b327fd3'
    MAXTEXT_DIR=RUN_ROOT/'patched-maxtext'
    PATCH_SHA256 = '07f27724c7e6d7267258eab99bfbe3172329ca359959a36f73fee29531acb8e6'
    OFFLINE_ARCHIVE_SHA256 = '5c67ce755065849c13003cf7b157a6aad90601c2bbdbb44329498d9fb61016d2'

    def file_sha256(path):
        h=hashlib.sha256()
        with open(path,'rb') as f:
            for chunk in iter(lambda:f.read(1024*1024), b''): h.update(chunk)
        return h.hexdigest()

    assert SUPPORT_DIR.is_dir(), f'Offline support dataset not attached: {SUPPORT_DIR}'
    PATCH_PATH = SUPPORT_DIR / 'source/maxtext-qwen38-production-text-vision.patch'
    assert PATCH_PATH.is_file(), f'Missing MaxText patch provenance file: {PATCH_PATH}'
    assert file_sha256(PATCH_PATH) == PATCH_SHA256, 'MaxText patch SHA mismatch'

    archive_name = 'maxtext-f0b2abe-qwen38-text-vision-patched.tar.gz'
    archive_candidates = [
        SUPPORT_DIR / 'source' / archive_name,
        SUPPORT_DIR / 'source' / (archive_name + '.blob'),
    ]
    archive = next((p for p in archive_candidates if p.is_file()), archive_candidates[0])
    assert archive.is_file(), f'Missing patched MaxText archive: {archive_candidates}'
    assert file_sha256(archive) == OFFLINE_ARCHIVE_SHA256, 'Offline MaxText archive SHA mismatch'

    assert not MAXTEXT_DIR.exists(), ('NON_DESTRUCTIVE_BOOTSTRAP_TARGET_EXISTS',str(MAXTEXT_DIR))
    MAXTEXT_DIR.mkdir(parents=True,exist_ok=False)
    with tarfile.open(archive,'r:gz') as tf:
        tf.extractall(MAXTEXT_DIR,filter='data')
    meta=json.loads((MAXTEXT_DIR/'QWEN38_OFFLINE_BUILD.json').read_text())
    assert meta['maxtext_commit'] == MAXTEXT_COMMIT
    # The embedded build marker records the original pre-cast patch. The archive and
    # separate integrated patch are pinned above; verify the actual loader bytes too.
    assert meta['patch_sha256']=='2ec8241c93b3a3e1d54e97e85eace644d27f90ba2147c1bc8059bf67998c38cf'
    loader_path=MAXTEXT_DIR/'src/maxtext/checkpoint_conversion/utils/tensor_handling.py'
    assert file_sha256(loader_path)=='40edea16fde56e199895119215442bc2bb560ae7e04f2e9d401bf2aa7e80734e'
    assert 'array.astype(leaf.dtype)' in loader_path.read_text()

    # The SHA256-verified source archive provides the pinned MaxText provenance.
    assert 'Q/K RMSNorm' in meta.get('qk_norm_fix','')
    source_mode='offline-support-dataset'

    marker = MAXTEXT_DIR / 'src/maxtext/configs/models/qwen3.8-27b.yml'
    assert marker.exists(), 'Qwen3.8 MaxText patch marker missing'
    print('MaxText commit:', MAXTEXT_COMMIT)
    print('Patch SHA256:', PATCH_SHA256)
    print('Source mode:', source_mode)
    print('Qwen3.8 patch: READY')

    # Reproduce the original SHA-verified Split-Vision patch; keep source integrity gates.
    import subprocess
    SPLIT_PATCH=SUPPORT_DIR/'helpers/split_vision_maxtext.patch'
    assert SPLIT_PATCH.is_file(), 'Split Vision patch missing; extract local integration kit first'
    assert file_sha256(SPLIT_PATCH)=='7ba8d1bdedbfab2c8338c101ca94f51580caf4d4ee80982225f58fe5daec5a65', 'Split patch SHA mismatch'
    for rel,original_sha,patched_sha in [('maxtext/models/models.py', 'aa2a59c5664eac1fe645b7d7f98fec4d3ee6185a12b82b62c1175ae0cadb2d8d', 'b97b56c3a0b1cdbc85aab832591c0eb88ae1f7471c29d6740f8ebdbf157b5fbb'), ('maxtext/inference/maxengine/maxengine.py', 'd7207bc6305033bc46dee8df8826a82b0934f9aadb7f445540776209d041aecc', 'c81c69712852fab4676ecde0451cec06469925cfaa59618150d92bee4b9ce8df')]:
        p=MAXTEXT_DIR/'src'/rel
        assert file_sha256(p)==original_sha, ('SPLIT_PATCH_SOURCE_BASE_MISMATCH',rel)
    subprocess.run(['git','apply','--check',str(SPLIT_PATCH)],cwd=MAXTEXT_DIR/'src',check=True)
    subprocess.run(['git','apply',str(SPLIT_PATCH)],cwd=MAXTEXT_DIR/'src',check=True)
    for rel,original_sha,patched_sha in [('maxtext/models/models.py', 'aa2a59c5664eac1fe645b7d7f98fec4d3ee6185a12b82b62c1175ae0cadb2d8d', 'b97b56c3a0b1cdbc85aab832591c0eb88ae1f7471c29d6740f8ebdbf157b5fbb'), ('maxtext/inference/maxengine/maxengine.py', 'd7207bc6305033bc46dee8df8826a82b0934f9aadb7f445540776209d041aecc', 'c81c69712852fab4676ecde0451cec06469925cfaa59618150d92bee4b9ce8df')]:
        assert file_sha256(MAXTEXT_DIR/'src'/rel)==patched_sha, ('SPLIT_PATCH_INTEGRITY_FAIL',rel)
    print('SPLIT_VISION_SOURCE_GATE=PASS',len([{'file': 'maxtext/models/models.py', 'original_sha256': 'aa2a59c5664eac1fe645b7d7f98fec4d3ee6185a12b82b62c1175ae0cadb2d8d', 'patched_sha256': 'b97b56c3a0b1cdbc85aab832591c0eb88ae1f7471c29d6740f8ebdbf157b5fbb', 'added_lines': 15, 'removed_lines': 1}, {'file': 'maxtext/inference/maxengine/maxengine.py', 'original_sha256': 'd7207bc6305033bc46dee8df8826a82b0934f9aadb7f445540776209d041aecc', 'patched_sha256': 'c81c69712852fab4676ecde0451cec06469925cfaa59618150d92bee4b9ce8df', 'added_lines': 32, 'removed_lines': 0}]))
    # Explicitly return the stage's named results to the workflow.
    return {'MAXTEXT_DIR': MAXTEXT_DIR, 'PATCH_SHA256': PATCH_SHA256, 'file_sha256': file_sha256}
