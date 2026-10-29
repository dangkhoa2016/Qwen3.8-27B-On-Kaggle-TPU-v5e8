"""Dependencies stage; runs only when explicitly invoked.

No TPU work occurs at import time.
"""
from __future__ import annotations

def execute(context):
    # Inputs from earlier stages
    BUNDLE_MANIFEST = context.require("BUNDLE_MANIFEST")
    PORTABLE_MANIFEST_SHA256 = context.require("PORTABLE_MANIFEST_SHA256")
    SUPPORT_DIR = context.require("SUPPORT_DIR")
    bundle = context.require("bundle")
    os = context.require("os")
    # The original audited stage implementation
    # 2. Offline-first dependency gate — never replace Kaggle's JAX/TPU stack.
    import hashlib
    import importlib.metadata as md
    import subprocess, sys, json
    from pathlib import Path

    # The portable support manifest and all 48 payload files were verified in Cell 1.
    assert SUPPORT_DIR.is_dir() and BUNDLE_MANIFEST.is_file(), ('SUPPORT_DATASET_MISSING',str(SUPPORT_DIR))
    assert len(bundle['files']) == 48
    print('SUPPORT_MANIFEST_VALIDATED',PORTABLE_MANIFEST_SHA256)

    WHEEL_INFO = {
        'pathwaysutils': ('0.1.12', 'pathwaysutils-0.1.12-py3-none-any.whl', '3aab3cd936ea9076c469b731cbdca43a453736594c65bdcee4b274c10ce51c34'),
        'aqtp': ('0.9.0', 'aqtp-0.9.0-py3-none-any.whl', '5efdccd24657e149d3ac17599bb3d4eb88297fdc6aab14eaba0c70625057a257'),
        'drjax': ('0.2.1', 'drjax-0.2.1-py3-none-any.whl', '9ec93541a7ba0af864d599cd3e14770a19e291a6291b8b79ddf6b92e8b173494'),
        'tokamax': ('0.0.14', 'tokamax-0.0.14-py3-none-any.whl', 'a10a91bbb6ddf406bb7006cb1a27af59a6faaa6cb06fe788249dd10d9eaac409'),
        'xprof': ('2.23.1', 'xprof-2.23.1-py3-none-manylinux_2_27_x86_64.whl', '96701f868b7a736a789d0a4b5efeeb269feaed63348117707e853a72e006c722'),
        'einshape': ('1.0', 'einshape-1.0-py3-none-any.whl', '42da4c2dea3a27f87ee45a7cee5072a636b97cb184bb07bf5d6412ba0ff7b965'),
        'typeguard': ('2.13.3', 'typeguard-2.13.3-py3-none-any.whl', '5e3e3be01e887e7eafae5af63d1f36c849aaa94e3a0112097312aabfa16284f1'),
        'tiktoken': ('0.14.0', 'tiktoken-0.14.0-cp312-cp312-manylinux_2_28_x86_64.whl', '7896eea257fe497a2b7134474d909156c6744ce8da35bce88011a960e008aa0d'),
    }

    # Additional version-pinned wheels used by the MaxText runtime.
    extra_manifest=json.loads((SUPPORT_DIR/'runtime/manifest.json').read_text())['wheels']
    assert {x['dist'] for x in extra_manifest}=={'jaxtyping','wadler-lindig','qwix'}
    for x in extra_manifest:
        WHEEL_INFO[x['dist']]=(x['version'],x['wheel'],x['sha256'])

    CORE_EXPECTED = {'jax':'0.11.0','jaxlib':'0.11.0','flax':'0.12.8','transformers':'5.12.1'}
    core_versions = {k:md.version(k) for k in CORE_EXPECTED}
    assert core_versions==CORE_EXPECTED,('BASELINE_RUNTIME_MISMATCH',core_versions,CORE_EXPECTED)
    print('CORE_BASELINE_PASS',core_versions)

    def sha256(path):
        h=hashlib.sha256()
        with open(path,'rb') as f:
            for chunk in iter(lambda:f.read(1024*1024), b''): h.update(chunk)
        return h.hexdigest()

    for dist, (ver, wheel_name, expected_sha) in WHEEL_INFO.items():
        try: current=md.version(dist)
        except md.PackageNotFoundError: current=None
        if current != ver:
            wheel = SUPPORT_DIR / 'wheels' / wheel_name
            assert wheel.is_file(), ('PINNED_WHEEL_NOT_ATTACHED',dist,wheel_name)
            assert sha256(wheel) == expected_sha, ('WHEEL_SHA_MISMATCH',wheel_name)
            subprocess.check_call([sys.executable,'-m','pip','install','-q','--no-index','--no-deps',str(wheel)])
        assert md.version(dist) == ver, (dist, md.version(dist), ver)

    # MaxText's pathwaysutils profiling module imports FastAPI transitively.
    # Pin the validated compatibility set without letting pip replace JAX/JAXLIB.
    WEB_IMPORT_DEPS = {'fastapi':'0.115.12', 'starlette':'0.46.2', 'uvicorn':'0.34.2'}
    allow_online=os.environ.get('QWEN38_ALLOW_PINNED_WEB_WHEELS','0')=='1'
    for dist,ver in WEB_IMPORT_DEPS.items():
        try: current=md.version(dist)
        except md.PackageNotFoundError: current=None
        if current != ver:
            candidates=sorted((SUPPORT_DIR/'wheels').glob(f'{dist.replace("-", "_")}-{ver}-*.whl'))
            if candidates:
                wheel=candidates[0]
                # Non-mutating integrity check above already pinned this file via full-dataset SHA256.
                subprocess.check_call([sys.executable,'-m','pip','install','-q','--no-index','--no-deps',str(wheel)])
            else:
                assert allow_online, ('WEB_IMPORT_WHEELS_MISSING',dist,ver,
                      'Attach the documented Full19 support Dataset, or enable QWEN38_ALLOW_PINNED_WEB_WHEELS=1 with Kaggle Internet.')
                subprocess.check_call([sys.executable,'-m','pip','install','-q','--no-deps',f'{dist}=={ver}'])
        assert md.version(dist)==ver
    print('MAXTEXT_TRANSITIVE_IMPORT_GATE=PASS', WEB_IMPORT_DEPS)

    CORE_EXPECTED = {'jax': '0.11.0', 'jaxlib': '0.11.0', 'transformers': '5.12.1'}
    versions = {k: md.version(k) for k in CORE_EXPECTED}
    print('Core versions:', versions)
    mismatches={k:(versions[k],v) for k,v in CORE_EXPECTED.items() if versions[k] != v}
    assert not mismatches, f'Fail-closed: core runtime differs from acceptance environment: {mismatches}. Do not pip-replace Kaggle JAX/JAXLIB in a live TPU runtime.'
    print('DEPENDENCY_GATE=PASS', 'offline_support='+str(SUPPORT_DIR.is_dir()))
    # Explicitly return the stage's named results to the workflow.
    return {}
