"""Preflight stage; runs only when explicitly invoked.

No TPU work occurs at import time.
"""
from __future__ import annotations

def execute(context):
    # Inputs from earlier stages
    pass  # This stage has no earlier-stage dependencies
    # The original audited stage implementation
    # 1. Fail-closed, offline and portable preflight: run before importing JAX.
    import os, json, hashlib, time
    # Optional notebook configuration: edit these values before Run All.
    # 1 enables a feature; 0 disables it. Existing environment overrides take priority.
    DEFAULT_ENABLE_VI = '1'               # 1 = 17 extra VI TPU inferences
    DEFAULT_RETRY_VI_INCOMPLETE = '0'    # 1 = retry failed VI rows when rerunning VI cell
    DEFAULT_RENDER_HTML = '1'           # 1 = create CPU-only static Gallery
    os.environ.setdefault('QWEN38_ENABLE_VI', DEFAULT_ENABLE_VI)
    os.environ.setdefault('QWEN38_RETRY_VI_INCOMPLETE', DEFAULT_RETRY_VI_INCOMPLETE)
    os.environ.setdefault('QWEN38_RENDER_HTML', DEFAULT_RENDER_HTML)
    print('NOTEBOOK_OPTIONS', {k: os.environ[k] for k in ('QWEN38_ENABLE_VI','QWEN38_RETRY_VI_INCOMPLETE','QWEN38_RENDER_HTML')})
    from datetime import datetime, timezone
    from pathlib import Path
    RUN_WALL_START=time.monotonic()
    RUN_STARTED_UTC=datetime.now(timezone.utc).isoformat()
    MODEL_DIR=Path(os.environ.get('QWEN38_MODEL_DIR',
        '/kaggle/input/models/dangkhoa2016/qwen-qwen3-8-27b/transformers/default/1'))
    assert MODEL_DIR.is_dir() and (MODEL_DIR/'model.safetensors.index.json').is_file(),('MODEL_MOUNT_MISSING',str(MODEL_DIR))

    # Resolve the seven organized ZIP archives attached as a Kaggle Dataset.
    import shutil, zipfile
    SUPPORT_CANDIDATES=[
        Path(os.environ['QWEN38_SUPPORT_DIR']) if os.environ.get('QWEN38_SUPPORT_DIR') else None,
        Path('/kaggle/input/datasets/dangkhoa2016/qwen38-27b-tpu-v5e8-full19-runtime-support'),
        Path('/kaggle/input/qwen38-27b-tpu-v5e8-full19-runtime-support'),
    ]
    def find_portable_support_folder(candidates, subdirectory='.'):
        for base in candidates:
            if base is None:
                continue
            for candidate in (base, base/subdirectory):
                if (candidate/'portable-bundle-manifest.json').is_file():
                    return candidate
        return None
    SUPPORT_MOUNT=find_portable_support_folder(SUPPORT_CANDIDATES)
    assert SUPPORT_MOUNT is not None, ('OFFLINE_SUPPORT_NOT_ATTACHED',
        [str(x) for x in SUPPORT_CANDIDATES if x is not None])
    BUNDLE_MANIFEST=SUPPORT_MOUNT/'portable-bundle-manifest.json'
    PORTABLE_MANIFEST_SHA256='fb2ee48dfe423733499112668d61e965b5d8d7466d52361b403c68dbc5768817'
    def file_sha256(path):
        h=hashlib.sha256()
        with Path(path).open('rb') as stream:
            for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
        return h.hexdigest()
    assert file_sha256(BUNDLE_MANIFEST)==PORTABLE_MANIFEST_SHA256, 'PORTABLE_MANIFEST_SHA256_MISMATCH'
    bundle=json.loads(BUNDLE_MANIFEST.read_text())
    assert bundle['format']=='QWEN38_FULL19_ORGANIZED_SUPPORT_V1'
    assert bundle['required_file_count']==len(bundle['files'])==48
    assert bundle['archive_count']==len(bundle['archives'])==7
    assert bundle['image_count']==16
    RUN_ID=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    RUN_ROOT=Path('/kaggle/working')/('qwen38-inference-'+RUN_ID)
    assert not RUN_ROOT.exists(), ('NO_OVERWRITE',str(RUN_ROOT))
    RUN_ROOT.mkdir(exist_ok=False)
    # Kaggle expands category ZIPs as folders. Accept that native mount layout.
    # For a manual ZIP distribution, verify archives before extraction instead.
    expected=set(bundle['files'])
    mount_ready=all((SUPPORT_MOUNT/path).is_file() for path in expected)
    total_verified=0
    if mount_ready:
        SUPPORT_DIR=SUPPORT_MOUNT
        for relative,item in sorted(bundle['files'].items()):
            rp=Path(relative)
            assert not rp.is_absolute() and '..' not in rp.parts
            file=SUPPORT_DIR/rp
            assert file.stat().st_size==item['bytes'], ('MOUNT_FILE_SIZE_MISMATCH',relative)
            assert file_sha256(file)==item['sha256'], ('MOUNT_FILE_SHA256_MISMATCH',relative)
            total_verified+=1
        support_mode='kaggle-extracted-directories'
    else:
        SUPPORT_DIR=RUN_ROOT/'support'
        SUPPORT_DIR.mkdir(exist_ok=False)
        for archive_name,spec in sorted(bundle['archives'].items()):
            assert archive_name.endswith('.zip') and '/' not in archive_name and '..' not in archive_name
            category=archive_name[:-4]
            members={Path(path).name for path in expected if path.startswith(category+'/')}
            assert len(members)==spec['file_count']
            pack=SUPPORT_MOUNT/archive_name
            assert pack.is_file() and pack.stat().st_size==spec['bytes'], ('PACKAGE_MISSING',archive_name)
            assert file_sha256(pack)==spec['sha256'], ('PACKAGE_SHA256_MISMATCH',archive_name)
            with zipfile.ZipFile(pack) as zf:
                infos=zf.infolist()
                assert len(infos)==len(members)
                assert {info.filename for info in infos}==members
                for info in infos:
                    relative=Path(info.filename)
                    assert not relative.is_absolute() and '..' not in relative.parts
                    assert info.filename==relative.name
                    assert not info.is_dir() and (info.external_attr >> 16) & 0o170000 != 0o120000
                    logical=f'{category}/{info.filename}'
                    assert info.file_size==bundle['files'][logical]['bytes']
                    target=SUPPORT_DIR/logical
                    target.parent.mkdir(parents=True,exist_ok=True)
                    assert not target.exists()
                    with zf.open(info) as reader,target.open('xb') as writer:
                        shutil.copyfileobj(reader,writer,1024*1024)
                    assert target.stat().st_size==bundle['files'][logical]['bytes']
                    assert file_sha256(target)==bundle['files'][logical]['sha256'], ('PAYLOAD_SHA256_MISMATCH',logical)
                    total_verified+=1
        support_mode='validated-zip-extraction'
    assert total_verified==48
    assert (SUPPORT_DIR/'helpers/split_vision_maxtext.patch').is_file()
    assert (SUPPORT_DIR/'cache/pinned-cache-manifest.json').is_file()
    print('ORGANIZED_SUPPORT_SHA256_GATE_PASS',total_verified,'files',support_mode,str(SUPPORT_DIR))

    os.environ.setdefault('QWEN38_ALLOW_PINNED_WEB_WHEELS','0')
    os.environ.update({
     'PJRT_DEVICE':'TPU','TPU_SKIP_MDS_QUERY':'1',
     'TPU_HOST_BOUNDS':'1,1,1','TPU_CHIPS_PER_HOST_BOUNDS':'2,4,1',
     'TPU_WORKER_HOSTNAMES':os.uname().nodename,'TPU_WORKER_ID':'0',
     'TPU_ACCELERATOR_TYPE':'v5e-8','JAX_PLATFORMS':'tpu,cpu',
    })
    MAXTEXT_DIR=RUN_ROOT/'patched-maxtext'
    print('RUN_ROOT',RUN_ROOT,'MODEL_DIR',MODEL_DIR)
    # Explicitly return the stage's named results to the workflow.
    return {'RUN_ROOT': RUN_ROOT, 'RUN_WALL_START': RUN_WALL_START, 'RUN_STARTED_UTC': RUN_STARTED_UTC, 'MODEL_DIR': MODEL_DIR, 'SUPPORT_DIR': SUPPORT_DIR, 'BUNDLE_MANIFEST': BUNDLE_MANIFEST, 'PORTABLE_MANIFEST_SHA256': PORTABLE_MANIFEST_SHA256, 'bundle': bundle, 'os': os, 'json': json, 'hashlib': hashlib, 'Path': Path, 'time': time, 'datetime': datetime, 'timezone': timezone, 'file_sha256': file_sha256, 'MAXTEXT_DIR': MAXTEXT_DIR}
