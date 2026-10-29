"""Inputs stage; runs only when explicitly invoked.

No TPU work occurs at import time.
"""
from __future__ import annotations

def execute(context):
    # Inputs from earlier stages
    Path = context.require("Path")
    SUPPORT_DIR = context.require("SUPPORT_DIR")
    hashlib = context.require("hashlib")
    json = context.require("json")
    # The original audited stage implementation
    # 3b. Verify image assets and case definitions from the attached support Dataset.
    EXAMPLES_DIR=SUPPORT_DIR/'images'  # all input images are verified by SHA256
    assets=json.loads((SUPPORT_DIR/'cases/assets_manifest.json').read_text())['assets']
    assert len(assets)==16,len(assets)
    for item in assets:
        p=EXAMPLES_DIR/Path(item['path']).name
        assert p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest()==item['sha256'],('ASSET_SHA_MISMATCH',p)
    rich_spec=json.loads((SUPPORT_DIR/'cases/vision_showcase_cases.json').read_text())['cases']
    photo_spec=json.loads((SUPPORT_DIR/'cases/photo_showcase_cases.json').read_text())['cases']
    assert len(rich_spec)==6 and len(photo_spec)==8
    print('SINGLE_DATASET_ASSET_GATE_PASS',len(assets),len(rich_spec),len(photo_spec))
    # Explicitly return the stage's named results to the workflow.
    return {'EXAMPLES_DIR': EXAMPLES_DIR, 'rich_spec': rich_spec, 'photo_spec': photo_spec}
