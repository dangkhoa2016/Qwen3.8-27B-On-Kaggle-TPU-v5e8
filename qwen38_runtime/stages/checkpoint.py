"""Checkpoint stage; runs only when explicitly invoked.

No TPU work occurs at import time.
"""
from __future__ import annotations

def execute(context):
    # Inputs from earlier stages
    MODEL_DIR = context.require("MODEL_DIR")
    # The original audited stage implementation
    # 4. Source checkpoint audit before TPU-heavy work.
    import json
    from pathlib import Path

    cfg = json.loads((MODEL_DIR / 'config.json').read_text())
    index = json.loads((MODEL_DIR / 'model.safetensors.index.json').read_text())
    shards = sorted(MODEL_DIR.glob('model-*.safetensors'))
    print('architecture:', cfg.get('architectures'))
    print('model_type:', cfg.get('model_type'))
    print('safetensors shards:', len(shards))
    print('declared total bytes:', index.get('metadata', {}).get('total_size'))
    assert len(shards) == 18, f'Expected 18 shards, found {len(shards)}'
    assert index.get('metadata', {}).get('total_size') == 55_562_855_904
    print('CHECKPOINT_AUDIT=PASS')
    # Explicitly return the stage's named results to the workflow.
    return {}
