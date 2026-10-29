"""Engine stage; runs only when explicitly invoked.

No TPU work occurs at import time.
"""
from __future__ import annotations

def execute(context):
    # Inputs from earlier stages
    EXAMPLES_DIR = context.require("EXAMPLES_DIR")
    MAXTEXT_REPO_ROOT = context.require("MAXTEXT_REPO_ROOT")
    MODEL_DIR = context.require("MODEL_DIR")
    MaxEngine = context.require("MaxEngine")
    RUN_ROOT = context.require("RUN_ROOT")
    SUPPORT_DIR = context.require("SUPPORT_DIR")
    file_sha256 = context.require("file_sha256")
    json = context.require("json")
    pyconfig = context.require("pyconfig")
    time = context.require("time")
    # The original audited stage implementation
    # 6. One multimodal engine, one BF16 load.
    from maxtext.multimodal import processor as mm_processor
    from maxtext.multimodal import processor_qwen3_omni as qwen_mm

    import importlib.util
    spec=importlib.util.spec_from_file_location('qwen38_engine_args',SUPPORT_DIR/'runtime/engine_args.py')
    e_args=importlib.util.module_from_spec(spec);spec.loader.exec_module(e_args)
    # Verified shape: 320-token prefill, 768-token target. Rich/Photography decode up to 448 tokens.
    e_args.MAX_TARGET_LEN=768
    e_args.PREFILL_LEN=320
    e_args.INFERENCE_WIDTH=512
    e_args.INFERENCE_HEIGHT=384
    PREFILL_LEN=e_args.PREFILL_LEN
    MAX_TARGET_LEN=e_args.MAX_TARGET_LEN
    print('COMPILE_TARGET_LENGTH',MAX_TARGET_LEN,'PREFILL',PREFILL_LEN)
    VISION_SEED_IMAGE=EXAMPLES_DIR/'vision_test_card_731_1280x960.png'
    assert VISION_SEED_IMAGE.is_file()
    args=e_args.build_engine_args(MAXTEXT_REPO_ROOT,MODEL_DIR,VISION_SEED_IMAGE)
    # Verify the optional compilation cache without modifying the user cache.
    STAGE_MANIFEST=SUPPORT_DIR/'cache/pinned-cache-manifest.json'
    STAGE_METADATA=json.loads(STAGE_MANIFEST.read_text())
    assert len(STAGE_METADATA['entries'])==3
    STAGE_CACHE=RUN_ROOT/'jax_cache'
    assert not STAGE_CACHE.exists()
    STAGE_CACHE.mkdir()
    import shutil
    for entry in STAGE_METADATA['entries']:
        source=SUPPORT_DIR/'cache'/entry['file']
        assert source.is_file() and file_sha256(source)==entry['sha256']
        target=STAGE_CACHE/entry['file']
        shutil.copyfile(source,target)
        assert target.stat().st_size==entry['bytes'] and file_sha256(target)==entry['sha256']
    assert not any(x.startswith('jax_cache_dir=') for x in args)
    args.append('jax_cache_dir='+str(STAGE_CACHE))
    print('JAX_CACHE_CANDIDATE_SHA256_PASS',len(STAGE_METADATA['entries']))
    print('NATIVE_JAX_SHARDING_LAYOUT_CHECKS_REQUIRED; CACHE_HIT_NOT_GUARANTEED')

    t0=time.time(); config=pyconfig.initialize(args); engine=MaxEngine(config); construct_s=time.time()-t0
    print(f'engine_construct_seconds={construct_s:.3f}')
    t0=time.time(); params=engine.load_params(); load_s=time.time()-t0
    print(f'load_params_seconds={load_s:.3f}')
    assert getattr(config,'use_multimodal',False) is True
    print('MODEL_LOAD=PASS')
    print('ONE_ENGINE_POLICY=PASS')
    # Explicitly return the stage's named results to the workflow.
    return {'PREFILL_LEN': PREFILL_LEN, 'MAX_TARGET_LEN': MAX_TARGET_LEN, 'STAGE_CACHE': STAGE_CACHE, 'STAGE_MANIFEST': STAGE_MANIFEST, 'STAGE_METADATA': STAGE_METADATA, 'e_args': e_args, 'config': config, 'engine': engine, 'params': params, 'mm_processor': mm_processor, 'qwen_mm': qwen_mm, 'construct_s': construct_s, 'load_s': load_s}
