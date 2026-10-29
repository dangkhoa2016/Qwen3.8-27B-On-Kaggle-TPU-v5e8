"""Ocr probe stage; runs only when explicitly invoked.

No TPU work occurs at import time.
"""
from __future__ import annotations

def execute(context):
    # Inputs from earlier stages
    EXAMPLES_DIR = context.require("EXAMPLES_DIR")
    PORTABLE_MANIFEST_SHA256 = context.require("PORTABLE_MANIFEST_SHA256")
    PREFILL_LEN = context.require("PREFILL_LEN")
    RUN_ROOT = context.require("RUN_ROOT")
    STAGE_CACHE = context.require("STAGE_CACHE")
    SUPPORT_DIR = context.require("SUPPORT_DIR")
    VISION_SUFFIX = context.require("VISION_SUFFIX")
    _decode = context.require("_decode")
    completion_info = context.require("completion_info")
    config = context.require("config")
    engine = context.require("engine")
    hashlib = context.require("hashlib")
    image_for_inference = context.require("image_for_inference")
    jax = context.require("jax")
    jnp = context.require("jnp")
    mm_processor = context.require("mm_processor")
    pad_id = context.require("pad_id")
    params = context.require("params")
    pyconfig = context.require("pyconfig")
    qwen_mm = context.require("qwen_mm")
    tok = context.require("tok")
    # The original audited stage implementation
    # One real OCR inference with phase milestones saved OUTSIDE the notebook.
    # No hardcoded model response. Sidecar survives a SIGALRM during TPU compilation.
    import json, time
    from datetime import datetime,timezone
    from pathlib import Path
    import numpy as np
    TRACE_PATH=RUN_ROOT/'ocr731-phase-events.json'
    TRACE_PATH.parent.mkdir(parents=True,exist_ok=True)
    assert not TRACE_PATH.exists(), 'Evidence already exists: do not overwrite or rerun.'
    TRACE={'status':'RUNNING','started_utc':datetime.now(timezone.utc).isoformat(),'events':[],'target':'real OCR 731, TPU v5e-8'}
    STEP_START=time.monotonic()
    def mark(event,**metadata):
        TRACE['events'].append({'phase':event,'elapsed_s':round(time.monotonic()-STEP_START,3),
            'utc':datetime.now(timezone.utc).isoformat(),**metadata})
        tmp=TRACE_PATH.with_suffix('.writing')
        tmp.write_text(json.dumps(TRACE,indent=2,ensure_ascii=False)+'\n')
        tmp.replace(TRACE_PATH)
        print('OCR_PHASE',event,round(time.monotonic()-STEP_START,3),metadata,flush=True)
    mark('BEGIN',jax_cache_dir=str(STAGE_CACHE),manifest_sha256=PORTABLE_MANIFEST_SHA256)
    ocr_source=EXAMPLES_DIR/'vision_test_card_731_1280x960.png'
    assert ocr_source.is_file()
    resized=image_for_inference(ocr_source)
    mark('IMAGE_RESIZED',image=str(resized))
    rendered_prompt='Read the large number in this image. Answer only the digits.'+VISION_SUFFIX
    pd=config._pydantic_config.model_copy(update={'image_path':str(resized),'prompt':rendered_prompt})
    cfg=pyconfig.HyperParameters(pd)
    po=mm_processor.preprocess_mm_data(cfg)
    off=int(mm_processor.get_image_offsets(config=cfg,processor_output=po))
    content=f'<|vision_start|><|image_pad|><|vision_end|>{rendered_prompt}'
    rendered=tok.apply_chat_template([{'role':'user','content':content}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
    base=tok(rendered,add_special_tokens=False)['input_ids']
    budget=PREFILL_LEN-off
    assert len(base)<=budget,(len(base),budget)
    padded=np.asarray(base+[pad_id]*(budget-len(base)),dtype=np.int32)
    fused=mm_processor.prepare_text_for_image_fusion(tokens=padded,config=cfg,processor_output=po)
    true_length=len(base)+off
    mask=(np.arange(len(fused))[None,:]<true_length).astype(np.int32)
    pos,delta=qwen_mm.get_rope_index(input_ids=fused[None,:],image_grid_thw=po.pixel_grid_thw,
        attention_mask=mask,spatial_merge_size=cfg.spatial_merge_size_for_vit,
        position_id_per_seconds=cfg.position_id_per_seconds,config=cfg)
    mark('PREPROCESS_READY',image_offset_tokens=off,prefill_tokens=true_length,
        grid=po.pixel_grid_thw.tolist(),pixel_shape=list(po.pixel_values.shape))
    # Compile/run Vision encoder separately. The exact result is forwarded to the decoder without modification.
    t=time.monotonic()
    mark('VISION_ENCODE_ENTER')
    image_embeddings,deepstack_features=engine.encode_vision(
        params=params,images=jnp.asarray(po.pixel_values))
    mark('VISION_ENCODE_DISPATCH_RETURNED')
    jax.block_until_ready(image_embeddings)
    for feature in (deepstack_features or []):
        jax.block_until_ready(feature)
    vision_s=time.monotonic()-t
    mark('VISION_ENCODE_SYNC_COMPLETE',vision_seconds=round(vision_s,3),
        embedding_shape=list(image_embeddings.shape),
        deepstack_count=len(deepstack_features or []))
    # JAX AOT diagnostic: individually instrument lowering, compile/loading, executable dispatch and sync.
    # Keep engine.prefill arguments identical when calling the compiled object directly.
    t=time.monotonic()
    if engine.rng is None:
        engine.rng=jax.random.PRNGKey(0)
    engine.rng,aot_rng=jax.random.split(engine.rng)
    aot_kwargs=dict(
        params=params,existing_prefix=None,
        padded_tokens=jnp.asarray(fused,dtype=jnp.int32),
        positions=jnp.asarray(pos),mrope_deltas=jnp.asarray(delta),
        images=None,image_embeddings=image_embeddings,deepstack_visual_embeds=deepstack_features,
        image_masks=None,videos=None,video_masks=None,video_grid_thw=None,
        audio_values=None,audio_masks=None,true_length=true_length,sampler=None,
        rng=aot_rng,slot=None,return_prompt_logp=False,algorithm='greedy',
        topk=None,nucleus_topp=None,temperature=None)
    static_keys=('video_grid_thw','return_prompt_logp','algorithm','topk','nucleus_topp')
    dynamic_kwargs={k:v for k,v in aot_kwargs.items() if k not in static_keys}
    mark('DECODER_AOT_LOWER_ENTER')
    lowered=engine._prefill_jit.lower(engine,**aot_kwargs)
    mark('DECODER_AOT_LOWER_RETURN')
    # Retain all native runtime correctness checks.
    import sys
    sys.path.insert(0,str(SUPPORT_DIR/'helpers'))
    from aot_jax_pxla_phase_hooks import install as install_aot_phase_hooks
    from aot_pxla_nested_sharding_hooks import install as install_nested_sharding_hooks
    from aot_native_metadata_hooks import install as install_native_metadata_hooks
    from aot_sharding_metadata_reuse_strict import install as install_metadata_reuse
    LAYOUT_HOOK_FILE=SUPPORT_DIR/'helpers/aot_native_layout_timing_hooks.py'
    assert LAYOUT_HOOK_FILE.is_file() and hashlib.sha256(LAYOUT_HOOK_FILE.read_bytes()).hexdigest()=='ac58045e8ede38f5434a9761abc2001472d68b9012451c847e1d6d841325bce8', 'LAYOUT_HOOK_HELPER_SHA_MISMATCH'
    sys.path.insert(0,str(LAYOUT_HOOK_FILE.parent))
    from aot_native_layout_timing_hooks import install as install_native_layout_timing
    from contextlib import ExitStack
    mark('DECODER_PXLA_HOOK_INSTALL_ENTER')
    with ExitStack() as cleanups:
        remove_aot_hooks=install_aot_phase_hooks(lambda label:mark(label))
        cleanups.callback(remove_aot_hooks)
        remove_native_hooks=install_native_metadata_hooks(lambda label:mark(label))
        cleanups.callback(remove_native_hooks)
        remove_reuse_hooks=install_metadata_reuse(lambda label:mark(label))
        def restore_memo():
            stats=remove_reuse_hooks()
            mark('DECODER_METADATA_REUSE_STATS',**stats)
        cleanups.callback(restore_memo)
        remove_nested_hooks=install_nested_sharding_hooks(lambda label:mark(label))
        cleanups.callback(remove_nested_hooks)
        remove_native_layout_timing=install_native_layout_timing(lambda label:mark(label))
        cleanups.callback(remove_native_layout_timing)
        mark('DECODER_PXLA_HOOK_INSTALL_RETURN')
        mark('DECODER_AOT_COMPILE_ENTER')
        compiled=lowered.compile()
        mark('DECODER_AOT_COMPILE_RETURN')
    mark('DECODER_PXLA_HOOK_REMOVED')
    mark('DECODER_EXEC_ENTER')
    prefix,first=compiled(**dynamic_kwargs)
    mark('DECODER_EXEC_RETURN')
    jax.block_until_ready(first.data)
    prefill_s=time.monotonic()-t
    mark('PREFILL_RETURN',decoder_aot_total_seconds=round(prefill_s,3))
    t=time.monotonic()
    mark('DECODE_ENTER')
    out,text,steps=_decode(prefix,first,true_length,max_decode_steps=12)
    mark('DECODE_RETURN',decode_seconds=round(time.monotonic()-t,3),decode_tokens=len(out))
    ocr_result={'prompt':rendered_prompt,'image':str(ocr_source),'generated_text':text,
        'generated_token_ids':out,'eos_reached':completion_info(out)['eos_reached'],
        'semantic_check':'731' in text,'vision_seconds':vision_s,'prefill_seconds':prefill_s,
        'decode_step_seconds':steps,'image_offset_tokens':off,'true_length':true_length}
    TRACE['status']='COMPLETE'
    TRACE['ocr_result']=ocr_result
    mark('DONE')
    print('LIVE_OCR',json.dumps({'generated_text':text,'semantic_check':'731' in text,
        'eos_reached':ocr_result['eos_reached'],'vision_seconds':vision_s,'prefill_seconds':prefill_s,
        'decode_tokens':len(out)},ensure_ascii=False),flush=True)
    # Explicitly return the stage's named results to the workflow.
    return {'TRACE': TRACE, 'compiled': compiled, 'dynamic_kwargs': dynamic_kwargs, 'ocr_result': ocr_result, 'ocr_source': ocr_source, 'po': po, 'image_embeddings': image_embeddings}
