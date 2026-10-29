"""Vision reuse stage; runs only when explicitly invoked.

No TPU work occurs at import time.
"""
from __future__ import annotations

def execute(context):
    # Inputs from earlier stages
    MAX_TARGET_LEN = context.require("MAX_TARGET_LEN")
    MODEL_DIR = context.require("MODEL_DIR")
    PATCH_SHA256 = context.require("PATCH_SHA256")
    PORTABLE_MANIFEST_SHA256 = context.require("PORTABLE_MANIFEST_SHA256")
    PREFILL_LEN = context.require("PREFILL_LEN")
    RUN_ROOT = context.require("RUN_ROOT")
    RUN_STARTED_UTC = context.require("RUN_STARTED_UTC")
    STAGE_CACHE = context.require("STAGE_CACHE")
    STAGE_MANIFEST = context.require("STAGE_MANIFEST")
    TRACE = context.require("TRACE")
    VISION_SUFFIX = context.require("VISION_SUFFIX")
    _decode = context.require("_decode")
    compiled = context.require("compiled")
    completion_info = context.require("completion_info")
    config = context.require("config")
    dynamic_kwargs = context.require("dynamic_kwargs")
    engine = context.require("engine")
    file_sha256 = context.require("file_sha256")
    image_embeddings = context.require("image_embeddings")
    image_for_inference = context.require("image_for_inference")
    jax = context.require("jax")
    jnp = context.require("jnp")
    mm_processor = context.require("mm_processor")
    ocr_result = context.require("ocr_result")
    ocr_source = context.require("ocr_source")
    pad_id = context.require("pad_id")
    params = context.require("params")
    po = context.require("po")
    pyconfig = context.require("pyconfig")
    qwen_mm = context.require("qwen_mm")
    tok = context.require("tok")
    # The original audited stage implementation
    # Split-Vision cached runtime reuse and incremental durable provenance.
    import json,hashlib,time,traceback,os
    from datetime import datetime,timezone
    from pathlib import Path
    import numpy as np
    from PIL import Image as PILImage
    assert TRACE['status']=='COMPLETE' and 'compiled' in context
    # Semantic and text-format mismatches are review findings, not AOT runtime failures.
    ocr_result['format_check']=ocr_result['generated_text']=='731<|im_end|>'
    if not (ocr_result['semantic_check'] and ocr_result['format_check'] and ocr_result['eos_reached']):
        print('OCR731_REVIEW_REQUIRED', {'semantic':ocr_result['semantic_check'],
              'format':ocr_result['format_check'],'eos':ocr_result['eos_reached']}, flush=True)
    assert 'compiled' in context and 'dynamic_kwargs' in context
    FULL19_DIR=RUN_ROOT/'cases'
    assert not FULL19_DIR.exists(), ('NO_OVERWRITE',str(FULL19_DIR))
    FULL19_DIR.mkdir(exist_ok=False)
    FULL19_SIDECAR=FULL19_DIR/'full19-case-events.json'
    FULL19_FINAL=FULL19_DIR/'full19-final.json'
    assert not FULL19_SIDECAR.exists() and not FULL19_FINAL.exists()
    stage_manifest=json.loads(STAGE_MANIFEST.read_text())
    cache_entries={p.name: p for p in STAGE_CACHE.glob('*') if p.is_file()}
    for entry in stage_manifest['entries']:
        assert entry['file'] in cache_entries
    generate_cache='jit__generate_jit-8583d9e06f6bcd9b55e0a85553b25aa59f5463d643d81b5fa7b8cd0ad361019b-cache'
    assert generate_cache in cache_entries
    assert file_sha256(cache_entries[generate_cache])=='ee76270eff0aa04a0128880cd30c44c3b2dfa99cfad69204fae64d4245698c57'
    runtime_signature=jax.tree_util.tree_structure(dynamic_kwargs)
    EXPECTED_VISION_EMBEDDING_SHAPE=tuple(image_embeddings.shape)
    EXPECTED_VISION_PIXEL_SHAPE=tuple(po.pixel_values.shape)
    cases_state={'status':'RUNNING','started_utc':datetime.now(timezone.utc).isoformat(),'model':'original BF16 27B',
                 'cases':[],'attempted_count':0,'device_count':jax.device_count(),'prefill_length':PREFILL_LEN,
                 'max_target_length':MAX_TARGET_LEN,'model_shards':18,'engine_count':1,
                 'portable_manifest_sha256':PORTABLE_MANIFEST_SHA256,
                 'checkpoint_index_sha256':file_sha256(MODEL_DIR/'model.safetensors.index.json'),
                 'maxtext_patch_sha256':PATCH_SHA256,
                 'jax_cache_manifest_sha256':file_sha256(STAGE_MANIFEST),
                 'run_started_utc':RUN_STARTED_UTC,
                 'model_files_attached':str(MODEL_DIR),
                 'device_descriptors':[str(x) for x in jax.devices()],
                 'ocr731_bootstrap_sha256':file_sha256(ocr_source),'first_ocr_result':ocr_result,
                 'actual_rng_seed':None,'rng_seed_note':'Uses MaxEngine RNG and per-call split; no separately attested seed.'}
    def save_cases():
        tmp=FULL19_SIDECAR.with_suffix('.writing')
        tmp.write_text(json.dumps(cases_state,indent=2,ensure_ascii=False,default=str)+'\n')
        tmp.replace(FULL19_SIDECAR)
    def add_case(group,item):
        assert group in ('text','ocr','rich','photo')
        assert 'id' in item and 'generated_token_ids' in item and 'eos_reached' in item
        assert not any(x['id']==item['id'] for x in cases_state['cases'])
        rec={'group':group,**item}
        cases_state['cases'].append(rec)
        cases_state['active_case']=None
        cases_state['attempted_count']=len(cases_state['cases'])
        save_cases()
        print('FULL19_CASE',len(cases_state['cases']),group,item['id'],
              'EOS',item['eos_reached'],'TOKENS',len(item['generated_token_ids']),flush=True)
    # The OCR731 was genuinely generated by the preceding unmodified accepted AOT path.
    add_case('ocr',{'id':'ocr-731','image_sha256':file_sha256(ocr_source),
        'prompt':ocr_result['prompt'],'generated_text':ocr_result['generated_text'],
        'generated_token_ids':ocr_result['generated_token_ids'],'eos_reached':ocr_result['eos_reached'],
        'semantic_check':ocr_result['semantic_check'],'format_check':ocr_result['format_check'],
        'vision_seconds':ocr_result['vision_seconds'],
        'prefill_seconds':ocr_result['prefill_seconds'],'decode_step_seconds':ocr_result['decode_step_seconds']})

    def begin_case(group,case_id):
        assert not cases_state.get('active_case')
        assert not any(x['id']==case_id for x in cases_state['cases'])
        cases_state['active_case']={'group':group,'id':case_id,'started_utc':datetime.now(timezone.utc).isoformat()}
        save_cases()

    def generate_vision_aot_reuse(image_path,prompt,max_decode_steps=224,suffix=VISION_SUFFIX):
        path=Path(image_path);assert path.is_file()
        resized=image_for_inference(path)
        prompt_full=prompt+suffix
        pd=config._pydantic_config.model_copy(update={'image_path':str(resized),'prompt':prompt_full})
        cfg=pyconfig.HyperParameters(pd)
        prepared=mm_processor.preprocess_mm_data(cfg)
        offset=int(mm_processor.get_image_offsets(config=cfg,processor_output=prepared))
        content=f'<|vision_start|><|image_pad|><|vision_end|>{prompt_full}'
        rendered=tok.apply_chat_template([{'role':'user','content':content}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
        ids=tok(rendered,add_special_tokens=False)['input_ids']
        budget=PREFILL_LEN-offset
        assert len(ids)<=budget,(path.name,len(ids),budget)
        padded=np.asarray(ids+[pad_id]*(budget-len(ids)),dtype=np.int32)
        fused=mm_processor.prepare_text_for_image_fusion(tokens=padded,config=cfg,processor_output=prepared)
        true_length=len(ids)+offset
        assert len(fused)==PREFILL_LEN and MAX_TARGET_LEN-true_length>=max_decode_steps
        attention_mask=(np.arange(len(fused))[None,:]<true_length).astype(np.int32)
        positions,delta=qwen_mm.get_rope_index(input_ids=fused[None,:],image_grid_thw=prepared.pixel_grid_thw,
             attention_mask=attention_mask,spatial_merge_size=cfg.spatial_merge_size_for_vit,
             position_id_per_seconds=cfg.position_id_per_seconds,config=cfg)
        assert tuple(prepared.pixel_values.shape)==EXPECTED_VISION_PIXEL_SHAPE
        start=time.monotonic()
        emb,deep=engine.encode_vision(params=params,images=jnp.asarray(prepared.pixel_values))
        jax.block_until_ready(emb)
        assert tuple(emb.shape)==EXPECTED_VISION_EMBEDDING_SHAPE and not deep
        vision_s=time.monotonic()-start
        engine.rng,key=jax.random.split(engine.rng)
        case_kwargs=dict(dynamic_kwargs)
        case_kwargs.update({'padded_tokens':jnp.asarray(fused,dtype=jnp.int32),
                            'positions':jnp.asarray(positions),'mrope_deltas':jnp.asarray(delta),
                            'image_embeddings':emb,'deepstack_visual_embeds':deep,
                            'rng':key,'true_length':true_length})
        assert jax.tree_util.tree_structure(case_kwargs)==runtime_signature
        for name in ('padded_tokens','positions','mrope_deltas','image_embeddings'):
            old_aval=dynamic_kwargs[name]
            new_aval=case_kwargs[name]
            assert tuple(new_aval.shape)==tuple(old_aval.shape),('AOT_SHAPE_MISMATCH',name,old_aval.shape,new_aval.shape)
            assert new_aval.dtype==old_aval.dtype,('AOT_DTYPE_MISMATCH',name,old_aval.dtype,new_aval.dtype)
        t=time.monotonic()
        prefix,first=compiled(**case_kwargs) # same exact AOT executable validated on OCR731
        jax.block_until_ready(first.data)
        prefill_s=time.monotonic()-t
        t=time.monotonic()
        out,text,steps=_decode(prefix,first,true_length,max_decode_steps)
        decode_s=time.monotonic()-t
        return {'image':str(path),'image_sha256':file_sha256(path),'prompt':prompt_full,
            'generated_text':text,'generated_token_ids':out,'eos_reached':completion_info(out)['eos_reached'],
            'image_offset_tokens':offset,'fused_true_length':true_length,'grid':prepared.pixel_grid_thw.tolist(),
            'pixel_shape':list(prepared.pixel_values.shape),'vision_seconds':vision_s,
            'prefill_seconds':prefill_s,'decode_seconds':decode_s,'decode_step_seconds':steps}
    print('FULL19_REUSE_SETUP_READY',len(cases_state['cases']))
    # Explicitly return the stage's named results to the workflow.
    return {'FULL19_DIR': FULL19_DIR, 'FULL19_FINAL': FULL19_FINAL, 'cases_state': cases_state, 'add_case': add_case, 'begin_case': begin_case, 'save_cases': save_cases, 'generate_vision_aot_reuse': generate_vision_aot_reuse}
