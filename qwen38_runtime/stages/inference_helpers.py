"""Inference helpers stage; runs only when explicitly invoked.

No TPU work occurs at import time.
"""
from __future__ import annotations

def execute(context):
    # Inputs from earlier stages
    AutoTokenizer = context.require("AutoTokenizer")
    MAX_TARGET_LEN = context.require("MAX_TARGET_LEN")
    MODEL_DIR = context.require("MODEL_DIR")
    PREFILL_LEN = context.require("PREFILL_LEN")
    Path = context.require("Path")
    RUN_ROOT = context.require("RUN_ROOT")
    config = context.require("config")
    e_args = context.require("e_args")
    engine = context.require("engine")
    jax = context.require("jax")
    jnp = context.require("jnp")
    mm_processor = context.require("mm_processor")
    params = context.require("params")
    pyconfig = context.require("pyconfig")
    qwen_mm = context.require("qwen_mm")
    time = context.require("time")
    # The original audited stage implementation
    # 7. Shared deterministic helpers.
    import numpy as np, statistics
    from IPython.display import display, Image as IPImage

    tok = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True, trust_remote_code=False)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else (tok.eos_token_id or 0)

    from PIL import Image as PILImage
    import hashlib
    RESIZED_DIR=RUN_ROOT/'resized-inputs-512x384'
    RESIZED_DIR.mkdir(exist_ok=True,parents=True)
    def image_for_inference(path):
        path=Path(path)
        assert path.is_file(),path
        dest=RESIZED_DIR/(path.stem+'-'+hashlib.sha256(path.read_bytes()).hexdigest()[:12]+'.png')
        if not dest.is_file():
            with PILImage.open(path) as im:
                converted=im.convert('RGB').resize((e_args.INFERENCE_WIDTH,e_args.INFERENCE_HEIGHT),PILImage.Resampling.LANCZOS)
                converted.save(dest)
        with PILImage.open(dest) as im:
            assert im.size==(e_args.INFERENCE_WIDTH,e_args.INFERENCE_HEIGHT)
        return dest

    def completion_info(ids):
        eos=int(tok.eos_token_id)
        return {'eos_reached':bool(ids and int(ids[-1])==eos),'completion_status':'EOS' if ids and int(ids[-1])==eos else 'MAX_STEPS'}

    TEXT_SUFFIX=' Answer briefly in one complete sentence.'
    VISION_SUFFIX=' Be concise, factual, and avoid speculation.'

    def _decode(prefix, first, true_length, max_decode_steps):
        state=engine.init_decode_state(); state=engine.insert(prefix,state,slot=0)
        out=[int(jax.device_get(first.data)[0,0])]; step_s=[]
        if out[0] == tok.eos_token_id: return out, tok.decode(out, skip_special_tokens=False), step_s
        for _ in range(max_decode_steps-1):
            t=time.time(); state,res=engine.generate(params=params,decode_state=state,algorithm='greedy'); jax.block_until_ready(res.data); step_s.append(time.time()-t)
            tid=int(jax.device_get(res.data)[0,0]); out.append(tid)
            if tid == tok.eos_token_id: break
        return out, tok.decode(out, skip_special_tokens=False), step_s

    def generate_text(prompt: str, max_decode_steps: int=96, suffix: str=TEXT_SUFFIX):
        rendered=tok.apply_chat_template([{'role':'user','content':prompt+suffix}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        ids=tok(rendered, add_special_tokens=False)['input_ids']
        assert len(ids) <= PREFILL_LEN, (len(ids), PREFILL_LEN, prompt)
        padded=np.asarray(ids+[pad_id]*(PREFILL_LEN-len(ids)),dtype=np.int32)
        t=time.time(); prefix,first=engine.prefill(params=params,padded_tokens=jnp.asarray(padded),true_length=len(ids),algorithm='greedy'); jax.block_until_ready(first.data); pre=time.time()-t
        out,text,steps=_decode(prefix,first,len(ids),max_decode_steps)
        return {'prompt':prompt,'prompt_token_count':len(ids),**completion_info(out),'generated_token_ids':out,'generated_text':text,'prefill_seconds':pre,'decode_step_seconds':steps}

    def generate_vision(image_path, prompt: str, max_decode_steps: int=224):
        image_path=Path(image_path); assert image_path.is_file(), image_path
        resized=image_for_inference(image_path)
        rendered_prompt=prompt+VISION_SUFFIX
        pd=config._pydantic_config.model_copy(update={'image_path':str(resized),'prompt':rendered_prompt})
        cfg=pyconfig.HyperParameters(pd)
        po=mm_processor.preprocess_mm_data(cfg)
        off=int(mm_processor.get_image_offsets(config=cfg,processor_output=po))
        content=f'<|vision_start|><|image_pad|><|vision_end|>{rendered_prompt}'
        rendered=tok.apply_chat_template([{'role':'user','content':content}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        base=tok(rendered,add_special_tokens=False)['input_ids']
        text_budget=PREFILL_LEN-off
        assert len(base) <= text_budget, {'text_tokens':len(base),'image_offset':off,'prefill':PREFILL_LEN}
        padded=np.asarray(base+[pad_id]*(text_budget-len(base)),dtype=np.int32)
        fused=mm_processor.prepare_text_for_image_fusion(tokens=padded,config=cfg,processor_output=po)
        true_length=len(base)+off
        mask=(np.arange(len(fused))[None,:] < true_length).astype(np.int32)
        pos,delta=qwen_mm.get_rope_index(
            input_ids=fused[None,:], image_grid_thw=po.pixel_grid_thw, attention_mask=mask,
            spatial_merge_size=cfg.spatial_merge_size_for_vit,
            position_id_per_seconds=cfg.position_id_per_seconds, config=cfg,
        )
        t=time.time(); prefix,first=engine.prefill(
            params=params,padded_tokens=jnp.asarray(fused,dtype=jnp.int32),positions=jnp.asarray(pos),mrope_deltas=jnp.asarray(delta),
            images=jnp.asarray(po.pixel_values),true_length=true_length,algorithm='greedy'); jax.block_until_ready(first.data); pre=time.time()-t
        max_decode_steps=min(max_decode_steps, MAX_TARGET_LEN-true_length)
        out,text,steps=_decode(prefix,first,true_length,max_decode_steps)
        return {'image':str(image_path),'prompt':prompt,**completion_info(out),'grid':po.pixel_grid_thw.tolist(),'image_offset_tokens':off,'fused_true_length':true_length,'generated_token_ids':out,'generated_text':text,'prefill_seconds':pre,'decode_step_seconds':steps}

    print('SHARED_GENERATION_HELPERS=READY')
    # Explicitly return the stage's named results to the workflow.
    return {'tok': tok, 'pad_id': pad_id, 'image_for_inference': image_for_inference, 'completion_info': completion_info, '_decode': _decode, 'generate_text': generate_text, 'generate_vision': generate_vision, 'VISION_SUFFIX': VISION_SUFFIX}
