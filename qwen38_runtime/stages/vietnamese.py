"""Vietnamese stage; runs only when explicitly invoked.

No TPU work occurs at import time.
"""
from __future__ import annotations

def execute(context):
    # Inputs from earlier stages
    EXAMPLES_DIR = context.require("EXAMPLES_DIR")
    FULL19_DIR = context.require("FULL19_DIR")
    FULL19_FINAL = context.require("FULL19_FINAL")
    MAX_TARGET_LEN = context.require("MAX_TARGET_LEN")
    PREFILL_LEN = context.require("PREFILL_LEN")
    cases_state = context.require("cases_state")
    generate_text = context.require("generate_text")
    generate_vision_aot_reuse = context.require("generate_vision_aot_reuse")
    photo_spec = context.require("photo_spec")
    rich_spec = context.require("rich_spec")
    tok = context.require("tok")
    # The original audited stage implementation
    import os
    import time
    from gallery.vi_state import (VI_SUFFIX, EN_RECORD_STATUSES, load_or_initialize, save, plan, expected_prompt, pending_ids, store_case, finalize, successful)
    VI_ENABLED=os.environ.get('QWEN38_ENABLE_VI','0').lower() in ('1','true','yes')
    RETRY_INCOMPLETE=os.environ.get('QWEN38_RETRY_VI_INCOMPLETE','0').lower() in ('1','true','yes')
    VI_FINAL=FULL19_DIR/'full19-vi.json'
    VI_PROMPT_SUFFIX=VI_SUFFIX
    assert FULL19_FINAL.is_file() and cases_state['status'] in EN_RECORD_STATUSES
    vi_state,en_by_id=load_or_initialize(VI_FINAL,FULL19_FINAL,enabled=VI_ENABLED)
    save(VI_FINAL,vi_state)
    if VI_ENABLED:
        for en in plan(en_by_id):
            prompt=expected_prompt(en,VI_SUFFIX)
            content=prompt if en['group']=='text' else '<|vision_start|><|image_pad|><|vision_end|>'+prompt
            rendered=tok.apply_chat_template([{'role':'user','content':content}],
                tokenize=False,add_generation_prompt=True,enable_thinking=False)
            length=len(tok(rendered,add_special_tokens=False)['input_ids'])
            offset=0 if en['group']=='text' else int(en['image_offset_tokens'])
            assert length+offset<=PREFILL_LEN,('VI_PREFILL_OVERFLOW',en['id'],length+offset)
            assert MAX_TARGET_LEN-length-offset>=(160 if en['group']=='text' else 448),('VI_DECODE_OVERFLOW',en['id'])
        todo=set(pending_ids(vi_state,en_by_id,retry_incomplete=RETRY_INCOMPLETE))
        print('FULL19_VI_RESUME',len(vi_state['cases']),'recorded',len(todo),'pending',flush=True)
        rich_map={r['id']:r for r in rich_spec}
        photo_map={r['id']:r for r in photo_spec}
        for en in plan(en_by_id):
            identifier=en['id']
            if identifier not in todo:
                continue
            group=en['group']
            prompt=expected_prompt(en,VI_SUFFIX)
            started=time.monotonic()
            vi_state['active_case']=identifier
            save(VI_FINAL,vi_state)
            try:
                if group=='text':
                    response=generate_text(en['prompt'],max_decode_steps=160,suffix=VI_SUFFIX)
                else:
                    spec=(rich_map if group=='rich' else photo_map)[identifier]
                    response=generate_vision_aot_reuse(EXAMPLES_DIR/spec['image'],
                          spec['prompt'],max_decode_steps=448,suffix=VI_SUFFIX)
                record={'id':identifier,'group':group,'prompt':prompt,
                        'generated_text':response['generated_text'],
                        'generated_token_ids':response['generated_token_ids'],
                        'eos_reached':response['eos_reached'],
                        'case_seconds':round(time.monotonic()-started,3),
                        'prefill_seconds':response.get('prefill_seconds'),
                        'decode_step_seconds':response.get('decode_step_seconds'),
                        'image_sha256':en.get('image_sha256'),
                        'quality_review':'NOT_AUTOMATICALLY_SCORED'}
            except Exception as exc:
                record={'id':identifier,'group':group,'prompt':prompt,
                        'image_sha256':en.get('image_sha256'),
                        'error':type(exc).__name__+': '+str(exc),
                        'eos_reached':False,'case_seconds':round(time.monotonic()-started,3),
                        'quality_review':'NOT_AVAILABLE'}
            store_case(vi_state,record,en_by_id,retry_incomplete=RETRY_INCOMPLETE)
            save(VI_FINAL,vi_state)
            print('FULL19_VI_CASE',len(vi_state['cases']),identifier,
                  'EOS',record['eos_reached'],'ERROR',bool(record.get('error')),flush=True)
        finalize(vi_state)
        save(VI_FINAL,vi_state)
    print('FULL19_VI_STATUS',vi_state['status'],
          sum(successful(r) for r in vi_state['cases']),'/17',flush=True)
    if VI_ENABLED and vi_state['status'] != 'COMPLETED':
        print('WARNING_VI_PARTIAL_OR_INCOMPLETE: review full19-vi.json before publishing.',flush=True)
    # Explicitly return the stage's named results to the workflow.
    return {'VI_FINAL': VI_FINAL}
