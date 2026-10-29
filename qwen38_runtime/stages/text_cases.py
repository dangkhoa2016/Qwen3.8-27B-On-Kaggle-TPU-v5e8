"""Text cases stage; runs only when explicitly invoked.

No TPU work occurs at import time.
"""
from __future__ import annotations

def execute(context):
    # Inputs from earlier stages
    FULL19_FINAL = context.require("FULL19_FINAL")
    RUN_WALL_START = context.require("RUN_WALL_START")
    add_case = context.require("add_case")
    begin_case = context.require("begin_case")
    cases_state = context.require("cases_state")
    datetime = context.require("datetime")
    generate_text = context.require("generate_text")
    json = context.require("json")
    save_cases = context.require("save_cases")
    time = context.require("time")
    timezone = context.require("timezone")
    # The original audited stage implementation
    # Three text prompts. Their text-prefill compiled program may differ from vision-prefill.
    TEXT_PROMPTS=['2+2','France capital','Ruby language']
    for p in TEXT_PROMPTS:
        t=time.monotonic()
        begin_case('text','text-'+p.lower().replace(' ','-').replace('+','plus'))
        try:
            row=generate_text(p,max_decode_steps=96)
            low=row['generated_text'].lower()
            ok=('4' in row['generated_text']) if p=='2+2' else (
                'paris' in low if p=='France capital' else (
                    'ruby' in low and ('programming' in low or 'language' in low)))
            row.update({'id':'text-'+p.lower().replace(' ','-').replace('+','plus'),
                        'semantic_check':bool(ok),'case_seconds':round(time.monotonic()-t,3)})
            add_case('text',row)
        except BaseException as exc:
            cases_state['status']='FAILED_TEXT_CASE'
            cases_state['error']={'prompt':p,'type':type(exc).__name__,'message':str(exc)}
            save_cases()
            raise
        if not (row['semantic_check'] and row['eos_reached']):
            # A model-quality finding cannot erase valid execution evidence.
            print('TEXT_REVIEW_REQUIRED',p,'EOS',row['eos_reached'],
                  'SEMANTIC',row['semantic_check'],flush=True)
    assert len(cases_state['cases'])==19
    from collections import Counter
    counts=Counter(x['group'] for x in cases_state['cases'])
    assert counts=={'text':3,'ocr':2,'rich':6,'photo':8},counts
    assert len({x['id'] for x in cases_state['cases']})==19
    all_eos=all(bool(x['eos_reached']) for x in cases_state['cases'])
    review_flags=[]
    for row in cases_state['cases']:
        if not row['eos_reached']:
            review_flags.append({'id':row['id'],'reason':'NO_EOS'})
        if row['group'] in ('text','ocr') and not row.get('semantic_check'):
            review_flags.append({'id':row['id'],'reason':'CONTENT_CHECK'})
        if row['id']=='ocr-731' and not row.get('format_check'):
            review_flags.append({'id':row['id'],'reason':'OCR_FORMAT'})
    cases_state['status']=('AUTOMATED_19_CASES_EOS_COMPLETED' if all_eos
                           else 'AUTOMATED_19_CASES_RECORDED_NEEDS_REVIEW')
    cases_state['finished_utc']=datetime.now(timezone.utc).isoformat()
    cases_state['total_wall_seconds']=round(time.monotonic()-RUN_WALL_START,3)
    cases_state['cold_compile_cache_hit_not_guaranteed']=True
    cases_state['group_counts']=dict(counts)
    cases_state['all_eos_reached']=all_eos
    cases_state['eos_count']=sum(bool(x['eos_reached']) for x in cases_state['cases'])
    cases_state['review_flags']=review_flags
    cases_state['automated_quality_checks_passed']=not review_flags
    cases_state['visual_human_review_pending']=True
    cases_state['rich_keyword_hits']=sum(x.get('check_hits',0) for x in cases_state['cases'] if x['group']=='rich')
    cases_state['rich_keyword_checks']=sum(x.get('check_count',0) for x in cases_state['cases'] if x['group']=='rich')
    assert not FULL19_FINAL.exists()
    FULL19_FINAL.write_text(json.dumps(cases_state,indent=2,ensure_ascii=False,default=str)+'\n')
    save_cases()
    print('FULL19_EXECUTION_SUMMARY',cases_state['status'],'cases',len(cases_state['cases']),
          'eos',cases_state['eos_count'],'review_flags',len(review_flags),
          'VISUAL_REVIEW_PENDING',flush=True)
    # Explicitly return the stage's named results to the workflow.
    return {}
