"""Image cases stage; runs only when explicitly invoked.

No TPU work occurs at import time.
"""
from __future__ import annotations

def execute(context):
    # Inputs from earlier stages
    EXAMPLES_DIR = context.require("EXAMPLES_DIR")
    SUPPORT_DIR = context.require("SUPPORT_DIR")
    add_case = context.require("add_case")
    begin_case = context.require("begin_case")
    cases_state = context.require("cases_state")
    generate_vision_aot_reuse = context.require("generate_vision_aot_reuse")
    json = context.require("json")
    save_cases = context.require("save_cases")
    time = context.require("time")
    # The original audited stage implementation
    # Remaining 15 image cases: OCR482 + 6 rich scenes + 8 photographs.
    rich_spec=json.loads((SUPPORT_DIR/'cases/vision_showcase_cases.json').read_text())['cases']
    photo_spec=json.loads((SUPPORT_DIR/'cases/photo_showcase_cases.json').read_text())['cases']
    assert len(rich_spec)==6 and len(photo_spec)==8
    image_jobs=[('ocr',{'id':'ocr-482','image':'vision_test_card_482_1280x960.png',
                         'prompt':'Read the large number in this image. Answer only the digits.','expected':'482','max_decode_steps':12})]
    image_jobs += [('rich',x) for x in rich_spec]
    image_jobs += [('photo',x) for x in photo_spec]
    assert len(image_jobs)==15
    for group,case in image_jobs:
        identifier=case['id']
        path=EXAMPLES_DIR/case['image']
        started=time.monotonic()
        begin_case(group,identifier)
        try:
            row=generate_vision_aot_reuse(path,case['prompt'],max_decode_steps=case.get('max_decode_steps',448))
            row['id']=identifier
            row['group']=group
            if group=='ocr':
                row['expected']=case['expected']
                row['semantic_check']=case['expected'] in row['generated_text']
            if group=='rich':
                hits={x:(x.lower() in row['generated_text'].lower()) for x in case.get('checks',[])}
                row['checks']=hits
                row['check_hits']=sum(hits.values())
                row['check_count']=len(hits)
            row['case_seconds']=round(time.monotonic()-started,3)
            add_case(group,row)
        except BaseException as err:
            cases_state['status']='FAILED_'+identifier
            cases_state['error']={'type':type(err).__name__,'message':str(err),'case_id':identifier}
            save_cases()
            raise
        # Keep the original output and process the next input even when an OCR
        # answer is incorrect or generation reaches its token limit.
        if (group=='ocr' and not row['semantic_check']) or not row['eos_reached']:
            print('CASE_REVIEW_REQUIRED',identifier,
                  'EOS',row['eos_reached'],
                  'SEMANTIC',row.get('semantic_check'),flush=True)
    print('FULL19_IMAGE_CASES_COMPLETED',len([x for x in cases_state['cases'] if x['group']!='text']))
    # Explicitly return the stage's named results to the workflow.
    return {'rich_spec': rich_spec, 'photo_spec': photo_spec}
