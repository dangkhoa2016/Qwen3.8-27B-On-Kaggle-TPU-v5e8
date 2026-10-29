"""Ocr evidence stage; runs only when explicitly invoked.

No TPU work occurs at import time.
"""
from __future__ import annotations

def execute(context):
    # Inputs from earlier stages
    RUN_ROOT = context.require("RUN_ROOT")
    construct_s = context.require("construct_s")
    json = context.require("json")
    load_s = context.require("load_s")
    ocr_result = context.require("ocr_result")
    # The original audited stage implementation
    # Final evidence is separate from the incremental sidecar.
    from datetime import datetime, timezone
    final=RUN_ROOT/'ocr731-bootstrap-summary.json'
    final.parent.mkdir(parents=True,exist_ok=True)
    assert not final.exists()
    final.write_text(json.dumps({'status':'COMPLETED_NEEDS_REVIEW',
     'utc':datetime.now(timezone.utc).isoformat(),
     'construct_seconds':construct_s,'checkpoint_load_seconds':load_s,
     'ocr':ocr_result,'full19_executed':False},indent=2,ensure_ascii=False)+'\n')
    print('FINAL_OCR_EVIDENCE_WRITTEN',final)
    # Explicitly return the stage's named results to the workflow.
    return {}
