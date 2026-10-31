import json
import pytest
from gallery import vi_state as v
from test_gallery import fixture_results

def setup(tmp):
    en,_=fixture_results(tmp)
    return en,v.verified_source(en)[1],tmp/'vi.json'

def record(en,text='SYNTHETIC_VI',eos=True):
    return dict(id=en['id'],group=en['group'],prompt=v.expected_prompt(en,v.VI_SUFFIX),image_sha256=en.get('image_sha256'),generated_text=text,generated_token_ids=[1,2],eos_reached=eos)

def test_resume_complete_and_retry(tmp_path):
    en,by_id,path=setup(tmp_path)
    state,_=v.load_or_initialize(path,en,enabled=False)
    v.save(path,state)
    state,_=v.load_or_initialize(path,en,enabled=True)
    assert len(v.pending_ids(state,by_id))==17
    first=v.plan(by_id)[0]
    v.store_case(state,record(first),by_id)
    v.save(path,state)
    again,_=v.load_or_initialize(path,en,enabled=True)
    assert len(v.pending_ids(again,by_id))==16
    with pytest.raises(ValueError):v.store_case(again,record(first),by_id,retry_incomplete=True)
    broken=v.plan(by_id)[1]
    failure=record(broken,'',False);failure['error']='SYNTHETIC_FAILURE'
    v.store_case(again,failure,by_id)
    assert broken['id'] not in v.pending_ids(again,by_id)
    assert broken['id'] in v.pending_ids(again,by_id,retry_incomplete=True)
    v.store_case(again,record(broken),by_id,retry_incomplete=True)
    assert again['retry_history'][0]['error']=='SYNTHETIC_FAILURE'

def test_provenance_and_17_case_completion(tmp_path):
    en,by_id,path=setup(tmp_path)
    state,_=v.load_or_initialize(path,en,enabled=True)
    first=v.plan(by_id)[0]
    incorrect=record(first);incorrect['prompt']='WRONG'
    with pytest.raises(ValueError,match='prompt'):v.store_case(state,incorrect,by_id)
    for row in v.plan(by_id):v.store_case(state,record(row),by_id)
    assert v.finalize(state)['status']=='COMPLETED'
    v.save(path,state)
    assert v.pending_ids(v.load_or_initialize(path,en,enabled=True)[0],by_id,retry_incomplete=True)==[]
    content=json.loads(en.read_text());content['cases'][0]['generated_text']='CHANGED';en.write_text(json.dumps(content))
    with pytest.raises(ValueError,match='hash'):v.load_or_initialize(path,en,enabled=True)


def test_vi_state_accepts_complete_en_case_inventory_with_missing_eos(tmp_path):
    en, _ = fixture_results(tmp_path)
    data = json.loads(en.read_text())
    data["status"] = "AUTOMATED_19_CASES_RECORDED_NEEDS_REVIEW"
    data["cases"][0]["eos_reached"] = False
    en.write_text(json.dumps(data))
    digest, by_id = v.verified_source(en)
    assert len(by_id) == 19
    assert digest == v.sha_file(en)
    state, source = v.load_or_initialize(tmp_path / "vi-review.json", en, enabled=True)
    assert len(v.pending_ids(state, source)) == 17
    assert state["source_full19_sha256"] == digest


def test_vi_state_rejects_false_metadata_and_missing_evidence(tmp_path):
    en, _ = fixture_results(tmp_path)
    data = json.loads(en.read_text())
    data["cases"][0].pop("generated_token_ids")
    en.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="output fields"):
        v.verified_source(en)
    data["cases"][0]["generated_token_ids"] = [1, 2]
    data["status"] = "RUNNING"
    en.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="must finish"):
        v.verified_source(en)
