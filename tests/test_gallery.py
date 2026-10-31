"""CPU-only tests; synthetic fixture outputs are NEVER published as model evidence."""
import json
from pathlib import Path
import pytest
from gallery.render import build_gallery, sha256
ROOT=Path(__file__).resolve().parents[1]
EXAMPLES=ROOT/'examples'
MANIFEST=EXAMPLES/'assets_manifest.json'

def fixture_results(tmp_path):
    rich=json.loads((EXAMPLES/'vision_showcase_cases.json').read_text())['cases']
    photo=json.loads((EXAMPLES/'photo_showcase_cases.json').read_text())['cases']
    hashes={Path(a['path']).name:a['sha256'] for a in json.loads(MANIFEST.read_text())['assets']}
    cases=[]
    for ident in ['text-2plus2','text-france-capital','text-ruby-language']:
        cases.append({'id':ident,'group':'text','prompt':ident,'generated_text':'SYNTHETIC_TEST_ONLY',
                      'generated_token_ids':[1,2],'eos_reached':True,'case_seconds':0.1})
    for n in ('731','482'):
        filename=f'vision_test_card_{n}_1280x960.png'
        cases.append({'id':'ocr-'+n,'group':'ocr','image':str(EXAMPLES/'images'/filename),
                      'image_sha256':hashes[filename],'prompt':'Read number','generated_text':n,
                      'generated_token_ids':[1,2],'eos_reached':True})
    for group,items in (('rich',rich),('photo',photo)):
        for case in items:
            cases.append({'id':case['id'],'group':group,
                          'image':str(EXAMPLES/'images'/case['image']),
                          'image_sha256':hashes[case['image']],
                          'prompt':case['prompt'],
                          'generated_text':"</script><script>alert('xss')</script>",
                          'generated_token_ids':[1,2],'eos_reached':True,'case_seconds':0.2})
    path=tmp_path/'en.json'
    path.write_text(json.dumps({'status':'AUTOMATED_19_CASES_EOS_COMPLETED',
                    'model':'SYNTHETIC_TEST_ONLY','total_wall_seconds':0.1,'cases':cases}))
    return path,cases

def test_gallery_never_fabricates_unrun_vi(tmp_path):
    en,rows=fixture_results(tmp_path)
    output=tmp_path/'out'
    build_gallery(en,EXAMPLES/'images',output,manifest=MANIFEST)
    page=(output/'index.html').read_text()
    data=json.loads((output/'evidence.json').read_text())
    assert len(data['cases'])==19
    assert data['summary']['vi_generated']==0
    assert sum(x['vi']['status']=='not_run' for x in data['cases'])==17
    assert sum(x['vi']['status']=='shared_numeric' for x in data['cases'])==2
    assert len(list((output/'assets').iterdir()))==16
    assert '</script><script>alert' not in page
    assert r'\u003c/script\u003e' in page
    assert 'VI not generated' in page and 'Chưa sinh VI' in page

def test_gallery_only_shows_actual_vi_records(tmp_path):
    en,rows=fixture_results(tmp_path)
    case=next(r for r in rows if r['group']=='photo')
    vi=tmp_path/'vi.json'
    vi.write_text(json.dumps({'status':'PARTIAL','source_full19':en.name,'source_full19_sha256':sha256(en),'cases':[
        {'id':case['id'],'group':'photo','prompt':'VI prompt',
         'generated_text':'VI TEST OUTPUT - NOT REAL',
         'generated_token_ids':[5],'eos_reached':True,'case_seconds':0.2,'image_sha256':case['image_sha256']},
        {'id':'text-2plus2','group':'text','error':'TEST_EXCEPTION','eos_reached':False}
    ]}))
    out=tmp_path/'out'
    build_gallery(en,EXAMPLES/'images',out,vi,MANIFEST)
    data=json.loads((out/'evidence.json').read_text())
    assert data['summary']['vi_generated']==1
    assert next(x for x in data['cases'] if x['id']==case['id'])['vi']['status']=='completed'
    err=next(x for x in data['cases'] if x['id']=='text-2plus2')
    assert err['vi']['status']=='incomplete' and err['vi']['output']==''

def test_gallery_rejects_wrong_image_sha(tmp_path):
    en,rows=fixture_results(tmp_path)
    rows[3]['image_sha256']='0'*64
    en.write_text(json.dumps({'status':'AUTOMATED_19_CASES_EOS_COMPLETED','cases':rows}))
    with pytest.raises(ValueError,match='SHA256 mismatch'):
        build_gallery(en,EXAMPLES/'images',tmp_path/'out',manifest=MANIFEST)

def test_gallery_rejects_incomplete_or_duplicate_cases(tmp_path):
    en,rows=fixture_results(tmp_path)
    for altered,error in ((rows[:-1],'19-case'),(rows[:-1]+[rows[0]],'duplicate')):
        en.write_text(json.dumps({'status':'AUTOMATED_19_CASES_EOS_COMPLETED','cases':altered}))
        with pytest.raises(ValueError,match=error):
            build_gallery(en,EXAMPLES/'images',tmp_path/'out',manifest=MANIFEST)

def test_notebook_inline_gallery_source_matches_cpu_builder():
    n=json.loads((ROOT/'notebooks/kaggle-tpu-v5e8-production-showcase.ipynb').read_text())
    assert ''.join(n['cells'][32]['source']) == "# Gallery\nworkflow.run('gallery')\n"
    assert len(n['cells'])==35
    vi_code=''.join(n['cells'][30]['source'])
    assert vi_code == "# Vietnamese\nworkflow.run('vietnamese')\n"
    vi_stage=(ROOT/'qwen38_runtime/stages/vietnamese.py').read_text()
    assert "VI_FINAL=FULL19_DIR/'full19-vi.json'" in vi_stage
    assert "pending_ids(vi_state,en_by_id" in vi_stage
    assert "MAX_TARGET_LEN-length-offset" in vi_stage
    assert "QWEN38_ENABLE_VI" in vi_stage
    assert "cases_state['cases'].append" not in vi_stage
    assert "FULL19_FINAL.write_text" in (ROOT/'qwen38_runtime/stages/text_cases.py').read_text()

def test_verified_refresh_replaces_only_same_run(tmp_path):
    from gallery.render import refresh_gallery
    en,rows=fixture_results(tmp_path)
    dest=tmp_path/'gallery'
    build_gallery(en,EXAMPLES/'images',dest,manifest=MANIFEST)
    vi=tmp_path/'vi.json'
    one=next(x for x in rows if x['group']=='photo')
    vi.write_text(json.dumps({'source_full19':en.name,'source_full19_sha256':sha256(en),'cases':[{'id':one['id'],'group':'photo','prompt':'VI TEST','generated_text':'SYNTHETIC_VI','eos_reached':True,'generated_token_ids':[1],'image_sha256':one['image_sha256']}]}))
    refresh_gallery(en,EXAMPLES/'images',dest,vi,MANIFEST)
    data=json.loads((dest/'evidence.json').read_text())
    assert data['summary']['vi_generated']==1
    assert len(list((dest/'assets').iterdir()))==16
    assert data['summary']['english_cases']==19
    original=json.loads(en.read_text())
    original['cases'][0]['generated_text']='DIFFERENT EN'
    en.write_text(json.dumps(original))
    with pytest.raises(ValueError,match='does not belong'):
        refresh_gallery(en,EXAMPLES/'images',dest,vi,MANIFEST)
    assert (dest/'index.html').exists()

def test_refresh_refuses_unmanaged_folder(tmp_path):
    from gallery.render import refresh_gallery
    en,_=fixture_results(tmp_path)
    output=tmp_path/'foreign';output.mkdir()
    (output/'user.txt').write_text('KEEP')
    with pytest.raises(ValueError,match='not a gallery'):
        refresh_gallery(en,EXAMPLES/'images',output,manifest=MANIFEST)
    assert (output/'user.txt').read_text()=='KEEP'

def test_optional_semantic_review_is_sha_bound_and_never_automatic(tmp_path):
    from collections import Counter
    import copy
    en, rows = fixture_results(tmp_path)
    vi_path = tmp_path / 'vi.json'
    eligible = [r for r in rows if r['group'] != 'ocr']
    vi = {'status': 'COMPLETED', 'source_full19': en.name,
          'source_full19_sha256': sha256(en), 'cases': [
        {'id': r['id'], 'group': r['group'], 'prompt': 'SYNTHETIC VI PROMPT',
         'generated_text': 'SYNTHETIC VI OUTPUT', 'eos_reached': True,
         'generated_token_ids': [1, 2], 'image_sha256': r.get('image_sha256')}
        for r in eligible]}
    vi_path.write_text(json.dumps(vi))
    baseline = tmp_path / 'baseline'
    build_gallery(en, EXAMPLES / 'images', baseline, vi_path, MANIFEST)
    assert not (baseline / 'semantic-review.v1.json').exists()
    assert not (baseline / 'manual-comparison.vi.md').exists()
    assert 'const AUDIT=null;' in (baseline / 'index.html').read_text()
    findings = [
        {'id': r['id'],
         'verdict': ('core_equivalent', 'partial_equivalent', 'material_divergence')[i % 3],
         'common': {'vi': 'SYNTHETIC Chung', 'en': 'SYNTHETIC Shared'},
         'reference': {'vi': 'SYNTHETIC dữ kiện', 'en': 'SYNTHETIC Reference'},
         'difference': {'vi': 'SYNTHETIC khác', 'en': 'SYNTHETIC Difference'}}
        for i, r in enumerate(eligible)]
    annotation = {'schema': 'qwen38-pairwise-review-v1',
                  'status': 'PRELIMINARY_MANUAL_REVIEW',
                  'methodology': {'vi': 'SYNTHETIC thử nghiệm', 'en': 'SYNTHETIC test'},
                  'source': {'en_sha256': sha256(en), 'vi_sha256': sha256(vi_path),
                             'evidence_sha256': sha256(baseline / 'evidence.json')},
                  'records': findings,
                  'counts': dict(Counter(r['verdict'] for r in findings))}
    review = tmp_path / 'semantic-review.v1.json'
    review.write_text(json.dumps(annotation, ensure_ascii=False))
    built = tmp_path / 'reviewed'
    build_gallery(en, EXAMPLES / 'images', built, vi_path, MANIFEST, review)
    assert (built / 'evidence.json').read_bytes() == (baseline / 'evidence.json').read_bytes()
    assert sha256(built / 'semantic-review.v1.json') == sha256(review)
    assert len((built / 'manual-comparison.vi.md').read_text().splitlines()) > 17
    assert len((built / 'manual-comparison.en.md').read_text().splitlines()) > 17
    assert 'const AUDIT=null;' not in (built / 'index.html').read_text()
    altered = copy.deepcopy(annotation)
    altered['source']['vi_sha256'] = '0' * 64
    review.write_text(json.dumps(altered))
    with pytest.raises(ValueError, match='source SHA256 mismatch'):
        build_gallery(en, EXAMPLES / 'images', tmp_path / 'bad-vi-hash', vi_path, MANIFEST, review)
    altered = copy.deepcopy(annotation)
    altered['records'][0]['id'] = 'ocr-731'
    review.write_text(json.dumps(altered))
    with pytest.raises(ValueError, match='17 unique non-OCR'):
        build_gallery(en, EXAMPLES / 'images', tmp_path / 'bad-inventory', vi_path, MANIFEST, review)
    partial = copy.deepcopy(vi)
    partial['status'] = 'PARTIAL'
    vi_path.write_text(json.dumps(partial))
    with pytest.raises(ValueError, match='17 completed VI'):
        build_gallery(en, EXAMPLES / 'images', tmp_path / 'bad-incomplete', vi_path, MANIFEST, review)


def test_notebook_renderer_remains_self_contained_with_optional_review():
    import ast
    import nbformat
    path = ROOT / 'notebooks/kaggle-tpu-v5e8-production-showcase.ipynb'
    n = nbformat.read(path, 4)
    nbformat.validate(n)
    assert len(n.cells) == 35
    assert 'const DATA=__GALLERY_DATA__;' in (ROOT / 'gallery/render.py').read_text()
    assert 'const AUDIT=__SEMANTIC_REVIEW_DATA__;' in (ROOT / 'gallery/render.py').read_text()
    stage=(ROOT/'qwen38_runtime/stages/gallery.py').read_text()
    assert 'from gallery.render import refresh_gallery' in stage
    assert 'review_file = RUN_ROOT / "semantic-review.v1.json"' in stage
    assert "review=review_file if review_file.is_file() else None" in stage
    assert len(n.cells[32].source.splitlines()) <= 3
    for c in n.cells:
        if c.cell_type == 'code':
            ast.parse(c.source)


def test_gallery_missing_vi_path_is_not_silently_treated_as_unrun(tmp_path):
    en, _ = fixture_results(tmp_path)
    target = tmp_path / 'out'
    with pytest.raises(FileNotFoundError, match='Requested VI evidence'):
        build_gallery(en, EXAMPLES / 'images', target,
                      tmp_path / 'typo-missing-vi.json', MANIFEST)
    assert not target.exists()


def test_gallery_invalid_review_fails_without_creating_destination(tmp_path):
    from collections import Counter
    en, rows = fixture_results(tmp_path)
    vi_path = tmp_path / 'vi.json'
    eligible = [r for r in rows if r['group'] != 'ocr']
    vi_path.write_text(json.dumps({
        'status':'COMPLETED', 'source_full19': en.name,
        'source_full19_sha256':sha256(en), 'cases':[
            {'id':r['id'],'group':r['group'],'prompt':'SYNTHETIC',
             'generated_text':'SYNTHETIC','generated_token_ids':[1],
             'eos_reached':True,'image_sha256':r.get('image_sha256')}
             for r in eligible]}))
    findings = [
        {'id':r['id'], 'verdict':'core_equivalent',
         'common':{'vi':'Dữ kiện','en':'Shared fact'},
         'reference':{'vi':'Tham chiếu','en':'Reference'},
         'difference':{'vi':'Khác biệt','en':'Difference'}}
        for r in eligible
    ]
    review_data = {
        'schema':'qwen38-pairwise-review-v1', 'status':'PRELIMINARY_MANUAL_REVIEW',
        'source':{'en_sha256':sha256(en),'vi_sha256':sha256(vi_path),
                  'evidence_sha256':'0'*64},
        'methodology':{'vi':'Đối chiếu sơ bộ','en':'Preliminary review'},
        'records':findings,'counts':{'core_equivalent':len(findings),
                                   'partial_equivalent':0,'material_divergence':0}
    }
    review_file=tmp_path/'review.json'
    review_file.write_text(json.dumps(review_data))
    dest=tmp_path/'wrong-review'
    with pytest.raises(ValueError,match='generated gallery evidence SHA256'):
        build_gallery(en,EXAMPLES/'images',dest,vi_path,MANIFEST,review_file)
    assert not dest.exists()
    review_data['methodology'].pop('vi')
    review_file.write_text(json.dumps(review_data))
    dest=tmp_path/'missing-methodology'
    with pytest.raises(ValueError,match='methodology in both languages'):
        build_gallery(en,EXAMPLES/'images',dest,vi_path,MANIFEST,review_file)
    assert not dest.exists()


def test_gallery_refresh_rejects_changed_prompt_and_file_digest(tmp_path):
    from gallery.render import refresh_gallery
    en, rows=fixture_results(tmp_path)
    dest=tmp_path/'gallery'
    build_gallery(en,EXAMPLES/'images',dest,manifest=MANIFEST)
    original=sha256(dest/'index.html')
    assert (dest/'source-full19.sha256').read_text().strip()==sha256(en)
    # Even if all generated answers are unchanged, altered prompts must be rejected.
    source=json.loads(en.read_text())
    source['cases'][0]['prompt']='CHANGED PROMPT, same output'
    en.write_text(json.dumps(source))
    with pytest.raises(ValueError,match='English source SHA256'):
        refresh_gallery(en,EXAMPLES/'images',dest,manifest=MANIFEST)
    assert sha256(dest/'index.html')==original
    # When the source sidecar is absent, verify individual prompts.
    (dest/'source-full19.sha256').unlink()
    with pytest.raises(ValueError,match='does not belong'):
        refresh_gallery(en,EXAMPLES/'images',dest,manifest=MANIFEST)
    assert sha256(dest/'index.html')==original


def test_gallery_refuses_symlink_output_path(tmp_path):
    en, _ = fixture_results(tmp_path)
    user=tmp_path/'empty-user-directory'
    user.mkdir()
    pointer=tmp_path/'symlink-output'
    pointer.symlink_to(user,target_is_directory=True)
    with pytest.raises(ValueError,match='symlink'):
        build_gallery(en,EXAMPLES/'images',pointer,manifest=MANIFEST)
    assert not list(user.iterdir())


def test_gallery_without_source_sidecar_checks_case_metadata(tmp_path):
    from gallery.render import refresh_gallery
    en, _ = fixture_results(tmp_path)
    dest = tmp_path / 'old-gallery'
    build_gallery(en, EXAMPLES / 'images', dest, manifest=MANIFEST)
    (dest / 'source-full19.sha256').unlink()  # Existing gallery from before the new sidecar.
    before = json.loads((dest / 'evidence.json').read_text())
    refreshed = refresh_gallery(en, EXAMPLES / 'images', dest, manifest=MANIFEST)
    assert refreshed.is_file()
    assert json.loads((dest / 'evidence.json').read_text()) == before
    assert (dest / 'source-full19.sha256').read_text().strip() == sha256(en)


def test_selected_source_bound_review_marks_only_verified_cases(tmp_path):
    # Synthetic QA fixture, never represented as real model evidence.
    en, rows = fixture_results(tmp_path)
    eligible = [row for row in rows if row['group'] != 'ocr']
    vi_path = tmp_path / 'vi.json'
    vi_path.write_text(json.dumps({
        'status': 'COMPLETED', 'source_full19': en.name,
        'source_full19_sha256': sha256(en),
        'cases': [
            {'id': r['id'], 'group': r['group'], 'prompt': 'SYNTHETIC VI',
             'generated_text': 'SYNTHETIC OUTPUT', 'eos_reached': True,
             'generated_token_ids': [4], 'image_sha256': r.get('image_sha256')}
            for r in eligible],
    }))
    base = tmp_path / 'base'
    build_gallery(en, EXAMPLES / 'images', base, vi_path, MANIFEST)
    source_hash = sha256(base / 'evidence.json')
    review_file = tmp_path / 'verified.json'
    review = {
        'schema': 'qwen38-selected-review-v1',
        'status': 'SELECTIVE_MANUAL_NOTES',
        'source': {'en_sha256': sha256(en),
                   'vi_sha256': sha256(vi_path),
                   'evidence_sha256': source_hash},
        'methodology': {
            'en': 'Only these selected cases have been independently checked',
            'vi': 'Chỉ một số case được kiểm tra độc lập',
        },
        'records': [{
            'id': eligible[0]['id'], 'verdict': 'partial_equivalent',
            'common': {'en': 'Some overlap', 'vi': 'Một số điểm giống'},
            'reference': {'en': 'Source image', 'vi': 'Ảnh gốc'},
            'difference': {'en': 'One issue', 'vi': 'Một điểm khác biệt'},
        }],
        'counts': {'core_equivalent': 0, 'partial_equivalent': 1,
                   'material_divergence': 0},
    }
    review_file.write_text(json.dumps(review))
    marked = tmp_path / 'marked'
    build_gallery(en, EXAMPLES / 'images', marked, vi_path, MANIFEST, review_file)
    html = (marked / 'index.html').read_text()
    assert 'qwen38-selected-review-v1' in html
    assert 'const AUDIT=null;' not in html
    assert (marked / 'semantic-review.v1.json').is_file()
    assert (marked / 'manual-comparison.vi.md').is_file()
    assert 'không bao phủ toàn bộ 17 case' in (marked / 'manual-comparison.vi.md').read_text()
    assert len(json.loads((marked / 'evidence.json').read_text())['cases']) == 19
    assert sha256(marked / 'evidence.json') == source_hash
    assert len(list((marked / 'assets').iterdir())) == 16
    # Prevent annotations from following unrelated inference source.
    review['source']['en_sha256'] = '0' * 64
    review_file.write_text(json.dumps(review))
    with pytest.raises(ValueError, match='source SHA256 mismatch'):
        build_gallery(en, EXAMPLES / 'images', tmp_path / 'reject', vi_path, MANIFEST, review_file)
    assert not (tmp_path / 'reject').exists()


def test_notebook_gallery_exact_source_parity():
    from support.sync_gallery_notebook import sync
    sync(write=False)
    notebook = json.loads((ROOT / 'notebooks/kaggle-tpu-v5e8-production-showcase.ipynb').read_text())
    assert ''.join(notebook['cells'][32]['source']) == "# Gallery\nworkflow.run('gallery')\n"
    source = ''.join(notebook['cells'][2]['source'])
    assert 'DEFAULT_ENABLE_VI = "1"' in source
    assert 'DEFAULT_RENDER_HTML = "1"' in source

def test_gallery_accepts_full_case_inventory_with_eos_review_flags(tmp_path):
    # All 19 input cases were attempted. A missing EOS is a review finding,
    # not a reason to silently drop a valid raw model response.
    en, rows = fixture_results(tmp_path)
    rows[0]["eos_reached"] = False
    rows[1]["semantic_check"] = False
    en.write_text(json.dumps({
        "status": "AUTOMATED_19_CASES_RECORDED_NEEDS_REVIEW",
        "review_flags": [{"id": rows[0]["id"], "reason": "NO_EOS"},
                         {"id": rows[1]["id"], "reason": "CONTENT_CHECK"}],
        "cases": rows
    }))
    out = tmp_path / "review-gallery"
    build_gallery(en, EXAMPLES / "images", out, manifest=MANIFEST)
    evidence = json.loads((out / "evidence.json").read_text())
    assert evidence["summary"]["english_cases"] == 19
    assert evidence["summary"]["en_status"] == "AUTOMATED_19_CASES_RECORDED_NEEDS_REVIEW"
    assert evidence["summary"]["en_eos_count"] == 18
    assert evidence["summary"]["en_review_flag_count"] == 2
    assert evidence["summary"]["en_review_flags"] == [
        {"id": rows[0]["id"], "reason": "NO_EOS"},
        {"id": rows[1]["id"], "reason": "CONTENT_CHECK"},
    ]
    first = next(x for x in evidence["cases"] if x["id"] == rows[0]["id"])
    assert first["en"]["eos"] is False
    assert first["en"]["review_flags"] == ["NO_EOS"]
    assert first["en"]["output"] == rows[0]["generated_text"]
    assert evidence["summary"]["vi_generated"] == 0


def test_gallery_rejects_false_claim_of_all_eos(tmp_path):
    en, rows = fixture_results(tmp_path)
    rows[0]["eos_reached"] = False
    en.write_text(json.dumps({
        "status": "AUTOMATED_19_CASES_EOS_COMPLETED",
        "cases": rows
    }))
    with pytest.raises(ValueError, match="contradicts"):
        build_gallery(en, EXAMPLES / "images", tmp_path / "not-built", manifest=MANIFEST)
    assert not (tmp_path / "not-built").exists()


def test_notebook_vi_state_matches_canonical_module():
    import subprocess
    from gallery import vi_state
    notebook = json.loads((ROOT / "notebooks/kaggle-tpu-v5e8-production-showcase.ipynb").read_text())
    cell = "".join(notebook["cells"][30]["source"])
    assert cell == "# Vietnamese\nworkflow.run('vietnamese')\n"
    source = (ROOT / "qwen38_runtime/stages/vietnamese.py").read_text()
    assert "from gallery.vi_state import" in source
    assert "EN_RECORD_STATUSES" in source
    subprocess.run(["python3", "-m", "support.sync_vi_state_notebook", "--check"],
                   cwd=ROOT, check=True, capture_output=True, text=True)
