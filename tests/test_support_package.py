"""Organized Kaggle support distribution: deterministic, verified ZIP packages."""
import hashlib
import json
import zipfile
from pathlib import Path

import pytest
from support.package_dataset import GROUPS, build, digest

ROOT=Path(__file__).resolve().parents[1]


def test_organized_manifest_paths_and_assets():
    m=json.loads((ROOT/'support/portable-bundle-manifest.json').read_text())
    assert len(m['files'])==48
    assert m['archive_count']==len(m['archives'])==len(GROUPS)==7
    assert set(m['archives'])=={g+'.zip' for g in GROUPS}
    assert sum(x['file_count'] for x in m['archives'].values())==48
    assets=json.loads((ROOT/'examples/assets_manifest.json').read_text())['assets']
    assert len(assets)==m['image_count']==16
    for a in assets:
        entry=m['files']['images/'+Path(a['path']).name]
        assert entry['sha256']==a['sha256'] and entry['bytes']==a['bytes']


def test_packager_rejects_incorrect_hash_before_writing(tmp_path):
    source=tmp_path/'flat'
    source.mkdir()
    (source/'portable-bundle-manifest.json').write_text(json.dumps({
        'required_file_count':48,
        'files':{'a.whl':{'bytes':2,'sha256':'incorrect'}}
    }))
    (source/'a.whl').write_bytes(b'ab')
    with pytest.raises(AssertionError):
        build(source,tmp_path/'output')


def test_all_seven_archives_have_expected_folder_structure(tmp_path):
    source=tmp_path/'flat'
    source.mkdir()
    names=[p.split('/')[-1] for p in json.loads((ROOT/'support/portable-bundle-manifest.json').read_text())['files']]
    assert len(set(names))==48
    files={}
    for i,name in enumerate(names):
        data=(f'safe fixture {i} {name}').encode()
        (source/name).write_bytes(data)
        files[name]={'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
    (source/'portable-bundle-manifest.json').write_text(json.dumps({
        'required_file_count':48,'files':files
    }))
    output=tmp_path/'output'
    result=build(source,output)
    assert set(result['archives'])=={g+'.zip' for g in GROUPS}
    for filename,meta in result['archives'].items():
        package=output/filename
        assert meta['sha256']==digest(package)
        with zipfile.ZipFile(package) as z:
            entries=z.namelist()
            assert all('/' not in p for p in entries)
            assert len(entries)==meta['file_count']
            for path in entries:
                assert z.read(path)==(source/path).read_bytes()
    assert len(result['files'])==48
    # New source-only Python bundle is separately pinned, preserving the 48
    # existing support-data payloads and seven data archive hashes.
    code_zip = output / "notebook_source.zip"
    code_manifest = json.loads((output / "notebook-source-manifest.json").read_text())
    assert code_zip.is_file()
    assert code_manifest["file_count"] == len(code_manifest["files"]) == 23
    assert digest(code_zip) == code_manifest["sha256"]
    with zipfile.ZipFile(code_zip) as zipped:
        assert set(zipped.namelist()) == set(code_manifest["files"])
    assert code_manifest["files"]["gallery/render.py"]["bytes"] > 1000
    # Extra third-party notices are sidecars, never counted as payloads.
    full_licenses = [
        "Apache-2.0.txt", "XPROF-2.23.1-LICENSE",
        "DAWN-20250713-BSD-LICENSE",
        "EMSCRIPTEN-4.0.6-LICENSE",
        "IMGUI-1.88-MIT-LICENSE",
        "PERFETTO-53.0-LICENSE",
        "ANGULAR-15.2.9-MIT-LICENSE",
        "CHROMIUM-BSD-3-CLAUSE-LICENSE",
        "LIT-BSD-3-CLAUSE-LICENSE",
        "TSLIB-0BSD-LICENSE",
    ]
    for name in full_licenses:
        target = output / "THIRD_PARTY_LICENSES" / name
        assert target.is_file() and target.stat().st_size > 100
    for name in ("THIRD_PARTY_NOTICES.md", "THIRD_PARTY_NOTICES.vi.md"):
        assert "Material Icons" in (output / name).read_text()
