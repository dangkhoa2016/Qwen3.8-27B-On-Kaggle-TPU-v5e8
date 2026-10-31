#!/usr/bin/env python3
"""Check that the notebook calls the reusable vietnamese source, never embeds its implementation."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
try:
    from .sync_notebook_bootstrap import sync as sync_bootstrap
except ImportError:  # direct CLI execution
    from sync_notebook_bootstrap import sync as sync_bootstrap

ROOT=Path(__file__).resolve().parents[1]
NOTEBOOK=ROOT/"notebooks/kaggle-tpu-v5e8-production-showcase.ipynb"
SOURCE=ROOT/"gallery/vi_state.py"
TARGET=ROOT/"qwen38_runtime/stages/vietnamese.py"
INDEX=30
EXPECTED="# Vietnamese\nworkflow.run('vietnamese')\n"

def sync(write=False):
    assert SOURCE.is_file() and TARGET.is_file()
    assert "from gallery.vi_state import" in TARGET.read_text()
    nb=json.loads(NOTEBOOK.read_text())
    assert len(nb["cells"])==35 and nb["cells"][INDEX]["cell_type"]=="code"
    current="".join(nb["cells"][INDEX]["source"])
    if write and current!=EXPECTED:
        nb["cells"][INDEX]["source"]=EXPECTED.splitlines(keepends=True)
        NOTEBOOK.write_text(json.dumps(nb,ensure_ascii=False,indent=1)+"\n")
    elif not write and current!=EXPECTED:
        raise SystemExit("VI_STATE_NOTEBOOK_PARITY_FAIL")
    sync_bootstrap(write=write)
    print("VIETNAMESE_NOTEBOOK_SOURCE_PASS")

if __name__=="__main__":
    p=argparse.ArgumentParser()
    mode=p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check",action="store_true")
    mode.add_argument("--write",action="store_true")
    args=p.parse_args()
    sync(write=args.write)
