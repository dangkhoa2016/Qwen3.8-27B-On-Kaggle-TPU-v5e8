"""Check the published image input and its provenance without conflating it with inference outcomes."""
import hashlib
import json
import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_crossing_image_provenance_and_case_definition():
    record = json.loads((ROOT / "examples/image_provenance.json").read_text())
    assets = json.loads((ROOT / "examples/assets_manifest.json").read_text())["assets"]
    cases = json.loads((ROOT / "examples/photo_showcase_cases.json").read_text())["cases"]
    item = next(a for a in assets if a["path"] == record["image_path"])
    file = ROOT / record["image_path"]
    data = file.read_bytes()
    assert hashlib.sha256(data).hexdigest() == record["sha256"] == item["sha256"]
    assert len(data) == item["bytes"]
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    assert list(struct.unpack(">II", data[16:24])) == record["dimensions"] == item["dimensions"]
    assert sum(x["id"] == record["case_id"] and
               x["image"] == file.name for x in cases) == 1
