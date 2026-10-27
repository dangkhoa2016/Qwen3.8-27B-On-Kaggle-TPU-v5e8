"""Release gate: current Git image != archived Full19 / Private support image."""
import hashlib
import json
from pathlib import Path
import struct

ROOT = Path(__file__).resolve().parents[1]


def test_selected_crossing_source_and_historical_evidence_are_distinct():
    policy = json.loads((ROOT / "examples/release_image_review_policy.json").read_text())
    assets = json.loads((ROOT / "examples/assets_manifest.json").read_text())["assets"]
    photos = json.loads((ROOT / "examples/photo_showcase_cases.json").read_text())["cases"]
    historical = policy["current_source"]
    candidate = policy["replacement"]
    matching = [a for a in assets if a["path"] == "examples/images/" + candidate["candidate_filename"]]
    assert len(matching) == 1
    asset = matching[0]
    image = ROOT / asset["path"]
    digest = hashlib.sha256(image.read_bytes()).hexdigest()
    assert digest == asset["sha256"] == candidate["candidate_sha256"]
    assert image.stat().st_size == asset["bytes"] == candidate["candidate_bytes"]
    png_bytes = image.read_bytes()
    assert png_bytes[:8] == b"\x89PNG\r\n\x1a\n"
    assert struct.unpack(">II", png_bytes[16:24]) == tuple(asset["dimensions"]) == (1280, 960)
    assert historical["sha256"] == "bba573de952942ec97ea6a32c352bdbd3770082b570cfe74520b1cd03b087763"
    assert historical["sha256"] != digest
    assert len([c for c in photos if c["id"] == historical["case_id"]
                and c["image"] == candidate["candidate_filename"]]) == 1
    assert policy["development_private_support_dataset"] == "unchanged_old_sha256"
    assert policy["release_allowed"] is False
    assert candidate["status"] == "USER_SELECTED_GIT_CANDIDATE_PENDING_TPU"
    assert policy["release_gates_status"]["fresh_tpu_inference_completed"] == "NO"
