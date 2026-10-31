"""Durable CPU-only state management for the 17 optional VI cases."""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

VI_SUFFIX = " Trả lời chỉ bằng tiếng Việt, rõ ràng và ngắn gọn."
COUNTS = {"ocr": 2, "rich": 6, "photo": 8, "text": 3}
EN_RECORD_STATUSES = frozenset({
    "AUTOMATED_19_CASES_EOS_COMPLETED",
    "AUTOMATED_19_CASES_RECORDED_NEEDS_REVIEW",
})


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as reader:
        for block in iter(lambda: reader.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verified_source(path):
    en = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = en.get("cases")
    if en.get("status") not in EN_RECORD_STATUSES or not isinstance(rows, list) or len(rows) != 19:
        raise ValueError("Original 19 English cases must finish first")
    by_id = {}
    for record in rows:
        cid = record.get("id")
        if (not isinstance(cid, str) or not cid or cid in by_id
                or not isinstance(record.get("eos_reached"), bool)
                or not isinstance(record.get("generated_text"), str)
                or not isinstance(record.get("generated_token_ids"), list)) :
            raise ValueError("Invalid English case ID, output fields or duplicates")
        by_id[cid] = record
    groups = {k: sum(r.get("group") == k for r in rows) for k in COUNTS}
    if groups != COUNTS:
        raise ValueError("Unexpected English group counts")
    return sha_file(path), by_id


def plan(by_id):
    return [r for group in ("rich", "photo", "text") for r in by_id.values()
            if r["group"] == group]


def successful(row):
    return bool(row.get("eos_reached") and not row.get("error") and
                isinstance(row.get("generated_text"), str) and row["generated_text"].strip() and
                isinstance(row.get("generated_token_ids"), list) and row["generated_token_ids"])


def expected_prompt(en, suffix):
    prompt = en["prompt"]
    end = " Be concise, factual, and avoid speculation."
    if en["group"] != "text" and prompt.endswith(end):
        prompt = prompt[:-len(end)]
    return prompt + suffix


def verify_record(record, by_id, suffix):
    cid = record.get("id") if isinstance(record, dict) else None
    if cid not in by_id:
        raise ValueError("Unknown VI case")
    en = by_id[cid]
    if en["group"] == "ocr" or en["group"] != record.get("group"):
        raise ValueError("Invalid VI case group")
    if record.get("prompt") != expected_prompt(en, suffix):
        raise ValueError("VI prompt changed or does not match EN source")
    if en["group"] != "text" and record.get("image_sha256") != en.get("image_sha256"):
        raise ValueError("VI source image hash does not match EN source")


def load_or_initialize(path, en_path, enabled=False, suffix=VI_SUFFIX):
    digest, by_id = verified_source(en_path)
    path = Path(path)
    if path.exists():
        state = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(state, dict) or state.get("language") != "vi":
            raise ValueError("Unexpected VI state file")
        if state.get("source_full19") != Path(en_path).name:
            raise ValueError("VI state points to another English file")
        known_hash = state.get("source_full19_sha256")
        if known_hash and known_hash != digest:
            raise ValueError("VI and EN input hashes do not match")
        if not known_hash and state.get("cases"):
            raise ValueError("Cannot resume VI cases without EN hash")
        if state.get("prompt_suffix") != suffix:
            raise ValueError("VI prompt version changed")
        if state.get("status") not in ("DISABLED", "RUNNING", "PARTIAL", "COMPLETED"):
            raise ValueError("Invalid VI state status")
        records = state.get("cases")
        if not isinstance(records, list) or len(records) > 17:
            raise ValueError("Invalid VI state records")
        seen = set()
        for record in records:
            verify_record(record, by_id, suffix)
            if record["id"] in seen:
                raise ValueError("Duplicated VI case")
            seen.add(record["id"])
        state["source_full19_sha256"] = digest
    else:
        state = {
            "status": "DISABLED", "language": "vi", "started_utc": utc_now(),
            "source_full19": Path(en_path).name, "source_full19_sha256": digest,
            "english_count": 19, "eligible_count": 17, "shared_numeric_ocr_count": 2,
            "prompt_suffix": suffix, "cases": [], "attempted_count": 0,
            "note": "Original English inference outputs remain unchanged.",
        }
    if enabled:
        state["status"] = "RUNNING"
        state.pop("finished_utc", None)
        state.pop("elapsed_seconds", None)
    elif not state["cases"]:
        state["status"] = "DISABLED"
    state["attempted_count"] = len(state["cases"])
    return state, by_id


def pending_ids(state, by_id, retry_incomplete=False):
    records = {r["id"]: r for r in state["cases"]}
    return [r["id"] for r in plan(by_id) if r["id"] not in records or
            (retry_incomplete and not successful(records[r["id"]]))]


def store_case(state, row, by_id, retry_incomplete=False):
    verify_record(row, by_id, state["prompt_suffix"])
    previous = next((r for r in state["cases"] if r["id"] == row["id"]), None)
    if previous is not None:
        if successful(previous) or not retry_incomplete:
            raise ValueError("Cannot replace previously recorded VI output")
        state["cases"].remove(previous)
        state.setdefault("retry_history", []).append(previous)
    state["cases"].append(row)
    state["attempted_count"] = len(state["cases"])
    state["active_case"] = None


def finalize(state):
    passed = sum(successful(r) for r in state["cases"])
    state["status"] = "COMPLETED" if passed == 17 else "PARTIAL"
    state["successful_count"] = passed
    state["finished_utc"] = utc_now()
    state["elapsed_seconds"] = round((
        datetime.fromisoformat(state["finished_utc"]) -
        datetime.fromisoformat(state["started_utc"])).total_seconds(), 3)
    state["active_case"] = None
    return state


def save(path, state):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".writing")
    with temp.open("w", encoding="utf-8") as writer:
        json.dump(state, writer, ensure_ascii=False, indent=2, default=str)
        writer.write("\n")
        writer.flush()
        os.fsync(writer.fileno())
    os.replace(temp, path)
