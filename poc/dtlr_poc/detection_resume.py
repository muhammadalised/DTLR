"""Validation helpers for safely resuming deterministic detection exports."""

import json
from pathlib import Path


def read_jsonl_for_resume(path: Path) -> list[dict]:
    if not path.exists():
        return []
    content = path.read_text(encoding="utf-8")
    if content and not content.endswith("\n"):
        raise ValueError("existing detection output ends with an incomplete JSONL line")
    records = []
    for line_number, line in enumerate(content.splitlines(), 1):
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSON in existing detection output line {line_number}") from error
    return records


def validate_resume_prefix(records: list[dict], examples: list[dict], expected: dict) -> None:
    """Require existing records to be an exact provenance-compatible prefix."""
    if len(records) > len(examples):
        raise ValueError("existing detection output is longer than the requested selection")
    expected_ids = [example["id"] for example in examples[:len(records)]]
    actual_ids = [record.get("line_id") for record in records]
    if actual_ids != expected_ids:
        raise ValueError("existing detection line IDs are not an exact selection prefix")
    if len(set(actual_ids)) != len(actual_ids):
        raise ValueError("existing detection output contains duplicate line IDs")
    for index, (record, example) in enumerate(zip(records, examples), 1):
        if record.get("transcription") != example["text"]:
            raise ValueError(f"transcription mismatch in existing detection line {index}")
        for key, value in expected.items():
            if record.get(key) != value:
                raise ValueError(f"provenance mismatch for {key!r} in existing detection line {index}")
