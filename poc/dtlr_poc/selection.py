"""Deterministic, outcome-blind IAM sample selection and verification."""

import hashlib
import json
import pickle
from pathlib import Path


SELECTION_RULE = "lowest-sha256-of-seed-colon-line-id-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_selection(labels_path: Path, split: str, count: int, seed: str) -> dict:
    if count <= 0:
        raise ValueError("count must be positive")
    labels = pickle.loads(labels_path.read_bytes())
    examples = labels["ground_truth"][split]
    if count > len(examples):
        raise ValueError(f"count {count} exceeds {split} split size {len(examples)}")

    ranked = sorted(
        examples,
        key=lambda row: (hashlib.sha256(f"{seed}:{row['id']}".encode()).digest(), row["id"]),
    )
    selected = ranked[:count]
    return {
        "schema_version": "dtlr.iam-selection.v1",
        "dataset": "IAM",
        "split": split,
        "selection_rule": SELECTION_RULE,
        "seed": seed,
        "requested_count": count,
        "labels_sha256": sha256_file(labels_path),
        "lines": [
            {"id": row["id"], "transcription_sha256": text_sha256(row["text"])}
            for row in selected
        ],
    }


def load_selected_examples(selection_path: Path, labels_path: Path, split: str) -> tuple[list[dict], dict]:
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection.get("schema_version") != "dtlr.iam-selection.v1":
        raise ValueError("unsupported IAM selection schema")
    if selection.get("dataset") != "IAM" or selection.get("split") != split:
        raise ValueError("selection dataset/split does not match the requested IAM split")
    if selection.get("labels_sha256") != sha256_file(labels_path):
        raise ValueError("labels.pkl hash differs from the file used to freeze the selection")

    labels = pickle.loads(labels_path.read_bytes())
    by_id = {row["id"]: row for row in labels["ground_truth"][split]}
    examples = []
    for selected in selection.get("lines", []):
        line_id = selected["id"]
        if line_id not in by_id:
            raise ValueError(f"selected line is absent from labels.pkl: {line_id}")
        row = by_id[line_id]
        if selected.get("transcription_sha256") != text_sha256(row["text"]):
            raise ValueError(f"transcription changed for selected line: {line_id}")
        examples.append(row)
    if len(examples) != selection.get("requested_count"):
        raise ValueError("selection line count differs from requested_count")
    return examples, selection
