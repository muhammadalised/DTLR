"""READ 2016 label normalization, deterministic selection, and image lookup."""

from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path, PurePosixPath

from .selection import sha256_file, text_sha256


SELECTION_RULE = "lowest-sha256-of-seed-colon-line-id-v1"
TRANSCRIPTION_NORMALIZATION = "remove-read-line-continuation-marker-u00ac-v1"


def normalize_transcription(text: str) -> str:
    """Apply the same explicit READ marker removal used by the upstream loader."""
    return text.replace("¬", "")


def _split_rows(labels: dict, split: str) -> list[dict]:
    ground_truth = labels["ground_truth"][split]
    rows = list(ground_truth.values()) if isinstance(ground_truth, dict) else list(ground_truth)
    return sorted(rows, key=lambda row: (int(row["idx"]), str(row.get("path", ""))))


def canonical_example(row: dict, split: str) -> dict:
    idx = int(row["idx"])
    raw_text = str(row["text"])
    stored_path = str(row.get("path", ""))
    return {
        "id": f"READ-{split}-{idx:06d}",
        "idx": idx,
        "text": normalize_transcription(raw_text),
        "raw_text": raw_text,
        "label_image_relpath": stored_path,
    }


def all_examples(labels_path: Path, split: str) -> list[dict]:
    labels = pickle.loads(labels_path.read_bytes())
    return [canonical_example(row, split) for row in _split_rows(labels, split)]


def build_read_selection(labels_path: Path, split: str, count: int, seed: str) -> dict:
    if count <= 0:
        raise ValueError("count must be positive")
    examples = all_examples(labels_path, split)
    if count > len(examples):
        raise ValueError(f"count {count} exceeds {split} split size {len(examples)}")
    ranked = sorted(
        examples,
        key=lambda row: (hashlib.sha256(f"{seed}:{row['id']}".encode()).digest(), row["id"]),
    )
    selected = ranked[:count]
    return {
        "schema_version": "dtlr.read-selection.v1",
        "dataset": "READ",
        "split": split,
        "selection_rule": SELECTION_RULE,
        "seed": seed,
        "requested_count": count,
        "labels_sha256": sha256_file(labels_path),
        "transcription_normalization": TRANSCRIPTION_NORMALIZATION,
        "lines": [
            {
                "id": row["id"],
                "idx": row["idx"],
                "label_image_relpath": row["label_image_relpath"],
                "raw_transcription_sha256": text_sha256(row["raw_text"]),
                "transcription_sha256": text_sha256(row["text"]),
            }
            for row in selected
        ],
    }


def load_selected_read_examples(
    selection_path: Path, labels_path: Path, split: str
) -> tuple[list[dict], dict]:
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection.get("schema_version") != "dtlr.read-selection.v1":
        raise ValueError("unsupported READ selection schema")
    if selection.get("dataset") != "READ" or selection.get("split") != split:
        raise ValueError("selection dataset/split does not match the requested READ split")
    if selection.get("labels_sha256") != sha256_file(labels_path):
        raise ValueError("labels.pkl hash differs from the file used to freeze the selection")
    if selection.get("transcription_normalization") != TRANSCRIPTION_NORMALIZATION:
        raise ValueError("READ transcription-normalization rule differs from the frozen selection")

    by_id = {row["id"]: row for row in all_examples(labels_path, split)}
    examples = []
    for selected in selection.get("lines", []):
        line_id = selected["id"]
        if line_id not in by_id:
            raise ValueError(f"selected READ line is absent from labels.pkl: {line_id}")
        row = by_id[line_id]
        checks = {
            "idx": row["idx"],
            "label_image_relpath": row["label_image_relpath"],
            "raw_transcription_sha256": text_sha256(row["raw_text"]),
            "transcription_sha256": text_sha256(row["text"]),
        }
        for key, expected in checks.items():
            if selected.get(key) != expected:
                raise ValueError(f"{key} changed for selected READ line: {line_id}")
        examples.append(row)
    if len(examples) != selection.get("requested_count"):
        raise ValueError("selection line count differs from requested_count")
    return examples, selection


def _safe_relative_path(value: str) -> Path:
    posix = PurePosixPath(value)
    if not value or posix.is_absolute() or ".." in posix.parts:
        raise ValueError(f"unsafe or empty READ image path in labels.pkl: {value!r}")
    return Path(*posix.parts)


def resolve_image(data_root: Path, example: dict, split: str) -> tuple[Path, str]:
    """Resolve the archive-recorded layout or the upstream loader's legacy layout."""
    candidates: list[tuple[Path, str]] = []
    if example.get("label_image_relpath"):
        candidates.append((_safe_relative_path(example["label_image_relpath"]), "labels-pkl-path"))
    candidates.append(
        (Path("READ_2016/images") / split / f"{example['idx']}.jpeg", "upstream-index-path")
    )

    existing: list[tuple[Path, str]] = []
    seen: set[Path] = set()
    for relpath, source in candidates:
        if relpath in seen:
            continue
        seen.add(relpath)
        if (data_root / relpath).is_file():
            existing.append((relpath, source))
    if not existing:
        rendered = ", ".join(str(data_root / relpath) for relpath, _ in candidates)
        raise FileNotFoundError(f"READ image not found; checked: {rendered}")
    if len(existing) > 1:
        first_bytes = (data_root / existing[0][0]).read_bytes()
        if any((data_root / relpath).read_bytes() != first_bytes for relpath, _ in existing[1:]):
            rendered = ", ".join(str(data_root / relpath) for relpath, _ in existing)
            raise ValueError(f"ambiguous READ image layouts contain different files: {rendered}")
    return existing[0]
