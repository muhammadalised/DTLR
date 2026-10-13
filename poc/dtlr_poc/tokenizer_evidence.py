"""Portable IAM connectivity evidence export; no vocabulary selection or fitting."""
import gzip
import hashlib
import json
from pathlib import Path
import unicodedata

from .selection import text_sha256, sha256_file

SCHEMA = "dtlr.iam-tokenizer-evidence.v1"
SOURCE_SHA256 = {
    "detections": "5e68b4c2ef1db15b5d1776ba1d4b7696ba9289f81dd4dd018598dd7d6aea2b32",
    "evidence": "9c75d6d240127c2e6a6c4dffd83c491beab6a31dc4be204461861c1913740bd3",
    "manifest": "82cb95a14e71ab5b70c868ca1676b3f628552fabac395fbd30039fc7798bf190",
    "scores": "45c86a9f703ebc3684563ec2a840e7c34241aa76e84d692be46cf11bfec27055",
    "selection": "7893dfba4febe6df99cf0bdb0c74fabe7b736d2a6af05033d8638b90455bc1c2",
}


def prepare_corpus(records, rows, selection, manifest):
    if (selection.get("schema_version") != "dtlr.iam-selection.v1"
            or selection.get("dataset") != "IAM" or selection.get("split") != "train"):
        raise ValueError("selection must be IAM train")
    if (manifest.get("schema_version") != "dtlr.export-manifest.v5"
            or manifest.get("datasets") != ["IAM"] or manifest.get("splits") != ["train"]
            or manifest.get("primary_connectivity_method") != "dominant-core-v3"):
        raise ValueError("manifest must declare sole IAM train dominant-core-v3 evidence")
    selected = selection["lines"]
    if (len(selected) != selection["requested_count"] or len(records) != len(selected)
            or len(records) != manifest["line_count"] or len(rows) != manifest["evidence_count"]):
        raise ValueError("selection, line, or evidence counts disagree")
    corpus, by_id = [], {}
    for record, item in zip(records, selected):
        if (record.get("dataset") != "IAM" or record.get("split") != "train"
                or record["line_id"] != item["id"]
                or text_sha256(record["transcription"]) != item["transcription_sha256"]):
            raise ValueError("detection text/order differs from frozen IAM train selection")
        text = record["transcription"]
        if unicodedata.normalize("NFC", text) != text:
            raise ValueError("non-NFC evidence text needs an explicit offset remapping")
        if record["line_id"] in by_id:
            raise ValueError("duplicate line ID")
        line = {"line_id": record["line_id"], "text": text, "edges": {}}
        corpus.append(line)
        by_id[line["line_id"]] = line
    for row in rows:
        if (row.get("schema_version") != "dtlr.bigram-evidence.v5"
                or row.get("dataset") != "IAM" or row.get("split") != "train"
                or row.get("connectivity_method") != "dominant-core-v3"):
            raise ValueError("row is not IAM train dominant-core-v3 evidence")
        if row["line_id"] not in by_id:
            raise ValueError("evidence line is outside frozen selection")
        line = by_id[row["line_id"]]
        start = row["left_gt_index"]
        if (not isinstance(start, int) or start < 0 or start + 1 >= len(line["text"])
                or row["right_gt_index"] != start + 1
                or line["text"][start:start + 2] != row["pair"]
                or any(c.isspace() for c in row["pair"])):
            raise ValueError("evidence offsets/text do not match transcription")
        if start in line["edges"]:
            raise ValueError("duplicate edge evidence")
        usable = (row["usable"] is True
                  and row["left_alignment"] == row["right_alignment"] == "match")
        left, right = row.get("left_dominant_component_id"), row.get("right_dominant_component_id")
        if usable and (type(left) is not int or left <= 0 or type(right) is not int or right <= 0
                       or row["connected"] is not (left == right)):
            raise ValueError("usable edge has inconsistent dominant component evidence")
        line["edges"][start] = (left, right) if usable else None
    for line in corpus:
        expected = {i for i in range(len(line["text"]) - 1)
                    if not any(c.isspace() for c in line["text"][i:i + 2])}
        if set(line["edges"]) != expected:
            raise ValueError("incomplete per-line non-whitespace edge evidence")
    return corpus


def build_export(records, rows, selection, manifest, scores, source_sha256):
    """Retain text, offsets, abstentions and component IDs, without selecting tokens."""
    corpus = prepare_corpus(records, rows, selection, manifest)
    counts = {}
    for row in rows:
        item = counts.setdefault(row["pair"], [0, 0, 0])
        item[0] += 1
        if row["usable"] and row["left_alignment"] == row["right_alignment"] == "match":
            item[1] += 1
            item[2] += int(row["connected"])
    if len(scores) != manifest["score_count"] or set(counts) != {r["pair"] for r in scores}:
        raise ValueError("score inventory differs from occurrence evidence")
    for row in scores:
        if (row.get("schema_version") != "dtlr.bigram-scores.v5"
                or row.get("dataset") != "IAM" or row.get("split") != "train"
                or row.get("connectivity_method") != "dominant-core-v3"):
            raise ValueError("scores must be IAM train dominant-core-v3")
        total, exact, connected = counts[row["pair"]]
        rate = connected / exact if exact else None
        if (row["n_total"] != total or row["n_exact_alignment"] != exact
                or row["exact_alignment_connected_rate"] != rate):
            raise ValueError("primary aggregates differ from occurrence evidence")
    return {
        "schema_version": SCHEMA, "dataset": "IAM", "split": "train",
        "connectivity_method": "dominant-core-v3", "unicode_normalization": "NFC",
        "source_sha256": source_sha256, "selection": selection,
        "manifest": manifest, "scores": scores,
        "lines": [{"line_id": line["line_id"], "text": line["text"],
                   "edges": [[i, *(edge if edge is not None else (None, None))]
                             for i, edge in sorted(line["edges"].items())]}
                  for line in corpus],
        "contract": {
            "edge_fields": ["left_text_offset", "left_component_id", "right_component_id"],
            "null_components": "unusable or non-exact evidence; abstain, never disconnected",
            "component_scope": "IDs are comparable only within the same line image",
            "vocabulary_selection_performed": False,
            "allowed_consumer": "TVA vocabulary training from IAM train evidence only",
        },
    }


def export_iam_evidence(source_directory, output_path):
    source = Path(source_directory)
    evidence = source / "bigrams-dominant-core-v3"
    paths = {"selection": source / "selection.json", "detections": source / "detections.jsonl",
             "manifest": evidence / "manifest.json", "scores": evidence / "bigram_scores.json",
             "evidence": evidence / "bigram_evidence.json"}
    hashes = {name: sha256_file(path) for name, path in paths.items()}
    if hashes != SOURCE_SHA256:
        raise ValueError("IAM v1 input SHA-256 differs; a new evidence release needs a new version")
    manifest = json.loads(paths["manifest"].read_text())
    if Path(manifest["source"]).resolve() != paths["detections"].resolve():
        raise ValueError("manifest detection source differs")
    selection = json.loads(paths["selection"].read_text())
    # Discard detector boxes and scores immediately; only aligned text is exported.
    records = []
    with paths["detections"].open() as stream:
        for raw in stream:
            if raw.strip():
                row = json.loads(raw)
                records.append({key: row[key] for key in ("dataset", "split", "line_id", "transcription")})
    with paths["evidence"].open() as stream:
        rows = json.load(stream)
    scores = json.loads(paths["scores"].read_text())
    bundle = build_export(records, rows, selection, manifest, scores, hashes)
    raw = (json.dumps(bundle, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    packed = gzip.compress(raw, compresslevel=9, mtime=0)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        if output.read_bytes() != packed:
            raise ValueError(f"Existing evidence export differs: {output}")
    else:
        with output.open("xb") as stream:
            stream.write(packed)
    return {"path": str(output), "sha256": hashlib.sha256(packed).hexdigest(),
            "uncompressed_sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(packed), "lines": len(bundle["lines"]),
            "edges": sum(len(line["edges"]) for line in bundle["lines"]),
            "scores": len(scores)}
