#!/usr/bin/env python3
"""Freeze exactly the audited IAM handwriting BPE/Unigram models; never refit."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "poc"))
from dtlr_poc.subword_freeze import (
    CANDIDATES_SHA256, MODELS, RELEASE_VERSION, SOURCE_SHA256,
    build_frozen_subword_artifact, build_release_manifest, canonical_bytes,
)


def read_pinned(path, expected):
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError(f"pinned SHA-256 differs: {path.name}")
    return json.loads(raw)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=REPO / "outputs/iam-handwriting-subwords-v1")
    parser.add_argument("--output-dir", type=Path, default=REPO / "poc/frozen")
    parser.add_argument("--check", action="store_true", help="verify existing release bytes without writing")
    args = parser.parse_args()
    source = args.source_dir.resolve()
    output = args.output_dir.resolve()
    if not source.is_relative_to(REPO):
        raise ValueError("source snapshots must be inside DTLR")
    if not (output.is_relative_to(REPO / "poc/frozen") or output.is_relative_to(REPO / "outputs")):
        raise ValueError("release output must stay within DTLR/poc/frozen or DTLR/outputs")
    bigram = REPO / "poc/frozen/iam-english-handwriting-bigram-v1.json"
    if hashlib.sha256(bigram.read_bytes()).hexdigest() != SOURCE_SHA256["frozen_bigram"]:
        raise ValueError("existing frozen bigram SHA-256 differs")
    candidates = read_pinned(source / "candidates.json", CANDIDATES_SHA256)
    if len(candidates) != 303:
        raise ValueError("candidate pool count differs")
    artifacts = {}
    for algorithm, item in MODELS.items():
        model = read_pinned(source / item["source_filename"], item["source_model_sha256"])
        if model.get("algorithm") != algorithm:
            raise ValueError("source algorithm differs")
        if any(candidates.get(row["token"]) != row for row in model["vocabulary"]):
            raise ValueError("selected token evidence differs from pinned candidate pool")
        artifacts[algorithm] = build_frozen_subword_artifact(model, item["source_model_sha256"])
    manifest = build_release_manifest(artifacts)
    planned = {model["model_version"] + ".json": canonical_bytes(model) for model in artifacts.values()}
    planned[RELEASE_VERSION + ".manifest.json"] = canonical_bytes(manifest)
    # Check every destination before any writes; preserve unrelated files.
    for filename, raw in planned.items():
        target = output / filename
        if target.exists() and target.read_bytes() != raw:
            raise ValueError(f"refusing to overwrite a different release: {filename}")
        if args.check and not target.exists():
            raise ValueError(f"release file missing: {filename}")
    if not args.check:
        output.mkdir(parents=True, exist_ok=True)
        for filename, raw in planned.items():
            target = output / filename
            if not target.exists():
                with target.open("xb") as handle:
                    handle.write(raw)
    print(json.dumps({"mode": "verified" if args.check else "frozen",
                      "output_dir": str(output), "files": {name: hashlib.sha256(raw).hexdigest()
                      for name, raw in planned.items()}}, indent=2))


if __name__ == "__main__":
    main()
