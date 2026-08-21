#!/usr/bin/env python3
import argparse
import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "poc"))

from dtlr_poc.evidence import aggregate, line_evidence, write_evidence  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--detections", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ink-threshold", type=int)
    args = parser.parse_args()
    records = [json.loads(line) for line in args.detections.read_text(encoding="utf-8").splitlines() if line]
    rows = [row for record in records for row in line_evidence(record, args.data_root, args.ink_threshold)]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_evidence(rows, args.output_dir / "bigram_evidence.csv", args.output_dir / "bigram_evidence.json")
    scores = aggregate(rows)
    write_evidence(scores, args.output_dir / "bigram_scores.csv", args.output_dir / "bigram_scores.json")
    manifest = {
        "schema_version": "dtlr.export-manifest.v3",
        "source": str(args.detections),
        "datasets": sorted({r["dataset"] for r in records}),
        "splits": sorted({r["split"] for r in records}),
        "line_count": len(records),
        "evidence_count": len(rows),
        "score_count": len(scores),
        "aggregation_keys": ["dataset", "split", "pair"],
        "primary_connectivity_method": "exclusive-core-v2.1",
        "retained_comparison_methods": ["box-intersection-v1", "exclusive-core-v2"],
        "v1_v2_1_disagreement_count": sum(
            row["usable"]
            and row["connected_box_intersection_v1"] != row["connected_exclusive_core_v2_1"]
            for row in rows
        ),
        "v2_v2_1_disagreement_count": sum(
            row["usable"]
            and row["connected_exclusive_core_v2"] != row["connected_exclusive_core_v2_1"]
            for row in rows
        ),
        "exclusive_core_unusable_count": sum(
            row["alignment_usable"] and not row["exclusive_core_usable"] for row in rows
        ),
        "note": "Scores remain split-specific; TVA must not fit thresholds on held-out splits.",
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
