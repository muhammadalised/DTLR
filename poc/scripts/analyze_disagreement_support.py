#!/usr/bin/env python3
"""Analyze shared-component support for the frozen v2.1/v3 disagreements."""

import argparse
import csv
import html
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "poc"))

from dtlr_poc.alignment import gt_detection_map  # noqa: E402
from dtlr_poc.ccl import label_ink, otsu_threshold, pair_component_evidence  # noqa: E402
from dtlr_poc.review import summarize_review  # noqa: E402
from dtlr_poc.selection import sha256_file  # noqa: E402
from dtlr_poc.support_analysis import (  # noqa: E402
    best_exploratory_candidate,
    distribution,
    shared_component_support,
    threshold_candidates,
)


ROW_FIELDS = [
    "pair_id", "line_id", "left_gt_index", "right_gt_index", "pair",
    "manual_alignment", "manual_visual_connectivity", "manual_v3_assessment",
    "evaluable", "otsu_threshold", "shared_component_id", "shared_left_share",
    "shared_right_share", "bidirectional_support", "left_dominant_share",
    "right_dominant_share", "connected_exclusive_core_v2_1",
    "connected_dominant_core_v3", "manual_visual_cause", "notes",
]


def read_jsonl(path: Path) -> dict[str, dict]:
    records = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if record["line_id"] in records:
                raise ValueError(f"duplicate line_id at detections line {line_number}: {record['line_id']}")
            records[record["line_id"]] = record
    return records


def recompute_row(queue_row: dict, annotation: dict, record: dict, data_root: Path) -> dict:
    detections = sorted(record["detections"], key=lambda item: item["box_xyxy"][0])
    mapping = gt_detection_map(record["transcription"], [item["predicted_char"] for item in detections])
    left_index, right_index = queue_row["left_gt_index"], queue_row["right_gt_index"]
    left_alignment, right_alignment = mapping[left_index], mapping[right_index]
    if left_alignment.detection_index is None or right_alignment.detection_index is None:
        raise ValueError(f"disagreement unexpectedly has a missing detection: {queue_row['pair_id']}")
    left_box = detections[left_alignment.detection_index]["box_xyxy"]
    right_box = detections[right_alignment.detection_index]["box_xyxy"]
    gray = np.asarray(Image.open(data_root / record["image_relpath"]).convert("L"))
    threshold = otsu_threshold(gray)
    labels, _ = label_ink(gray, threshold)
    evidence = pair_component_evidence(labels, left_box, right_box)
    for field in ("connected_exclusive_core_v2_1", "connected_dominant_core_v3"):
        if evidence[field] != queue_row[field]:
            raise ValueError(
                f"recomputed {field} does not match frozen queue for {queue_row['pair_id']}"
            )
    support = shared_component_support(evidence)
    evaluable = (
        annotation["alignment"] == "correct"
        and annotation["visual_connectivity"] in ("connected", "disconnected")
        and annotation["v3_assessment"] in ("correct", "incorrect")
    )
    return {
        "pair_id": queue_row["pair_id"],
        "line_id": queue_row["line_id"],
        "left_gt_index": left_index,
        "right_gt_index": right_index,
        "pair": queue_row["pair"],
        "manual_alignment": annotation["alignment"],
        "manual_visual_connectivity": annotation["visual_connectivity"],
        "manual_v3_assessment": annotation["v3_assessment"],
        "evaluable": evaluable,
        "otsu_threshold": threshold,
        "shared_component_id": support["component_id"],
        "shared_left_share": support["left_share"],
        "shared_right_share": support["right_share"],
        "bidirectional_support": support["bidirectional_support"],
        "left_dominant_share": evidence["left_dominant_share"],
        "right_dominant_share": evidence["right_dominant_share"],
        "connected_exclusive_core_v2_1": evidence["connected_exclusive_core_v2_1"],
        "connected_dominant_core_v3": evidence["connected_dominant_core_v3"],
        "manual_visual_cause": annotation.get("visual_cause", ""),
        "notes": annotation.get("notes", ""),
    }


def write_csv(rows: list[dict], path: Path, fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def display(value) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_html(summary: dict, path: Path) -> None:
    candidate = summary["exploratory_best_threshold_candidate"]
    candidate_text = "No evaluable rows" if candidate is None else ", ".join(
        f"{key}={display(candidate[key])}" for key in (
            "threshold", "true_connected", "false_connected", "true_disconnected",
            "false_disconnected", "precision", "recall", "balanced_accuracy"
        )
    )
    rows = "".join(
        "<tr>" + "".join(f"<td>{html.escape(display(row[key]))}</td>" for key in (
            "pair_id", "pair", "manual_alignment", "manual_visual_connectivity",
            "bidirectional_support", "shared_left_share", "shared_right_share"
        )) + "</tr>"
        for row in sorted(summary["rows"], key=lambda item: (-item["bidirectional_support"], item["pair_id"]))
    )
    path.write_text(f"""<!doctype html><meta charset="utf-8"><title>IAM disagreement support analysis</title>
<style>body{{font:14px system-ui,sans-serif;margin:20px}}table{{border-collapse:collapse}}th,td{{border:1px solid #bbb;padding:4px 8px}}.warning{{background:#fff3cd;padding:10px}}</style>
<h1>IAM v2.1/v3 disagreement support analysis</h1>
<p class="warning">Exploratory validation analysis only. The selected cutoff is not a frozen method and the two ad-hoc cases are excluded.</p>
<p>Frozen disagreements: {summary['disagreement_count']}; evaluable: {summary['evaluable_count']}.</p>
<p>Connected support: {html.escape(json.dumps(summary['support_distributions']['connected']))}</p>
<p>Disconnected support: {html.escape(json.dumps(summary['support_distributions']['disconnected']))}</p>
<p>Best exploratory candidate: {html.escape(candidate_text)}</p>
<table><thead><tr><th>pair ID</th><th>pair</th><th>alignment</th><th>manual connectivity</th><th>bidirectional support</th><th>left share</th><th>right share</th></tr></thead><tbody>{rows}</tbody></table>
""", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--detections", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--manual-review", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    queue = json.loads(args.queue.read_text(encoding="utf-8"))
    review = json.loads(args.manual_review.read_text(encoding="utf-8"))
    validation = summarize_review(queue, review)
    if validation["status"] != "complete":
        raise ValueError("manual review is incomplete or internally inconsistent")
    disagreements = [row for row in queue["rows"] if row["queue_group"] == "v2.1-v3-disagreement"]
    records = read_jsonl(args.detections)
    annotations = review["annotations"]
    rows = []
    for queue_row in disagreements:
        if queue_row["line_id"] not in records:
            raise ValueError(f"line absent from detections: {queue_row['line_id']}")
        rows.append(recompute_row(queue_row, annotations[queue_row["pair_id"]], records[queue_row["line_id"]], args.data_root))
    evaluable = [row for row in rows if row["evaluable"]]
    connected = [row["bidirectional_support"] for row in evaluable if row["manual_visual_connectivity"] == "connected"]
    disconnected = [row["bidirectional_support"] for row in evaluable if row["manual_visual_connectivity"] == "disconnected"]
    candidates = threshold_candidates(evaluable)
    summary = {
        "schema_version": "dtlr.disagreement-support-analysis.v1",
        "analysis_scope": "frozen v2.1-v3 disagreements only; ad-hoc discoveries excluded",
        "status": "exploratory-not-frozen",
        "queue_sha256": sha256_file(args.queue),
        "manual_review_sha256": sha256_file(args.manual_review),
        "detections_sha256": sha256_file(args.detections),
        "score_definition": "maximum across shared components of min(left_core_share, right_core_share)",
        "disagreement_count": len(rows),
        "evaluable_count": len(evaluable),
        "excluded_count": len(rows) - len(evaluable),
        "excluded_reason_counts": dict(sorted(Counter(
            "alignment-incorrect" if row["manual_alignment"] == "incorrect"
            else "alignment-uncertain" if row["manual_alignment"] == "uncertain"
            else "visual-connectivity-uncertain" if row["manual_visual_connectivity"] == "uncertain"
            else "v3-assessment-uncertain"
            for row in rows if not row["evaluable"]
        ).items())),
        "label_counts": dict(sorted(Counter(row["manual_visual_connectivity"] for row in evaluable).items())),
        "support_distributions": {
            "connected": distribution(connected),
            "disconnected": distribution(disconnected),
        },
        "exploratory_best_threshold_candidate": best_exploratory_candidate(candidates),
        "note": "Candidate metrics are development diagnostics, not held-out performance and not authorization to change v3.",
        "rows": rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "disagreement_support_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_csv(rows, args.output_dir / "disagreement_support_rows.csv", ROW_FIELDS)
    write_csv(candidates, args.output_dir / "threshold_candidates.csv", list(candidates[0]) if candidates else ["threshold"])
    write_html(summary, args.output_dir / "index.html")
    print(json.dumps({key: value for key, value in summary.items() if key != "rows"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
