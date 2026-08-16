import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from .alignment import gt_detection_map
from .ccl import component_ids_in_box, label_ink


SCHEMA_VERSION = "dtlr.bigram-evidence.v1"


def line_evidence(record: dict, data_root: Path, threshold: int | None = None) -> list[dict]:
    transcription = record["transcription"]
    detections = sorted(record["detections"], key=lambda item: item["box_xyxy"][0])
    mapping = gt_detection_map(transcription, [item["predicted_char"] for item in detections])
    image_path = data_root / record["image_relpath"]
    gray = np.asarray(Image.open(image_path).convert("L"))
    labels, component_count = label_ink(gray, threshold)
    rows = []
    for left_index in range(len(transcription) - 1):
        right_index = left_index + 1
        left_char, right_char = transcription[left_index], transcription[right_index]
        if left_char.isspace() or right_char.isspace():
            continue
        left_alignment, right_alignment = mapping[left_index], mapping[right_index]
        row = {
            "schema_version": SCHEMA_VERSION,
            "dataset": record["dataset"],
            "split": record["split"],
            "line_id": record["line_id"],
            "left_gt_index": left_index,
            "right_gt_index": right_index,
            "left_char": left_char,
            "right_char": right_char,
            "pair": left_char + right_char,
            "left_alignment": left_alignment.operation,
            "right_alignment": right_alignment.operation,
            "usable": False,
            "connected": None,
            "shared_component_count": None,
            "ccl_component_count": component_count,
        }
        if left_alignment.detection_index is not None and right_alignment.detection_index is not None:
            left_box = detections[left_alignment.detection_index]["box_xyxy"]
            right_box = detections[right_alignment.detection_index]["box_xyxy"]
            shared = component_ids_in_box(labels, left_box) & component_ids_in_box(labels, right_box)
            row.update({
                "usable": True,
                "connected": bool(shared),
                "shared_component_count": len(shared),
                "left_score": detections[left_alignment.detection_index]["score"],
                "right_score": detections[right_alignment.detection_index]["score"],
                "left_box_xyxy": left_box,
                "right_box_xyxy": right_box,
            })
        rows.append(row)
    return rows


def write_evidence(rows: list[dict], csv_path: Path, json_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, list) else v for k, v in row.items()})
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def aggregate(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["dataset"], row["split"], row["pair"])].append(row)
    result = []
    for (dataset, split, pair), items in sorted(groups.items()):
        usable = [item for item in items if item["usable"]]
        exact = [item for item in usable if item["left_alignment"] == item["right_alignment"] == "match"]
        result.append({
            "schema_version": "dtlr.bigram-scores.v1",
            "dataset": dataset,
            "split": split,
            "pair": pair,
            "left_char": pair[0],
            "right_char": pair[1],
            "n_total": len(items),
            "n_usable": len(usable),
            "n_exact_alignment": len(exact),
            "connected_rate": (sum(bool(x["connected"]) for x in usable) / len(usable)) if usable else None,
            "exact_alignment_connected_rate": (sum(bool(x["connected"]) for x in exact) / len(exact)) if exact else None,
        })
    return result
