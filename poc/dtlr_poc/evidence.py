import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from .alignment import gt_detection_map
from .ccl import label_ink, pair_component_evidence


SCHEMA_VERSION = "dtlr.bigram-evidence.v4"
CONNECTIVITY_METHOD = "dominant-core-v3"


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
            "connectivity_method": CONNECTIVITY_METHOD,
            "alignment_usable": False,
            "usable": False,
            "connected": None,
            "shared_component_count": None,
            "connected_box_intersection_v1": None,
            "shared_component_count_box_intersection_v1": None,
            "exclusive_core_usable_v2": False,
            "exclusive_core_usable": False,
            "connected_exclusive_core_v2": None,
            "shared_component_count_exclusive_core_v2": None,
            "connected_exclusive_core_v2_1": None,
            "shared_component_count_exclusive_core_v2_1": None,
            "dominant_core_usable": False,
            "connected_dominant_core_v3": None,
            "left_dominant_component_id": None,
            "right_dominant_component_id": None,
            "left_dominant_component_share": None,
            "right_dominant_component_share": None,
            "ccl_component_count": component_count,
        }
        if left_alignment.detection_index is not None and right_alignment.detection_index is not None:
            left_box = detections[left_alignment.detection_index]["box_xyxy"]
            right_box = detections[right_alignment.detection_index]["box_xyxy"]
            component_evidence = pair_component_evidence(labels, left_box, right_box)
            core_usable = component_evidence["exclusive_core_usable"]
            dominant_usable = component_evidence["dominant_core_usable"]
            shared_core = component_evidence["shared_core_components"]
            row.update({
                "alignment_usable": True,
                "usable": dominant_usable,
                "connected": component_evidence["connected_dominant_core_v3"],
                "shared_component_count": (
                    int(component_evidence["connected_dominant_core_v3"])
                    if dominant_usable else None
                ),
                "connected_box_intersection_v1": component_evidence["connected_box_intersection_v1"],
                "shared_component_count_box_intersection_v1": len(component_evidence["shared_full_components"]),
                "exclusive_core_usable_v2": component_evidence["exclusive_core_usable_v2"],
                "exclusive_core_usable": core_usable,
                "connected_exclusive_core_v2": component_evidence["connected_exclusive_core_v2"],
                "shared_component_count_exclusive_core_v2": (
                    len(component_evidence["shared_core_components_v2"])
                    if component_evidence["exclusive_core_usable_v2"] else None
                ),
                "connected_exclusive_core_v2_1": component_evidence["connected_exclusive_core_v2_1"],
                "shared_component_count_exclusive_core_v2_1": len(shared_core) if core_usable else None,
                "dominant_core_usable": dominant_usable,
                "connected_dominant_core_v3": component_evidence["connected_dominant_core_v3"],
                "left_dominant_component_id": component_evidence["left_dominant_component"],
                "right_dominant_component_id": component_evidence["right_dominant_component"],
                "left_dominant_component_share": component_evidence["left_dominant_share"],
                "right_dominant_component_share": component_evidence["right_dominant_share"],
                "left_core_component_pixel_counts": component_evidence["left_core_component_pixel_counts"],
                "right_core_component_pixel_counts": component_evidence["right_core_component_pixel_counts"],
                "left_score": detections[left_alignment.detection_index]["score"],
                "right_score": detections[right_alignment.detection_index]["score"],
                "left_box_xyxy": left_box,
                "right_box_xyxy": right_box,
                "left_exclusive_core_v2_xyxy": component_evidence["left_core_box_v2"],
                "right_exclusive_core_v2_xyxy": component_evidence["right_core_box_v2"],
                "left_exclusive_core_xyxy": component_evidence["left_core_box"],
                "right_exclusive_core_xyxy": component_evidence["right_core_box"],
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
            writer.writerow({
                k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v
                for k, v in row.items()
            })
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def aggregate(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["dataset"], row["split"], row["pair"])].append(row)
    result = []
    for (dataset, split, pair), items in sorted(groups.items()):
        usable = [item for item in items if item["usable"]]
        exact = [item for item in usable if item["left_alignment"] == item["right_alignment"] == "match"]
        v1_usable = [item for item in items if item.get("alignment_usable", item["usable"])]
        v1_v2_1_disagreements = [
            item for item in usable
            if item.get("connected_box_intersection_v1", item["connected"])
            != item.get("connected_exclusive_core_v2_1", item["connected"])
        ]
        v2_v2_1_disagreements = [
            item for item in usable
            if item.get("connected_exclusive_core_v2", item["connected"])
            != item.get("connected_exclusive_core_v2_1", item["connected"])
        ]
        v2_1_v3_disagreements = [
            item for item in usable
            if item.get("connected_exclusive_core_v2_1", item["connected"])
            != item.get("connected_dominant_core_v3", item["connected"])
        ]
        result.append({
            "schema_version": "dtlr.bigram-scores.v4",
            "dataset": dataset,
            "split": split,
            "pair": pair,
            "left_char": pair[0],
            "right_char": pair[1],
            "connectivity_method": CONNECTIVITY_METHOD,
            "n_total": len(items),
            "n_alignment_usable": len(v1_usable),
            "n_usable": len(usable),
            "n_exact_alignment": len(exact),
            "n_v1_v2_1_disagreement": len(v1_v2_1_disagreements),
            "n_v2_v2_1_disagreement": len(v2_v2_1_disagreements),
            "n_v2_1_v3_disagreement": len(v2_1_v3_disagreements),
            "connected_rate": (sum(bool(x["connected"]) for x in usable) / len(usable)) if usable else None,
            "exact_alignment_connected_rate": (sum(bool(x["connected"]) for x in exact) / len(exact)) if exact else None,
            "box_intersection_v1_connected_rate": (
                sum(bool(x.get("connected_box_intersection_v1", x["connected"])) for x in v1_usable)
                / len(v1_usable)
            ) if v1_usable else None,
        })
    return result
