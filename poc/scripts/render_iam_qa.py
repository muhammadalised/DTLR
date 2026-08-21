#!/usr/bin/env python3
"""Render side-by-side DTLR alignment and CCL overlays for manual QA."""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "poc"))

from dtlr_poc.alignment import gt_detection_map  # noqa: E402
from dtlr_poc.ccl import component_ids_in_box, label_ink, otsu_threshold  # noqa: E402


def component_image(gray: np.ndarray, labels: np.ndarray) -> Image.Image:
    """Return a deterministic color rendering of connected ink components."""
    rgb = np.repeat(gray[:, :, None], 3, axis=2).astype(np.uint16)
    ink = labels > 0
    colors = np.empty((*labels.shape, 3), dtype=np.uint16)
    colors[:, :, 0] = 32 + (labels * 53) % 192
    colors[:, :, 1] = 32 + (labels * 97) % 192
    colors[:, :, 2] = 32 + (labels * 151) % 192
    rgb[ink] = (rgb[ink] + 3 * colors[ink]) // 4
    return Image.fromarray(rgb.astype(np.uint8), mode="RGB")


def add_header(image: Image.Image, text: str, height: int = 42) -> Image.Image:
    canvas = Image.new("RGB", (image.width, image.height + height), "white")
    canvas.paste(image, (0, height))
    ImageDraw.Draw(canvas).text((6, 5), text, fill="black")
    return canvas


def shifted_box(box: list[float], y_offset: int) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    return round(x0), round(y0) + y_offset, round(x1), round(y1) + y_offset


def render_record(record: dict, data_root: Path, output: Path, fixed_threshold: int | None) -> dict:
    image_path = data_root / record["image_relpath"]
    original = Image.open(image_path).convert("RGB")
    gray = np.asarray(original.convert("L"))
    threshold = otsu_threshold(gray) if fixed_threshold is None else fixed_threshold
    labels, component_count = label_ink(gray, threshold)
    detections = sorted(record["detections"], key=lambda item: item["box_xyxy"][0])
    mapping = gt_detection_map(record["transcription"], [item["predicted_char"] for item in detections])
    detection_to_gt = {
        item.detection_index: (gt_index, item)
        for gt_index, item in mapping.items()
        if item.detection_index is not None
    }

    left = add_header(original, "Boxes: green=match, amber=substitution, blue=extra")
    right = add_header(component_image(gray, labels), "CCL: green link=shared component, red link=not shared")
    left_draw, right_draw = ImageDraw.Draw(left), ImageDraw.Draw(right)
    header = 42
    operation_counts = Counter(item.operation for item in mapping.values())

    for index, detection in enumerate(detections):
        mapped = detection_to_gt.get(index)
        color = "#31823f" if mapped and mapped[1].operation == "match" else "#d88900" if mapped else "#1976d2"
        box = shifted_box(detection["box_xyxy"], header)
        left_draw.rectangle(box, outline=color, width=2)
        right_draw.rectangle(box, outline=color, width=2)
        if mapped:
            gt_index, alignment = mapped
            gt = record["transcription"][gt_index]
            label = f"{gt}/{detection['predicted_char']} {alignment.operation[0]}"
        else:
            label = f"-/{detection['predicted_char']} i"
        left_draw.text((box[0], max(header, box[1] - 11)), label, fill=color)

    usable_pairs = connected_pairs = 0
    for gt_index in range(len(record["transcription"]) - 1):
        if record["transcription"][gt_index].isspace() or record["transcription"][gt_index + 1].isspace():
            continue
        first, second = mapping[gt_index], mapping[gt_index + 1]
        if first.detection_index is None or second.detection_index is None:
            continue
        first_box = detections[first.detection_index]["box_xyxy"]
        second_box = detections[second.detection_index]["box_xyxy"]
        shared = component_ids_in_box(labels, first_box) & component_ids_in_box(labels, second_box)
        usable_pairs += 1
        connected_pairs += bool(shared)
        x0 = round((first_box[0] + first_box[2]) / 2)
        y0 = round((first_box[1] + first_box[3]) / 2) + header
        x1 = round((second_box[0] + second_box[2]) / 2)
        y1 = round((second_box[1] + second_box[3]) / 2) + header
        right_draw.line((x0, y0, x1, y1), fill="#00a02b" if shared else "#d02020", width=2)

    combined = Image.new("RGB", (left.width + right.width, max(left.height, right.height)), "white")
    combined.paste(left, (0, 0))
    combined.paste(right, (left.width, 0))
    combined.save(output)
    return {
        "line_id": record["line_id"],
        "image": output.name,
        "transcription": record["transcription"],
        "detection_count": len(detections),
        "alignment_operations": dict(sorted(operation_counts.items())),
        "ccl_threshold": threshold,
        "ccl_component_count": component_count,
        "usable_pair_count": usable_pairs,
        "connected_pair_count": connected_pairs,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--detections", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ink-threshold", type=int)
    args = parser.parse_args()
    records = [json.loads(line) for line in args.detections.read_text(encoding="utf-8").splitlines() if line]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        render_record(record, args.data_root, args.output_dir / f"{record['line_id']}.qa.png", args.ink_threshold)
        for record in records
    ]
    manifest = {
        "schema_version": "dtlr.qa-manifest.v1",
        "source": str(args.detections),
        "line_count": len(lines),
        "ink_threshold": args.ink_threshold if args.ink_threshold is not None else "otsu-per-line",
        "status": "pending-manual-review",
        "lines": lines,
    }
    (args.output_dir / "qa_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in manifest.items() if key != "lines"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
