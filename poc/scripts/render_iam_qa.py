#!/usr/bin/env python3
"""Render side-by-side DTLR alignment and CCL overlays for manual QA."""
import argparse
import html
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "poc"))

from dtlr_poc.alignment import gt_detection_map  # noqa: E402
from dtlr_poc.ccl import label_ink, otsu_threshold, pair_component_evidence  # noqa: E402


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


def diagnostic_component_image(
    gray: np.ndarray,
    labels: np.ndarray,
    left_components: set[int],
    right_components: set[int],
) -> Image.Image:
    """Dim unrelated ink and highlight the components relevant to one pair."""
    rgb = np.full((*gray.shape, 3), 255, dtype=np.uint8)
    rgb[labels > 0] = (205, 205, 205)
    shared = left_components & right_components
    if shared:
        rgb[np.isin(labels, list(shared))] = (210, 0, 145)
    else:
        rgb[np.isin(labels, list(left_components))] = (0, 119, 187)
        rgb[np.isin(labels, list(right_components))] = (238, 119, 51)
    return Image.fromarray(rgb, mode="RGB")


def render_pair_crops(
    record: dict,
    data_root: Path,
    output_dir: Path,
    fixed_threshold: int | None,
) -> list[dict]:
    image_path = data_root / record["image_relpath"]
    original = Image.open(image_path).convert("RGB")
    gray = np.asarray(original.convert("L"))
    threshold = otsu_threshold(gray) if fixed_threshold is None else fixed_threshold
    labels, _ = label_ink(gray, threshold)
    detections = sorted(record["detections"], key=lambda item: item["box_xyxy"][0])
    mapping = gt_detection_map(record["transcription"], [item["predicted_char"] for item in detections])
    line_dir = output_dir / "pairs" / record["line_id"]
    line_dir.mkdir(parents=True, exist_ok=True)
    results = []

    for left_index in range(len(record["transcription"]) - 1):
        right_index = left_index + 1
        left_char = record["transcription"][left_index]
        right_char = record["transcription"][right_index]
        if left_char.isspace() or right_char.isspace():
            continue
        left_alignment, right_alignment = mapping[left_index], mapping[right_index]
        item = {
            "left_gt_index": left_index,
            "right_gt_index": right_index,
            "pair": left_char + right_char,
            "left_alignment": left_alignment.operation,
            "right_alignment": right_alignment.operation,
            "connectivity_method": "dominant-core-v3",
            "usable": False,
            "connected": None,
            "shared_component_count": None,
            "connected_box_intersection_v1": None,
            "connected_exclusive_core_v2": None,
            "connected_exclusive_core_v2_1": None,
            "connected_dominant_core_v3": None,
            "image": None,
        }
        if left_alignment.detection_index is None or right_alignment.detection_index is None:
            results.append(item)
            continue

        left_box = detections[left_alignment.detection_index]["box_xyxy"]
        right_box = detections[right_alignment.detection_index]["box_xyxy"]
        component_evidence = pair_component_evidence(labels, left_box, right_box)
        left_components = (
            {component_evidence["left_dominant_component"]}
            if component_evidence["left_dominant_component"] is not None else set()
        )
        right_components = (
            {component_evidence["right_dominant_component"]}
            if component_evidence["right_dominant_component"] is not None else set()
        )
        shared = component_evidence["shared_core_components"]
        dominant_usable = component_evidence["dominant_core_usable"]
        pad = 12
        crop_left = max(0, int(np.floor(min(left_box[0], right_box[0]))) - pad)
        crop_top = max(0, int(np.floor(min(left_box[1], right_box[1]))) - pad)
        crop_right = min(original.width, int(np.ceil(max(left_box[2], right_box[2]))) + pad)
        crop_bottom = min(original.height, int(np.ceil(max(left_box[3], right_box[3]))) + pad)
        crop_bounds = (crop_left, crop_top, crop_right, crop_bottom)
        original_crop = original.crop(crop_bounds)
        diagnostic = diagnostic_component_image(gray, labels, left_components, right_components).crop(crop_bounds)

        header = 48
        panel_width = max(320, original_crop.width)
        canvas = Image.new("RGB", (panel_width * 2, original_crop.height + header), "white")
        left_offset = (panel_width - original_crop.width) // 2
        right_offset = panel_width + (panel_width - diagnostic.width) // 2
        canvas.paste(original_crop, (left_offset, header))
        canvas.paste(diagnostic, (right_offset, header))
        draw = ImageDraw.Draw(canvas)
        left_share = component_evidence["left_dominant_share"]
        right_share = component_evidence["right_dominant_share"]
        support = (
            f"support={left_share:.2f}/{right_share:.2f}"
            if left_share is not None and right_share is not None else "support=ambiguous"
        )
        title = (
            f"{record['line_id']} [{left_index}:{right_index}] {left_char + right_char!r}  "
            f"{left_alignment.operation}/{right_alignment.operation}  "
            f"dominant-v3={component_evidence['connected_dominant_core_v3']} "
            f"core-v2.1={component_evidence['connected_exclusive_core_v2_1']} "
            f"{support}"
        )
        draw.text((6, 4), title, fill="black")
        draw.text((6, 22), "original: complete DTLR boxes", fill="black")
        draw.text((panel_width + 6, 22), "diagnostic: unique dominant core components", fill="black")
        for panel_offset, boxes in (
            (left_offset, (left_box, right_box)),
            (right_offset, (component_evidence["left_core_box"], component_evidence["right_core_box"])),
        ):
            for box, color in zip(boxes, ("#0077bb", "#ee7733")):
                translated = (
                    round(box[0]) - crop_left + panel_offset,
                    round(box[1]) - crop_top + header,
                    round(box[2]) - crop_left + panel_offset,
                    round(box[3]) - crop_top + header,
                )
                draw.rectangle(translated, outline=color, width=2)

        filename = f"pair_{left_index:04d}.png"
        canvas.save(line_dir / filename)
        item.update({
            "usable": dominant_usable,
            "connected": component_evidence["connected_dominant_core_v3"],
            "shared_component_count": (
                int(component_evidence["connected_dominant_core_v3"])
                if dominant_usable else None
            ),
            "connected_box_intersection_v1": component_evidence["connected_box_intersection_v1"],
            "connected_exclusive_core_v2": component_evidence["connected_exclusive_core_v2"],
            "connected_exclusive_core_v2_1": component_evidence["connected_exclusive_core_v2_1"],
            "connected_dominant_core_v3": component_evidence["connected_dominant_core_v3"],
            "exclusive_core_usable": component_evidence["exclusive_core_usable"],
            "dominant_core_usable": dominant_usable,
            "shared_component_count_box_intersection_v1": len(component_evidence["shared_full_components"]),
            "shared_component_count_exclusive_core_v2": (
                len(component_evidence["shared_core_components_v2"])
                if component_evidence["exclusive_core_usable_v2"] else None
            ),
            "shared_component_count_exclusive_core_v2_1": (
                len(shared) if component_evidence["exclusive_core_usable"] else None
            ),
            "left_component_count": len(component_evidence["left_core_components"]),
            "right_component_count": len(component_evidence["right_core_components"]),
            "left_dominant_component_id": component_evidence["left_dominant_component"],
            "right_dominant_component_id": component_evidence["right_dominant_component"],
            "left_dominant_component_share": component_evidence["left_dominant_share"],
            "right_dominant_component_share": component_evidence["right_dominant_share"],
            "left_exclusive_core_v2_xyxy": component_evidence["left_core_box_v2"],
            "right_exclusive_core_v2_xyxy": component_evidence["right_core_box_v2"],
            "left_exclusive_core_xyxy": component_evidence["left_core_box"],
            "right_exclusive_core_xyxy": component_evidence["right_core_box"],
            "image": str(Path("pairs") / record["line_id"] / filename),
        })
        results.append(item)
    return results


def write_review_index(lines: list[dict], output: Path) -> None:
    sections = []
    for line in lines:
        cards = []
        for pair in line["pairs"]:
            state = "unusable" if not pair["usable"] else "connected" if pair["connected"] else "disconnected"
            image = (
                f'<a href="{html.escape(pair["image"])}"><img loading="lazy" src="{html.escape(pair["image"])}"></a>'
                if pair["image"] else "<p>No pair image: one or both detections are missing.</p>"
            )
            cards.append(
                f'<article class="{state}"><h3>{pair["left_gt_index"]}: '
                f'{html.escape(pair["pair"])} — dominant-v3 {state}; '
                f'core-v2.1={pair["connected_exclusive_core_v2_1"]}; '
                f'core-v2={pair["connected_exclusive_core_v2"]}; '
                f'box-v1={pair["connected_box_intersection_v1"]}</h3>{image}</article>'
            )
        sections.append(
            f'<details><summary>{html.escape(line["line_id"])} — '
            f'{len(line["pairs"])} pairs</summary>{"".join(cards)}</details>'
        )
    document = f"""<!doctype html>
<meta charset="utf-8">
<title>IAM pair-level QA</title>
<style>
body {{ font: 14px system-ui, sans-serif; margin: 20px; }}
summary {{ cursor: pointer; font-size: 18px; font-weight: 600; margin: 12px 0; }}
article {{ border-left: 5px solid #888; margin: 12px 0; padding: 4px 10px; }}
article.connected {{ border-color: #b00078; }}
article.disconnected {{ border-color: #666; }}
article.unusable {{ border-color: #d88900; }}
h3 {{ margin: 3px 0; }}
img {{ width: min(100%, 960px); image-rendering: auto; }}
</style>
<h1>IAM pair-level QA</h1>
<p>The primary method is dominant-core-v3. Magenta is the unique largest
component in both raster-safe cores. When disconnected, blue is dominant in the
left core and orange in the right. Rejected core-v2.1, core-v2, and box-v1
results remain for comparison.</p>
{"".join(sections)}
"""
    output.write_text(document, encoding="utf-8")


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
    right = add_header(component_image(gray, labels), "CCL dominant-core-v3: green=connected, red=disconnected")
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

    aligned_pairs = usable_pairs = connected_pairs = disagreements = 0
    for gt_index in range(len(record["transcription"]) - 1):
        if record["transcription"][gt_index].isspace() or record["transcription"][gt_index + 1].isspace():
            continue
        first, second = mapping[gt_index], mapping[gt_index + 1]
        if first.detection_index is None or second.detection_index is None:
            continue
        first_box = detections[first.detection_index]["box_xyxy"]
        second_box = detections[second.detection_index]["box_xyxy"]
        component_evidence = pair_component_evidence(labels, first_box, second_box)
        aligned_pairs += 1
        if not component_evidence["dominant_core_usable"]:
            continue
        connected = component_evidence["connected_dominant_core_v3"]
        usable_pairs += 1
        connected_pairs += bool(connected)
        disagreements += (
            component_evidence["connected_exclusive_core_v2_1"] != connected
        )
        x0 = round((first_box[0] + first_box[2]) / 2)
        y0 = round((first_box[1] + first_box[3]) / 2) + header
        x1 = round((second_box[0] + second_box[2]) / 2)
        y1 = round((second_box[1] + second_box[3]) / 2) + header
        right_draw.line((x0, y0, x1, y1), fill="#00a02b" if connected else "#d02020", width=2)

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
        "alignment_usable_pair_count": aligned_pairs,
        "usable_pair_count": usable_pairs,
        "connected_pair_count": connected_pairs,
        "v2_1_v3_disagreement_count": disagreements,
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
    lines = []
    for record in records:
        line = render_record(
            record,
            args.data_root,
            args.output_dir / f"{record['line_id']}.qa.png",
            args.ink_threshold,
        )
        line["pairs"] = render_pair_crops(record, args.data_root, args.output_dir, args.ink_threshold)
        lines.append(line)
    write_review_index(lines, args.output_dir / "index.html")
    manifest = {
        "schema_version": "dtlr.qa-manifest.v4",
        "source": str(args.detections),
        "line_count": len(lines),
        "pair_count": sum(len(line["pairs"]) for line in lines),
        "usable_pair_count": sum(pair["usable"] for line in lines for pair in line["pairs"]),
        "primary_connectivity_method": "dominant-core-v3",
        "retained_comparison_methods": [
            "box-intersection-v1", "exclusive-core-v2", "exclusive-core-v2.1"
        ],
        "v2_1_v3_disagreement_count": sum(
            pair["usable"]
            and pair["connected_exclusive_core_v2_1"] != pair["connected_dominant_core_v3"]
            for line in lines for pair in line["pairs"]
        ),
        "ink_threshold": args.ink_threshold if args.ink_threshold is not None else "otsu-per-line",
        "status": "pending-manual-review",
        "review_index": "index.html",
        "lines": lines,
    }
    (args.output_dir / "qa_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in manifest.items() if key != "lines"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
