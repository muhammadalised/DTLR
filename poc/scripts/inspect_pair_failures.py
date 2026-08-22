#!/usr/bin/env python3
"""Render threshold and component diagnostics for an explicit pair selection.

This is an analysis tool. It reports existing v1/v2.1/v3 behavior without
changing connectivity decisions or updating the frozen manual review.
"""

import argparse
import html
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "poc"))

from dtlr_poc.alignment import gt_detection_map  # noqa: E402
from dtlr_poc.ccl import label_ink, otsu_threshold, pair_component_evidence  # noqa: E402
from dtlr_poc.selection import sha256_file  # noqa: E402


COLORS = {
    "left": (0, 119, 187),
    "right": (238, 119, 51),
    "shared": (210, 0, 145),
    "other": (205, 205, 205),
}


def parse_offsets(value: str) -> list[int]:
    try:
        offsets = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError("threshold offsets must be comma-separated integers") from error
    if not offsets:
        raise argparse.ArgumentTypeError("at least one threshold offset is required")
    return offsets


def read_jsonl(path: Path) -> dict[str, dict]:
    records = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            line_id = record["line_id"]
            if line_id in records:
                raise ValueError(f"duplicate line_id {line_id!r} at line {line_number}")
            records[line_id] = record
    return records


def split_pair_id(identifier: str) -> tuple[str, int, int]:
    try:
        line_id, left, right = identifier.rsplit(":", 2)
        left_index, right_index = int(left), int(right)
    except (ValueError, TypeError) as error:
        raise ValueError(f"invalid pair_id {identifier!r}") from error
    if right_index != left_index + 1:
        raise ValueError(f"pair_id is not adjacent: {identifier!r}")
    return line_id, left_index, right_index


def drawable_box(box: list[float], bounds: tuple[int, int, int, int], scale: int) -> tuple[int, int, int, int] | None:
    x0, y0, x1, y1 = box
    if not np.all(np.isfinite(box)) or x1 < x0 or y1 < y0:
        return None
    crop_left, crop_top, _, _ = bounds
    return tuple(round(value * scale) for value in (
        x0 - crop_left, y0 - crop_top, x1 - crop_left, y1 - crop_top
    ))


def add_boxes(image: Image.Image, boxes: tuple[list[float], list[float]], bounds: tuple[int, int, int, int], scale: int) -> Image.Image:
    output = image.copy()
    draw = ImageDraw.Draw(output)
    for box, color in zip(boxes, (COLORS["left"], COLORS["right"])):
        drawable = drawable_box(box, bounds, scale)
        if drawable is not None:
            draw.rectangle(drawable, outline=color, width=max(2, scale))
    return output


def component_panel(gray: np.ndarray, labels: np.ndarray) -> Image.Image:
    rgb = np.repeat(gray[:, :, None], 3, axis=2).astype(np.uint16)
    ink = labels > 0
    colors = np.empty((*labels.shape, 3), dtype=np.uint16)
    colors[:, :, 0] = 32 + (labels * 53) % 192
    colors[:, :, 1] = 32 + (labels * 97) % 192
    colors[:, :, 2] = 32 + (labels * 151) % 192
    rgb[ink] = (rgb[ink] + 3 * colors[ink]) // 4
    return Image.fromarray(rgb.astype(np.uint8), mode="RGB")


def evidence_panel(gray: np.ndarray, labels: np.ndarray, evidence: dict) -> Image.Image:
    rgb = np.full((*gray.shape, 3), 255, dtype=np.uint8)
    rgb[labels > 0] = COLORS["other"]
    left = evidence["left_dominant_component"]
    right = evidence["right_dominant_component"]
    shared = evidence["shared_core_components"]
    if left is not None:
        rgb[labels == left] = COLORS["left"]
    if right is not None:
        rgb[labels == right] = COLORS["right"]
    if shared:
        rgb[np.isin(labels, list(shared))] = COLORS["shared"]
    return Image.fromarray(rgb, mode="RGB")


def crop_bounds(image: Image.Image, left_box: list[float], right_box: list[float], pad: int = 12) -> tuple[int, int, int, int]:
    return (
        max(0, int(np.floor(min(left_box[0], right_box[0]))) - pad),
        max(0, int(np.floor(min(left_box[1], right_box[1]))) - pad),
        min(image.width, int(np.ceil(max(left_box[2], right_box[2]))) + pad),
        min(image.height, int(np.ceil(max(left_box[3], right_box[3]))) + pad),
    )


def scaled_crop(image: Image.Image, bounds: tuple[int, int, int, int], scale: int) -> Image.Image:
    crop = image.crop(bounds)
    return crop.resize((crop.width * scale, crop.height * scale), Image.Resampling.NEAREST)


def component_rows(evidence: dict) -> list[dict]:
    left_counts = evidence["left_core_component_pixel_counts"]
    right_counts = evidence["right_core_component_pixel_counts"]
    left_total, right_total = sum(left_counts.values()), sum(right_counts.values())
    rows = []
    for component in sorted(set(left_counts) | set(right_counts)):
        left_pixels, right_pixels = left_counts.get(component, 0), right_counts.get(component, 0)
        rows.append({
            "component_id": component,
            "left_core_pixels": left_pixels,
            "right_core_pixels": right_pixels,
            "left_core_share": left_pixels / left_total if left_total else 0.0,
            "right_core_share": right_pixels / right_total if right_total else 0.0,
            "shared_between_cores": bool(left_pixels and right_pixels),
            "left_dominant": component == evidence["left_dominant_component"],
            "right_dominant": component == evidence["right_dominant_component"],
        })
    return rows


def compact_evidence(evidence: dict) -> dict:
    return {
        "connected_box_intersection_v1": evidence["connected_box_intersection_v1"],
        "connected_exclusive_core_v2_1": evidence["connected_exclusive_core_v2_1"],
        "connected_dominant_core_v3": evidence["connected_dominant_core_v3"],
        "dominant_core_usable": evidence["dominant_core_usable"],
        "left_dominant_component_id": evidence["left_dominant_component"],
        "right_dominant_component_id": evidence["right_dominant_component"],
        "left_dominant_share": evidence["left_dominant_share"],
        "right_dominant_share": evidence["right_dominant_share"],
        "shared_core_component_ids": sorted(evidence["shared_core_components"]),
        "components": component_rows(evidence),
        "unusable_reason_codes": evidence["unusable_reason_codes"],
    }


def make_main_figure(
    original: Image.Image,
    gray: np.ndarray,
    labels: np.ndarray,
    evidence: dict,
    boxes: tuple[list[float], list[float]],
    bounds: tuple[int, int, int, int],
    scale: int,
    title: str,
) -> Image.Image:
    binary = Image.fromarray(np.where(labels > 0, 0, 255).astype(np.uint8), mode="L").convert("RGB")
    panels = [
        add_boxes(scaled_crop(original, bounds, scale), boxes, bounds, scale),
        add_boxes(scaled_crop(binary, bounds, scale), boxes, bounds, scale),
        add_boxes(scaled_crop(component_panel(gray, labels), bounds, scale), boxes, bounds, scale),
        add_boxes(
            scaled_crop(evidence_panel(gray, labels, evidence), bounds, scale),
            (evidence["left_core_box"], evidence["right_core_box"]), bounds, scale,
        ),
    ]
    labels_text = ("grayscale + full boxes", "Otsu binary + full boxes", "all 8-connected components", "core evidence: magenta=shared")
    panel_width, panel_height = panels[0].size
    header = 54
    canvas = Image.new("RGB", (panel_width * 4, panel_height + header), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 4), title, fill="black")
    for index, (panel, label) in enumerate(zip(panels, labels_text)):
        x = index * panel_width
        draw.text((x + 6, 25), label, fill="black")
        canvas.paste(panel, (x, header))
    return canvas


def make_sweep_figure(
    gray: np.ndarray,
    boxes: tuple[list[float], list[float]],
    bounds: tuple[int, int, int, int],
    thresholds: list[int],
    scale: int,
) -> Image.Image:
    panels, captions = [], []
    for threshold in thresholds:
        labels, _ = label_ink(gray, threshold)
        evidence = pair_component_evidence(labels, *boxes)
        panel = evidence_panel(gray, labels, evidence)
        panels.append(add_boxes(
            scaled_crop(panel, bounds, scale),
            (evidence["left_core_box"], evidence["right_core_box"]), bounds, scale,
        ))
        captions.append(
            f"t={threshold} v2.1={evidence['connected_exclusive_core_v2_1']} "
            f"v3={evidence['connected_dominant_core_v3']}"
        )
    width, height = panels[0].size
    header = 28
    canvas = Image.new("RGB", (width * len(panels), height + header), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (panel, caption) in enumerate(zip(panels, captions)):
        x = index * width
        draw.text((x + 4, 5), caption, fill="black")
        canvas.paste(panel, (x, header))
    return canvas


def inspect_case(case: dict, record: dict, data_root: Path, output_dir: Path, offsets: list[int], scale: int) -> dict:
    line_id, left_index, right_index = split_pair_id(case["pair_id"])
    transcription = record["transcription"]
    if right_index >= len(transcription):
        raise ValueError(f"pair index outside transcription: {case['pair_id']}")
    if transcription[left_index].isspace() or transcription[right_index].isspace():
        raise ValueError(f"pair crosses whitespace: {case['pair_id']}")
    detections = sorted(record["detections"], key=lambda item: item["box_xyxy"][0])
    mapping = gt_detection_map(transcription, [item["predicted_char"] for item in detections])
    left_alignment, right_alignment = mapping[left_index], mapping[right_index]
    if left_alignment.detection_index is None or right_alignment.detection_index is None:
        raise ValueError(f"pair has missing detection and cannot be rendered: {case['pair_id']}")
    left_box = detections[left_alignment.detection_index]["box_xyxy"]
    right_box = detections[right_alignment.detection_index]["box_xyxy"]
    boxes = (left_box, right_box)
    image_path = data_root / record["image_relpath"]
    original = Image.open(image_path).convert("RGB")
    gray = np.asarray(original.convert("L"))
    otsu = otsu_threshold(gray)
    thresholds = sorted(set(max(0, min(255, otsu + offset)) for offset in offsets))
    if otsu not in thresholds:
        thresholds.append(otsu)
        thresholds.sort()
    threshold_results = []
    otsu_labels = otsu_evidence = None
    for threshold in thresholds:
        labels, component_count = label_ink(gray, threshold)
        evidence = pair_component_evidence(labels, left_box, right_box)
        threshold_results.append({
            "threshold": threshold,
            "offset_from_otsu": threshold - otsu,
            "line_component_count": component_count,
            **compact_evidence(evidence),
        })
        if threshold == otsu:
            otsu_labels, otsu_evidence = labels, evidence
    assert otsu_labels is not None and otsu_evidence is not None
    bounds = crop_bounds(original, left_box, right_box)
    safe_id = case["pair_id"].replace(":", "_")
    main_name, sweep_name = f"{safe_id}_diagnostic.png", f"{safe_id}_threshold-sweep.png"
    title = (
        f"{case['pair_id']} {transcription[left_index:right_index + 1]!r} "
        f"alignment={left_alignment.operation}/{right_alignment.operation} Otsu={otsu}"
    )
    make_main_figure(original, gray, otsu_labels, otsu_evidence, boxes, bounds, scale, title).save(output_dir / main_name)
    make_sweep_figure(gray, boxes, bounds, thresholds, scale).save(output_dir / sweep_name)
    return {
        **case,
        "line_id": line_id,
        "left_gt_index": left_index,
        "right_gt_index": right_index,
        "pair": transcription[left_index:right_index + 1],
        "left_alignment": left_alignment.operation,
        "right_alignment": right_alignment.operation,
        "left_box_xyxy": left_box,
        "right_box_xyxy": right_box,
        "image_relpath": record["image_relpath"],
        "otsu_threshold": otsu,
        "otsu_evidence": compact_evidence(otsu_evidence),
        "threshold_sweep": threshold_results,
        "diagnostic_image": main_name,
        "threshold_sweep_image": sweep_name,
    }


def write_html(manifest: dict, path: Path) -> None:
    cards = []
    for case in manifest["cases"]:
        rows = "".join(
            "<tr>" + "".join(f"<td>{html.escape(str(row[key]))}</td>" for key in (
                "threshold", "offset_from_otsu", "connected_exclusive_core_v2_1",
                "connected_dominant_core_v3", "left_dominant_share", "right_dominant_share"
            )) + "</tr>"
            for row in case["threshold_sweep"]
        )
        cards.append(f"""
<section><h2>{html.escape(case['pair_id'])} — {html.escape(case['pair'])} — {html.escape(case['source'])}</h2>
<p>alignment={html.escape(case['left_alignment'])}/{html.escape(case['right_alignment'])}; Otsu={case['otsu_threshold']}</p>
<img src="{html.escape(case['diagnostic_image'])}">
<img src="{html.escape(case['threshold_sweep_image'])}">
<table><thead><tr><th>threshold</th><th>offset</th><th>v2.1</th><th>v3</th><th>left support</th><th>right support</th></tr></thead><tbody>{rows}</tbody></table></section>
""")
    path.write_text("""<!doctype html><meta charset="utf-8"><title>IAM v3 failure diagnostics</title>
<style>body{font:14px system-ui,sans-serif;margin:20px}img{display:block;max-width:100%;margin:8px 0;border:1px solid #bbb}section{margin:30px 0}table{border-collapse:collapse}th,td{border:1px solid #bbb;padding:4px 8px}</style>
<h1>IAM dominant-core-v3 failure diagnostics</h1>
<p>Analysis only: the frozen review and connectivity implementation are unchanged.</p>
""" + "\n".join(cards), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--detections", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threshold-offsets", type=parse_offsets, default=parse_offsets("-20,-10,-5,0,5,10,20"))
    parser.add_argument("--scale", type=int, default=3)
    args = parser.parse_args()
    if args.scale < 1:
        parser.error("--scale must be at least 1")
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    if selection.get("schema_version") != "dtlr.failure-inspection-selection.v1":
        raise ValueError("unsupported failure inspection selection schema")
    records = read_jsonl(args.detections)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases = []
    for case in selection["cases"]:
        line_id, _, _ = split_pair_id(case["pair_id"])
        if line_id not in records:
            raise ValueError(f"line_id {line_id!r} is absent from detections")
        cases.append(inspect_case(case, records[line_id], args.data_root, args.output_dir, args.threshold_offsets, args.scale))
    manifest = {
        "schema_version": "dtlr.failure-inspection.v1",
        "selection": str(args.selection),
        "selection_sha256": sha256_file(args.selection),
        "detections": str(args.detections),
        "detections_sha256": sha256_file(args.detections),
        "data_root": str(args.data_root),
        "threshold_method": "otsu-per-line-with-explicit-offset-sweep",
        "threshold_offsets_requested": args.threshold_offsets,
        "connectivity_implementation": "unchanged dominant-core-v3 with retained v2.1 comparison",
        "case_count": len(cases),
        "cases": cases,
        "review_index": "index.html",
    }
    (args.output_dir / "failure_diagnostics.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    write_html(manifest, args.output_dir / "index.html")
    print(json.dumps({key: value for key, value in manifest.items() if key != "cases"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
