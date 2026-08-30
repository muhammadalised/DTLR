#!/usr/bin/env python3
"""Create frozen READ line crops from PAGE XML without changing GT identity."""

import argparse
import io
import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import PIL
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "poc"))

from dtlr_poc.read_dataset import all_examples, load_selected_read_examples  # noqa: E402
from dtlr_poc.selection import sha256_file  # noqa: E402


RAW_LAYOUT = {
    "train": ("Training", Path("page/page")),
    "valid": ("Validation", Path("page/page")),
    "test": ("Test", Path("page")),
}
LINE_PATH = re.compile(r"^READ_2016/images/(train|valid|test)/\1_(\d+)_(\d+)\.jpeg$")
CORRECTED_NULL_LABELS = {
    "line_816fb2ce-06b0-4e00-bb28-10c8b9c367f2": "16",
    "line_a5f4ab4e-2ea0-4c65-840c-4a89b04bd477": "108",
    "line_e1288df8-8a0d-40df-be91-4b4a332027ec": "196",
    "line_455330f3-9e27-4340-ae86-9d6c448dc091": "199",
    "line_ecbbccee-e8c2-495d-ac47-0aff93f3d9ac": "202",
    "line_e918616d-64f8-43d2-869c-f687726212be": "214",
    "line_ebd8f850-1da5-45b1-b59c-9349497ecc8e": "216",
}
CROP_RULE = "page-textline-coords-inclusive-aabb-v1"
JPEG_SETTINGS = {
    "format": "JPEG",
    "quality": 95,
    "subsampling": 0,
    "optimize": False,
    "progressive": False,
}


def local_name(element: ET.Element) -> str:
    return element.tag.rsplit("}", 1)[-1]


def direct_child(element: ET.Element, name: str) -> ET.Element | None:
    return next((child for child in element if local_name(child) == name), None)


def page_line_index(example: dict, split: str) -> tuple[int, int]:
    match = LINE_PATH.fullmatch(example["label_image_relpath"])
    if match is None or match.group(1) != split:
        raise ValueError(f"unexpected READ line path: {example['label_image_relpath']}")
    return int(match.group(2)), int(match.group(3))


def line_records(xml_path: Path) -> list[dict]:
    root = ET.parse(xml_path).getroot()
    records = []
    for element in root.iter():
        if local_name(element) != "TextLine":
            continue
        line_id = element.attrib.get("id", "")
        text_equiv = direct_child(element, "TextEquiv")
        unicode_node = direct_child(text_equiv, "Unicode") if text_equiv is not None else None
        text = unicode_node.text if unicode_node is not None else None
        if line_id in CORRECTED_NULL_LABELS:
            text = CORRECTED_NULL_LABELS[line_id]
        if text is None:
            continue
        coords = direct_child(element, "Coords")
        if coords is None or not coords.attrib.get("points"):
            raise ValueError(f"TextLine {line_id!r} has no PAGE coordinates in {xml_path}")
        points = []
        for point in coords.attrib["points"].split():
            x_text, y_text = point.split(",", 1)
            points.append((int(x_text), int(y_text)))
        if not points:
            raise ValueError(f"TextLine {line_id!r} has empty PAGE coordinates in {xml_path}")
        xs, ys = zip(*points)
        records.append({
            "page_line_id": line_id,
            "xml_transcription": text.strip(),
            "crop_xyxy_inclusive": [min(xs), min(ys), max(xs), max(ys)],
        })
    return records


def encode_crop(image: Image.Image, bounds: list[int]) -> tuple[bytes, list[int]]:
    left, top, right, bottom = bounds
    clipped = [
        max(0, left),
        max(0, top),
        min(image.width - 1, right),
        min(image.height - 1, bottom),
    ]
    if clipped[2] < clipped[0] or clipped[3] < clipped[1]:
        raise ValueError(f"empty line crop after clipping: {bounds} -> {clipped}")
    # PAGE maxima are coordinates of pixels. PIL's right/bottom crop bounds are
    # exclusive, hence the explicit +1 encoded in CROP_RULE.
    crop = image.crop((clipped[0], clipped[1], clipped[2] + 1, clipped[3] + 1)).convert("RGB")
    buffer = io.BytesIO()
    crop.save(buffer, **JPEG_SETTINGS)
    return buffer.getvalue(), clipped


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    args = parser.parse_args()

    selection = json.loads(args.selection_manifest.read_text(encoding="utf-8"))
    split = selection.get("split")
    if selection.get("dataset") != "READ" or split not in RAW_LAYOUT:
        raise SystemExit("selection manifest is not a supported READ split")
    labels_path = args.data_root / "READ_2016/labels.pkl"
    examples, _ = load_selected_read_examples(args.selection_manifest, labels_path, split)

    archive_name, xml_subdir = RAW_LAYOUT[split]
    split_root = args.raw_root / "PublicData" / archive_name
    xml_paths = sorted((split_root / xml_subdir).glob("Seite*.xml"))
    image_root = split_root / "Images"

    all_split_examples = all_examples(labels_path, split)
    by_page: dict[int, list[dict]] = {}
    for example in all_split_examples:
        page_index, line_index = page_line_index(example, split)
        by_page.setdefault(page_index, []).append({**example, "page_line_index": line_index})
    for page_index, rows in by_page.items():
        rows.sort(key=lambda row: row["page_line_index"])
        if [row["page_line_index"] for row in rows] != list(range(len(rows))):
            raise ValueError(f"non-contiguous labels.pkl line indices on {split} page {page_index}")
    if len(xml_paths) != len(by_page):
        raise ValueError(
            f"PAGE XML count {len(xml_paths)} differs from labels.pkl page count {len(by_page)}"
        )

    selected_by_page: dict[int, list[tuple[dict, int]]] = {}
    for example in examples:
        page_index, line_index = page_line_index(example, split)
        selected_by_page.setdefault(page_index, []).append((example, line_index))

    output_records = []
    created = existing = 0
    for page_index in sorted(selected_by_page):
        xml_path = xml_paths[page_index]
        source_image_path = image_root / f"{xml_path.stem}.JPG"
        if not source_image_path.is_file():
            raise FileNotFoundError(f"READ page image is absent: {source_image_path}")
        xml_lines = line_records(xml_path)
        label_lines = by_page[page_index]
        if len(xml_lines) != len(label_lines):
            raise ValueError(
                f"line count mismatch on {split} page {page_index} ({xml_path.name}): "
                f"XML={len(xml_lines)}, labels.pkl={len(label_lines)}"
            )
        mismatches = [
            index for index, (xml_line, label_line) in enumerate(zip(xml_lines, label_lines))
            if xml_line["xml_transcription"] != label_line["raw_text"]
        ]
        if mismatches:
            preview = ", ".join(str(index) for index in mismatches[:10])
            raise ValueError(
                f"transcription mismatch on {split} page {page_index} ({xml_path.name}) "
                f"at line indices: {preview}"
            )

        with Image.open(source_image_path) as source_image:
            source_image.load()
            page_xml_sha256 = sha256_file(xml_path)
            source_image_sha256 = sha256_file(source_image_path)
            for example, line_index in sorted(selected_by_page[page_index], key=lambda item: item[1]):
                xml_line = xml_lines[line_index]
                encoded, clipped = encode_crop(
                    source_image, xml_line["crop_xyxy_inclusive"]
                )
                output_path = args.data_root / example["label_image_relpath"]
                output_path.parent.mkdir(parents=True, exist_ok=True)
                if output_path.exists():
                    if output_path.read_bytes() != encoded:
                        raise ValueError(f"refusing to overwrite a different READ line crop: {output_path}")
                    existing += 1
                else:
                    output_path.write_bytes(encoded)
                    created += 1
                output_records.append({
                    "line_id": example["id"],
                    "read_idx": example["idx"],
                    "page_index": page_index,
                    "page_line_index": line_index,
                    "page_xml": str(xml_path.relative_to(args.raw_root)),
                    "page_xml_sha256": page_xml_sha256,
                    "source_image": str(source_image_path.relative_to(args.raw_root)),
                    "source_image_sha256": source_image_sha256,
                    "page_line_id": xml_line["page_line_id"],
                    "crop_xyxy_inclusive": xml_line["crop_xyxy_inclusive"],
                    "crop_xyxy_inclusive_clipped": clipped,
                    "output": example["label_image_relpath"],
                    "output_sha256": sha256_file(output_path),
                })

    manifest = {
        "schema_version": "dtlr.read-preprocessing.v1",
        "selection_manifest": str(args.selection_manifest),
        "selection_manifest_sha256": sha256_file(args.selection_manifest),
        "labels_sha256": sha256_file(labels_path),
        "split": split,
        "record_count": len(output_records),
        "crop_rule": CROP_RULE,
        "jpeg_settings": JPEG_SETTINGS,
        "pillow_version": PIL.__version__,
        "records": output_records,
    }
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    if args.output_manifest.exists() and args.output_manifest.read_text(encoding="utf-8") != rendered:
        raise ValueError(f"refusing to overwrite a different preprocessing manifest: {args.output_manifest}")
    args.output_manifest.write_text(rendered, encoding="utf-8")
    summary = {key: value for key, value in manifest.items() if key != "records"}
    summary["created_this_run"] = created
    summary["existing_this_run"] = existing
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
