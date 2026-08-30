#!/usr/bin/env python3
"""Export READ character boxes with explicit dataset and checkpoint provenance."""

import argparse
import json
import pickle
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from datasets.READ import make_coco_transforms  # noqa: E402
from finetuning import build_model_main  # noqa: E402
from poc.dtlr_poc.checkpoint_structure import classify_decoder_class_embed  # noqa: E402
from poc.dtlr_poc.detection_resume import read_jsonl_for_resume, validate_resume_prefix  # noqa: E402
from poc.dtlr_poc.read_dataset import (  # noqa: E402
    TRANSCRIPTION_NORMALIZATION,
    all_examples,
    load_selected_read_examples,
    resolve_image,
)
from poc.dtlr_poc.selection import sha256_file  # noqa: E402
from util.slconfig import SLConfig  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--checkpoint-kind", required=True,
        choices=("read-finetuned", "german-pretrained"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path)
    parser.add_argument("--split", choices=("train", "valid", "test"), default="valid")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--nms", type=float, default=0.5)
    parser.add_argument(
        "--resume", action="store_true",
        help="append only after validating that existing output is an exact compatible prefix",
    )
    args = parser.parse_args()

    if args.checkpoint_kind != "read-finetuned":
        print(
            "WARNING: german-pretrained is synthetic-language pretraining and does not "
            "verify READ-finetuned localization.",
            file=sys.stderr,
        )
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; run in the Conda CUDA environment on the RTX host")
    if not args.checkpoint.is_file():
        raise SystemExit(f"checkpoint not found: {args.checkpoint}")

    labels_path = args.data_root / "READ_2016/labels.pkl"
    if not labels_path.is_file():
        raise SystemExit(f"READ labels file not found: {labels_path}")
    labels = pickle.loads(labels_path.read_bytes())
    charset = [chr(value) for value in labels["charset"]]
    if "¬" in charset:
        raise SystemExit("READ charset unexpectedly contains the removed continuation marker '¬'")

    config = SLConfig.fromfile(str(REPO / "config/Latin_CTC.py"))
    config.dataset_file = "READ"
    config.charset = charset
    config.num_classes = len(charset)
    config.device = "cuda:0"
    config.fix_size = False

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    if "model" not in checkpoint:
        raise KeyError("checkpoint has no 'model' state dictionary")
    checkpoint_structure = classify_decoder_class_embed(
        checkpoint["model"].keys(), config.dec_layers
    )

    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    checkpoint_sha256 = sha256_file(args.checkpoint)
    selection_provenance = None
    if args.selection_manifest:
        if args.start != 0 or args.limit != 8:
            raise SystemExit("do not combine --selection-manifest with --start/--limit")
        examples, selection = load_selected_read_examples(
            args.selection_manifest, labels_path, args.split
        )
        selection_provenance = {
            "schema_version": selection["schema_version"],
            "sha256": sha256_file(args.selection_manifest),
        }
    else:
        examples = all_examples(labels_path, args.split)[args.start:args.start + args.limit]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() and not args.resume:
        raise SystemExit(
            f"refusing to overwrite existing output; use --resume after verifying the run: {args.output}"
        )
    existing = read_jsonl_for_resume(args.output) if args.resume else []
    expected = {
        "schema_version": "dtlr.detections.v3",
        "dataset": "READ",
        "split": args.split,
        "checkpoint": {"kind": args.checkpoint_kind, "sha256": checkpoint_sha256},
        "checkpoint_structure": checkpoint_structure,
        "selection_manifest": selection_provenance,
        "repo_commit": commit,
        "threshold": args.threshold,
        "nms_iou": args.nms,
        "transcription_normalization": TRANSCRIPTION_NORMALIZATION,
        "labels_sha256": sha256_file(labels_path),
    }
    validate_resume_prefix(existing, examples, expected)
    for position, (record, example) in enumerate(zip(existing, examples), 1):
        relpath, path_source = resolve_image(args.data_root, example, args.split)
        read_specific = {
            "read_idx": example["idx"],
            "raw_transcription": example["raw_text"],
            "image_relpath": str(relpath),
            "image_path_source": path_source,
        }
        for key, value in read_specific.items():
            if record.get(key) != value:
                raise ValueError(
                    f"READ resume provenance mismatch for {key!r} in existing line {position}"
                )
    if len(existing) == len(examples):
        print(f"Detection export already complete: {len(existing)}/{len(examples)} records")
        return 0

    transform = make_coco_transforms("test", args=config)

    model, _, postprocessors = build_model_main(config)
    if checkpoint_structure["decoder_class_embed"] == "single-linear":
        weight = checkpoint["model"]["transformer.decoder.class_embed.weight"]
        bias = checkpoint["model"]["transformer.decoder.class_embed.bias"]
        expected_shape = (len(charset), model.class_embed[0].in_features)
        if tuple(weight.shape) != expected_shape or tuple(bias.shape) != (len(charset),):
            raise ValueError(
                "READ checkpoint decoder classifier shape does not match the dataset charset "
                f"and model hidden size: weight={tuple(weight.shape)}, bias={tuple(bias.shape)}, "
                f"expected={expected_shape}/{(len(charset),)}"
            )
        model.transformer.decoder.class_embed = nn.Linear(
            expected_shape[1], expected_shape[0]
        )
    incompatible = model.load_state_dict(checkpoint["model"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(str(incompatible))
    model.eval().cuda()
    postprocessors["bbox"].num_select = 900
    postprocessors["bbox"].nms_iou_threshold = args.nms

    mode = "a" if args.output.exists() else "x"
    with args.output.open(mode, encoding="utf-8") as out:
        for position, example in enumerate(examples[len(existing):], len(existing) + 1):
            relpath, path_source = resolve_image(args.data_root, example, args.split)
            image = Image.open(args.data_root / relpath).convert("RGB")
            width, height = image.size
            tensor, _ = transform(image, {"boxes": torch.empty((0, 4))})
            with torch.no_grad():
                outputs = model(tensor[None].cuda())
                result = postprocessors["bbox"](
                    outputs, torch.tensor([[height, width]], device="cuda")
                )[0]
            keep = result["scores"] > args.threshold
            order = torch.argsort(result["boxes"][keep, 0])
            detections = []
            for rank, index in enumerate(order.tolist()):
                label = int(result["labels"][keep][index])
                if not 0 <= label < len(charset):
                    raise ValueError(f"predicted READ label {label} is outside charset size {len(charset)}")
                detections.append({
                    "rank": rank,
                    "label": label,
                    "predicted_char": charset[label],
                    "score": float(result["scores"][keep][index].cpu()),
                    "box_xyxy": [float(value) for value in result["boxes"][keep][index].cpu()],
                })
            record = {
                **expected,
                "line_id": example["id"],
                "read_idx": example["idx"],
                "raw_transcription": example["raw_text"],
                "transcription": example["text"],
                "image_relpath": str(relpath),
                "image_path_source": path_source,
                "image_width": width,
                "image_height": height,
                "detections": detections,
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            marker_count = example["raw_text"].count("¬")
            print(
                f"[{position}/{len(examples)}] {example['id']}: {len(detections)} detections "
                f"for {len(example['text'])} normalized GT characters; removed_markers={marker_count}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
