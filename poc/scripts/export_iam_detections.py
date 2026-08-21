#!/usr/bin/env python3
"""Run a small IAM batch and export DTLR boxes without changing model behavior."""
import argparse
import hashlib
import json
import pickle
import subprocess
import sys
from pathlib import Path

import torch
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from datasets.IAM import make_coco_transforms  # noqa: E402
from finetuning import build_model_main  # noqa: E402
from util.slconfig import SLConfig  # noqa: E402
from poc.dtlr_poc.selection import load_selected_examples, sha256_file  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-kind", required=True, choices=("iam-finetuned", "english-pretrained"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path)
    parser.add_argument("--split", choices=("train", "valid", "test"), default="test")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--nms", type=float, default=0.5)
    args = parser.parse_args()
    if args.checkpoint_kind != "iam-finetuned":
        print("WARNING: English-pretrained is a language/synthetic pretraining checkpoint; it does not verify the IAM fine-tuned milestone.", file=sys.stderr)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; run inside environment/cuda-linux on the RTX 4060 host")

    label_path = args.data_root / "IAM_new/labels.pkl"
    labels = pickle.loads(label_path.read_bytes())
    charset = json.loads((REPO / "datasets/default_charset.json").read_text())
    config = SLConfig.fromfile(str(REPO / "config/Latin_CTC.py"))
    config.dataset_file = "IAM"
    config.charset = charset
    config.device = "cuda:0"
    config.fix_size = False
    transform = make_coco_transforms("test", args=config)

    model, _, postprocessors = build_model_main(config)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    incompatible = model.load_state_dict(checkpoint["model"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(str(incompatible))
    model.eval().cuda()
    postprocessors["bbox"].num_select = 900
    postprocessors["bbox"].nms_iou_threshold = args.nms

    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    checkpoint_sha256 = sha256(args.checkpoint)
    selection_provenance = None
    if args.selection_manifest:
        if args.start != 0 or args.limit != 8:
            raise SystemExit("do not combine --selection-manifest with --start/--limit")
        examples, selection = load_selected_examples(args.selection_manifest, label_path, args.split)
        selection_provenance = {
            "schema_version": selection["schema_version"],
            "sha256": sha256_file(args.selection_manifest),
        }
    else:
        examples = labels["ground_truth"][args.split][args.start:args.start + args.limit]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as out:
        for example in examples:
            relpath = Path("IAM_new/data/imgs/lines") / f"{example['id']}.jpg"
            image = Image.open(args.data_root / relpath).convert("RGB")
            width, height = image.size
            tensor, _ = transform(image, {"boxes": torch.empty((0, 4))})
            with torch.no_grad():
                outputs = model(tensor[None].cuda())
                result = postprocessors["bbox"](outputs, torch.tensor([[height, width]], device="cuda"))[0]
            keep = result["scores"] > args.threshold
            order = torch.argsort(result["boxes"][keep, 0])
            detections = []
            for rank, index in enumerate(order.tolist()):
                label = int(result["labels"][keep][index])
                detections.append({
                    "rank": rank,
                    "label": label,
                    "predicted_char": charset[label],
                    "score": float(result["scores"][keep][index].cpu()),
                    "box_xyxy": [float(v) for v in result["boxes"][keep][index].cpu()],
                })
            record = {
                "schema_version": "dtlr.detections.v1",
                "dataset": "IAM",
                "split": args.split,
                "line_id": example["id"],
                "transcription": example["text"],
                "image_relpath": str(relpath),
                "image_width": width,
                "image_height": height,
                "checkpoint": {"kind": args.checkpoint_kind, "sha256": checkpoint_sha256},
                "selection_manifest": selection_provenance,
                "repo_commit": commit,
                "threshold": args.threshold,
                "nms_iou": args.nms,
                "detections": detections,
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"{example['id']}: {len(detections)} detections for {len(example['text'])} GT characters")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
