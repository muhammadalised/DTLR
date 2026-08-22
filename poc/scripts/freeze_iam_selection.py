#!/usr/bin/env python3
"""Freeze IAM line IDs before inference or visual inspection."""

import argparse
import json
import pickle
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "poc"))

from dtlr_poc.selection import build_selection  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "valid", "test"), required=True)
    size = parser.add_mutually_exclusive_group(required=True)
    size.add_argument("--count", type=int)
    size.add_argument("--all", action="store_true", help="select the complete declared split")
    parser.add_argument("--seed", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    labels_path = args.data_root / "IAM_new/labels.pkl"
    if args.all:
        labels = pickle.loads(labels_path.read_bytes())
        count = len(labels["ground_truth"][args.split])
    else:
        count = args.count
    selection = build_selection(labels_path, args.split, count, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        existing = json.loads(args.output.read_text(encoding="utf-8"))
        if existing != selection:
            raise SystemExit(f"refusing to overwrite a different frozen selection: {args.output}")
        print(f"Selection already frozen and unchanged: {args.output}")
        return 0
    args.output.write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8")
    print(f"Frozen {len(selection['lines'])} {args.split} line IDs: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
