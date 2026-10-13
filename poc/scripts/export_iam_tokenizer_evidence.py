#!/usr/bin/env python3
"""Export compact, pinned IAM training connectivity evidence for TVA."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dtlr_poc.tokenizer_evidence import export_iam_evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iam-train-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export_iam_evidence(args.iam_train_dir, args.output), indent=2))


if __name__ == "__main__":
    main()
