#!/usr/bin/env python3
"""Build a provisional IAM+READ handwriting-aware bigram tokenizer."""

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "poc"))

from dtlr_poc.selection import sha256_file  # noqa: E402
from dtlr_poc.tokenizer import build_combined_model  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iam-scores", type=Path, required=True)
    parser.add_argument("--read-scores", type=Path, required=True)
    parser.add_argument("--minimum-count", type=int, default=20)
    parser.add_argument("--rate-threshold", type=float, default=0.5)
    parser.add_argument(
        "--pair-policy", choices=("letters-only", "non-whitespace"), default="letters-only"
    )
    parser.add_argument("--unicode-normalization", choices=("none", "NFC"), default="NFC")
    parser.add_argument(
        "--required-character",
        action="append",
        default=[],
        help="single-character fallback to include; repeat as needed",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    score_paths = {"IAM": args.iam_scores, "READ": args.read_scores}
    scores = {
        dataset: json.loads(path.read_text(encoding="utf-8"))
        for dataset, path in score_paths.items()
    }
    required_characters = tuple(dict.fromkeys((" ", *args.required_character)))
    model = build_combined_model(
        scores,
        {dataset: sha256_file(path) for dataset, path in score_paths.items()},
        minimum_count=args.minimum_count,
        rate_threshold=args.rate_threshold,
        pair_policy=args.pair_policy,
        required_characters=required_characters,
        unicode_normalization=args.unicode_normalization,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(model, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    summary = {key: value for key, value in model.items() if key != "vocabulary"}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
