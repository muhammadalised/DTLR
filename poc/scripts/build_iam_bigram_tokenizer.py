#!/usr/bin/env python3
"""Build a provisional IAM handwriting-aware bigram tokenizer model."""

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "poc"))

from dtlr_poc.selection import sha256_file  # noqa: E402
from dtlr_poc.tokenizer import build_model  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--minimum-count", type=int, default=20)
    parser.add_argument("--rate-threshold", type=float, default=0.5)
    parser.add_argument("--pair-policy", choices=("letters-only", "non-whitespace"), default="letters-only")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    scores = json.loads(args.scores.read_text(encoding="utf-8"))
    model = build_model(
        scores,
        source_sha256=sha256_file(args.scores),
        minimum_count=args.minimum_count,
        rate_threshold=args.rate_threshold,
        pair_policy=args.pair_policy,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(model, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in model.items() if key != "vocabulary"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
