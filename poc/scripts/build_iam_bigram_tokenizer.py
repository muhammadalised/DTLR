#!/usr/bin/env python3
"""Build or reproducibly freeze an IAM handwriting-aware bigram model."""

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "poc"))

from dtlr_poc.selection import sha256_file  # noqa: E402
from dtlr_poc.tokenizer import (  # noqa: E402
    FROZEN_IAM_MODEL_VERSION,
    build_frozen_iam_artifact,
    build_model,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--source-model", type=Path)
    parser.add_argument(
        "--model-version",
        choices=(FROZEN_IAM_MODEL_VERSION,),
        help="freeze the pinned generic IAM-only artifact instead of building a provisional model",
    )
    parser.add_argument("--minimum-count", type=int, default=20)
    parser.add_argument("--rate-threshold", type=float, default=0.5)
    parser.add_argument("--pair-policy", choices=("letters-only", "non-whitespace"), default="letters-only")
    parser.add_argument(
        "--required-character",
        action="append",
        default=[],
        help="single-character fallback to include in addition to characters observed in score pairs; repeat as needed",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    required_characters = tuple(dict.fromkeys((" ", *args.required_character)))
    scores = json.loads(args.scores.read_text(encoding="utf-8"))
    if args.model_version:
        if args.manifest is None or args.source_model is None:
            parser.error("--model-version requires --manifest and --source-model")
        if (
            args.minimum_count != 20
            or args.rate_threshold != 0.5
            or args.pair_policy != "letters-only"
            or required_characters != (" ",)
        ):
            parser.error("the frozen IAM v1 policy is fixed at count=20, rate=0.5, letters-only, and space")
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        source_model = json.loads(args.source_model.read_text(encoding="utf-8"))
        model = build_frozen_iam_artifact(
            scores=scores,
            manifest=manifest,
            source_model=source_model,
            source_scores_sha256=sha256_file(args.scores),
            source_manifest_sha256=sha256_file(args.manifest),
            source_model_sha256=sha256_file(args.source_model),
        )
    else:
        if args.manifest is not None or args.source_model is not None:
            parser.error("--manifest and --source-model require --model-version")
        model = build_model(
            scores,
            source_sha256=sha256_file(args.scores),
            minimum_count=args.minimum_count,
            rate_threshold=args.rate_threshold,
            pair_policy=args.pair_policy,
            required_characters=required_characters,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(model, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in model.items() if key != "vocabulary"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
