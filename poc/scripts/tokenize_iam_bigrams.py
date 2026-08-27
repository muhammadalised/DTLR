#!/usr/bin/env python3
"""Tokenize text with a train-derived IAM bigram model."""

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "poc"))

from dtlr_poc.tokenizer import HandwritingBigramTokenizer  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--text", action="append", required=True, help="text to tokenize; may be repeated")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    model = json.loads(args.model.read_text(encoding="utf-8"))
    tokenizer = HandwritingBigramTokenizer(model)
    results = []
    for text in args.text:
        result = tokenizer.segment(text)
        token_ids = tokenizer.encode(text)
        result.update({
            "token_ids": token_ids,
            "decoded_text": tokenizer.decode(token_ids),
        })
        results.append(result)
    payload = {
        "schema_version": "dtlr.tokenization-demo.v1",
        "model": str(args.model),
        "model_status": model.get("status"),
        "tokenizer_size": tokenizer.size,
        "results": results,
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
