#!/usr/bin/env python3
"""Retired training entry point: vocabulary fitting now belongs to TVA."""


def main():
    raise SystemExit(
        "Tokenizer training moved to TVA. Run train_ed_tokenizers.ipynb there. "
        "DTLR now exports the input evidence with poc/scripts/export_iam_tokenizer_evidence.py. "
        "Historical frozen artifacts and inference interfaces remain available."
    )


if __name__ == "__main__":
    main()
