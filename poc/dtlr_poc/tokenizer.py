"""Train-derived IAM bigram vocabulary and deterministic text segmentation."""

from dataclasses import dataclass
import json
from pathlib import Path


MODEL_SCHEMA = "dtlr.handwriting-bigram-tokenizer.v1"


def pair_allowed(pair: str, policy: str) -> bool:
    if len(pair) != 2:
        return False
    if policy == "letters-only":
        return all(char.isalpha() for char in pair)
    if policy == "non-whitespace":
        return not any(char.isspace() for char in pair)
    raise ValueError(f"unsupported pair policy: {policy}")


def build_model(
    scores: list[dict],
    source_sha256: str,
    minimum_count: int,
    rate_threshold: float,
    pair_policy: str = "letters-only",
    required_characters: tuple[str, ...] = (" ",),
) -> dict:
    if minimum_count <= 0:
        raise ValueError("minimum_count must be positive")
    if not 0.0 <= rate_threshold <= 1.0:
        raise ValueError("rate_threshold must be between zero and one")
    if not scores:
        raise ValueError("bigram score input is empty")
    if any(len(char) != 1 for char in required_characters):
        raise ValueError("required characters must each contain exactly one character")
    pairs = set()
    characters = set(required_characters)
    vocabulary = []
    for row in scores:
        if row.get("schema_version") != "dtlr.bigram-scores.v5":
            raise ValueError("unsupported bigram score schema")
        if row.get("dataset") != "IAM" or row.get("split") != "train":
            raise ValueError("tokenizer vocabulary must be derived from IAM train scores only")
        if row.get("connectivity_method") != "dominant-core-v3":
            raise ValueError("tokenizer scores do not use frozen dominant-core-v3")
        pair = row["pair"]
        if len(pair) != 2:
            raise ValueError(f"score pair must contain exactly two characters: {pair!r}")
        if pair in pairs:
            raise ValueError(f"duplicate score row for pair {pair!r}")
        pairs.add(pair)
        characters.update(pair)
        count = row.get("n_exact_alignment", 0)
        rate = row.get("exact_alignment_connected_rate")
        if (
            pair_allowed(pair, pair_policy)
            and count >= minimum_count
            and rate is not None
            and rate >= rate_threshold
        ):
            vocabulary.append({
                "token": pair,
                "n_exact_alignment": count,
                "exact_alignment_connected_rate": rate,
                "utility": rate,
            })
    vocabulary.sort(key=lambda row: row["token"])
    # TVA reserves ID 0 for the CTC blank, represented by the empty string.
    ordered_tokens = [""] + sorted(characters) + [row["token"] for row in vocabulary]
    vocab = {token: index for index, token in enumerate(ordered_tokens)}
    idx_token = {str(index): token for token, index in vocab.items()}
    return {
        "schema_version": MODEL_SCHEMA,
        "status": "provisional-demo-policy",
        "dataset": "IAM",
        "training_split": "train",
        "connectivity_method": "dominant-core-v3",
        "score_field": "exact_alignment_connected_rate",
        "count_field": "n_exact_alignment",
        "blank_token": "",
        "blank_id": 0,
        "source_scores_sha256": source_sha256,
        "policy": {
            "minimum_count": minimum_count,
            "rate_threshold": rate_threshold,
            "pair_policy": pair_policy,
            "overlap_resolution": "maximum-total-utility-non-overlapping-v1",
            "single_character_utility": 0.0,
            "required_characters": list(required_characters),
        },
        "vocab": vocab,
        "idx_token": idx_token,
        "size": len(vocab),
        "eligible_bigram_count": len(vocabulary),
        "vocabulary": vocabulary,
        "note": (
            "Training-derived demonstration model. Policy thresholds are not held-out "
            "performance and must be validated before thesis claims."
        ),
    }


class HandwritingBigramTokenizer:
    """TVA-familiar tokenizer interface with handwriting-aware DP segmentation.

    The public shape intentionally mirrors TVA's BigramTokenizer (`vocab`,
    `idx_token`, `size`, `load`, `encode`, and `decode`). Token selection still
    uses this project's maximum-total-utility dynamic program, not TVA's greedy
    left-to-right bigram matching.
    """

    def __init__(self, model: dict | None = None) -> None:
        self.model: dict = {}
        self.vocab: dict[str, int] = {}
        self.idx_token: dict[int, str] = {}
        if model is not None:
            self.load_model(model)

    @property
    def size(self) -> int:
        if not self.vocab:
            raise ValueError("Tokenizer not trained or loaded.")
        return len(self.vocab)

    def load_model(self, model: dict) -> None:
        if model.get("schema_version") != MODEL_SCHEMA:
            raise ValueError("unsupported tokenizer model schema")
        if "vocab" not in model or "idx_token" not in model:
            raise ValueError(
                "tokenizer model lacks TVA-compatible mappings; rebuild it from train scores"
            )
        vocab = model["vocab"]
        idx_token = {int(index): token for index, token in model["idx_token"].items()}
        if set(vocab.values()) != set(range(len(vocab))):
            raise ValueError("vocabulary IDs must be contiguous and zero-based")
        if len(vocab) != len(idx_token) or any(
            idx_token.get(index) != token for token, index in vocab.items()
        ):
            raise ValueError("vocab and idx_token are not inverse mappings")
        if model.get("size") != len(vocab):
            raise ValueError("tokenizer size does not match vocabulary")
        if model.get("blank_token") != "" or model.get("blank_id") != 0:
            raise ValueError("TVA-compatible CTC blank must be the empty token at ID 0")
        self.model = model
        self.vocab = dict(vocab)
        self.idx_token = idx_token

    def load(self, path_config: str | Path) -> None:
        path = Path(path_config)
        self.load_model(json.loads(path.read_text(encoding="utf-8")))

    def segment(self, text: str) -> dict:
        if not self.model:
            raise ValueError("Tokenizer not trained or loaded.")
        return tokenize(text, self.model)

    def encode(self, text: str) -> list[int]:
        result = self.segment(text)
        missing = sorted({token for token in result["tokens"] if token not in self.vocab})
        if missing:
            raise ValueError(f"characters are absent from tokenizer vocabulary: {missing!r}")
        return [self.vocab[token] for token in result["tokens"]]

    def decode(self, ids: list[int]) -> str:
        if not self.idx_token:
            raise ValueError("Tokenizer not trained or loaded.")
        try:
            return "".join(self.idx_token[index] for index in ids)
        except KeyError as error:
            raise ValueError(f"unknown token id: {error.args[0]}") from error


@dataclass(frozen=True)
class Segmentation:
    utility: float
    bigram_count: int
    segments: tuple[dict, ...]


def tokenize(text: str, model: dict) -> dict:
    if model.get("schema_version") != MODEL_SCHEMA:
        raise ValueError("unsupported tokenizer model schema")
    vocabulary = {row["token"]: row for row in model["vocabulary"]}
    best: list[Segmentation | None] = [None] * (len(text) + 1)
    best[len(text)] = Segmentation(0.0, 0, ())
    for index in range(len(text) - 1, -1, -1):
        suffix = best[index + 1]
        assert suffix is not None
        single = {
            "token": text[index],
            "start": index,
            "end": index + 1,
            "kind": "single",
            "utility": 0.0,
        }
        winner = Segmentation(suffix.utility, suffix.bigram_count, (single, *suffix.segments))
        pair = text[index:index + 2]
        if len(pair) == 2 and pair in vocabulary:
            suffix = best[index + 2]
            assert suffix is not None
            entry = vocabulary[pair]
            segment = {
                "token": pair,
                "start": index,
                "end": index + 2,
                "kind": "handwriting-bigram",
                "utility": entry["utility"],
                "n_exact_alignment": entry["n_exact_alignment"],
                "exact_alignment_connected_rate": entry["exact_alignment_connected_rate"],
            }
            candidate = Segmentation(
                suffix.utility + entry["utility"],
                suffix.bigram_count + 1,
                (segment, *suffix.segments),
            )
            if (candidate.utility, candidate.bigram_count) >= (winner.utility, winner.bigram_count):
                winner = candidate
        best[index] = winner
    result = best[0]
    assert result is not None
    return {
        "text": text,
        "tokens": [segment["token"] for segment in result.segments],
        "total_utility": result.utility,
        "bigram_token_count": result.bigram_count,
        "segments": list(result.segments),
    }
