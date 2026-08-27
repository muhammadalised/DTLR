"""Train-derived IAM bigram vocabulary and deterministic text segmentation."""

from dataclasses import dataclass


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
) -> dict:
    if minimum_count <= 0:
        raise ValueError("minimum_count must be positive")
    if not 0.0 <= rate_threshold <= 1.0:
        raise ValueError("rate_threshold must be between zero and one")
    if not scores:
        raise ValueError("bigram score input is empty")
    pairs = set()
    vocabulary = []
    for row in scores:
        if row.get("schema_version") != "dtlr.bigram-scores.v5":
            raise ValueError("unsupported bigram score schema")
        if row.get("dataset") != "IAM" or row.get("split") != "train":
            raise ValueError("tokenizer vocabulary must be derived from IAM train scores only")
        if row.get("connectivity_method") != "dominant-core-v3":
            raise ValueError("tokenizer scores do not use frozen dominant-core-v3")
        pair = row["pair"]
        if pair in pairs:
            raise ValueError(f"duplicate score row for pair {pair!r}")
        pairs.add(pair)
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
    return {
        "schema_version": MODEL_SCHEMA,
        "status": "provisional-demo-policy",
        "dataset": "IAM",
        "training_split": "train",
        "connectivity_method": "dominant-core-v3",
        "score_field": "exact_alignment_connected_rate",
        "count_field": "n_exact_alignment",
        "source_scores_sha256": source_sha256,
        "policy": {
            "minimum_count": minimum_count,
            "rate_threshold": rate_threshold,
            "pair_policy": pair_policy,
            "overlap_resolution": "maximum-total-utility-non-overlapping-v1",
            "single_character_utility": 0.0,
        },
        "eligible_bigram_count": len(vocabulary),
        "vocabulary": vocabulary,
        "note": (
            "Training-derived demonstration model. Policy thresholds are not held-out "
            "performance and must be validated before thesis claims."
        ),
    }


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
