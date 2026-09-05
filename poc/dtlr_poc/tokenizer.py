"""Train-derived handwriting bigram vocabularies and text segmentation."""

from dataclasses import dataclass
import json
from pathlib import Path
import unicodedata


MODEL_SCHEMA = "dtlr.handwriting-bigram-tokenizer.v1"
COMBINED_MODEL_SCHEMA = "dtlr.handwriting-bigram-tokenizer.v2"
SUPPORTED_MODEL_SCHEMAS = {MODEL_SCHEMA, COMBINED_MODEL_SCHEMA}


def normalize_text(text: str, policy: str) -> str:
    if policy == "none":
        return text
    if policy == "NFC":
        return unicodedata.normalize("NFC", text)
    raise ValueError(f"unsupported Unicode normalization policy: {policy}")


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


def _exact_connected_count(row: dict) -> tuple[int, int]:
    count = row.get("n_exact_alignment", 0)
    rate = row.get("exact_alignment_connected_rate")
    if not isinstance(count, int) or count < 0:
        raise ValueError("n_exact_alignment must be a non-negative integer")
    if count == 0:
        if rate is not None:
            raise ValueError("zero exact-alignment count must have a null connected rate")
        return 0, 0
    if not isinstance(rate, (int, float)) or not 0.0 <= rate <= 1.0:
        raise ValueError("nonzero exact-alignment count must have a connected rate in [0, 1]")
    raw_connected = count * rate
    connected = round(raw_connected)
    if abs(raw_connected - connected) > 1e-7:
        raise ValueError("connected rate cannot be recovered as an integer observation count")
    return count, connected


def build_combined_model(
    scores_by_dataset: dict[str, list[dict]],
    source_sha256_by_dataset: dict[str, str],
    minimum_count: int,
    rate_threshold: float,
    pair_policy: str = "letters-only",
    required_characters: tuple[str, ...] = (" ",),
    unicode_normalization: str = "NFC",
) -> dict:
    """Build an IAM+READ model from train-only, dataset-retained score rows."""
    required_datasets = {"IAM", "READ"}
    if set(scores_by_dataset) != required_datasets:
        raise ValueError("combined model requires exactly IAM and READ score inputs")
    if set(source_sha256_by_dataset) != required_datasets:
        raise ValueError("combined model requires one source hash for IAM and READ")
    if minimum_count <= 0:
        raise ValueError("minimum_count must be positive")
    if not 0.0 <= rate_threshold <= 1.0:
        raise ValueError("rate_threshold must be between zero and one")
    if unicode_normalization not in ("none", "NFC"):
        raise ValueError("combined tokenizer Unicode normalization must be none or NFC")

    normalized_required = tuple(
        dict.fromkeys(normalize_text(char, unicode_normalization) for char in required_characters)
    )
    if any(len(char) != 1 for char in normalized_required):
        raise ValueError("required characters must normalize to exactly one code point")

    # pair -> dataset -> [exact observations, connected exact observations]
    observations: dict[str, dict[str, list[int]]] = {}
    original_pairs: dict[str, set[str]] = {dataset: set() for dataset in required_datasets}
    characters = set(normalized_required)
    for dataset in sorted(required_datasets):
        scores = scores_by_dataset[dataset]
        if not scores:
            raise ValueError(f"{dataset} bigram score input is empty")
        for row in scores:
            if row.get("schema_version") != "dtlr.bigram-scores.v5":
                raise ValueError("unsupported bigram score schema")
            if row.get("dataset") != dataset or row.get("split") != "train":
                raise ValueError(
                    f"{dataset} tokenizer input must contain {dataset} train scores only"
                )
            if row.get("connectivity_method") != "dominant-core-v3":
                raise ValueError("combined tokenizer scores do not use frozen dominant-core-v3")
            source_pair = row["pair"]
            if len(source_pair) != 2:
                raise ValueError(
                    f"score pair must contain exactly two code points: {source_pair!r}"
                )
            if source_pair in original_pairs[dataset]:
                raise ValueError(f"duplicate {dataset} score row for pair {source_pair!r}")
            original_pairs[dataset].add(source_pair)
            pair = normalize_text(source_pair, unicode_normalization)
            if len(pair) != 2:
                raise ValueError(
                    "Unicode normalization changes a score pair boundary; regenerate "
                    f"evidence after normalization instead of pooling {source_pair!r} as {pair!r}"
                )
            characters.update(pair)
            count, connected = _exact_connected_count(row)
            dataset_counts = observations.setdefault(pair, {}).setdefault(dataset, [0, 0])
            dataset_counts[0] += count
            dataset_counts[1] += connected

    vocabulary = []
    for pair in sorted(observations):
        dataset_statistics = {}
        total_count = total_connected = 0
        for dataset in sorted(required_datasets):
            count, connected = observations[pair].get(dataset, [0, 0])
            total_count += count
            total_connected += connected
            dataset_statistics[dataset] = {
                "n_exact_alignment": count,
                "n_exact_alignment_connected": connected,
                "exact_alignment_connected_rate": connected / count if count else None,
            }
        pooled_rate = total_connected / total_count if total_count else None
        if (
            pair_allowed(pair, pair_policy)
            and total_count >= minimum_count
            and pooled_rate is not None
            and pooled_rate >= rate_threshold
        ):
            vocabulary.append({
                "token": pair,
                "n_exact_alignment": total_count,
                "n_exact_alignment_connected": total_connected,
                "exact_alignment_connected_rate": pooled_rate,
                "utility": pooled_rate,
                "dataset_statistics": dataset_statistics,
            })

    ordered_tokens = [""] + sorted(characters) + [row["token"] for row in vocabulary]
    vocab = {token: index for index, token in enumerate(ordered_tokens)}
    idx_token = {str(index): token for token, index in vocab.items()}
    return {
        "schema_version": COMBINED_MODEL_SCHEMA,
        "model_version": "iam-read-combined-v1",
        "status": "provisional-combined-policy",
        "datasets": ["IAM", "READ"],
        "training_splits": {"IAM": "train", "READ": "train"},
        "connectivity_method": "dominant-core-v3",
        "score_field": "exact_alignment_connected_rate",
        "count_field": "n_exact_alignment",
        "combination_rule": "pooled-exact-alignment-observations-v1",
        "connected_count_recovery": "round-count-times-rate-with-integrality-check-v1",
        "text_normalization": unicode_normalization,
        "blank_token": "",
        "blank_id": 0,
        "source_scores": {
            dataset: {"sha256": source_sha256_by_dataset[dataset]}
            for dataset in sorted(required_datasets)
        },
        "policy": {
            "minimum_count": minimum_count,
            "rate_threshold": rate_threshold,
            "pair_policy": pair_policy,
            "overlap_resolution": "maximum-total-utility-non-overlapping-v1",
            "single_character_utility": 0.0,
            "required_characters": list(normalized_required),
        },
        "vocab": vocab,
        "idx_token": idx_token,
        "size": len(vocab),
        "eligible_bigram_count": len(vocabulary),
        "vocabulary": vocabulary,
        "note": (
            "IAM+READ train-derived model. Pooled policy thresholds are provisional and "
            "must be evaluated on data excluded from vocabulary construction."
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
        if model.get("schema_version") not in SUPPORTED_MODEL_SCHEMAS:
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
    if model.get("schema_version") not in SUPPORTED_MODEL_SCHEMAS:
        raise ValueError("unsupported tokenizer model schema")
    input_text = text
    text = normalize_text(text, model.get("text_normalization", "none"))
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
    output = {
        "text": text,
        "tokens": [segment["token"] for segment in result.segments],
        "total_utility": result.utility,
        "bigram_token_count": result.bigram_count,
        "segments": list(result.segments),
    }
    if input_text != text:
        output["input_text"] = input_text
    return output
