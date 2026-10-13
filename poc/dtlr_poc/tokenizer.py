"""Historical IAM bigram inference and compatibility with TVA-owned selection."""

from dataclasses import dataclass
import json
from pathlib import Path
import unicodedata
from ._tva_training import load_training


MODEL_SCHEMA = "dtlr.handwriting-bigram-tokenizer.v1"
FROZEN_IAM_MODEL_VERSION = "iam-english-handwriting-bigram-v1"
FROZEN_IAM_SCORES_SHA256 = "45c86a9f703ebc3684563ec2a840e7c34241aa76e84d692be46cf11bfec27055"
FROZEN_IAM_MANIFEST_SHA256 = "82cb95a14e71ab5b70c868ca1676b3f628552fabac395fbd30039fc7798bf190"
FROZEN_IAM_SOURCE_MODEL_SHA256 = "36db922860cbd501a6c5a1297dbf7d998abdbe1a337513acb00b44e74fbd9334"


def pair_allowed(*args, **kwargs):
    """Historical API; selection policy is implemented in TVA."""
    return load_training().pair_allowed(*args, **kwargs)


def build_model(*args, **kwargs):
    """Historical API; bigram selection is implemented in TVA."""
    return load_training().build_bigram_model(*args, **kwargs)


def build_frozen_iam_artifact(
    scores: list[dict],
    manifest: dict,
    source_model: dict,
    source_scores_sha256: str,
    source_manifest_sha256: str,
    source_model_sha256: str,
) -> dict:
    """Reproduce and formalize the pinned IAM-only experimental vocabulary.

    This freeze fixes a pre-downstream experiment configuration. It does not
    claim that the provisional count/rate thresholds are statistically optimal.
    """
    expected_hashes = {
        "score file": (source_scores_sha256, FROZEN_IAM_SCORES_SHA256),
        "evidence manifest": (source_manifest_sha256, FROZEN_IAM_MANIFEST_SHA256),
        "source model": (source_model_sha256, FROZEN_IAM_SOURCE_MODEL_SHA256),
    }
    for source, (actual, expected) in expected_hashes.items():
        if actual != expected:
            raise ValueError(f"unexpected {source} SHA-256: {actual}; expected {expected}")

    if manifest.get("schema_version") != "dtlr.export-manifest.v5":
        raise ValueError("unsupported IAM evidence manifest schema")
    if manifest.get("datasets") != ["IAM"] or manifest.get("splits") != ["train"]:
        raise ValueError("frozen artifact requires IAM train as its sole evidence source")
    if manifest.get("primary_connectivity_method") != "dominant-core-v3":
        raise ValueError("IAM evidence manifest does not use dominant-core-v3")
    if manifest.get("score_count") != len(scores):
        raise ValueError("IAM evidence manifest score count does not match the score file")

    reconstructed = build_model(
        scores,
        source_sha256=source_scores_sha256,
        minimum_count=20,
        rate_threshold=0.5,
        pair_policy="letters-only",
        required_characters=(" ",),
    )
    if reconstructed != source_model:
        raise ValueError("pinned IAM source model is not exactly reconstructible from the scores")
    if reconstructed["eligible_bigram_count"] != 145 or reconstructed["size"] != 222:
        raise ValueError("pinned IAM vocabulary counts differ from the audited 145/222 counts")

    artifact = dict(reconstructed)
    artifact.update({
        "model_version": FROZEN_IAM_MODEL_VERSION,
        "status": "frozen-experiment-policy",
        "unicode_normalization": "NFC",
        "evidence_provenance": {
            "sole_dataset": "IAM",
            "sole_split": "train",
            "connectivity_method": "dominant-core-v3",
            "line_count": manifest["line_count"],
            "evidence_count": manifest["evidence_count"],
            "score_count": manifest["score_count"],
            "source_scores_sha256": source_scores_sha256,
            "source_manifest_sha256": source_manifest_sha256,
            "source_model_sha256": source_model_sha256,
        },
        "reconstruction": {
            "source_model_exact_match": True,
            "deterministic_order": "blank, Unicode-sorted singletons, token-sorted eligible bigrams",
        },
        "downstream_contract": {
            "vocabulary_selection": "frozen from IAM train handwriting evidence only",
            "reference_segmentation": "maximum-total-utility-non-overlapping-v1",
            "task_alphabet": "consumer responsibility",
            "segmentation_policy": "consumer responsibility; must be declared separately",
        },
        "note": (
            "Frozen IAM-only experiment policy. The 20-observation and 0.5-rate thresholds "
            "were provisional demonstration settings and are not a held-out optimality claim. "
            "No downstream data or results were used to select or validate the bigrams."
        ),
    })
    return artifact


class HandwritingBigramTokenizer:
    """TVA-familiar tokenizer interface with handwriting-aware DP segmentation.

    The public shape intentionally mirrors TVA's BigramTokenizer (`vocab`,
    `idx_token`, `size`, `load`, `encode`, and `decode`). This historical IAM
    reference uses maximum-total-utility DP for text segmentation. Vocabulary
    selection is implemented in TVA and reached through the compatibility API.
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
        normalization = self.model.get("unicode_normalization")
        if normalization is not None:
            if normalization != "NFC":
                raise ValueError(f"unsupported Unicode normalization: {normalization}")
            text = unicodedata.normalize(normalization, text)
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
