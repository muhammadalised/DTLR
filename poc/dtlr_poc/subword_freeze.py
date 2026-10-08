"""Freeze the audited IAM handwriting BPE/Unigram snapshot without refitting."""
import copy
import hashlib
import json
from pathlib import Path

FROZEN_STATUS = "frozen-experiment-policy"
FREEZE_SCHEMA = "dtlr.handwriting-subword-freeze.v1"
RELEASE_SCHEMA = "dtlr.handwriting-subword-release.v1"
RELEASE_VERSION = "iam-english-handwriting-subwords-v1"
REVIEWED_COMMIT = "9cecdf3574c7b3193a743f367fa61f12c2839239"
CANDIDATES_SHA256 = "a0d7cd4569aa8feffe65a8816b41e6579ab937efab8d9cf4ab25513b1762aebc"
SOURCE_SHA256 = {
    "detections": "5e68b4c2ef1db15b5d1776ba1d4b7696ba9289f81dd4dd018598dd7d6aea2b32",
    "evidence": "9c75d6d240127c2e6a6c4dffd83c491beab6a31dc4be204461861c1913740bd3",
    "frozen_bigram": "2fbb81479211454d8ad028d8af76eb26e76b75b80408a1a1e1f4efa15ab26a92",
    "manifest": "82cb95a14e71ab5b70c868ca1676b3f628552fabac395fbd30039fc7798bf190",
    "scores": "45c86a9f703ebc3684563ec2a840e7c34241aa76e84d692be46cf11bfec27055",
    "selection": "7893dfba4febe6df99cf0bdb0c74fabe7b736d2a6af05033d8638b90455bc1c2",
}
MODELS = {
    "handwriting-aware-bpe": {
        "model_version": "iam-english-handwriting-bpe-v1",
        "source_filename": "hw-bpe.json",
        "source_model_sha256": "9dedba1087e8e011633251b22159f857fb2e80babc15c877f7ffb03d7a985aad",
        "reference_segmentation": "ordered-BPE-merges",
    },
    "handwriting-aware-unigram": {
        "model_version": "iam-english-handwriting-unigram-v1",
        "source_filename": "hw-unigram.json",
        "source_model_sha256": "1ea98cc926fa3658dc6455a8d9e24ebb3a42b72e15848e9aca0bb7bfa00753c5",
        "reference_segmentation": "maximum-log-probability-Viterbi",
    },
}


FROZEN_ARTIFACT_SHA256 = {
    "handwriting-aware-bpe": "a504c2306200faf6eab98f3feef6f98b81e6e6c6c7f707cf31aa3399a660be47",
    "handwriting-aware-unigram": "af2bc8455afd796367737857f3a36f1deb45f07ec9783a4bbd7f15057f2b9476",
}


def canonical_bytes(value):
    """Original snapshot serialization, also used for deterministic releases."""
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True,
                       allow_nan=False) + "\n").encode("utf-8")


def digest(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def freeze_metadata(algorithm):
    item = MODELS[algorithm]
    return {
        "schema_version": FREEZE_SCHEMA,
        "release_version": RELEASE_VERSION,
        "release_date": "2026-10-08",
        "reviewed_implementation_commit": REVIEWED_COMMIT,
        "source_model_sha256": item["source_model_sha256"],
        "source_model_status": "experimental-not-frozen",
        "candidate_pool_sha256": CANDIDATES_SHA256,
        "candidate_pool_count": 303,
        "singleton_count": 79,
        "subword_count": 145,
        "ctc_class_count": 225,
        "source_model_exact_reconstruction": True,
        "reference_segmentation": item["reference_segmentation"],
        "inference_inputs": "text only; NFC; case, spaces and punctuation preserved",
        "review_status": "22-span AI-assisted train pilot and informal user qualitative review; case-level human annotations pending",
        "scientific_limitations": [
            "Frozen experimental configuration; not statistically optimal or held-out validated.",
            "Count 20/rate 0.5 are inherited exploratory settings; length cap 6 is exploratory.",
            "Long-span pair-facing component consistency is conservative and has localization/abstention limitations.",
            "Handwriting constrains Unigram candidates and initialization; its EM objective is text likelihood, not joint visual likelihood.",
            "Training comparisons reuse selection evidence and do not establish recognition gains.",
        ],
        "consumer_contract": {
            "frozen_content": "preserve vocabulary, IDs, evidence counts, policies, merges/probabilities and provenance",
            "derived_variants": "alphabet, IDs or encoding changes require a separately versioned derived artifact",
            "unknown_characters": "raise an error; no silent unknown or byte fallback",
            "blank": "empty token at ID 0; never emitted by encoding",
        },
    }


def validate_source_model(source, source_model_sha256):
    algorithm = source.get("algorithm")
    if algorithm not in MODELS:
        raise ValueError("freeze requires the pinned IAM handwriting-aware BPE or Unigram")
    expected = MODELS[algorithm]["source_model_sha256"]
    if source_model_sha256 != expected or digest(source) != expected:
        raise ValueError("pinned source model SHA-256 or exact model reconstruction differs")
    if (source.get("schema_version") != "dtlr.handwriting-subwords.v1"
            or source.get("status") != "experimental-not-frozen"
            or source.get("dataset") != "IAM" or source.get("training_split") != "train"
            or source.get("connectivity_method") != "dominant-core-v3"
            or source.get("span_connectivity_method") != "pairwise-consistent-dominant-component-v1"
            or source.get("unicode_normalization") != "NFC"):
        raise ValueError("source is not the audited IAM train handwriting snapshot")
    provenance = source["evidence_provenance"]
    if (provenance.get("source_sha256") != SOURCE_SHA256
            or provenance.get("sole_dataset") != "IAM" or provenance.get("sole_split") != "train"
            or provenance.get("line_count") != 5694):
        raise ValueError("source does not carry the pinned sole-IAM-train evidence")
    if (source["size"] != 225 or len(source["vocabulary"]) != 145
            or sum(len(t) == 1 for t in source["vocab"]) != 79):
        raise ValueError("source vocabulary counts differ from 79/145/225")
    # General encoder validation checks inverse IDs, ordered merges/probabilities,
    # normalization, complete fallback coverage and blank semantics.
    from .handwriting_subwords import HandwritingSubwordTokenizer
    HandwritingSubwordTokenizer(source)


def build_frozen_subword_artifact(source, source_model_sha256):
    validate_source_model(source, source_model_sha256)
    artifact = copy.deepcopy(source)
    algorithm = source["algorithm"]
    artifact["model_version"] = MODELS[algorithm]["model_version"]
    artifact["status"] = FROZEN_STATUS
    artifact["freeze"] = freeze_metadata(algorithm)
    return artifact


def reconstruct_source_model(artifact):
    """Restore every original model/provenance field from the release."""
    source = copy.deepcopy(artifact)
    del source["model_version"], source["freeze"]
    source["status"] = "experimental-not-frozen"
    return source


def validate_frozen_subword_artifact(artifact):
    algorithm = artifact.get("algorithm")
    if algorithm not in MODELS:
        raise ValueError("unknown frozen handwriting subword algorithm")
    if (artifact.get("status") != FROZEN_STATUS
            or artifact.get("model_version") != MODELS[algorithm]["model_version"]
            or artifact.get("freeze") != freeze_metadata(algorithm)):
        raise ValueError("frozen subword version/status/metadata differs from the release")
    if digest(artifact) != FROZEN_ARTIFACT_SHA256[algorithm]:
        raise ValueError("frozen subword canonical artifact SHA-256 differs")
    validate_source_model(reconstruct_source_model(artifact), MODELS[algorithm]["source_model_sha256"])


def load_frozen_subword_model(path):
    raw = Path(path).read_bytes()
    model = json.loads(raw)
    validate_frozen_subword_artifact(model)
    if hashlib.sha256(raw).hexdigest() != FROZEN_ARTIFACT_SHA256[model["algorithm"]]:
        raise ValueError("frozen subword file SHA-256 differs")
    return model


def build_release_manifest(artifacts):
    if set(artifacts) != set(MODELS):
        raise ValueError("release must contain both pinned handwriting models")
    entries = []
    for algorithm, model in sorted(artifacts.items()):
        validate_frozen_subword_artifact(model)
        entries.append({"algorithm": algorithm, "model_version": model["model_version"],
                        "filename": model["model_version"] + ".json", "sha256": digest(model),
                        "source_model_sha256": MODELS[algorithm]["source_model_sha256"],
                        "ctc_class_count": 225, "singleton_count": 79, "subword_count": 145})
    return {"schema_version": RELEASE_SCHEMA, "release_version": RELEASE_VERSION,
            "status": FROZEN_STATUS, "release_date": "2026-10-08",
            "sole_dataset": "IAM", "sole_split": "train", "models": entries,
            "candidate_pool_sha256": CANDIDATES_SHA256,
            "source_sha256": SOURCE_SHA256,
            "existing_bigram_artifact_unchanged": True,
            "scientific_status": "Frozen experimental release before recognition evaluation; no optimality or recognition-performance claim."}
