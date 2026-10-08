import copy
import hashlib
import json
from pathlib import Path
import unittest

from dtlr_poc.handwriting_subwords import HandwritingSubwordTokenizer
from dtlr_poc.subword_freeze import (
    MODELS, SOURCE_SHA256, RELEASE_VERSION, build_frozen_subword_artifact,
    build_release_manifest, canonical_bytes, digest, load_frozen_subword_model,
    reconstruct_source_model, validate_frozen_subword_artifact,
)

ROOT = Path(__file__).resolve().parents[2]


class FrozenSubwordTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.artifacts = {algorithm: load_frozen_subword_model(
            ROOT / "poc/frozen" / (item["model_version"] + ".json"))
            for algorithm, item in MODELS.items()}

    def test_source_reconstruction_and_release_bytes_are_exact(self):
        for algorithm, artifact in self.artifacts.items():
            with self.subTest(algorithm=algorithm):
                source = reconstruct_source_model(artifact)
                expected = MODELS[algorithm]["source_model_sha256"]
                self.assertEqual(digest(source), expected)
                before = copy.deepcopy(source)
                rebuilt = build_frozen_subword_artifact(source, expected)
                self.assertEqual(source, before)
                path = ROOT / "poc/frozen" / (artifact["model_version"] + ".json")
                self.assertEqual(canonical_bytes(rebuilt), path.read_bytes())

    def test_manifest_checksums_match_both_artifacts_and_bigram_is_unchanged(self):
        path = ROOT / "poc/frozen" / (RELEASE_VERSION + ".manifest.json")
        self.assertEqual(canonical_bytes(build_release_manifest(self.artifacts)), path.read_bytes())
        manifest = json.loads(path.read_text())
        for entry in manifest["models"]:
            raw = (path.parent / entry["filename"]).read_bytes()
            self.assertEqual(hashlib.sha256(raw).hexdigest(), entry["sha256"])
        bigram = ROOT / "poc/frozen/iam-english-handwriting-bigram-v1.json"
        self.assertEqual(hashlib.sha256(bigram.read_bytes()).hexdigest(), SOURCE_SHA256["frozen_bigram"])

    def test_all_pieces_round_trip_with_identical_ids_and_no_blank(self):
        for algorithm, artifact in self.artifacts.items():
            tokenizer = HandwritingSubwordTokenizer(artifact)
            source = HandwritingSubwordTokenizer(reconstruct_source_model(artifact))
            self.assertEqual(tokenizer.size, 225)
            self.assertEqual(tokenizer.decode([0]), "")
            for token in artifact["vocab"]:
                ids = tokenizer.encode(token)
                self.assertEqual(ids, source.encode(token))
                self.assertNotIn(0, ids)
                self.assertEqual(tokenizer.decode(ids), token)
            with self.assertRaisesRegex(ValueError, "absent"):
                tokenizer.encode("\u2603")

    def test_changed_ids_evidence_policy_or_dataset_are_rejected(self):
        original = self.artifacts["handwriting-aware-bpe"]
        mutations = [
            lambda m: m.update(dataset="other"),
            lambda m: m.update(training_split="test"),
            lambda m: m["candidate_policy"].update(rate_threshold=0.6),
            lambda m: m["candidate_policy"].update(max_token_length=5),
            lambda m: m["vocabulary"][0].update(n_connected=999),
            lambda m: m["evidence_provenance"].update(sole_split="valid"),
            lambda m: m["vocab"].update({"a": m["vocab"]["b"], "b": m["vocab"]["a"]}),
            lambda m: m.update(unicode_normalization="NFKC"),
        ]
        for index, mutate in enumerate(mutations):
            model = copy.deepcopy(original)
            mutate(model)
            with self.subTest(index=index), self.assertRaises(ValueError):
                HandwritingSubwordTokenizer(model)

    def test_reordering_valid_bpe_merges_is_rejected(self):
        model = copy.deepcopy(self.artifacts["handwriting-aware-bpe"])
        # he and er are both valid first merges; their order is part of the freeze.
        model["merges"][0], model["merges"][1] = model["merges"][1], model["merges"][0]
        with self.assertRaises(ValueError):
            HandwritingSubwordTokenizer(model)

    def test_renormalized_unigram_probability_changes_are_rejected(self):
        import math
        model = copy.deepcopy(self.artifacts["handwriting-aware-unigram"])
        a, b = "a", "b"
        pa, pb = math.exp(model["log_probs"][a]), math.exp(model["log_probs"][b])
        delta = min(pa, pb) * 0.01
        model["log_probs"][a] = math.log(pa + delta)
        model["log_probs"][b] = math.log(pb - delta)
        with self.assertRaises(ValueError):
            HandwritingSubwordTokenizer(model)

    def test_version_metadata_and_status_cannot_be_relabelled(self):
        original = self.artifacts["handwriting-aware-unigram"]
        for field, value in (("status", "experimental-not-frozen"),
                             ("model_version", "iam-english-handwriting-unigram-v2"),
                             ("freeze", {})):
            model = copy.deepcopy(original)
            model[field] = value
            with self.assertRaises(ValueError):
                validate_frozen_subword_artifact(model)
            with self.assertRaises(ValueError):
                HandwritingSubwordTokenizer(model)

    def test_builder_rejects_wrong_source_hash_or_modified_snapshot(self):
        for algorithm, artifact in self.artifacts.items():
            source = reconstruct_source_model(artifact)
            with self.assertRaises(ValueError):
                build_frozen_subword_artifact(source, "0" * 64)
            source["policy"]["subword_budget"] = 144
            with self.assertRaises(ValueError):
                build_frozen_subword_artifact(source, MODELS[algorithm]["source_model_sha256"])

    def test_release_requires_both_models(self):
        with self.assertRaises(ValueError):
            build_release_manifest({"handwriting-aware-bpe": self.artifacts["handwriting-aware-bpe"]})

    def test_strict_loader_rejects_changed_file_bytes(self):
        from unittest.mock import patch
        model = self.artifacts["handwriting-aware-bpe"]
        # Even a semantically identical reserialization has a different file hash.
        with patch("pathlib.Path.read_bytes", return_value=canonical_bytes(model) + b" "):
            with self.assertRaisesRegex(ValueError, "file SHA-256"):
                load_frozen_subword_model("unused.json")
