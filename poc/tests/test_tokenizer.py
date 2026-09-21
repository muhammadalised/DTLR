import json
import tempfile
import unittest
from pathlib import Path

from dtlr_poc.tokenizer import (
    FROZEN_IAM_MANIFEST_SHA256,
    FROZEN_IAM_SCORES_SHA256,
    FROZEN_IAM_SOURCE_MODEL_SHA256,
    HandwritingBigramTokenizer,
    build_frozen_iam_artifact,
    build_model,
    tokenize,
)


def score(pair, count, rate, split="train"):
    return {
        "schema_version": "dtlr.bigram-scores.v5",
        "dataset": "IAM",
        "split": split,
        "pair": pair,
        "connectivity_method": "dominant-core-v3",
        "n_exact_alignment": count,
        "exact_alignment_connected_rate": rate,
    }


class TokenizerTests(unittest.TestCase):
    def test_model_filters_by_count_rate_and_letters(self):
        model = build_model([
            score("th", 30, 0.8),
            score("he", 5, 0.9),
            score("er", 30, 0.4),
            score("r,", 30, 0.9),
        ], "source", minimum_count=20, rate_threshold=0.5)
        self.assertEqual([row["token"] for row in model["vocabulary"]], ["th"])
        self.assertEqual(model["status"], "provisional-demo-policy")
        self.assertEqual(model["size"], len(model["vocab"]))
        self.assertEqual(model["vocab"][""], 0)
        self.assertEqual(model["idx_token"]["0"], "")
        self.assertEqual(
            {int(index): token for index, token in model["idx_token"].items()},
            {index: token for token, index in model["vocab"].items()},
        )
        self.assertLess(model["vocab"]["t"], model["vocab"]["th"])

    def test_rejects_non_training_scores(self):
        with self.assertRaisesRegex(ValueError, "train scores only"):
            build_model([score("th", 30, 0.8, split="valid")], "source", 20, 0.5)

    def test_dynamic_programming_resolves_overlaps_by_utility(self):
        model = build_model([
            score("th", 30, 0.7),
            score("he", 30, 0.9),
        ], "source", minimum_count=20, rate_threshold=0.5)
        result = tokenize("the", model)
        self.assertEqual(result["tokens"], ["t", "he"])
        self.assertEqual(result["bigram_token_count"], 1)

    def test_selects_multiple_non_overlapping_bigrams_and_preserves_space(self):
        model = build_model([
            score("ab", 30, 0.8),
            score("cd", 30, 0.7),
        ], "source", minimum_count=20, rate_threshold=0.5)
        result = tokenize("abcd ab", model)
        self.assertEqual(result["tokens"], ["ab", "cd", " ", "ab"])
        self.assertAlmostEqual(result["total_utility"], 2.3)

    def test_tva_familiar_interface_preserves_dp_and_round_trip(self):
        model = build_model([
            score("th", 30, 0.7),
            score("he", 30, 0.9),
        ], "source", minimum_count=20, rate_threshold=0.5)
        tokenizer = HandwritingBigramTokenizer(model)
        ids = tokenizer.encode("the")
        self.assertEqual([tokenizer.idx_token[index] for index in ids], ["t", "he"])
        self.assertEqual(tokenizer.decode(ids), "the")
        self.assertEqual(tokenizer.decode([0, *ids, 0]), "the")
        self.assertEqual(tokenizer.size, model["size"])

    def test_tva_familiar_interface_rejects_unknown_characters(self):
        tokenizer = HandwritingBigramTokenizer(build_model(
            [score("ab", 30, 0.8)], "source", minimum_count=20, rate_threshold=0.5
        ))
        with self.assertRaisesRegex(ValueError, "absent from tokenizer vocabulary"):
            tokenizer.encode("az")

    def test_tva_familiar_interface_rejects_legacy_model_without_mappings(self):
        model = build_model(
            [score("ab", 30, 0.8)], "source", minimum_count=20, rate_threshold=0.5
        )
        del model["vocab"]
        with self.assertRaisesRegex(ValueError, "rebuild it from train scores"):
            HandwritingBigramTokenizer(model)

    def test_tva_familiar_interface_loads_json(self):
        model = build_model(
            [score("ab", 30, 0.8)], "source", minimum_count=20, rate_threshold=0.5
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.json"
            path.write_text(json.dumps(model), encoding="utf-8")
            tokenizer = HandwritingBigramTokenizer()
            tokenizer.load(path)
        self.assertEqual(tokenizer.decode(tokenizer.encode("ab ab")), "ab ab")

    def frozen_inputs(self):
        characters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
        characters.extend(list("αβγδεζηθικλμνξοπρστυφχψ"))
        self.assertEqual(len(characters), 75)
        pairs = [characters[0] + char for char in characters]
        pairs.extend(characters[1] + char for char in characters[:70])
        scores = [score(pair, 20, 0.5) for pair in pairs]
        self.assertEqual(len(scores), 145)
        source_model = build_model(scores, FROZEN_IAM_SCORES_SHA256, 20, 0.5)
        manifest = {
            "schema_version": "dtlr.export-manifest.v5",
            "datasets": ["IAM"],
            "splits": ["train"],
            "line_count": 5694,
            "evidence_count": 160536,
            "score_count": len(scores),
            "primary_connectivity_method": "dominant-core-v3",
        }
        return scores, manifest, source_model

    def build_frozen(self):
        scores, manifest, source_model = self.frozen_inputs()
        return build_frozen_iam_artifact(
            scores,
            manifest,
            source_model,
            FROZEN_IAM_SCORES_SHA256,
            FROZEN_IAM_MANIFEST_SHA256,
            FROZEN_IAM_SOURCE_MODEL_SHA256,
        )

    def test_frozen_artifact_is_deterministic_and_preserves_vocabulary(self):
        first = self.build_frozen()
        second = self.build_frozen()
        _, _, source_model = self.frozen_inputs()
        self.assertEqual(first, second)
        self.assertEqual(first["vocabulary"], source_model["vocabulary"])
        self.assertEqual(first["vocab"], source_model["vocab"])
        self.assertEqual(first["eligible_bigram_count"], 145)
        self.assertEqual(first["size"], 222)
        self.assertEqual(first["vocab"][""], 0)
        self.assertEqual(set(first["vocab"].values()), set(range(first["size"])))
        self.assertEqual(first["unicode_normalization"], "NFC")
        self.assertEqual(first["evidence_provenance"]["sole_dataset"], "IAM")

    def test_frozen_artifact_rejects_wrong_hash_dataset_split_and_schema(self):
        scores, manifest, source_model = self.frozen_inputs()
        arguments = [
            scores, manifest, source_model, FROZEN_IAM_SCORES_SHA256,
            FROZEN_IAM_MANIFEST_SHA256, FROZEN_IAM_SOURCE_MODEL_SHA256,
        ]
        for index in (3, 4, 5):
            changed = list(arguments)
            changed[index] = "0" * 64
            with self.assertRaisesRegex(ValueError, "unexpected .* SHA-256"):
                build_frozen_iam_artifact(*changed)
        for field, value in (("datasets", ["READ"]), ("splits", ["valid"])):
            changed_manifest = dict(manifest)
            changed_manifest[field] = value
            with self.assertRaisesRegex(ValueError, "IAM train as its sole"):
                build_frozen_iam_artifact(*[scores, changed_manifest, *arguments[2:]])
        changed_manifest = dict(manifest)
        changed_manifest["schema_version"] = "unknown"
        with self.assertRaisesRegex(ValueError, "manifest schema"):
            build_frozen_iam_artifact(*[scores, changed_manifest, *arguments[2:]])

        for field, value in (("dataset", "READ"), ("split", "valid"), ("schema_version", "unknown")):
            changed_scores = [dict(row) for row in scores]
            changed_scores[0][field] = value
            with self.assertRaises(ValueError):
                build_frozen_iam_artifact(*[changed_scores, *arguments[1:]])

    def test_frozen_artifact_rejects_non_reconstructible_source_model(self):
        scores, manifest, source_model = self.frozen_inputs()
        changed_model = json.loads(json.dumps(source_model))
        changed_model["vocabulary"][0]["utility"] = 0.75
        with self.assertRaisesRegex(ValueError, "not exactly reconstructible"):
            build_frozen_iam_artifact(
                scores, manifest, changed_model, FROZEN_IAM_SCORES_SHA256,
                FROZEN_IAM_MANIFEST_SHA256, FROZEN_IAM_SOURCE_MODEL_SHA256,
            )

    def test_frozen_artifact_normalizes_before_existing_dp_segmentation(self):
        scores = [score("éa", 20, 0.8)]
        model = build_model(scores, "source", 20, 0.5)
        model["unicode_normalization"] = "NFC"
        tokenizer = HandwritingBigramTokenizer(model)
        ids = tokenizer.encode("e\u0301a")
        self.assertEqual([tokenizer.idx_token[index] for index in ids], ["éa"])
        self.assertEqual(tokenizer.decode(ids), "éa")


if __name__ == "__main__":
    unittest.main()
