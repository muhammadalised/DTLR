import json
import tempfile
import unittest
from pathlib import Path

from dtlr_poc.tokenizer import (
    HandwritingBigramTokenizer,
    build_combined_model,
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


def dataset_score(dataset, pair, count, rate, split="train"):
    row = score(pair, count, rate, split)
    row["dataset"] = dataset
    return row


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

    def test_combined_model_pools_integer_observations_and_retains_sources(self):
        model = build_combined_model(
            {
                "IAM": [
                    dataset_score("IAM", "th", 20, 0.8),
                    dataset_score("IAM", "ab", 10, 0.4),
                ],
                "READ": [
                    dataset_score("READ", "th", 10, 0.4),
                    dataset_score("READ", "äß", 25, 0.8),
                ],
            },
            {"IAM": "iam-hash", "READ": "read-hash"},
            minimum_count=20,
            rate_threshold=0.5,
        )
        vocabulary = {row["token"]: row for row in model["vocabulary"]}
        self.assertEqual(set(vocabulary), {"th", "äß"})
        self.assertEqual(vocabulary["th"]["n_exact_alignment"], 30)
        self.assertEqual(vocabulary["th"]["n_exact_alignment_connected"], 20)
        self.assertAlmostEqual(vocabulary["th"]["utility"], 2 / 3)
        self.assertEqual(
            vocabulary["th"]["dataset_statistics"]["IAM"]["n_exact_alignment"], 20
        )
        self.assertEqual(model["source_scores"]["READ"]["sha256"], "read-hash")
        self.assertEqual(model["model_version"], "iam-read-combined-v1")

    def test_combined_model_rejects_nontraining_or_boundary_changing_normalization(self):
        with self.assertRaisesRegex(ValueError, "train scores only"):
            build_combined_model(
                {
                    "IAM": [dataset_score("IAM", "th", 20, 0.8, split="valid")],
                    "READ": [dataset_score("READ", "th", 20, 0.8)],
                },
                {"IAM": "iam", "READ": "read"},
                20,
                0.5,
            )
        with self.assertRaisesRegex(ValueError, "changes a score pair boundary"):
            build_combined_model(
                {
                    "IAM": [
                        dataset_score("IAM", "A\N{COMBINING RING ABOVE}", 20, 0.8)
                    ],
                    "READ": [dataset_score("READ", "th", 20, 0.8)],
                },
                {"IAM": "iam", "READ": "read"},
                20,
                0.5,
            )

    def test_combined_tokenizer_normalizes_input_explicitly(self):
        model = build_combined_model(
            {
                "IAM": [dataset_score("IAM", "üb", 20, 0.8)],
                "READ": [dataset_score("READ", "üb", 20, 0.9)],
            },
            {"IAM": "iam", "READ": "read"},
            20,
            0.5,
        )
        tokenizer = HandwritingBigramTokenizer(model)
        result = tokenizer.segment("u\N{COMBINING DIAERESIS}b")
        self.assertEqual(result["text"], "üb")
        self.assertEqual(result["input_text"], "u\N{COMBINING DIAERESIS}b")
        self.assertEqual(result["tokens"], ["üb"])
        self.assertEqual(tokenizer.decode(tokenizer.encode("u\N{COMBINING DIAERESIS}b")), "üb")


if __name__ == "__main__":
    unittest.main()
