import json
import tempfile
import unittest
from pathlib import Path

from dtlr_poc.tokenizer import HandwritingBigramTokenizer, build_model, tokenize


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


if __name__ == "__main__":
    unittest.main()
