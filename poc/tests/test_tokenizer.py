import unittest

from dtlr_poc.tokenizer import build_model, tokenize


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


if __name__ == "__main__":
    unittest.main()
