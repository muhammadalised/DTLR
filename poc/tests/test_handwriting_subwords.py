import copy
import json
import math
import unittest

from dtlr_poc.handwriting_subwords import (
    HandwritingSubwordTokenizer, collect_candidates, compare_tokenizers,
    forward_backward, prepare_corpus, span_connected, train_bpe, train_unigram, removal_losses, extend_bigram_training_alphabet,
)
from dtlr_poc.selection import text_sha256


def line(text, components, name="line"):
    return {"line_id": name, "text": text,
            "edges": {i: None if pair is None else tuple(pair) for i, pair in enumerate(components)}}


class HandwritingSubwordsTests(unittest.TestCase):
    def test_long_span_requires_same_component_not_just_connected_edges(self):
        consistent = line("abc", [(1, 1), (1, 1)])
        inconsistent = line("abc", [(1, 1), (2, 2)])
        missing = line("abc", [(1, 1), None])
        self.assertTrue(span_connected(consistent, 0, 3))
        self.assertFalse(span_connected(inconsistent, 0, 3))
        self.assertIsNone(span_connected(missing, 0, 3))
        candidates = collect_candidates([consistent, inconsistent, missing], 3, 1, 0.5)
        self.assertEqual(candidates["abc"]["n_total"], 3)
        self.assertEqual(candidates["abc"]["n_exact_alignment"], 2)
        self.assertEqual(candidates["abc"]["connected_rate"], 0.5)

    def test_candidates_never_cross_space_or_punctuation(self):
        corpus = [line("ab cd!ef", [(1, 1)] * 7)]
        candidates = collect_candidates(corpus, 6, 1, 0.5)
        self.assertEqual(set(candidates), {"ab", "cd", "ef"})

    def test_bpe_prioritizes_visual_support_over_text_frequency(self):
        corpus = [line("ab", [(1, 2)], str(i)) for i in range(10)]
        corpus += [line("cd", [(3, 3)], str(i + 10)) for i in range(3)]
        candidates = collect_candidates(corpus, 2, 1, 0)
        model = train_bpe(corpus, candidates, budget=1)
        self.assertEqual(model["merges"][0]["token"], "cd")
        self.assertEqual(model["merges"][0]["connected_occurrences_at_merge"], 3)

    def test_bpe_can_learn_longer_token_and_applies_ordered_merges(self):
        corpus = [line("abc abc", [(1, 1), (1, 1), None, None, (2, 2), (2, 2)])]
        candidates = collect_candidates(corpus, 3, 1, 0.5)
        model = train_bpe(corpus, candidates, budget=2)
        tokenizer = HandwritingSubwordTokenizer(model)
        self.assertEqual(tokenizer.segment("abc abc")["tokens"], ["abc", " ", "abc"])
        self.assertNotIn(0, tokenizer.encode("abc abc"))
        self.assertEqual(tokenizer.decode(tokenizer.encode("abc abc")), "abc abc")
        self.assertEqual(tokenizer.segment("")["tokens"], [])

    def test_different_handwriting_changes_unigram_candidate_vocabulary(self):
        connected = [line("abc", [(1, 1), (1, 1)])]
        disconnected = [line("abc", [(1, 2), (2, 3)])]
        eligible = collect_candidates(connected, 3, 1, 0.5)
        ineligible = collect_candidates(disconnected, 3, 1, 0.5)
        model = train_unigram(connected, eligible, 3, iterations=3)
        control = train_unigram(disconnected, ineligible, 3, iterations=3)
        self.assertIn("abc", model["vocab"])
        self.assertFalse(any(len(t) > 1 for t in control["vocab"]))
        tokenizer = HandwritingSubwordTokenizer(model)
        self.assertEqual(tokenizer.segment("abc")["tokens"], ["abc"])
        self.assertEqual(tokenizer.decode(tokenizer.encode("abc")), "abc")

    def test_forward_backward_matches_exhaustive_segmentation(self):
        probabilities = {"a": 0.2, "b": 0.3, "ab": 0.5}
        score, counts = forward_backward("ab", {t: math.log(p) for t, p in probabilities.items()})
        self.assertAlmostEqual(math.exp(score), 0.5 + 0.2 * 0.3)
        self.assertAlmostEqual(counts["ab"], 0.5 / 0.56)
        self.assertAlmostEqual(counts["a"], 0.06 / 0.56)
        self.assertAlmostEqual(counts["b"], 0.06 / 0.56)

    def test_em_likelihood_increases_within_each_fixed_vocabulary_stage(self):
        corpus = [line("abcabc", [(1, 1)] * 5)] * 4
        candidates = collect_candidates(corpus, 4, 1, 0.5)
        model = train_unigram(corpus, candidates, budget=2, iterations=5)
        self.assertEqual(len(model["vocabulary"]), 2)
        for stage in model["training_curve"]:
            history = stage["pre_update_log_likelihoods"]
            self.assertTrue(all(right >= left - 1e-8 for left, right in zip(history, history[1:])))
        HandwritingSubwordTokenizer(model)

    def test_unigram_viterbi_matches_exhaustive_search(self):
        corpus = [line("abab", [(1, 1)] * 3)] * 3
        candidates = collect_candidates(corpus, 3, 1, 0.5)
        model = train_unigram(corpus, candidates, budget=3, iterations=2)
        tokenizer = HandwritingSubwordTokenizer(model)
        text = "abab"
        def enumerate_paths(start):
            if start == len(text):
                return [([], 0.0)]
            result = []
            for token, score in model["log_probs"].items():
                if text.startswith(token, start):
                    for suffix, value in enumerate_paths(start + len(token)):
                        result.append(([token] + suffix, score + value))
            return result
        tokens = tokenizer.segment(text)["tokens"]
        value = sum(model["log_probs"][t] for t in tokens)
        self.assertAlmostEqual(value, max(v for _, v in enumerate_paths(0)))

    def test_models_are_deterministic_and_preserve_ctc_singletons(self):
        corpus = [line("éab!", [(1, 1), (1, 1), (1, 2)])]
        candidates = collect_candidates(corpus, 3, 1, 0.5)
        for train in (train_bpe, train_unigram):
            first, second = train(corpus, candidates, budget=2), train(corpus, candidates, budget=2)
            self.assertEqual(json.dumps(first, sort_keys=True), json.dumps(second, sort_keys=True))
            tokenizer = HandwritingSubwordTokenizer(first)
            self.assertEqual(tokenizer.decode(tokenizer.encode("e\u0301ab!")), "éab!")
            self.assertEqual(tokenizer.decode([0]), "")
            with self.assertRaisesRegex(ValueError, "absent"):
                tokenizer.encode("z")
            broken = copy.deepcopy(first)
            broken["vocab"][""] = 1
            with self.assertRaises(ValueError):
                HandwritingSubwordTokenizer(broken)

    def test_comparison_counts_abstentions_separately(self):
        corpus = [line("abc", [(1, 1), None])]
        candidates = collect_candidates(corpus, 2, 1, 0.5)
        tokenizer = HandwritingSubwordTokenizer(train_bpe(corpus, candidates, 1))
        report = compare_tokenizers(corpus, {"bpe": tokenizer})["tokenizers"]["bpe"]
        self.assertEqual(report["known_internal_edges"], 1)
        self.assertEqual(report["connected_edges_cut"], 0)
        self.assertEqual(report["known_boundary_edges"], 0)
        self.assertEqual(report["round_trip_or_blank_failures"], 0)

    def test_removal_loss_matches_full_forward_likelihood_difference(self):
        scores = {"a": math.log(0.2), "b": math.log(0.3), "ab": math.log(0.5)}
        loss = removal_losses({"ab": 7}, scores, {"ab"})["ab"]
        original, _ = forward_backward("ab", scores)
        without, _ = forward_backward("ab", {t: v for t, v in scores.items() if t != "ab"})
        self.assertAlmostEqual(loss, 7 * (original - without))

    def test_derived_alphabet_copy_keeps_frozen_ids_and_inventory(self):
        from dtlr_poc.tokenizer import build_model, HandwritingBigramTokenizer
        score = {"schema_version": "dtlr.bigram-scores.v5", "dataset": "IAM", "split": "train",
                 "connectivity_method": "dominant-core-v3", "pair": "ab",
                 "n_exact_alignment": 20, "exact_alignment_connected_rate": 0.8}
        frozen = build_model([score], "hash", 20, 0.5)
        before = copy.deepcopy(frozen)
        derived = extend_bigram_training_alphabet(frozen, {"a", "b", "#"})
        self.assertEqual(frozen, before)
        for token, index in frozen["vocab"].items():
            self.assertEqual(derived["vocab"][token], index)
        self.assertEqual(derived["vocabulary"], frozen["vocabulary"])
        self.assertEqual(derived["size"], frozen["size"] + 1)
        tokenizer = HandwritingBigramTokenizer(derived)
        self.assertEqual(tokenizer.decode(tokenizer.encode("ab#")), "ab#")

    def test_corpus_guard_rejects_leakage_and_changed_text_and_evidence(self):
        record = {"dataset": "IAM", "split": "train", "line_id": "id", "transcription": "ab"}
        selection = {"schema_version": "dtlr.iam-selection.v1", "dataset": "IAM", "split": "train",
                     "requested_count": 1, "lines": [{"id": "id", "transcription_sha256": text_sha256("ab")}]}
        manifest = {"schema_version": "dtlr.export-manifest.v5", "datasets": ["IAM"],
                    "splits": ["train"], "primary_connectivity_method": "dominant-core-v3",
                    "line_count": 1, "evidence_count": 1}
        row = {"schema_version": "dtlr.bigram-evidence.v5", "dataset": "IAM", "split": "train",
               "connectivity_method": "dominant-core-v3", "line_id": "id", "left_gt_index": 0,
               "right_gt_index": 1, "pair": "ab", "usable": True, "left_alignment": "match",
               "right_alignment": "match", "left_dominant_component_id": 1,
               "right_dominant_component_id": 1, "connected": True}
        corpus = prepare_corpus([record], [row], selection, manifest)
        self.assertTrue(span_connected(corpus[0], 0, 2))
        for key, value in (("dataset", "other"), ("split", "valid"), ("transcription", "ac")):
            changed = dict(record, **{key: value})
            with self.assertRaises(ValueError):
                prepare_corpus([changed], [row], selection, manifest)
        for key, value in (("split", "test"), ("pair", "ac"), ("connected", False)):
            changed = dict(row, **{key: value})
            with self.assertRaises(ValueError):
                prepare_corpus([record], [changed], selection, manifest)
        duplicate_manifest = dict(manifest, evidence_count=2)
        with self.assertRaisesRegex(ValueError, "duplicate edge"):
            prepare_corpus([record], [row, row], selection, duplicate_manifest)
        mismatched = dict(row, left_alignment="substitution")
        abstain = prepare_corpus([record], [mismatched], selection, manifest)
        self.assertIsNone(span_connected(abstain[0], 0, 2))


if __name__ == "__main__":
    unittest.main()
