import unittest

from dtlr_poc.handwriting_subwords import (
    HandwritingSubwordTokenizer, collect_text_candidates, train_bpe, train_unigram,
)


class TextSubwordControlTests(unittest.TestCase):
    def test_candidate_counts_use_text_and_preserve_boundaries(self):
        corpus = [{"text": "abc abc!Abc"}]
        candidates = collect_text_candidates(corpus, 3, 2)
        self.assertEqual(set(candidates), {"ab", "abc", "bc"})
        self.assertEqual(candidates["abc"], {"token": "abc", "n_total": 2})
        self.assertFalse(any(" " in t or "!" in t for t in candidates))

    def test_bpe_frequency_control_never_requires_edges(self):
        corpus = [{"text": "ab ab ab cd cd"}]
        candidates = collect_text_candidates(corpus, 2, 1)
        model = train_bpe(corpus, candidates, 1, handwriting_aware=False)
        self.assertEqual(model["algorithm"], "text-only-bpe")
        self.assertEqual(model["merges"][0]["token"], "ab")
        self.assertEqual(model["merges"][0]["text_occurrences_at_merge"], 3)
        self.assertNotIn("connected_occurrences_at_merge", model["merges"][0])
        self.assertNotIn("connectivity_method", model)
        tokenizer = HandwritingSubwordTokenizer(model)
        self.assertEqual(tokenizer.segment("ab cd")["tokens"], ["ab", " ", "c", "d"])
        self.assertEqual(tokenizer.decode(tokenizer.encode("ab cd")), "ab cd")

    def test_unigram_control_fits_without_edges_and_preserves_symbols(self):
        corpus = [{"text": "abc abc!"}]
        candidates = collect_text_candidates(corpus, 3, 1)
        model = train_unigram(corpus, candidates, 2, iterations=4, handwriting_aware=False)
        self.assertEqual(model["schema_version"], "dtlr.text-subwords.v1")
        self.assertEqual(model["algorithm"], "text-only-unigram")
        self.assertEqual(model["policy"]["seed_priority"], "text-occurrence-count")
        tokenizer = HandwritingSubwordTokenizer(model)
        ids = tokenizer.encode("abc abc!")
        self.assertNotIn(0, ids)
        self.assertEqual(tokenizer.decode(ids), "abc abc!")
        self.assertNotIn("span_connectivity_method", model)

    def test_control_vocabularies_do_not_change_when_edges_change(self):
        text = [{"text": "abc abc abd"}]
        candidates = collect_text_candidates(text, 3, 1)
        disconnected = [{**text[0], "edges": {0: (1, 2)}}]
        connected = [{**text[0], "edges": {0: (1, 1)}}]
        for train in (train_bpe, train_unigram):
            first = train(disconnected, candidates, 2, handwriting_aware=False)
            second = train(connected, candidates, 2, handwriting_aware=False)
            self.assertEqual(first, second)

    def test_schema_algorithm_mismatch_rejected(self):
        corpus = [{"text": "abc"}]
        model = train_bpe(corpus, collect_text_candidates(corpus, 3, 1), 1, handwriting_aware=False)
        model["algorithm"] = "handwriting-aware-bpe"
        with self.assertRaises(ValueError):
            HandwritingSubwordTokenizer(model)
