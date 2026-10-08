import unittest

from dtlr_poc.span_review import select_span_review, span_group


class SpanReviewTests(unittest.TestCase):
    def test_interior_disagreement_is_distinct_from_broken_edge(self):
        line = {"edges": {0: (1, 1), 1: (2, 2)}}
        self.assertEqual(span_group(line, 0, 3), "interior-component-disagreement")
        line["edges"][1] = (1, 2)
        self.assertEqual(span_group(line, 0, 3), "disconnected")
        line["edges"][1] = None
        self.assertEqual(span_group(line, 0, 3), "abstained")
        line["edges"][1] = (1, 1)
        self.assertEqual(span_group(line, 0, 3), "supported")

    def test_sampling_reproducible_and_covers_each_stratum(self):
        corpus = [{"line_id": str(i), "text": "abcd", "edges":
                   {0: (i + 1, i + 1), 1: (i + 1, i + 1), 2: (i + 1, i + 1)}}
                  for i in range(8)]
        first, counts = select_span_review(corpus, {"abc", "abcd"}, "fixed", 2)
        second, _ = select_span_review(list(reversed(corpus)), {"abcd", "abc"}, "fixed", 2)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 4)
        self.assertEqual(counts, {"length-3/supported": 8, "length-4/supported": 8})
        self.assertEqual(len({r["span_id"] for r in first}), 4)

    def test_stress_sample_is_separate_and_shortage_is_explicit(self):
        corpus = [{"line_id": "one", "text": "abc",
                   "edges": {0: (1, 1), 1: (2, 2)}}]
        rows, counts = select_span_review(corpus, {"abc"}, "fixed", 2, 4)
        self.assertEqual(len(rows), 1)
        self.assertEqual(counts["length-3/interior-component-disagreement"], 1)
        self.assertEqual(rows[0]["group"], "interior-component-disagreement")

    def test_nonletter_and_short_tokens_rejected(self):
        for tokens in ({"ab"}, {"a b"}, {"abcdefg"}):
            with self.assertRaises(ValueError):
                select_span_review([], tokens, "fixed")
