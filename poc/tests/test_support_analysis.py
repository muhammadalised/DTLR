import unittest

from dtlr_poc.support_analysis import (
    best_exploratory_candidate,
    distribution,
    shared_component_support,
    threshold_candidates,
)


class SupportAnalysisTests(unittest.TestCase):
    def test_selects_shared_component_with_best_bidirectional_support(self):
        evidence = {
            "left_core_component_pixel_counts": {1: 80, 2: 20},
            "right_core_component_pixel_counts": {1: 10, 2: 40, 3: 50},
            "shared_core_components": {1, 2},
        }
        result = shared_component_support(evidence)
        self.assertEqual(result["component_id"], 2)
        self.assertAlmostEqual(result["left_share"], 0.2)
        self.assertAlmostEqual(result["right_share"], 0.4)
        self.assertAlmostEqual(result["bidirectional_support"], 0.2)

    def test_empty_shared_set_has_zero_support(self):
        result = shared_component_support({
            "left_core_component_pixel_counts": {1: 10},
            "right_core_component_pixel_counts": {2: 10},
            "shared_core_components": set(),
        })
        self.assertIsNone(result["component_id"])
        self.assertEqual(result["bidirectional_support"], 0.0)

    def test_threshold_metrics_separate_example_labels(self):
        rows = [
            {"bidirectional_support": 0.45, "manual_visual_connectivity": "connected"},
            {"bidirectional_support": 0.30, "manual_visual_connectivity": "connected"},
            {"bidirectional_support": 0.05, "manual_visual_connectivity": "disconnected"},
            {"bidirectional_support": 0.01, "manual_visual_connectivity": "disconnected"},
        ]
        candidates = threshold_candidates(rows)
        best = best_exploratory_candidate(candidates)
        self.assertEqual(best["threshold"], 0.30)
        self.assertEqual(best["true_connected"], 2)
        self.assertEqual(best["false_connected"], 0)
        self.assertEqual(best["balanced_accuracy"], 1.0)

    def test_distribution_reports_quartiles(self):
        result = distribution([0.0, 0.25, 0.5, 1.0])
        self.assertEqual(result["count"], 4)
        self.assertEqual(result["minimum"], 0.0)
        self.assertEqual(result["maximum"], 1.0)
        self.assertAlmostEqual(result["median"], 0.375)


if __name__ == "__main__":
    unittest.main()
