import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image

from dtlr_poc.alignment import align_monotonic, gt_detection_map
from dtlr_poc.ccl import component_ids_in_box, label_ink, pair_component_evidence
from dtlr_poc.evidence import aggregate, line_evidence


class AlignmentTests(unittest.TestCase):
    def test_identity(self):
        items = align_monotonic("ink", list("ink"))
        self.assertEqual([item.operation for item in items], ["match"] * 3)

    def test_missing_detection_preserves_gt_index(self):
        mapping = gt_detection_map("ink", list("ik"))
        self.assertEqual(mapping[1].operation, "missing_detection")
        self.assertEqual(mapping[2].detection_index, 1)


class CCLTests(unittest.TestCase):
    def test_shared_ink_component(self):
        image = np.full((5, 8), 255, dtype=np.uint8)
        image[2, 1:7] = 0
        labels, count = label_ink(image, threshold=100)
        self.assertEqual(count, 1)
        left = component_ids_in_box(labels, [0, 0, 5, 5])
        right = component_ids_in_box(labels, [4, 0, 8, 5])
        self.assertEqual(left & right, {1})

    def test_raster_safe_cores_do_not_regain_fractional_overlap(self):
        image = np.full((7, 11), 255, dtype=np.uint8)
        image[3, 1:5] = 0
        image[3, 6:10] = 0
        labels, _ = label_ink(image, threshold=100)
        result = pair_component_evidence(
            labels,
            [0.1, 0, 4.915, 7],
            [4.882, 0, 9.9, 7],
        )
        self.assertTrue(result["connected_exclusive_core_v2"])
        self.assertFalse(result["connected_exclusive_core_v2_1"])
        self.assertEqual(result["left_core_box"][2], 4)
        self.assertEqual(result["right_core_box"][0], 5)


class AggregationTests(unittest.TestCase):
    def test_splits_are_never_combined(self):
        base = {"dataset": "IAM", "pair": "th", "usable": True, "left_alignment": "match", "right_alignment": "match"}
        rows = [{**base, "split": "train", "connected": True}, {**base, "split": "test", "connected": False}]
        scores = aggregate(rows)
        self.assertEqual(len(scores), 2)
        self.assertEqual({item["split"] for item in scores}, {"train", "test"})

    def test_end_to_end_line_evidence_uses_gt_identity(self):
        image = np.full((5, 8), 255, dtype=np.uint8)
        image[2, 1:7] = 0
        with TemporaryDirectory() as directory:
            Image.fromarray(image).save(Path(directory) / "line.png")
            record = {
                "dataset": "IAM",
                "split": "train",
                "line_id": "synthetic",
                "transcription": "ab",
                "image_relpath": "line.png",
                "detections": [
                    {"predicted_char": "a", "score": 0.9, "box_xyxy": [0, 0, 5, 5]},
                    {"predicted_char": "x", "score": 0.8, "box_xyxy": [4, 0, 8, 5]},
                ],
            }
            row = line_evidence(record, Path(directory), threshold=100)[0]
        self.assertEqual(row["pair"], "ab")
        self.assertEqual(row["right_alignment"], "substitute")
        self.assertTrue(row["connected"])
        self.assertEqual(row["connectivity_method"], "exclusive-core-v2.1")

    def test_exclusive_cores_reject_overlap_false_positive(self):
        image = np.full((7, 12), 255, dtype=np.uint8)
        image[3, 1:6] = 0
        image[3, 7:11] = 0
        with TemporaryDirectory() as directory:
            Image.fromarray(image).save(Path(directory) / "line.png")
            record = {
                "dataset": "IAM",
                "split": "valid",
                "line_id": "overlap",
                "transcription": "bi",
                "image_relpath": "line.png",
                "detections": [
                    {"predicted_char": "b", "score": 0.9, "box_xyxy": [0, 0, 6, 7]},
                    {"predicted_char": "i", "score": 0.9, "box_xyxy": [4, 0, 12, 7]},
                ],
            }
            row = line_evidence(record, Path(directory), threshold=100)[0]
        self.assertTrue(row["connected_box_intersection_v1"])
        self.assertTrue(row["exclusive_core_usable"])
        self.assertFalse(row["connected_exclusive_core_v2"])
        self.assertFalse(row["connected_exclusive_core_v2_1"])
        self.assertFalse(row["connected"])
        self.assertEqual(row["left_exclusive_core_xyxy"], [0, 0, 4, 7])
        self.assertEqual(row["right_exclusive_core_xyxy"], [6, 0, 12, 7])


if __name__ == "__main__":
    unittest.main()
