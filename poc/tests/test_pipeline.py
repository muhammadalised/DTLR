import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image

from dtlr_poc.alignment import align_monotonic, gt_detection_map
from dtlr_poc.ccl import component_ids_in_box, label_ink
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


if __name__ == "__main__":
    unittest.main()
