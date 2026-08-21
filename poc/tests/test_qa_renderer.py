import json
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image


REPO = Path(__file__).resolve().parents[2]


class QARendererTests(unittest.TestCase):
    def run_renderer(self, root, record):
        detections = root / "detections.jsonl"
        detections.write_text(json.dumps(record) + "\n", encoding="utf-8")
        output = root / "qa"
        subprocess.run(
            [
                sys.executable,
                str(REPO / "poc/scripts/render_iam_qa.py"),
                "--detections",
                str(detections),
                "--data-root",
                str(root),
                "--output-dir",
                str(output),
                "--ink-threshold",
                "100",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return output, json.loads((output / "qa_manifest.json").read_text(encoding="utf-8"))

    def test_pair_crop_highlights_shared_component_and_writes_index(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            image = np.full((24, 48), 255, dtype=np.uint8)
            image[12, 5:43] = 0
            Image.fromarray(image).save(root / "line.png")
            record = {
                "dataset": "IAM",
                "split": "valid",
                "line_id": "synthetic",
                "transcription": "ab",
                "image_relpath": "line.png",
                "detections": [
                    {"predicted_char": "a", "score": 0.9, "box_xyxy": [3, 4, 27, 20]},
                    {"predicted_char": "b", "score": 0.9, "box_xyxy": [22, 4, 45, 20]},
                ],
            }
            output, manifest = self.run_renderer(root, record)
            pair = manifest["lines"][0]["pairs"][0]
            self.assertEqual(manifest["schema_version"], "dtlr.qa-manifest.v5")
            self.assertEqual(manifest["pair_count"], 1)
            self.assertEqual(manifest["usable_pair_count"], 1)
            self.assertEqual(manifest["primary_connectivity_method"], "dominant-core-v3")
            self.assertTrue(pair["connected"])
            self.assertTrue(pair["connected_box_intersection_v1"])
            self.assertTrue(pair["connected_exclusive_core_v2"])
            self.assertTrue(pair["connected_exclusive_core_v2_1"])
            self.assertTrue(pair["connected_dominant_core_v3"])
            self.assertEqual(pair["shared_component_count"], 1)
            self.assertTrue((output / pair["image"]).is_file())
            self.assertIn("IAM pair-level QA", (output / "index.html").read_text(encoding="utf-8"))

    def test_inverted_exclusive_core_is_omitted_without_changing_evidence(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            image = np.full((24, 48), 255, dtype=np.uint8)
            image[12, 5:43] = 0
            Image.fromarray(image).save(root / "line.png")
            record = {
                "dataset": "IAM",
                "split": "valid",
                "line_id": "nested-boxes",
                "transcription": "ab",
                "image_relpath": "line.png",
                "detections": [
                    {"predicted_char": "a", "score": 0.9, "box_xyxy": [3, 4, 45, 20]},
                    {"predicted_char": "b", "score": 0.9, "box_xyxy": [10, 4, 20, 20]},
                ],
            }
            output, manifest = self.run_renderer(root, record)
            pair = manifest["lines"][0]["pairs"][0]
            self.assertEqual(manifest["invalid_exclusive_core_geometry_count"], 1)
            self.assertFalse(pair["usable"])
            self.assertFalse(pair["exclusive_core_geometry_valid"])
            self.assertEqual(pair["right_exclusive_core_xyxy"], [45, 4, 20, 20])
            self.assertTrue((output / pair["image"]).is_file())


if __name__ == "__main__":
    unittest.main()
