import json
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image


REPO = Path(__file__).resolve().parents[2]


class FailureInspectionTests(unittest.TestCase):
    def test_renders_explicit_pair_and_threshold_metrics(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            image = np.full((28, 56), 255, dtype=np.uint8)
            image[8:21, 6:22] = 20
            image[8:21, 34:50] = 20
            image[14, 20:36] = 110
            Image.fromarray(image).save(root / "line.png")
            detections = root / "detections.jsonl"
            detections.write_text(json.dumps({
                "dataset": "IAM",
                "split": "valid",
                "line_id": "synthetic",
                "transcription": "ab",
                "image_relpath": "line.png",
                "detections": [
                    {"predicted_char": "a", "score": 0.9, "box_xyxy": [4, 5, 31, 23]},
                    {"predicted_char": "b", "score": 0.9, "box_xyxy": [27, 5, 52, 23]},
                ],
            }) + "\n", encoding="utf-8")
            selection = root / "selection.json"
            selection.write_text(json.dumps({
                "schema_version": "dtlr.failure-inspection-selection.v1",
                "dataset": "IAM",
                "split": "valid",
                "cases": [{
                    "pair_id": "synthetic:0:1",
                    "source": "test",
                    "manual_visual_connectivity": "connected",
                }],
            }) + "\n", encoding="utf-8")
            output = root / "diagnostics"
            subprocess.run([
                sys.executable,
                str(REPO / "poc/scripts/inspect_pair_failures.py"),
                "--detections", str(detections),
                "--data-root", str(root),
                "--selection", str(selection),
                "--output-dir", str(output),
                "--threshold-offsets=-20,0,20",
                "--scale", "2",
            ], check=True, capture_output=True, text=True)

            manifest = json.loads((output / "failure_diagnostics.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["schema_version"], "dtlr.failure-inspection.v1")
            self.assertEqual(manifest["case_count"], 1)
            case = manifest["cases"][0]
            self.assertEqual(case["pair_id"], "synthetic:0:1")
            self.assertGreaterEqual(len(case["threshold_sweep"]), 3)
            self.assertIn("components", case["otsu_evidence"])
            self.assertTrue((output / case["diagnostic_image"]).is_file())
            self.assertTrue((output / case["threshold_sweep_image"]).is_file())
            self.assertIn("Analysis only", (output / "index.html").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
