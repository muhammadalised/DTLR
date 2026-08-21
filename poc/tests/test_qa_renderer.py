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
            manifest = json.loads((output / "qa_manifest.json").read_text(encoding="utf-8"))
            pair = manifest["lines"][0]["pairs"][0]
            self.assertEqual(manifest["pair_count"], 1)
            self.assertEqual(manifest["usable_pair_count"], 1)
            self.assertTrue(pair["connected"])
            self.assertEqual(pair["shared_component_count"], 1)
            self.assertTrue((output / pair["image"]).is_file())
            self.assertIn("IAM pair-level QA", (output / "index.html").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
