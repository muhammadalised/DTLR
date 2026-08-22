import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from dtlr_poc.review import build_review_rows


REPO = Path(__file__).resolve().parents[2]


def pair(index, usable=True, v2=False, v3=False):
    return {
        "left_gt_index": index,
        "right_gt_index": index + 1,
        "pair": "ab",
        "usable": usable,
        "connected_exclusive_core_v2_1": v2 if usable else None,
        "connected_dominant_core_v3": v3 if usable else None,
    }


class ReviewQueueTests(unittest.TestCase):
    def test_includes_mandatory_cases_and_fixed_agreement_audit(self):
        lines = [{
            "line_id": "line",
            "pairs": [
                pair(0, usable=False),
                pair(1, v2=True, v3=False),
                *[pair(index) for index in range(2, 12)],
            ],
        }]
        first = build_review_rows(lines, 4, "fixed-seed")
        second = build_review_rows(lines, 4, "fixed-seed")
        self.assertEqual(first, second)
        groups = [row["queue_group"] for row in first]
        self.assertEqual(groups.count("unusable"), 1)
        self.assertEqual(groups.count("v2.1-v3-disagreement"), 1)
        self.assertEqual(groups.count("agreement-audit"), 4)

    def test_rejects_audit_larger_than_pool(self):
        with self.assertRaisesRegex(ValueError, "exceeds pool size"):
            build_review_rows([{"line_id": "line", "pairs": [pair(0)]}], 2, "seed")

    def test_cli_writes_frozen_queue_and_browser_page(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            qa = root / "qa_manifest.json"
            qa.write_text(json.dumps({
                "lines": [{
                    "line_id": "line",
                    "pairs": [
                        pair(0, usable=False), pair(1, v2=True, v3=False),
                        pair(2), pair(3), pair(4),
                    ],
                }],
            }), encoding="utf-8")
            output = root / "review"
            subprocess.run([
                sys.executable, str(REPO / "poc/scripts/build_qa_review_queue.py"),
                "--qa-manifest", str(qa), "--agreement-audit-count", "2",
                "--seed", "fixed-seed", "--output-dir", str(output),
            ], check=True, capture_output=True, text=True)
            queue = json.loads((output / "review_queue.json").read_text(encoding="utf-8"))
            self.assertEqual(queue["queue_count"], 4)
            self.assertTrue((output / "review_queue.csv").is_file())
            self.assertIn("IAM validation manual review", (output / "review_queue.html").read_text())


if __name__ == "__main__":
    unittest.main()
