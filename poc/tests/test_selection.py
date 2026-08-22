import json
import pickle
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from dtlr_poc.selection import build_selection, load_selected_examples


REPO = Path(__file__).resolve().parents[2]


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.labels_path = self.root / "labels.pkl"
        labels = {
            "ground_truth": {
                "valid": [
                    {"id": f"line-{index}", "text": f"text {index}"}
                    for index in range(10)
                ]
            }
        }
        self.labels_path.write_bytes(pickle.dumps(labels))

    def tearDown(self):
        self.temp.cleanup()

    def test_selection_is_deterministic_and_verifiable(self):
        first = build_selection(self.labels_path, "valid", 4, "fixed-seed")
        second = build_selection(self.labels_path, "valid", 4, "fixed-seed")
        self.assertEqual(first, second)
        selection_path = self.root / "selection.json"
        selection_path.write_text(json.dumps(first), encoding="utf-8")
        examples, loaded = load_selected_examples(selection_path, self.labels_path, "valid")
        self.assertEqual([row["id"] for row in examples], [row["id"] for row in first["lines"]])
        self.assertEqual(loaded, first)

    def test_changed_labels_are_rejected(self):
        selection = build_selection(self.labels_path, "valid", 2, "fixed-seed")
        selection_path = self.root / "selection.json"
        selection_path.write_text(json.dumps(selection), encoding="utf-8")
        labels = pickle.loads(self.labels_path.read_bytes())
        labels["ground_truth"]["valid"][0]["text"] = "changed"
        self.labels_path.write_bytes(pickle.dumps(labels))
        with self.assertRaisesRegex(ValueError, "labels.pkl hash differs"):
            load_selected_examples(selection_path, self.labels_path, "valid")

    def test_freeze_cli_can_select_complete_split(self):
        data_root = self.root / "data-root"
        iam_root = data_root / "IAM_new"
        iam_root.mkdir(parents=True)
        (iam_root / "labels.pkl").write_bytes(self.labels_path.read_bytes())
        output = self.root / "all-valid.json"
        subprocess.run([
            sys.executable,
            str(REPO / "poc/scripts/freeze_iam_selection.py"),
            "--data-root", str(data_root),
            "--split", "valid",
            "--all",
            "--seed", "complete-valid-split",
            "--output", str(output),
        ], check=True, capture_output=True, text=True)
        selection = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(selection["requested_count"], 10)
        self.assertEqual(len(selection["lines"]), 10)


if __name__ == "__main__":
    unittest.main()
