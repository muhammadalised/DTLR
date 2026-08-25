import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from dtlr_poc.detection_resume import read_jsonl_for_resume, validate_resume_prefix


class DetectionResumeTests(unittest.TestCase):
    def setUp(self):
        self.examples = [
            {"id": "line-1", "text": "one"},
            {"id": "line-2", "text": "two"},
        ]
        self.expected = {
            "schema_version": "dtlr.detections.v1",
            "dataset": "IAM",
            "split": "train",
            "checkpoint": {"kind": "iam-finetuned", "sha256": "checkpoint"},
            "selection_manifest": {"schema_version": "selection", "sha256": "selection"},
            "repo_commit": "commit",
            "threshold": 0.3,
            "nms_iou": 0.5,
        }

    def record(self, line_id="line-1", transcription="one"):
        return {**self.expected, "line_id": line_id, "transcription": transcription, "detections": []}

    def test_accepts_exact_prefix(self):
        validate_resume_prefix([self.record()], self.examples, self.expected)

    def test_rejects_wrong_order_or_provenance(self):
        with self.assertRaisesRegex(ValueError, "exact selection prefix"):
            validate_resume_prefix([self.record("line-2", "two")], self.examples, self.expected)
        changed = self.record()
        changed["threshold"] = 0.4
        with self.assertRaisesRegex(ValueError, "provenance mismatch"):
            validate_resume_prefix([changed], self.examples, self.expected)

    def test_rejects_incomplete_last_line(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "detections.jsonl"
            path.write_text(json.dumps(self.record()), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "incomplete JSONL line"):
                read_jsonl_for_resume(path)

    def test_reads_complete_jsonl(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "detections.jsonl"
            path.write_text(json.dumps(self.record()) + "\n", encoding="utf-8")
            self.assertEqual(read_jsonl_for_resume(path), [self.record()])


if __name__ == "__main__":
    unittest.main()
