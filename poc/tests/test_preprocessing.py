import tempfile
import unittest
from pathlib import Path

from scripts.prepare_iam_selection_images import find_raw_line


class PreprocessingTests(unittest.TestCase):
    def test_finds_nested_raw_iam_line(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "n02" / "n02-000" / "n02-000-05.png"
            source.parent.mkdir(parents=True)
            source.write_bytes(b"image")
            self.assertEqual(find_raw_line(root, "n02-000-05"), source)

    def test_rejects_duplicate_raw_lines(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for folder in ("one", "two"):
                path = root / folder / "n02-000-05.png"
                path.parent.mkdir()
                path.write_bytes(b"image")
            with self.assertRaisesRegex(RuntimeError, "multiple raw IAM lines"):
                find_raw_line(root, "n02-000-05")


if __name__ == "__main__":
    unittest.main()
