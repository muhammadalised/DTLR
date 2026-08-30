import json
import pickle
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image

from dtlr_poc.read_dataset import (
    all_examples,
    build_read_selection,
    load_selected_read_examples,
    normalize_transcription,
    resolve_image,
)


REPO = Path(__file__).resolve().parents[2]


class ReadDatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.labels_path = self.root / "labels.pkl"
        labels = {
            "ground_truth": {
                "valid": {
                    0: {
                        "idx": 0,
                        "path": "READ_2016/images/valid/valid_0_0.jpeg",
                        "text": "Bestett¬",
                    },
                    1: {
                        "idx": 1,
                        "path": "READ_2016/images/valid/valid_0_1.jpeg",
                        "text": "Rāth",
                    },
                }
            },
            "charset": [ord(char) for char in "BestRāth"],
        }
        self.labels_path.write_bytes(pickle.dumps(labels))

    def tearDown(self):
        self.temp.cleanup()

    def test_marker_normalization_is_explicit(self):
        self.assertEqual(normalize_transcription("Bestett¬ vnd¬"), "Bestett vnd")
        examples = all_examples(self.labels_path, "valid")
        self.assertEqual(examples[0]["raw_text"], "Bestett¬")
        self.assertEqual(examples[0]["text"], "Bestett")

    def test_selection_is_deterministic_and_verifiable(self):
        first = build_read_selection(self.labels_path, "valid", 2, "read-seed")
        second = build_read_selection(self.labels_path, "valid", 2, "read-seed")
        self.assertEqual(first, second)
        selection_path = self.root / "selection.json"
        selection_path.write_text(json.dumps(first), encoding="utf-8")
        examples, loaded = load_selected_read_examples(
            selection_path, self.labels_path, "valid"
        )
        self.assertEqual([row["id"] for row in examples], [row["id"] for row in first["lines"]])
        self.assertEqual(loaded, first)

    def test_resolves_recorded_path_and_rejects_different_duplicate_layout(self):
        example = all_examples(self.labels_path, "valid")[0]
        recorded = self.root / example["label_image_relpath"]
        recorded.parent.mkdir(parents=True)
        recorded.write_bytes(b"recorded")
        relpath, source = resolve_image(self.root, example, "valid")
        self.assertEqual(relpath, Path(example["label_image_relpath"]))
        self.assertEqual(source, "labels-pkl-path")

        legacy = self.root / "READ_2016/images/valid/0.jpeg"
        legacy.write_bytes(b"different")
        with self.assertRaisesRegex(ValueError, "ambiguous READ image layouts"):
            resolve_image(self.root, example, "valid")

    def test_freeze_cli_can_select_complete_split(self):
        data_root = self.root / "data-root"
        read_root = data_root / "READ_2016"
        read_root.mkdir(parents=True)
        (read_root / "labels.pkl").write_bytes(self.labels_path.read_bytes())
        output = self.root / "all-valid.json"
        subprocess.run(
            [
                sys.executable,
                str(REPO / "poc/scripts/freeze_read_selection.py"),
                "--data-root", str(data_root),
                "--split", "valid",
                "--all",
                "--seed", "complete-read-valid",
                "--output", str(output),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        selection = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(selection["dataset"], "READ")
        self.assertEqual(selection["requested_count"], 2)
        self.assertEqual(len(selection["lines"]), 2)

    def test_prepares_selected_page_xml_crops_reproducibly(self):
        data_root = self.root / "data-root"
        labels_root = data_root / "READ_2016"
        labels_root.mkdir(parents=True)
        labels_root.joinpath("labels.pkl").write_bytes(self.labels_path.read_bytes())
        selection = build_read_selection(labels_root / "labels.pkl", "valid", 2, "crop-seed")
        selection_path = self.root / "selection.json"
        selection_path.write_text(json.dumps(selection), encoding="utf-8")

        raw_root = self.root / "raw"
        split_root = raw_root / "PublicData/Validation"
        xml_root = split_root / "page/page"
        image_root = split_root / "Images"
        xml_root.mkdir(parents=True)
        image_root.mkdir(parents=True)
        namespace = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"
        (xml_root / "Seite0001.xml").write_text(
            f'''<PcGts xmlns="{namespace}"><Page imageWidth="20" imageHeight="12">
            <TextRegion><TextLine id="line-0"><Coords points="1,2 8,2 8,6 1,6"/>
            <TextEquiv><Unicode>Bestett¬ </Unicode></TextEquiv></TextLine>
            <TextLine id="line-1"><Coords points="9,3 18,3 18,9 9,9"/>
            <TextEquiv><Unicode>Rāth</Unicode></TextEquiv></TextLine></TextRegion>
            </Page></PcGts>''',
            encoding="utf-8",
        )
        Image.new("RGB", (20, 12), "white").save(image_root / "Seite0001.JPG")
        manifest_path = self.root / "preprocessing.json"
        command = [
            sys.executable,
            str(REPO / "poc/scripts/prepare_read_selection_images.py"),
            "--selection-manifest", str(selection_path),
            "--raw-root", str(raw_root),
            "--data-root", str(data_root),
            "--output-manifest", str(manifest_path),
        ]
        first = subprocess.run(command, check=True, capture_output=True, text=True)
        second = subprocess.run(command, check=True, capture_output=True, text=True)
        self.assertIn('"created_this_run": 2', first.stdout)
        self.assertIn('"existing_this_run": 2', second.stdout)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["schema_version"], "dtlr.read-preprocessing.v1")
        self.assertEqual(manifest["record_count"], 2)
        for row in manifest["records"]:
            self.assertTrue((data_root / row["output"]).is_file())

        first_output = data_root / manifest["records"][0]["output"]
        first_output.write_bytes(b"different")
        failed = subprocess.run(command, capture_output=True, text=True)
        self.assertNotEqual(failed.returncode, 0)
        self.assertIn("refusing to overwrite a different READ line crop", failed.stderr)


if __name__ == "__main__":
    unittest.main()
