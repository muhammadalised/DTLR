import copy
import unittest

from dtlr_poc.selection import text_sha256
from dtlr_poc.tokenizer_evidence import build_export


class TokenizerEvidenceTests(unittest.TestCase):
    def inputs(self):
        records = [{'dataset': 'IAM', 'split': 'train', 'line_id': 'id', 'transcription': 'ab'}]
        selection = {'schema_version': 'dtlr.iam-selection.v1', 'dataset': 'IAM', 'split': 'train',
                     'requested_count': 1, 'lines': [{'id': 'id', 'transcription_sha256': text_sha256('ab')}]}
        manifest = {'schema_version': 'dtlr.export-manifest.v5', 'datasets': ['IAM'], 'splits': ['train'],
                    'primary_connectivity_method': 'dominant-core-v3', 'line_count': 1, 'evidence_count': 1, 'score_count': 1}
        rows = [{'schema_version': 'dtlr.bigram-evidence.v5', 'dataset': 'IAM', 'split': 'train',
                 'connectivity_method': 'dominant-core-v3', 'line_id': 'id', 'left_gt_index': 0,
                 'right_gt_index': 1, 'pair': 'ab', 'usable': True, 'left_alignment': 'match',
                 'right_alignment': 'match', 'left_dominant_component_id': 1,
                 'right_dominant_component_id': 1, 'connected': True}]
        scores = [{'schema_version': 'dtlr.bigram-scores.v5', 'dataset': 'IAM', 'split': 'train',
                   'connectivity_method': 'dominant-core-v3', 'pair': 'ab', 'n_total': 1,
                   'n_exact_alignment': 1, 'exact_alignment_connected_rate': 1.0}]
        return records, rows, selection, manifest, scores, {}

    def test_export_preserves_low_count_evidence_without_selection(self):
        bundle = build_export(*self.inputs())
        self.assertEqual(bundle['lines'][0]['edges'], [[0, 1, 1]])
        self.assertEqual(bundle['scores'][0]['n_exact_alignment'], 1)
        self.assertFalse(bundle['contract']['vocabulary_selection_performed'])
        self.assertNotIn('vocabulary', bundle)

    def test_unusable_evidence_remains_abstention(self):
        inputs = self.inputs()
        inputs[1][0]['usable'] = False
        inputs[4][0].update(n_exact_alignment=0, exact_alignment_connected_rate=None)
        bundle = build_export(*inputs)
        self.assertEqual(bundle['lines'][0]['edges'], [[0, None, None]])

    def test_changed_split_duplicate_edges_or_aggregate_fail(self):
        for mode in ('split', 'duplicate', 'aggregate'):
            with self.subTest(mode=mode):
                inputs = self.inputs()
                if mode == 'split': inputs[0][0]['split'] = 'valid'
                elif mode == 'duplicate':
                    inputs[1].append(copy.deepcopy(inputs[1][0]))
                    inputs[3]['evidence_count'] = 2
                else: inputs[4][0]['n_total'] = 2
                with self.assertRaises(ValueError):
                    build_export(*inputs)


if __name__ == '__main__':
    unittest.main()
