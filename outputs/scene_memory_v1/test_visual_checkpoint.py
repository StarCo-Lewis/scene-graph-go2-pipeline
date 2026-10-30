import unittest
from unittest.mock import patch
from pathlib import Path
from visual_checkpoint import fixture_contract, actual_bbox_cg


class CheckpointTests(unittest.TestCase):
    def test_coordinate_conversion_uses_physical_com_not_map_estimate(self):
        contract = {'kettle_com': [1, 2, 3], 'bbox_minus_com_offset': [0, .1, 0]}
        self.assertEqual(actual_bbox_cg(contract), [1, -3, 2.1])

    def test_failed_or_wrong_checkpoint_rejected(self):
        with patch('visual_checkpoint.read', side_effect=[{}, {'status': 'failed'}]):
            with self.assertRaises(ValueError):
                fixture_contract(Path('memory'), Path('previous'))
        with patch('visual_checkpoint.read', side_effect=[{}, {'status': 'passed', 'repaired_memory': 'other'}]):
            with self.assertRaises(ValueError):
                fixture_contract(Path('memory'), Path('previous'))

    def test_recursive_chain_bounded(self):
        with self.assertRaises(ValueError):
            fixture_contract(Path('missing'), depth=21)
