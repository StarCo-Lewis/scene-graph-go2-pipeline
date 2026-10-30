import unittest
from update_memory import initialize, apply_events
from run_memory_multiround_ablation import observed_move, item_from_memory


class MultiRoundTests(unittest.TestCase):
    def test_two_moves_keep_id_and_latest_position(self):
        original = initialize([{'id': 67, 'bbox_center': [1, 2, 3]},
                               {'id': 13, 'bbox_center': [4, 5, 6]}], [], 'fixture')
        first, event = observed_move(original, [4, 5, -6], [0, .1, 0], 'e1',
            '2026-09-26T01:00:00Z', 'sim-readback.json')
        second, _ = observed_move(first, [7, 8, -9], [0, .1, 0], 'e2',
            '2026-09-26T01:00:01Z', 'sim-readback2.json')
        self.assertEqual(item_from_memory(second), {'element_id': 67, 'cg_center': [7., 9., 8.1]})
        self.assertEqual(item_from_memory(first)['cg_center'], [4., 6., 5.1])
        self.assertEqual(original['version'], 0)
        self.assertEqual(second['version'], 2)
        self.assertEqual(second['edges'], [])  # kinematic placement is not support evidence
        self.assertEqual(apply_events(first, [event]), first)

    def test_deleted_object_cannot_be_selected(self):
        state = initialize([{'id': 67, 'bbox_center': [1, 2, 3]}], [], 'fixture')
        state['entities']['67']['active'] = False
        with self.assertRaisesRegex(ValueError, 'NO object'):
            item_from_memory(state)
