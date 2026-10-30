import unittest
from run_graph_memory_test import move_fixture
from run_graph_relation_test import relation_fixture
from update_memory import initialize, digest, apply_events


class MemoryWorkflowTests(unittest.TestCase):
    def test_move_changes_edges_without_overwriting(self):
        nodes, edges = relation_fixture([
            {'id': 2, 'object_tag': 'main table', 'caption': 'Table', 'bbox_center': [3, 1, .4]},
            {'id': 4, 'object_tag': 'coke can', 'caption': 'Red can', 'bbox_center': [2, 1, .8]}])
        state = initialize(nodes, edges, 'fixture'); before = digest(state)
        updated, event = move_fixture(state, 6, 7, [1.3, 1.4, .82], 'test_move')
        self.assertEqual(digest(state), before)
        self.assertEqual(updated['version'], 1)
        self.assertFalse(any(e['source'] == 6 and e['target'] == 2 for e in updated['edges']))
        self.assertTrue(any(e['source'] == 6 and e['target'] == 7 for e in updated['edges']))
        self.assertEqual(updated['entities']['6']['node']['bbox_center'], [1.3, 1.4, .82])
        self.assertEqual(apply_events(updated, [event]), updated)
