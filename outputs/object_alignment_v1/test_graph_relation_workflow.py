import unittest
from run_graph_relation_test import relation_fixture, relation_selection_passed


class WorkflowTests(unittest.TestCase):
    def test_fixture_and_acceptance(self):
        nodes, edges = relation_fixture([
            {'id': 2, 'object_tag': 'main table', 'caption': 'Table', 'bbox_center': [3, 1, .4]},
            {'id': 4, 'object_tag': 'coke can', 'caption': 'Red can', 'bbox_center': [2, 1, .8]}])
        self.assertEqual(len(nodes), 4)
        self.assertEqual(edges[0], {'source': 6, 'target': 2, 'relation': 'a on b'})
        alignment = {'status': 'success', 'context_bindings': {'can': {'element_ids': [6]}, 'table': {'element_ids': [2]}},
                     'alignment': {'relation_checks': [{'relation': 'on', 'status': 'observed_true'}]}}
        step = {'skill_id': 'navigate_to_object', 'arguments': {'target': {'element_id': 6}}}
        plan = {'subtask_plans': [{'steps': [step]}]}
        self.assertTrue(relation_selection_passed({'status': 'planning_passed'}, alignment, plan))
        alignment['alignment']['relation_checks'] = []
        self.assertFalse(relation_selection_passed({'status': 'planning_passed'}, alignment, plan))
