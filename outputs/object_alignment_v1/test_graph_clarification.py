import unittest
from run_graph_clarification_test import check_result


class ClarificationTests(unittest.TestCase):
    def test_target_and_skill_must_agree(self):
        summary = {'status': 'planning_passed'}
        alignment = {'status': 'success', 'context_bindings': {'can': {'element_ids': [6]}}}
        step = {'skill_id': 'navigate_to_object', 'arguments': {'target': {'element_id': 6}}}
        plan = {'subtask_plans': [{'steps': [step]}]}
        self.assertTrue(check_result(summary, alignment, plan))
        step['arguments']['target']['element_id'] = 4
        self.assertFalse(check_result(summary, alignment, plan))
