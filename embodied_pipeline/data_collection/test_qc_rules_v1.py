"""Regression tests for review rules that could otherwise create false positives."""
from pathlib import Path
import tempfile
import unittest

from .batch import write
from .qc_rules_v1 import audit_record


class QualityRulesTest(unittest.TestCase):
    def audit(self, task, goals, actions, queries=None):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            run = root / 'cases/scene200/tasks/scene200_00000001/model_run'
            roles = [{'role_id': 'payload', 'semantic_query_en': (queries or {}).get('payload', 'object ID 1'), 'quantity': 1},
                     {'role_id': 'destination', 'semantic_query_en': 'destination ID 2', 'quantity': 1}]
            write(run.parent / 'task.json', dict(instruction='test', constraints={}, **task))
            write(run / 'task_decomposition.json', {'task_roles': roles, 'goal_state': goals,
                                                    'initial_scene_constraints': [], 'process_constraints': []})
            write(run / 'object_alignment.json', {'context_bindings': {'payload': {'element_ids': [1]}, 'destination': {'element_ids': [2]}}})
            write(run / 'engineering_plan.json', {'plan': {'subtask_plans': [{'steps': actions}], 'deferred': {'goal_state': goals}}})
            write(run / 'execution_skill_library.json', {'skills': [{'skill_id': s} for s in ['navigate_to_object', 'grasp_object', 'lift_held_object', 'position_arm_for_object']]})
            write(root / 'cases/scene200/graph/scene_graph.json', [{'id': 1}, {'id': 2}])
            write(root / 'cases/scene200/graph/scene_graph_edges.json', [])
            return audit_record(root, 'scene200', 'scene200_00000001')['findings_for_review']

    def test_payload_near_with_held_is_equivalent(self):
        goals = [dict(subject_role='robot', relation='held', object_role='payload', expected_value=True, required=True),
                 dict(subject_role='payload', relation='near', object_role='destination', required=True)]
        task = {'goal': {'final_predicates': [{'predicate': 'held', 'object_id': 1, 'expected': True}, {'predicate': 'near', 'object_id': 2}]}}
        issues = self.audit(task, goals, [])
        self.assertFalse(any(i['code'] == 'missing_required_final_near' for i in issues))

    def test_source_id_in_query_is_not_second_binding(self):
        issues = self.audit({'goal': {'final_predicates': []}}, [], [], {'payload': 'object ID 1 on support ID 2'})
        self.assertFalse(any(i['stage'] == 'B' for i in issues))

    def test_intermediate_visits_not_final_conditions(self):
        goals = [dict(subject_role='robot', relation='near', object_role=r, required=True) for r in ('payload', 'destination')]
        issues = self.audit({'task_family': 'ordered_visit', 'goal': {'final_predicates': [{'predicate': 'near', 'object_id': 2}]}}, goals, [])
        self.assertTrue(any(i['code'] == 'intermediate_visits_encoded_as_final' for i in issues))
        self.assertTrue(any(i['code'] == 'inherited_intermediate_final_goals' for i in issues))

    def test_no_free_pass_for_grasp_without_preparation(self):
        step = {'skill_id': 'grasp_object', 'arguments': {'target': {'role_id': 'payload', 'element_id': 1}}}
        issues = self.audit({'goal': {'final_predicates': []}}, [], [step])
        self.assertTrue(any(i['code'] == 'grasp_prerequisites_missing' for i in issues))


if __name__ == '__main__':
    unittest.main()
