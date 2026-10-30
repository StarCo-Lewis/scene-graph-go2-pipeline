import importlib.util
import json
import sys
import unittest
from unittest.mock import patch
from pathlib import Path

HERE = Path(__file__).parent
SPEC = importlib.util.spec_from_file_location("match_skills", HERE / "match_skills.py")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
LIBRARY = json.loads((HERE.parent / "quadruped_atomic_skill_library.v0.1.json").read_text(encoding="utf-8"))


def successful_alignment():
    return {
        "status": "success",
        "task_decomposition": {"instruction": "Pick up the cup.",
                               "subtasks": [{"subtask_id": "st_01", "description": "Pick up cup", "dependencies": []}],
                               "initial_scene_constraints": [], "goal_state": []},
        "context_bindings": {"cup": {"element_ids": [102]}},
    }


class SkillMatchingTests(unittest.TestCase):
    def test_upstream_missing_object_short_circuits(self):
        result = MODULE.build_result({"status": "error", "error": {"code": "NO_OBJECT"}}, LIBRARY, None)
        self.assertEqual(result["status"], "unresolved")
        self.assertEqual(result["stage"], "object_alignment")

    def test_valid_grounded_skill_is_only_a_proposal(self):
        proposal = {"subtask_plans": [{"subtask_id": "st_01", "intent": "Pick up cup", "steps": [
            {"skill_id": "grasp_object", "arguments": {"target": {"role_id": "cup", "element_id": 102}}}
        ]}]}
        result = MODULE.build_result(successful_alignment(), LIBRARY, proposal)
        self.assertEqual(result["status"], "proposal_needs_environment_validation")
        step = result["subtask_plans"][0]["steps"][0]
        self.assertEqual(step["feasibility"], "unknown")
        self.assertEqual(step["execution_status"], "not_executed")

    def test_hallucinated_object_rejected(self):
        proposal = {"subtask_plans": [{"subtask_id": "st_01", "intent": "Pick up cup", "steps": [
            {"skill_id": "grasp_object", "arguments": {"target": {"role_id": "cup", "element_id": 999}}}
        ]}]}
        with self.assertRaisesRegex(MODULE.PlanError, "ungrounded object"):
            MODULE.validate_proposal(proposal, successful_alignment(), LIBRARY)

    def test_hallucinated_skill_rejected(self):
        proposal = {"subtask_plans": [{"subtask_id": "st_01", "intent": "Pick up cup", "steps": [
            {"skill_id": "teleport", "arguments": {}}
        ]}]}
        with self.assertRaisesRegex(MODULE.PlanError, "unknown skill"):
            MODULE.validate_proposal(proposal, successful_alignment(), LIBRARY)

    def test_empty_subtask_is_unresolved(self):
        proposal = {"subtask_plans": [{"subtask_id": "st_01", "intent": "Pick up cup", "steps": []}]}
        result = MODULE.build_result(successful_alignment(), LIBRARY, proposal)
        self.assertEqual(result["status"], "unresolved")
        self.assertEqual(result["unplanned_subtasks"], ["st_01"])

    def test_navigation_only_prompt_uses_flat_grounded_schema(self):
        alignment = successful_alignment()
        alignment['task_decomposition']['instruction'] = 'Go to the cup.'
        library = {'skills': [next(s for s in LIBRARY['skills'] if s['skill_id'] == 'navigate_to_object')]}
        captured = {}
        def reply(request, timeout, attempts):
            captured.update(json.loads(request.data.decode('utf-8')))
            return {'choices': [{'message': {'content': json.dumps({'subtask_plans': [
                {'subtask_id': 'st_01', 'intent': 'Go to cup', 'steps': [
                    {'skill_id': 'navigate_to_object', 'arguments':
                     {'target': {'role_id': 'cup', 'element_id': 102}}}]}]})}}]}
        with patch.dict(MODULE.os.environ, {'TEST_SKILL_KEY': 'test-only'}), patch.object(MODULE, 'request_json_with_retry', side_effect=reply):
            MODULE.request_plan(alignment, library, 'test-model', 'https://example.invalid', 'TEST_SKILL_KEY',
                                validation_error='unknown skill at st_01 step 1',
                                rejected_proposal={'subtask_plans': [{'subtask_id': 'st_01', 'steps': [{'type': 'navigate_to'}]}]})
        prompt = json.loads(captured['messages'][1]['content'])
        example_step = prompt['example']['subtask_plans'][0]['steps'][0]
        self.assertEqual(example_step['skill_id'], 'navigate_to_object')
        self.assertEqual(example_step['arguments']['target']['element_id'], 102)
        self.assertEqual(prompt['previous_validation_error'], 'unknown skill at st_01 step 1')
        self.assertNotIn('previous_rejected_proposal', prompt)
        self.assertEqual(prompt['required_subtask_plan_order'], ['st_01'])

    def test_missing_skill_is_not_reported_as_unknown_library_skill(self):
        for step in ({}, {'skill_id': None}, {'skill_id': []}, {'skill_id': ' '}, 'navigate'):
            with self.subTest(step=step):
                proposal = {'subtask_plans': [{'subtask_id': 'st_01', 'intent': 'Pick up cup', 'steps': [step]}]}
                with self.assertRaises(MODULE.PlanError) as failure:
                    MODULE.validate_proposal(proposal, successful_alignment(), LIBRARY)
                self.assertIn('subtask_plans[0].steps[0]', str(failure.exception))
                self.assertNotIn('unknown skill', str(failure.exception))
                self.assertEqual(proposal['subtask_plans'][0]['steps'][0], step)

    def test_recursive_subtask_is_rejected_without_flattening(self):
        nested = {'subtask_id': 'st_01', 'intent': 'Pick up cup', 'steps': [
            {'skill_id': 'grasp_object', 'arguments': {'target': {'role_id': 'cup', 'element_id': 102}}}]}
        proposal = {'subtask_plans': [{'subtask_id': 'st_01', 'intent': 'Pick up cup', 'steps': [nested]}]}
        with self.assertRaisesRegex(MODULE.PlanError, 'nested subtask/wrapper'):
            MODULE.validate_proposal(proposal, successful_alignment(), LIBRARY)
        self.assertIs(proposal['subtask_plans'][0]['steps'][0], nested)

    def test_atomic_action_in_subtask_array_reports_actual_ids_and_keys(self):
        proposal = {'subtask_plans': [{'skill_id': 'navigate_to_object', 'arguments': {}}]}
        with self.assertRaises(MODULE.PlanError) as failure:
            MODULE.validate_proposal(proposal, successful_alignment(), LIBRARY)
        self.assertIn("expected=['st_01']; received=[None]", str(failure.exception))
        self.assertIn('skill_id', str(failure.exception))

    def test_multiskill_retry_shows_full_root_and_omits_bad_response(self):
        captured = {}
        def reply(request, timeout, attempts):
            captured.update(json.loads(request.data.decode('utf-8')))
            return {'choices': [{'message': {'content': '{"subtask_plans":[]}'}}]}
        with patch.dict(MODULE.os.environ, {'TEST_SKILL_KEY': 'test-only'}), patch.object(
                MODULE, 'request_json_with_retry', side_effect=reply):
            MODULE.request_plan(successful_alignment(), LIBRARY, 'test-model', 'https://example.invalid',
                'TEST_SKILL_KEY', validation_error='steps[0].skill_id missing',
                rejected_proposal={'unique_bad_marker': {'steps': [{}]}})
        prompt = json.loads(captured['messages'][1]['content'])
        example = prompt['example']
        self.assertEqual(example['schema_version'], '1.0')
        self.assertNotIn('steps', example)
        self.assertTrue(all(set(s) == {'skill_id', 'arguments'}
                            for s in example['subtask_plans'][0]['steps']))
        self.assertNotIn('unique_bad_marker', json.dumps(captured))
        self.assertIn('steps[0].skill_id missing', prompt['previous_validation_error'])
        self.assertIn('never nest subtask objects', captured['messages'][0]['content'])

    def test_unique_navigation_compiles_from_validated_goal_and_edge(self):
        alignment = successful_alignment()
        alignment['task_decomposition']['instruction'] = 'Go to the cup on the table using observed edges.'
        alignment['task_decomposition']['goal_state'] = [
            {'constraint_id': 'goal', 'required': True, 'subject_role': 'robot',
             'relation': 'near', 'object_role': 'cup'}]
        alignment['task_decomposition']['initial_scene_constraints'] = [
            {'constraint_id': 'init', 'required': True, 'subject_role': 'cup',
             'relation': 'on', 'object_role': 'table'}]
        alignment['alignment'] = {'relation_checks': [{'constraint_id': 'init', 'status': 'observed_true'}]}
        library = {'skills': [next(s for s in LIBRARY['skills'] if s['skill_id'] == 'navigate_to_object')],
                   'execution_contract': {'adapter_mode': 'navigation',
                                          'supported_ordered_sequence': ['navigate_to_object']}}
        proposal = MODULE.deterministic_navigation_proposal(alignment, library)
        self.assertEqual(proposal['subtask_plans'][0]['steps'][0]['arguments']['target'],
                         {'role_id': 'cup', 'element_id': 102})
        self.assertEqual(MODULE.build_result(alignment, library, proposal)['status'],
                         'proposal_needs_environment_validation')
        alignment['alignment']['relation_checks'] = []
        self.assertIsNone(MODULE.deterministic_navigation_proposal(alignment, library))
        alignment['alignment']['relation_checks'] = [{'constraint_id': 'init', 'status': 'observed_true'}]
        alignment['task_decomposition']['instruction'] += ' Inspect it.'
        self.assertIsNone(MODULE.deterministic_navigation_proposal(alignment, library))


if __name__ == "__main__":
    unittest.main()
