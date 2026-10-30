import copy
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from decomposition_wire import from_flat_roles
from align_objects import call_llm, validate_decomposition, AlignmentError
from llm_graph_binding import validate_bindings, BindingError, check_binding_relations


def sample():
    return {'schema_version': '1.0', 'instruction': 'Go to the kettle and pick it up.', 'task_name': 'Pick up kettle',
        'object_roles': [{'role_id': 'kettle', 'name': 'electric kettle', 'semantic_query_en': 'electric kettle',
                         'semantic_labels': ['kettle'], 'quantity': 1, 'required': True}],
        'subtasks': [{'subtask_id': 'st_01', 'description': 'Go near the kettle.', 'dependencies': [], 'required_role_ids': ['kettle']},
                     {'subtask_id': 'st_02', 'description': 'Pick up the kettle.', 'dependencies': ['st_01'], 'required_role_ids': ['kettle']}],
        'initial_scene_constraints': [],
        'goal_state': [{'constraint_id': 'goal_01', 'subject_role': 'robot', 'relation': 'near', 'object_role': 'kettle', 'required': True}]}


class FlatRolesTests(unittest.TestCase):
    def protected_sample(self):
        data = sample()
        data['instruction'] += ' Leave the tape dispenser on the workbench; do not move it.'
        for rid in ('tape_dispenser', 'workbench'):
            data['object_roles'].append({'role_id': rid, 'name': rid.replace('_', ' '),
                'semantic_query_en': rid.replace('_', ' '), 'semantic_labels': [rid.replace('_', ' ')],
                'quantity': 1, 'required': True})
        data['goal_state'].append({'constraint_id': 'protect_tape', 'subject_role': 'tape_dispenser',
            'relation': 'on', 'object_role': 'workbench', 'required': True})
        return data

    def test_constraint_subject_and_support_are_grounded_without_adding_actions(self):
        data = self.protected_sample()
        before = copy.deepcopy(data)
        converted = from_flat_roles(data)
        result = validate_decomposition(converted)
        self.assertEqual(data, before)
        self.assertEqual([r['role_id'] for r in result['constraint_roles']], ['tape_dispenser', 'workbench'])
        self.assertEqual({r['role_id'] for r in result['task_roles']},
                         {'kettle', 'tape_dispenser', 'workbench'})
        self.assertEqual(result['object_requirements'], result['task_roles'])
        self.assertEqual(result['goal_state'], before['goal_state'])
        self.assertEqual(len(result['subtasks']), len(before['subtasks']))
        for actual, original in zip(result['subtasks'], before['subtasks']):
            self.assertEqual(actual['description'], original['description'])
            self.assertEqual(actual['dependencies'], original['dependencies'])
            self.assertEqual([r['role_id'] for r in actual['required_roles']], original['required_role_ids'])
        # The command-line path validates the converted decomposition a second time.
        self.assertEqual(validate_decomposition(copy.deepcopy(result)), result)

    def test_initial_constraint_only_roles_keep_initial_and_goal_scopes_separate(self):
        data = self.protected_sample()
        constraint = data['goal_state'].pop()
        data['initial_scene_constraints'] = [constraint]
        result = validate_decomposition(from_flat_roles(data))
        self.assertEqual(result['initial_scene_constraints'], [constraint])
        self.assertEqual(result['goal_state'], data['goal_state'])
        self.assertEqual(len(result['constraint_roles']), 2)
        bindings = {'tape_dispenser': {'element_ids': [230]}, 'workbench': {'element_ids': [165]}}
        _, errors = check_binding_relations(bindings, result['initial_scene_constraints'], [])
        self.assertEqual(errors[0]['reason'], 'relation_unverified')
        _, errors = check_binding_relations(bindings, result['initial_scene_constraints'],
            [{'source': 230, 'target': 165, 'relation': 'a on b'}])
        self.assertEqual(errors, [])

    def test_binding_requires_a_decision_for_every_constraint_role(self):
        data = validate_decomposition(from_flat_roles(self.protected_sample()))
        ids = {'kettle': 67, 'tape_dispenser': 230, 'workbench': 165}
        nodes = [{'id': value, 'object_tag': role, 'caption': role} for role, value in ids.items()]
        proposal = {'bindings': [{'role_id': role, 'status': 'matched', 'element_ids': [value],
                                  'reason': 'Explicit unit fixture, not a model result'}
                                 for role, value in ids.items()]}
        bindings, errors = validate_bindings(proposal, data['task_roles'], nodes)
        self.assertEqual(errors, [])
        self.assertEqual(bindings['workbench']['element_ids'], [165])
        proposal['bindings'].pop()
        with self.assertRaisesRegex(BindingError, 'Missing role decisions'):
            validate_bindings(proposal, data['task_roles'], nodes)

    def test_malformed_constraint_cannot_make_an_unused_role_valid(self):
        for change in ('relation', 'unknown_role', 'robot_subject', 'required', 'container', 'reference_type'):
            data = self.protected_sample()
            constraint = data['goal_state'][-1]
            if change == 'relation':
                constraint['relation'] = 'protect'
            elif change == 'unknown_role':
                constraint['subject_role'] = 'missing'
            elif change == 'robot_subject':
                constraint['subject_role'] = 'robot'
            elif change == 'required':
                constraint['required'] = 'true'
            elif change == 'container':
                data['goal_state'] = {'constraint': constraint}
            else:
                constraint['object_role'] = ['workbench']
            with self.subTest(change=change), self.assertRaises((ValueError, AlignmentError)):
                validate_decomposition(from_flat_roles(data))

    def test_native_constraint_role_table_cannot_smuggle_unused_or_duplicate_objects(self):
        for change in ('unused', 'duplicate', 'action_overlap', 'bad_definition', 'bad_container'):
            data = from_flat_roles(self.protected_sample())
            if change == 'unused':
                data['goal_state'].pop()
            elif change == 'duplicate':
                data['constraint_roles'].append(copy.deepcopy(data['constraint_roles'][0]))
            elif change == 'action_overlap':
                data['constraint_roles'].append(copy.deepcopy(data['subtasks'][0]['required_roles'][0]))
            elif change == 'bad_definition':
                data['constraint_roles'][0]['semantic_labels'] = []
            else:
                data['constraint_roles'] = {}
            with self.subTest(change=change), self.assertRaises(AlignmentError):
                validate_decomposition(data)

    def test_saved_workbench_failure_shapes_now_validate_without_rewriting_history(self):
        folder = (Path(__file__).resolve().parents[1] / 'vlm_full_scene_v1/runs'
                  / 'two_scene_20261015_004957_179366/trials/c07_r01/round_1/planning')
        if not folder.is_dir():
            self.skipTest('Historical user run not present')
        for attempt in (1, 2, 3):
            record = json.loads((folder / f'llm_decomposition_attempt_{attempt}.json').read_text(encoding='utf-8'))
            original = json.loads(record['response_content'])
            before = copy.deepcopy(original)
            with self.subTest(attempt=attempt):
                result = validate_decomposition(from_flat_roles(original))
                self.assertEqual(original, before)
                self.assertEqual([r['role_id'] for r in result['constraint_roles']], ['workbench_165'])
                self.assertIn('workbench_165', {r['role_id'] for r in result['task_roles']})
                self.assertEqual(len(result['subtasks']), len(original['subtasks']))
                self.assertEqual(result['goal_state'], original['goal_state'])
                for actual, source in zip(result['subtasks'], original['subtasks']):
                    self.assertEqual([r['role_id'] for r in actual['required_roles']], source['required_role_ids'])

    def carry_sample(self):
        data = sample()
        data['instruction'] = 'Carry the held kettle to the sideboard. Do not place it; keep holding it.'
        data['task_name'] = 'Carry held kettle'
        data['object_roles'].append({'role_id': 'sideboard', 'name': 'sideboard',
            'semantic_query_en': 'sideboard', 'semantic_labels': ['sideboard'],
            'quantity': 1, 'required': True})
        data['subtasks'] = [{'subtask_id': 'st_01', 'description': data['instruction'],
            'dependencies': [], 'required_role_ids': ['kettle', 'sideboard']}]
        data['goal_state'][0]['object_role'] = 'sideboard'
        return data

    def test_held_payload_and_destination_share_one_intent_without_extra_actions(self):
        data = self.carry_sample()
        saved = copy.deepcopy(data)
        result = validate_decomposition(from_flat_roles(data))
        self.assertEqual(data, saved)
        self.assertEqual(len(result['subtasks']), 1)
        self.assertEqual([r['role_id'] for r in result['subtasks'][0]['required_roles']],
                         ['kettle', 'sideboard'])

    def test_missing_held_role_is_rejected_not_auto_inserted(self):
        data = self.carry_sample()
        data['subtasks'][0]['required_role_ids'] = ['sideboard']
        saved = copy.deepcopy(data)
        with self.assertRaisesRegex(ValueError, "unreferenced_role_ids=\\['kettle'\\]"):
            from_flat_roles(data)
        self.assertEqual(data, saved)

    @patch.dict(os.environ, {'DASHSCOPE_API_KEY': 'test-only'})
    @patch('align_objects.request_json')
    def test_retry_names_missing_payload_and_keeps_original_instruction(self, request):
        good = self.carry_sample()
        bad = copy.deepcopy(good)
        bad['subtasks'][0]['required_role_ids'] = ['sideboard']
        request.side_effect = [{'choices': [{'message': {'content': json.dumps(value)},
                                           'finish_reason': 'stop'}]} for value in (bad, good)]
        result = call_llm(good['instruction'], 'test-model', 'https://invalid.test',
            'DASHSCOPE_API_KEY', decomposition_format='flat_roles', max_attempts=2)
        self.assertEqual(len(result['subtasks']), 1)
        self.assertEqual(request.call_count, 2)
        payload = json.loads(request.call_args_list[1].args[0].data)
        self.assertIn("unreferenced_role_ids=['kettle']", payload['messages'][1]['content'])
        self.assertIn('both the payload and destination', payload['messages'][1]['content'])
        self.assertIn(good['instruction'], payload['messages'][1]['content'])
        self.assertIn('ALREADY held', payload['messages'][0]['content'])

    def test_shared_role_expands_without_mutating_or_inventing(self):
        original = sample()
        saved = copy.deepcopy(original)
        result = validate_decomposition(from_flat_roles(original))
        self.assertEqual(original, saved)
        self.assertEqual(len(result['task_roles']), 1)
        self.assertEqual([s['required_roles'][0]['role_id'] for s in result['subtasks']], ['kettle', 'kettle'])
        result['subtasks'][0]['required_roles'][0]['name'] = 'changed'
        self.assertEqual(result['subtasks'][1]['required_roles'][0]['name'], 'electric kettle')

    def test_empty_roles_and_recursive_subtask_as_role_are_rejected(self):
        for role in ({}, sample()['subtasks'][0]):
            data = sample()
            data['object_roles'] = [role]
            with self.subTest(role=role), self.assertRaises(ValueError):
                from_flat_roles(data)

    def test_unknown_or_nested_role_reference_not_guessed(self):
        for references in (['missing'], [{}], ['kettle', 'kettle']):
            data = sample()
            data['subtasks'][0]['required_role_ids'] = references
            with self.subTest(references=references), self.assertRaises(ValueError):
                from_flat_roles(data)

    def test_duplicate_robot_unused_role_and_instance_ids_are_rejected(self):
        for change in ('duplicate', 'robot', 'unused', 'element_id'):
            data = sample()
            role = data['object_roles'][0]
            if change == 'duplicate':
                data['object_roles'].append(copy.deepcopy(role))
            elif change == 'robot':
                role['role_id'] = 'robot'
            elif change == 'unused':
                unused = copy.deepcopy(role)
                unused['role_id'] = 'cup'
                data['object_roles'].append(unused)
            else:
                role['element_id'] = 67
            with self.subTest(change=change), self.assertRaises(ValueError):
                from_flat_roles(data)

    def test_mixed_old_schema_and_bad_fields_rejected(self):
        for change in ('nested_roles', 'root_skills', 'quantity_bool', 'no_description'):
            data = sample()
            if change == 'nested_roles':
                data['subtasks'][0]['required_roles'] = [{}]
            elif change == 'root_skills':
                data['skills'] = []
            elif change == 'quantity_bool':
                data['object_roles'][0]['quantity'] = True
            else:
                data['subtasks'][0]['description'] = ''
            with self.subTest(change=change), self.assertRaises(ValueError):
                from_flat_roles(data)

    def test_native_role_goal_and_dependency_checks_still_apply(self):
        data = sample()
        data['goal_state'][0]['object_role'] = 'missing'
        with self.assertRaises(AlignmentError):
            validate_decomposition(from_flat_roles(data))
        data = sample()
        data['subtasks'][0]['dependencies'] = ['missing']
        with self.assertRaises(AlignmentError):
            validate_decomposition(from_flat_roles(data))

    @patch.dict(os.environ, {'DASHSCOPE_API_KEY': 'test-only'})
    @patch('align_objects.request_json')
    def test_bad_reply_not_included_in_correction_and_diagnostics_preserved(self, request):
        bad = sample()
        bad['object_roles'] = [{}]
        def reply(data):
            return {'choices': [{'message': {'content': json.dumps(data)}, 'finish_reason': 'stop'}],
                    'usage': {'completion_tokens': 200}, 'model': 'test-model-response'}
        request.side_effect = [reply(bad), reply(sample())]
        with tempfile.TemporaryDirectory() as directory:
            result = call_llm(sample()['instruction'], 'test-model', 'https://example.test', 'DASHSCOPE_API_KEY',
                              decomposition_format='flat_roles', diagnostics_dir=directory)
            self.assertEqual(result['decomposition_wire_format'], 'flat_roles')
            raw = json.loads((Path(directory) / 'llm_decomposition_attempt_1.json').read_text())
            self.assertEqual(raw['finish_reason'], 'stop')
            self.assertEqual(raw['usage']['completion_tokens'], 200)
        second = json.loads(request.call_args_list[1].args[0].data)
        self.assertEqual([m['role'] for m in second['messages']], ['system', 'user'])
        self.assertNotIn(json.dumps(bad), second['messages'][1]['content'])
        self.assertIn('Previous output failed validation', second['messages'][1]['content'])

    @patch.dict(os.environ, {'DASHSCOPE_API_KEY': 'test-only'})
    @patch('align_objects.request_json')
    def test_truncated_reply_rejected_and_attempt_budget_stays_bounded(self, request):
        request.return_value = {'choices': [{'message': {'content': json.dumps(sample())}, 'finish_reason': 'length'}]}
        with self.assertRaisesRegex(AlignmentError, 'truncated'):
            call_llm(sample()['instruction'], 'test-model', 'https://example.test', 'DASHSCOPE_API_KEY',
                     decomposition_format='flat_roles', max_attempts=2)
        self.assertEqual(request.call_count, 2)


if __name__ == '__main__':
    unittest.main()
