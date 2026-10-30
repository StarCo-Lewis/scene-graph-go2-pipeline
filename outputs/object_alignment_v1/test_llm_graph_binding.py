import unittest
import copy
import json
import tempfile
from pathlib import Path
from unittest.mock import patch
from llm_graph_binding import (validate_bindings, BindingError, request_bindings, request_validated_bindings,
                               partition_initial_constraints, check_binding_relations)


class GraphTests(unittest.TestCase):
    def setUp(self):
        self.nodes = [{'id': 2, 'object_tag': 'main table', 'caption': 'Workbench'},
                      {'id': 4, 'object_tag': 'coke can', 'caption': 'A red can'},
                      {'id': 5, 'object_tag': 'rolling office chair', 'caption': 'Black chair'}]
        self.roles = [{'role_id': 'can', 'quantity': 1, 'required': True}]

    def proposal(self, status, ids):
        return {'bindings': [{'role_id': 'can', 'status': status, 'element_ids': ids, 'reason': 'Fixture'}]}

    def test_bind(self):
        binding, errors = validate_bindings(self.proposal('matched', [4]), self.roles, self.nodes)
        self.assertEqual(binding['can']['element_ids'], [4]); self.assertFalse(errors)

    def test_id_types_rejected_with_path_and_original_value_unchanged(self):
        for value in ('4', 4.0, True, [4], {'id': 4}, None):
            proposal = self.proposal('matched', [value])
            original = copy.deepcopy(proposal)
            with self.subTest(value=value), self.assertRaisesRegex(
                    BindingError, r'bindings\[0\].element_ids\[0\].*JSON integer'):
                validate_bindings(proposal, self.roles, self.nodes)
            self.assertEqual(proposal, original)

    def test_duplicate_integer_ids_have_different_diagnostic(self):
        with self.assertRaisesRegex(BindingError, 'contains duplicate IDs'):
            validate_bindings(self.proposal('matched', [4, 4]), self.roles, self.nodes)

    def test_bounded_correction_keeps_original_bad_response_and_same_input(self):
        bad, good = self.proposal('matched', ['4']), self.proposal('matched', [4])
        original = copy.deepcopy(bad)
        with tempfile.TemporaryDirectory() as directory, patch(
                'llm_graph_binding.request_bindings', side_effect=[bad, good]) as request:
            proposal, bindings, errors, attempt = request_validated_bindings(
                self.roles, self.nodes, 'test', 'https://invalid.test', 'TEST_KEY',
                instruction='Use can ID 4.', diagnostics_dir=Path(directory))
            self.assertEqual(attempt, 2)
            self.assertFalse(errors)
            self.assertEqual(bindings['can']['element_ids'], [4])
            self.assertIs(proposal, good)
            self.assertEqual(bad, original)
            self.assertEqual(json.loads((Path(directory)/'llm_binding_proposal_attempt_1.json').read_text()), original)
            self.assertEqual(json.loads((Path(directory)/'llm_binding_proposal_attempt_2.json').read_text()), good)
            first, second = request.call_args_list
            self.assertEqual(first.args, second.args)
            self.assertEqual(second.kwargs['instruction'], first.kwargs['instruction'])
            self.assertIn('JSON integer', second.kwargs['validation_error'])
            self.assertEqual(second.kwargs['validation_attempt'], 2)

    def test_repeated_bad_types_stop_after_three_without_converting_ids(self):
        bad = self.proposal('matched', ['4'])
        with patch('llm_graph_binding.request_bindings', return_value=bad) as request:
            with self.assertRaisesRegex(BindingError, 'remained invalid after 3 attempts'):
                request_validated_bindings(self.roles, self.nodes, 'test', 'https://invalid.test', 'TEST_KEY')
            self.assertEqual(request.call_count, 3)
        self.assertEqual(bad['bindings'][0]['element_ids'], ['4'])

    def test_semantic_missing_or_ambiguity_is_terminal_not_fished_for_success(self):
        for status, ids in [('missing', []), ('ambiguous', [4])]:
            with self.subTest(status=status), patch('llm_graph_binding.request_bindings',
                    return_value=self.proposal(status, ids)) as request:
                _, bindings, errors, attempt = request_validated_bindings(
                    self.roles, self.nodes, 'test', 'https://invalid.test', 'TEST_KEY')
                self.assertFalse(bindings)
                self.assertEqual(errors[0]['reason'], status)
                self.assertEqual(attempt, 1)
                request.assert_called_once()

    def test_transport_failure_does_not_get_extra_schema_retries(self):
        with patch('llm_graph_binding.request_bindings', side_effect=RuntimeError('fixture transport failure')) as request:
            with self.assertRaisesRegex(RuntimeError, 'fixture transport failure'):
                request_validated_bindings(self.roles, self.nodes, 'test', 'https://invalid.test', 'TEST_KEY')
            request.assert_called_once()

    def test_correction_prompt_requires_integer_syntax_without_bad_proposal(self):
        def transport(request, **kwargs):
            payload = json.loads(request.data)
            system = payload['messages'][0]['content']
            data = json.loads(payload['messages'][1]['content'])
            self.assertIn('never strings, floats or booleans', system)
            self.assertEqual(data['previous_validation_error'], 'element_ids[0] was a string')
            self.assertEqual(data['instruction'], 'Use can ID 4.')
            self.assertEqual(data['nodes'], self.nodes)
            self.assertNotIn('previous_rejected_proposal', data)
            return {'choices': [{'message': {'content': json.dumps(self.proposal('matched', [4]))}}]}
        with patch.dict('os.environ', {'TEST_KEY': 'test-only'}), \
                patch('llm_graph_binding.request_json', side_effect=transport):
            request_bindings(self.roles, self.nodes, 'test', 'https://invalid.test', 'TEST_KEY',
                instruction='Use can ID 4.', validation_error='element_ids[0] was a string', validation_attempt=2)

    def test_schema_retry_budget_cannot_be_expanded(self):
        for attempts in (0, 4, True):
            with self.subTest(attempts=attempts), patch('llm_graph_binding.request_bindings') as request:
                with self.assertRaises(BindingError):
                    request_validated_bindings(self.roles, self.nodes, 'test', 'https://invalid.test',
                                               'TEST_KEY', max_validation_attempts=attempts)
                request.assert_not_called()

    def test_missing_and_ambiguous(self):
        for status in ('missing', 'ambiguous'):
            binding, errors = validate_bindings(self.proposal(status, []), self.roles, self.nodes)
            self.assertFalse(binding); self.assertEqual(errors[0]['reason'], status)

    def test_fabricated_id(self):
        with self.assertRaises(BindingError):
            validate_bindings(self.proposal('matched', [99]), self.roles, self.nodes)

    def test_missing_role(self):
        with self.assertRaises(BindingError):
            validate_bindings({'bindings': []}, self.roles, self.nodes)

    def test_inactive(self):
        self.nodes[1]['active'] = False
        with self.assertRaises(BindingError):
            validate_bindings(self.proposal('matched', [4]), self.roles, self.nodes)

    def test_ambiguous_candidates_never_bound(self):
        bindings, errors = validate_bindings(self.proposal('ambiguous', [4]), self.roles, self.nodes)
        self.assertFalse(bindings)
        self.assertEqual(errors[0]['candidate_ids'], [4])
        self.assertFalse(errors[0]['candidate_ids_are_bindings'])

    def test_ambiguous_unknown_id_rejected(self):
        with self.assertRaises(BindingError):
            validate_bindings(self.proposal('ambiguous', [99]), self.roles, self.nodes)

    def test_robot_constraints_routed_without_mutating_or_dropping_original(self):
        constraints = [dict(constraint_id='robot_near', subject_role='robot', relation='near', object_role='can', required=True),
                       dict(constraint_id='can_on', subject_role='can', relation='on', object_role='table', required=True)]
        original = copy.deepcopy(constraints)
        graph, runtime = partition_initial_constraints(constraints)
        self.assertEqual([c['constraint_id'] for c in graph], ['can_on'])
        self.assertEqual([c['constraint_id'] for c in runtime], ['robot_near'])
        runtime[0]['required'] = False
        self.assertEqual(constraints, original)

    def test_binding_http_contains_object_constraints_only_and_explicit_scope(self):
        constraints = [dict(constraint_id='robot_near', subject_role='robot', relation='near', object_role='can', required=True),
                       dict(constraint_id='can_on', subject_role='can', relation='on', object_role='table', required=True)]
        def transport(request, **kwargs):
            payload = json.loads(request.data)
            data = json.loads(payload['messages'][1]['content'])
            self.assertEqual([c['constraint_id'] for c in data['initial_scene_constraints']], ['can_on'])
            self.assertEqual(data['constraint_scope'], 'object_to_object_scene_relations_only')
            self.assertEqual(data['robot_state_validation'], 'deferred_to_runtime_not_object_binding')
            self.assertIn('Never demand robot nodes', payload['messages'][0]['content'])
            return {'choices': [{'message': {'content': json.dumps(self.proposal('matched', [4]))}}]}
        with patch.dict('os.environ', {'DASHSCOPE_API_KEY': 'test-only-not-a-key'}), \
                patch('llm_graph_binding.request_json', side_effect=transport):
            proposal = request_bindings(self.roles, self.nodes, 'test', 'https://invalid.test',
                'DASHSCOPE_API_KEY', instruction='Carry the held can.', initial_constraints=constraints)
        self.assertEqual(proposal['bindings'][0]['element_ids'], [4])

    def test_no_robot_edges_not_an_object_binding_error_but_missing_object_edges_still_reject(self):
        bindings = {'can': {'element_ids': [4]}, 'table': {'element_ids': [2]}}
        near = dict(constraint_id='robot_near', subject_role='robot', relation='near', object_role='can', required=True)
        self.assertEqual(check_binding_relations(bindings, [near], []), ([], []))
        on = dict(constraint_id='can_on', subject_role='can', relation='on', object_role='table', required=True)
        checks, errors = check_binding_relations(bindings, [near, on], [])
        self.assertEqual(checks[0]['status'], 'unknown')
        self.assertEqual(errors[0]['reason'], 'relation_unverified')

    def test_bad_constraint_container_not_silently_ignored(self):
        for bad in ({'subject_role': 'robot'}, [None], [{}]):
            with self.subTest(bad=bad), self.assertRaises(BindingError):
                partition_initial_constraints(bad)
