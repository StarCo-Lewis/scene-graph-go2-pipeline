import copy
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_json_long_horizon import row, nodes
from run_json_long_horizon import initial_state, prepare_round
from run_navigation_demo import read_json, write_json
from action_matrix import make_codebook, decode_matrix, matrix_payload, request_matrix
from match_skills import PlanError, build_result
from json_symbolic_review import review_symbolic
import run_action_matrix_ab as experiment


def inputs():
    descriptor, library = row([('navigate_to_object', 13), ('position_arm_for_object', 13),
                                ('place_object_on_surface', 67, 13)], support=13)
    alignment = descriptor['alignment']
    instruction = 'Carry the held kettle ID 67 to sideboard ID 13 and place it there.'
    descriptor['instruction'] = instruction
    alignment['task_decomposition'].update(instruction=instruction,
        subtasks=[dict(subtask_id='st_01', description='Carry to the sideboard', dependencies=[]),
                  dict(subtask_id='st_02', description='Place on the sideboard', dependencies=['st_01'])])
    state = initial_state(nodes())
    state.update(held_object_id=67, lifted=True, prepared_target_id=67)
    book = make_codebook(alignment, library)
    skills = {s['skill_id']: s['code'] for s in book['skills']}
    objects = {o['role_id']: o['code'] for o in book['objects']}
    matrix = {'matrix': [[1, skills['navigate_to_object'], objects['office'], 0],
                          [2, skills['position_arm_for_object'], objects['office'], 0],
                          [2, skills['place_object_on_surface'], objects['kettle'], objects['office']]]}
    retrieval = {'subtasks': [{'subtask_id': t['subtask_id'],
        'candidate_skill_ids': [s['skill_id'] for s in library['skills']]} for t in alignment['task_decomposition']['subtasks']]}
    return descriptor, library, state, book, matrix, retrieval


class MatrixTests(unittest.TestCase):
    def bottle_inputs(self, alias=False):
        d, lib, _, _, _, _ = inputs()
        alignment = d['alignment']
        alignment['context_bindings'] = {'black_water_bottle_121': {'element_ids': [121]}}
        if alias:
            alignment['context_bindings']['another_bottle_role'] = {'element_ids': [121]}
        alignment['task_decomposition'].update(
            instruction='Pick up and lift the black water bottle (ID 121).',
            goal_state=[], subtasks=[dict(subtask_id='st_01', description='Grasp the bottle', dependencies=[])])
        book = make_codebook(alignment, lib)
        grasp = next(s['code'] for s in book['skills'] if s['skill_id'] == 'grasp_object')
        retrieval = {'subtasks': [{'subtask_id': 'st_01', 'candidate_skill_ids': ['grasp_object']}]}
        return alignment, lib, book, grasp, retrieval

    def test_scene_id_confusion_reports_exact_slot_and_bound_code_without_conversion(self):
        alignment, lib, book, grasp, _ = self.bottle_inputs()
        raw = {'matrix': [[1, grasp, 121, 0]]}
        before = copy.deepcopy((alignment, lib, book, raw))
        with self.assertRaises(PlanError) as caught:
            decode_matrix(raw, book, alignment, lib)
        error = str(caught.exception)
        for text in ('unknown object code 121', 'arg1_code (grasp_object.target)',
                     'allowed object codes=[1]', 'scene element_id, NOT a matrix object code',
                     '"code": 1', 'black_water_bottle_121', 'no automatic ID conversion'):
            self.assertIn(text, error)
        self.assertEqual((alignment, lib, book, raw), before)
        decoded = decode_matrix({'matrix': [[1, grasp, 1, 0]]}, book, alignment, lib)
        self.assertEqual(decoded['subtask_plans'][0]['steps'][0]['arguments']['target'],
                         {'role_id': 'black_water_bottle_121', 'element_id': 121})

    def test_unknown_object_code_does_not_get_a_guessed_mapping(self):
        alignment, lib, book, grasp, _ = self.bottle_inputs()
        with self.assertRaises(PlanError) as caught:
            decode_matrix({'matrix': [[1, grasp, 999, 0]]}, book, alignment, lib)
        self.assertIn('allowed object codes=[1]', str(caught.exception))
        self.assertNotIn('is a scene element_id', str(caught.exception))

    def test_shared_scene_id_feedback_lists_roles_without_choosing_one(self):
        alignment, lib, book, grasp, _ = self.bottle_inputs(alias=True)
        with self.assertRaises(PlanError) as caught:
            decode_matrix({'matrix': [[1, grasp, 121, 0]]}, book, alignment, lib)
        error = str(caught.exception)
        self.assertIn('allowed object codes=[1, 2]', error)
        self.assertIn('multiple roles share this scene ID', error)
        self.assertIn('another_bottle_role', error)
        self.assertIn('black_water_bottle_121', error)

    def test_namespace_guidance_is_last_and_repeated_in_correction_prompt(self):
        import json
        alignment, lib, book, grasp, retrieval = self.bottle_inputs()
        with self.assertRaises(PlanError) as caught:
            decode_matrix({'matrix': [[1, grasp, 121, 0]]}, book, alignment, lib)
        before = copy.deepcopy((alignment, lib, book, retrieval))
        payload = matrix_payload(alignment, lib, book, 'offline', retrieval, True,
                                 validation_error=str(caught.exception))
        self.assertIn('element_id and matrix object code are DIFFERENT namespaces',
                      payload['messages'][0]['content'])
        prompt = json.loads(payload['messages'][1]['content'])
        self.assertEqual(list(prompt)[-1], 'matrix_encoding_contract')
        contract = prompt['matrix_encoding_contract']
        self.assertEqual(contract['allowed_object_slot_codes'], [1])
        self.assertIn('scene element_id=121 -> matrix object code=1',
                      contract['bound_object_lookup'][0])
        self.assertIn('unknown object code 121', prompt['previous_validation_error'])
        self.assertIn('NEVER the scene element_id', prompt['correction_requirement'])
        self.assertEqual((alignment, lib, book, retrieval), before)

    def test_decode_then_original_symbolic_review(self):
        d, lib, state, book, matrix, _ = inputs()
        before = copy.deepcopy((d, lib, state, book, matrix))
        decoded = decode_matrix(matrix, book, d['alignment'], lib)
        plan = build_result(d['alignment'], lib, decoded)
        review = review_symbolic(dict(d, plan=plan), lib, nodes(), state, 1.5)
        self.assertEqual(review['navigation_targets'], [13])
        self.assertIsNone(review['predicted_end_state']['held_object_id'])
        self.assertEqual(review['predicted_end_state']['backend']['objects']['67']['support_id'], 13)
        self.assertEqual((d, lib, state, book, matrix), before)
        self.assertFalse(review['simulation_executed'])

    def test_row_counter_error_requires_grouping_without_removing_actions(self):
        d, lib, _, book, matrix, retrieval = inputs()
        invalid = copy.deepcopy(matrix)
        invalid['matrix'][-1][0] = 3  # Three actions but only TWO supplied subtasks.
        with self.assertRaises(PlanError) as caught:
            decode_matrix(invalid, book, d['alignment'], lib)
        error = str(caught.exception)
        self.assertIn('unknown subtask code 3', error)
        self.assertIn('allowed subtask codes=[1, 2]', error)
        self.assertIn('NOT the action row number', error)
        self.assertIn('delete necessary actions', error)
        import json
        payload = matrix_payload(d['alignment'], lib, book, 'offline', retrieval, True, error)
        prompt = json.loads(payload['messages'][1]['content'])
        self.assertIn('Do not remove preparation', prompt['correction_requirement'])
        self.assertIn('row count may exceed subtask count',
                      prompt['matrix_encoding_contract']['action_grouping_rule'])
        self.assertEqual(prompt['matrix_encoding_contract']['allowed_subtask_codes'], [1, 2])
        self.assertEqual(len(decode_matrix(matrix, book, d['alignment'], lib)['subtask_plans'][1]['steps']), 2)

    def test_serial_placement_prerequisites_and_targeted_feedback_are_advertised(self):
        import json
        d, lib, _, _, _, retrieval = inputs()
        lib['execution_contract']['adapter_mode'] = 'json_symbolic_test_only'
        book = make_codebook(d['alignment'], lib)
        payload = matrix_payload(d['alignment'], lib, book, 'offline', retrieval, True,
                                 'SURFACE_PROXY_PREPARATION_REQUIRED')
        prompt = json.loads(payload['messages'][1]['content'])
        rules = prompt['matrix_encoding_contract']['serial_proxy_preconditions']
        self.assertIn('clears prepared_target_id', rules['navigate_to_object'])
        self.assertIn('prepared_target_id equal to surface', rules['place_object_on_surface'])
        self.assertIn('SAME placement subtask code', rules['place_object_on_surface'])
        self.assertIn('reuse the owning placement subtask code for both actions',
                      prompt['missing_precondition_correction'])
        # Nonserial libraries do not inherit this proxy-specific state machine.
        lib['execution_contract']['adapter_mode'] = 'another_adapter'
        payload = matrix_payload(d['alignment'], lib, book, 'offline', retrieval, True,
                                 'SURFACE_PROXY_PREPARATION_REQUIRED')
        prompt = json.loads(payload['messages'][1]['content'])
        self.assertIsNone(prompt['matrix_encoding_contract']['serial_proxy_preconditions'])
        self.assertNotIn('missing_precondition_correction', prompt)

    def test_reject_empty_wrong_shape_types_and_codes(self):
        d, lib, _, book, matrix, _ = inputs()
        for bad in ({}, {'matrix': []}, {'matrix': [[]]}, {'matrix': [{}]},
            {'matrix': [[1, 1, 2]]}, {'matrix': [[True, 1, 2, 0]]},
            {'matrix': [[1, 1, 2., 0]]}, {'matrix': [[1, 1, '2', 0]]},
            {'matrix': [[0, 1, 2, 0]]}, {'matrix': [[1, 999, 2, 0]]},
            {'matrix': [[1, 1, 999, 0]]}, {'matrix': [[1, 1, 0, 0]]},
            {'matrix': matrix['matrix'], 'extra': 1}):
            with self.subTest(bad=bad), self.assertRaises(PlanError):
                decode_matrix(bad, book, d['alignment'], lib)

    def test_missing_subtask_reversed_order_and_missing_surface_rejected(self):
        d, lib, _, book, matrix, _ = inputs()
        for rows in (matrix['matrix'][:1], list(reversed(matrix['matrix'])),
                     [matrix['matrix'][0], matrix['matrix'][2][:-1]+[0]],
                     [matrix['matrix'][0], matrix['matrix'][1][:-1]+[1], matrix['matrix'][2]]):
            with self.assertRaises(PlanError):
                decode_matrix({'matrix': rows}, book, d['alignment'], lib)

    def test_codebook_changes_detected(self):
        d, lib, _, book, matrix, _ = inputs()
        book['objects'][0]['element_id'] = 999
        with self.assertRaisesRegex(PlanError, 'codebook differs'):
            decode_matrix(matrix, book, d['alignment'], lib)

    def test_optional_numeric_preset_not_literal_or_object_code(self):
        d, lib, _, book, matrix, _ = inputs()
        preset = next(v for v in book['values'] if v['parameter'] == 'standoff_m')
        valid = copy.deepcopy(matrix)
        valid['matrix'][0][3] = preset['code']
        proposal = decode_matrix(valid, book, d['alignment'], lib)
        self.assertEqual(proposal['subtask_plans'][0]['steps'][0]['arguments']['standoff_m'], .6)
        self.assertNotIn('standoff_m', decode_matrix(matrix, book, d['alignment'], lib)['subtask_plans'][0]['steps'][0]['arguments'])
        for code in (matrix['matrix'][0][2], .6):
            invalid = copy.deepcopy(matrix)
            invalid['matrix'][0][3] = code
            with self.assertRaises(PlanError):
                decode_matrix(invalid, book, d['alignment'], lib)

    def test_alias_roles_for_same_instance_keep_explicit_binding(self):
        d, lib, _, _, _, _ = inputs()
        d['alignment']['context_bindings']['another_kettle_role'] = {'element_ids': [67]}
        book = make_codebook(d['alignment'], lib)
        codes = [o['code'] for o in book['objects'] if o['element_id'] == 67]
        self.assertEqual(len(codes), 2)
        self.assertNotEqual(*codes)

    def test_repeated_actions_are_not_removed(self):
        d, lib, _, book, matrix, _ = inputs()
        matrix['matrix'].insert(0, copy.deepcopy(matrix['matrix'][0]))
        proposal = decode_matrix(matrix, book, d['alignment'], lib)
        self.assertEqual(len(proposal['subtask_plans'][0]['steps']), 2)

    def test_unsupported_continuous_required_parameter_rejected(self):
        d, lib, _, _, _, _ = inputs()
        lib['skills'][0]['parameters'][1]['required'] = True
        with self.assertRaisesRegex(PlanError, 'nonobject slots'):
            make_codebook(d['alignment'], lib)

    def test_candidates_and_codebook_stable_on_expansion(self):
        d, lib, _, book, _, retrieval = inputs()
        for t in retrieval['subtasks']:
            t['candidate_skill_ids'] = ['navigate_to_object']
        first = matrix_payload(d['alignment'], lib, book, 'offline', retrieval, False)
        second = matrix_payload(d['alignment'], lib, book, 'offline', retrieval, True)
        import json
        a, b = [json.loads(p['messages'][1]['content']) for p in (first, second)]
        self.assertEqual(a['codebook_sha256'], b['codebook_sha256'])
        self.assertEqual(a['object_codes'], b['object_codes'])
        self.assertEqual(len(a['skills']), 1)
        self.assertEqual(len(b['skills']), len(lib['skills']))
        self.assertNotIn('SKILL JSON:', a['execution_contract'].get('strict_prompt_v2',''))

    def test_matrix_raw_http_content_audited_without_decoding(self):
        d, lib, _, book, matrix, retrieval = inputs()
        import json
        def transport(req, timeout, attempts, audit):
            body = {'choices': [{'message': {'content': json.dumps(matrix)}, 'finish_reason': 'stop'}],
                    'usage': {'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15}}
            audit.received(1, json.dumps(body))
            return body
        with tempfile.TemporaryDirectory() as tmp, patch.dict('os.environ', {'TEST_KEY':'fixture_key_not_real'}), \
             patch('action_matrix.request_json_with_retry', side_effect=transport):
            value = request_matrix(d['alignment'], lib, book, 'offline', 'https://invalid.test',
                'TEST_KEY', retrieval, diagnostics_dir=Path(tmp))
            self.assertEqual(value, matrix)
            from token_usage_audit import usage
            self.assertEqual(usage(Path(tmp))['total_tokens'], 15)
            records = [read_json(p) for p in (Path(tmp)/'llm_response_audit').glob('*.json')]
            request = next(r for r in records if 'model_input' in r)
            self.assertEqual(request['model_input']['response_format'], {'type':'json_object'})
            self.assertNotIn('Authorization', json.dumps(records))

    def test_json_mode_requirement_does_not_depend_on_incidental_context(self):
        import re
        d, lib, _, _, _, retrieval = inputs()
        # No strict_prompt_v2/SKILL JSON text: reproduce the live entry's context.
        lib['execution_contract'] = {}
        book = make_codebook(d['alignment'], lib)
        payload = matrix_payload(d['alignment'], lib, book, 'fixture', retrieval, False)
        self.assertEqual(payload['response_format'], {'type': 'json_object'})
        system = payload['messages'][0]['content']
        self.assertRegex(system.lower(), r'\bjson\b')
        self.assertIn('"matrix":', system)
        # The word must be supplied by our system contract, not by unrelated input.
        self.assertFalse(re.search(r'\bjson\b', payload['messages'][1]['content'], re.IGNORECASE))

    def test_wrong_object_code_still_fails_runtime(self):
        d, lib, state, book, matrix, _ = inputs()
        # Same type and valid code, but wrong semantic choice: cannot navigate to held kettle.
        matrix['matrix'][0][2] = next(o['code'] for o in book['objects'] if o['role_id'] == 'kettle')
        proposal = decode_matrix(matrix, book, d['alignment'], lib)
        with self.assertRaises(ValueError):
            review_symbolic(dict(d, plan=build_result(d['alignment'], lib, proposal)), lib, nodes(), state, 1.5)

    def test_worker_real_decode_and_review_no_external_calls(self):
        d, lib, state, book, matrix, retrieval = inputs()
        with tempfile.TemporaryDirectory() as tmp:
            source, dest = Path(tmp)/'input', Path(tmp)/'result'
            source.mkdir()
            dest.mkdir()
            for name in experiment.FILES:
                write_json(source/name, {'fixture': name})
            before = experiment.input_hashes(source)
            args = SimpleNamespace(worker_input=source, output_dir=dest, arm='matrix',
                                   model='offline', base_url='https://invalid.test')
            with patch.object(experiment, 'load_inputs', return_value=(state,d,lib,nodes(),d['alignment'],retrieval)), \
                 patch.object(experiment, 'request_matrix', return_value=matrix) as request:
                self.assertEqual(experiment.worker(args), 0)
            request.assert_called_once()
            self.assertEqual(before, experiment.input_hashes(source))
            self.assertEqual(read_json(dest/'raw_proposal_attempt_1.json'), matrix)
            self.assertEqual(read_json(dest/'skill_plan.json')['proposal_origin'], 'llm_matrix_decoded')
            self.assertTrue(read_json(dest/'trial_summary.json')['first_proposal_valid'])

    def test_worker_rejected_matrix_saved_then_retry_widens(self):
        d, lib, state, book, matrix, retrieval = inputs()
        with tempfile.TemporaryDirectory() as tmp:
            source, dest = Path(tmp)/'input', Path(tmp)/'result'
            source.mkdir()
            dest.mkdir()
            for name in experiment.FILES:
                write_json(source/name, {'fixture': name})
            args = SimpleNamespace(worker_input=source, output_dir=dest, arm='matrix',
                                   model='offline', base_url='https://invalid.test')
            with patch.object(experiment, 'load_inputs', return_value=(state,d,lib,nodes(),d['alignment'],retrieval)), \
                 patch.object(experiment, 'request_matrix', side_effect=[{'matrix':[[]]}, matrix]) as request:
                self.assertEqual(experiment.worker(args), 0)
            self.assertEqual(read_json(dest/'raw_proposal_attempt_1.json'), {'matrix':[[]]})
            self.assertFalse(request.call_args_list[0].kwargs['widen_inventory'])
            self.assertTrue(request.call_args_list[1].kwargs['widen_inventory'])
            self.assertFalse(read_json(dest/'trial_summary.json')['first_proposal_valid'])


if __name__ == '__main__':
    unittest.main()
