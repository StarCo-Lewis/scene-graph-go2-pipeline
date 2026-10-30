"""Offline signature/prompt/codec tests; no real model or execution claims."""
import copy
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from adapter_parameter_contract import project_library, assert_contract, signature, VERSION
from test_action_matrix_ab import inputs
from test_json_long_horizon import nodes
from action_matrix import make_codebook, decode_matrix, matrix_payload
from match_skills import build_result, PlanError, compact_skills, validate_proposal
from json_symbolic_review import review_symbolic
from sequential_proxy_core import validate_step, ExecutionRejected, serial_state_contract


class ParameterContractTests(unittest.TestCase):
    def fixture(self):
        d, library, state, _, matrix, retrieval = inputs()
        library['execution_contract']['adapter_mode'] = 'json_symbolic_test_only'
        return d, library, state, matrix, retrieval

    def test_projection_only_changes_copy_and_excludes_optional_declaration(self):
        _, library, _, _, _ = self.fixture()
        before = copy.deepcopy(library)
        scoped = project_library(library)
        self.assertEqual(library, before)
        nav = next(s for s in scoped['skills'] if s['skill_id'] == 'navigate_to_object')
        self.assertEqual([p['name'] for p in nav['parameters']], ['target'])
        excluded = scoped['adapter_parameter_projection']['excluded_optional_declarations']
        self.assertEqual([(e['skill_id'], e['parameter']) for e in excluded], [('navigate_to_object', 'standoff_m')])
        self.assertTrue(assert_contract(scoped))
        self.assertEqual(scoped['execution_contract']['allowed_parameters_by_skill'], signature())
        self.assertEqual(project_library(scoped), scoped)

    def test_unsupported_required_parameter_is_not_silently_removed(self):
        _, library, _, _, _ = self.fixture()
        library['skills'][0]['parameters'][1]['required'] = True
        with self.assertRaisesRegex(ValueError, 'Unsupported required adapter parameter'):
            project_library(library)

    def test_missing_or_wrong_required_object_parameter_fails_preflight(self):
        _, library, _, _, _ = self.fixture()
        for mutation in ('missing', 'optional', 'numeric', 'duplicate'):
            bad = copy.deepcopy(library)
            if mutation == 'missing':
                bad['skills'][0]['parameters'].pop(0)
            elif mutation == 'optional':
                bad['skills'][0]['parameters'][0]['required'] = False
            elif mutation == 'numeric':
                bad['skills'][0]['parameters'][0]['type'] = 'number'
            else:
                bad['skills'][0]['parameters'].append(copy.deepcopy(bad['skills'][0]['parameters'][0]))
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                project_library(bad)

    def test_codec_and_json_prompt_inventory_use_same_exact_parameters(self):
        d, library, state, matrix, retrieval = self.fixture()
        scoped = project_library(library)
        book = make_codebook(d['alignment'], scoped)
        self.assertEqual(book['values'], [])
        self.assertEqual([p['name'] for p in book['skills'][0]['slots']], ['target'])
        nav = next(s for s in compact_skills(scoped) if s['skill_id'] == 'navigate_to_object')
        self.assertEqual([p['name'] for p in nav['parameters']], ['target'])
        payload = matrix_payload(d['alignment'], scoped, book, 'offline', retrieval, False)
        self.assertEqual(json.loads(payload['messages'][1]['content'])['optional_numeric_preset_codes'], [])
        self.assertEqual(json.loads(payload['messages'][1]['content'])['execution_contract']['state_semantics_contract'],
                         serial_state_contract())
        proposal = decode_matrix(matrix, book, d['alignment'], scoped)
        plan = build_result(d['alignment'], scoped, proposal)
        review = review_symbolic(dict(d, plan=plan), scoped, nodes(), state, 1.5)
        self.assertEqual(review['status'], 'symbolically_valid_not_executed')

    def test_old_numeric_code_is_rejected_not_discarded(self):
        d, library, _, matrix, _ = self.fixture()
        old = make_codebook(d['alignment'], library)
        scoped = project_library(library)
        matrix['matrix'][0][3] = old['values'][0]['code']
        before = copy.deepcopy(matrix)
        with self.assertRaisesRegex(PlanError, 'unused parameter slot must be 0'):
            decode_matrix(matrix, make_codebook(d['alignment'], scoped), d['alignment'], scoped)
        self.assertEqual(matrix, before)

    def test_json_optional_parameter_is_rejected_without_mutating_output(self):
        d, library, _, matrix, _ = self.fixture()
        scoped = project_library(library)
        proposal = decode_matrix(matrix, make_codebook(d['alignment'], scoped), d['alignment'], scoped)
        proposal['subtask_plans'][0]['steps'][0]['arguments']['standoff_m'] = .6
        before = copy.deepcopy(proposal)
        with self.assertRaisesRegex(PlanError, 'unknown parameters'):
            validate_proposal(proposal, d['alignment'], scoped)
        self.assertEqual(proposal, before)

    def test_executor_checks_stay_strict(self):
        d, library, _, matrix, _ = self.fixture()
        scoped = project_library(library)
        proposal = decode_matrix(matrix, make_codebook(d['alignment'], scoped), d['alignment'], scoped)
        step = proposal['subtask_plans'][0]['steps'][0]
        validate_step(step, {'office': {'element_ids': [13]}}, {13}, {13})
        step['arguments']['standoff_m'] = .6
        with self.assertRaises(ExecutionRejected):
            validate_step(step, {'office': {'element_ids': [13]}}, {13}, {13})

    def test_contract_tampering_and_wrong_adapter_fail(self):
        _, library, _, _, _ = self.fixture()
        scoped = project_library(library)
        scoped['execution_contract']['allowed_parameters_by_skill']['navigate_to_object'].append('standoff_m')
        with self.assertRaises(ValueError):
            assert_contract(scoped)
        scoped = project_library(library)
        scoped['execution_contract']['state_semantics_contract']['successful_action_effects']['navigate_to_object']['held_object_id'] = 'null'
        with self.assertRaisesRegex(ValueError, 'Advertised state semantics'):
            assert_contract(scoped)
        library['execution_contract']['adapter_mode'] = 'some_other_robot'
        with self.assertRaises(ValueError):
            project_library(library)


if __name__ == '__main__':
    unittest.main()
