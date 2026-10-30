import copy
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_action_matrix_ab import inputs
from test_json_long_horizon import nodes
from run_navigation_demo import read_json, write_json
from action_matrix import decode_matrix
from match_skills import build_result, PlanError
from skill_output_normalization import normalize, normalize_and_record, NormalizationError
import run_action_matrix_ab as experiment


class NormalizationTests(unittest.TestCase):
    def setUp(self):
        self.d, self.lib, self.state, self.book, self.matrix, self.retrieval = inputs()
        self.proposal = decode_matrix(self.matrix, self.book, self.d['alignment'], self.lib)

    def alias_proposal(self):
        value = copy.deepcopy(self.proposal)
        for task in value['subtask_plans']:
            for step in task['steps']:
                step['s'] = step.pop('skill_id')
        return value

    def test_bare_matrix_preserved_and_decoded(self):
        raw = copy.deepcopy(self.matrix['matrix'])
        value, record = normalize(raw, 'matrix', self.lib)
        self.assertEqual(value, self.matrix)
        self.assertEqual(raw, self.matrix['matrix'])
        self.assertEqual(decode_matrix(value, self.book, self.d['alignment'], self.lib), self.proposal)
        self.assertEqual(record['raw_action_count'], record['normalized_action_count'])

    def test_known_skill_alias_preserves_parameters_and_order(self):
        raw = self.alias_proposal()
        before = copy.deepcopy(raw)
        value, record = normalize(raw, 'nested_json', self.lib)
        self.assertEqual(value, self.proposal)
        self.assertEqual(raw, before)
        self.assertEqual(len(record['changes']), 3)
        build_result(self.d['alignment'], self.lib, value)

    def test_missing_skill_not_inferred_from_arguments(self):
        raw = copy.deepcopy(self.proposal)
        raw['subtask_plans'][0]['steps'][0] = {'s': {'target': {'element_id': 13}}}
        value, record = normalize(raw, 'nested_json', self.lib)
        self.assertEqual(value, raw)
        self.assertFalse(record['changed'])
        with self.assertRaises(PlanError):
            build_result(self.d['alignment'], self.lib, value)

    def test_empty_unknown_and_whitespace_aliases_not_filled(self):
        for step in ({}, {'s': ''}, {'s': None}, {'s': 'unknown'}, {'s': ' navigate_to_object '}):
            raw = copy.deepcopy(self.proposal)
            raw['subtask_plans'][0]['steps'][0] = step
            value, record = normalize(raw, 'nested_json', self.lib)
            self.assertEqual(value, raw)
            self.assertFalse(record['changed'])
            with self.assertRaises(PlanError):
                build_result(self.d['alignment'], self.lib, value)

    def test_ambiguous_alias_rejects_whole_proposal_and_records_it(self):
        raw = self.alias_proposal()
        raw['subtask_plans'][1]['steps'][0]['skill_id'] = raw['subtask_plans'][1]['steps'][0]['s']
        value, record = normalize(raw, 'nested_json', self.lib)
        self.assertEqual(value, raw)
        self.assertEqual(record['changes'], [])
        self.assertEqual(record['status'], 'rejected_ambiguous')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(NormalizationError):
                normalize_and_record(raw, 'nested_json', self.lib, root, 'test', 'raw.json')
            saved = read_json(root/'test.normalization.json')
            self.assertEqual(saved['raw_sha256'], saved['normalized_sha256'])
            self.assertFalse((root/'test.normalized.json').exists())

    def test_no_recursive_argument_rename_or_wrapper_flattening(self):
        raw = copy.deepcopy(self.proposal)
        raw['subtask_plans'][0]['steps'][0]['arguments']['s'] = 'navigate_to_object'
        value, record = normalize(raw, 'nested_json', self.lib)
        self.assertEqual(value, raw)
        self.assertFalse(record['changed'])
        raw['subtask_plans'][0]['steps'] = [{'subtask_id': 'st_01', 'steps': self.alias_proposal()['subtask_plans'][0]['steps']}]
        value, record = normalize(raw, 'nested_json', self.lib)
        self.assertEqual(value, raw)
        self.assertFalse(record['changed'])

    def test_repeated_actions_and_invalid_matrix_codes_not_repaired(self):
        raw = copy.deepcopy(self.matrix['matrix'])
        raw.insert(0, copy.deepcopy(raw[0]))
        value, record = normalize(raw, 'matrix', self.lib)
        self.assertEqual(value['matrix'], raw)
        self.assertEqual(record['normalized_action_count'], 4)
        for invalid in ([[True, 1, 1, 0]], [[1, 999, 1, 0]], [[]]):
            value, _ = normalize(invalid, 'matrix', self.lib)
            with self.assertRaises(PlanError):
                decode_matrix(value, self.book, self.d['alignment'], self.lib)

    def test_missing_parameter_not_filled_after_alias_rename(self):
        raw = self.alias_proposal()
        raw['subtask_plans'][0]['steps'][0]['arguments'] = {}
        value, record = normalize(raw, 'nested_json', self.lib)
        self.assertTrue(record['changed'])
        with self.assertRaisesRegex(PlanError, 'missing required parameter'):
            build_result(self.d['alignment'], self.lib, value)

    def test_idempotent_canonical_output(self):
        for raw, kind in ((self.matrix, 'matrix'), (self.proposal, 'nested_json')):
            value, record = normalize(raw, kind, self.lib)
            self.assertEqual(value, raw)
            self.assertFalse(record['changed'])
            self.assertEqual(record['raw_sha256'], record['normalized_sha256'])
            self.assertEqual(normalize(value, kind, self.lib)[0], value)

    def worker_trial(self, arm, responses):
        with tempfile.TemporaryDirectory() as tmp:
            source, dest = Path(tmp)/'input', Path(tmp)/'result'
            source.mkdir()
            dest.mkdir()
            for name in experiment.FILES:
                write_json(source/name, {'fixture': name})
            args = SimpleNamespace(worker_input=source, output_dir=dest, arm=arm,
                                   model='offline', base_url='https://invalid.test')
            method = 'request_matrix' if arm == 'matrix' else 'request_plan'
            with patch.object(experiment, 'load_inputs', return_value=(self.state,self.d,self.lib,nodes(),self.d['alignment'],self.retrieval)), \
                 patch.object(experiment, method, side_effect=responses) as request:
                code = experiment.worker(args)
            artifacts = {p.name: read_json(p) for p in dest.glob('*.json')}
            return code, request.call_count, artifacts

    def test_worker_bare_matrix_accepted_without_another_request(self):
        raw = self.matrix['matrix']
        code, calls, files = self.worker_trial('matrix', [raw])
        self.assertEqual((code, calls), (0, 1))
        self.assertEqual(files['raw_proposal_attempt_1.json'], raw)
        self.assertEqual(files['proposal_attempt_1.normalized.json'], self.matrix)
        summary = files['trial_summary.json']
        self.assertFalse(summary['first_proposal_strict_valid'])
        self.assertTrue(summary['first_proposal_normalized_valid'])
        self.assertTrue(summary['accepted_after_normalization'])

    def test_worker_alias_accepted_without_another_request(self):
        raw = self.alias_proposal()
        code, calls, files = self.worker_trial('nested_json', [raw])
        self.assertEqual((code, calls), (0, 1))
        self.assertEqual(files['raw_proposal_attempt_1.json'], raw)
        self.assertEqual(files['proposal_attempt_1.normalized.json'], self.proposal)
        self.assertTrue(files['trial_summary.json']['first_proposal_normalized_valid'])

    def test_worker_missing_skill_requires_retry_not_guessing(self):
        raw = copy.deepcopy(self.proposal)
        raw['subtask_plans'][0]['steps'][0] = {'s': {'target': {'element_id': 13}}}
        code, calls, files = self.worker_trial('nested_json', [raw, self.proposal])
        self.assertEqual((code, calls), (0, 2))
        self.assertFalse(files['trial_summary.json']['first_proposal_valid'])
        self.assertEqual(files['raw_proposal_attempt_1.json'], raw)
        self.assertFalse(files['proposal_attempt_1.normalization.json']['changed'])

    def test_live_planner_alias_and_ambiguous_audit(self):
        import plan_live_round as live
        from json_symbolic_review import review_symbolic
        raw = self.alias_proposal()
        ambiguous = copy.deepcopy(raw)
        ambiguous['subtask_plans'][0]['steps'][0]['skill_id'] = 'navigate_to_object'
        for responses, expected_calls in (([raw], 1), ([ambiguous, raw], 2)):
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                for name, value in [('round_input.json', self.d), ('runtime_state.json', self.state),
                    ('execution_skill_library.json', self.lib), ('input_memory/scene/scene_graph.json', nodes())]:
                    write_json(root/name, value)
                args = SimpleNamespace(output_dir=root, round_input=root/'round_input.json', model='offline',
                                       base_url='https://invalid.test', reach=1.5)
                def retain(plan, critique):
                    return copy.deepcopy(plan)
                with patch.object(live, 'decompose_and_bind', return_value=self.d['alignment']), \
                     patch.object(live, 'request_plan', side_effect=responses) as request, \
                     patch.object(live, 'request_order', return_value={'reason': 'Retain'}), \
                     patch.object(live, 'candidate_plan', side_effect=retain):
                    self.assertEqual(live.plan(args, reviewer=review_symbolic), 0)
                self.assertEqual(request.call_count, expected_calls)
                summary = read_json(root/'planning_summary.json')
                self.assertEqual(len(summary['skill_format_normalizations']), expected_calls)
                self.assertTrue(read_json(root/'skill_plan_final.json')['format_normalization']['changed'])
                self.assertEqual(read_json(root/f'llm_skill_proposal_semantic_1_attempt_{expected_calls}.json'), raw)
                if expected_calls == 2:
                    self.assertEqual(summary['skill_format_normalizations'][0]['status'], 'rejected_ambiguous')


if __name__ == '__main__':
    unittest.main()
