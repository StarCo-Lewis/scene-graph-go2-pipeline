"""Synthetic local fixtures only; no real embedding/LLM/simulator verification."""
import copy
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_action_matrix_type_suite as suite
from test_action_matrix_ab import inputs
from test_json_long_horizon import nodes
from run_json_long_horizon import prepare_round
from run_navigation_demo import write_json, read_json
from action_matrix import decode_matrix, make_codebook


def saved_fixture(source):
    d, _, state, _, matrix, _ = inputs()
    descriptor = {k: v for k, v in d.items() if k not in ('alignment', 'plan')}
    descriptor.update(round=1, expected_navigation_targets=[13], expected_support_id=13)
    prepare_round(source, descriptor, state, nodes(), 'offline', 'https://invalid.test')
    library = read_json(source/'execution_skill_library.json')
    proposal = decode_matrix(matrix, make_codebook(d['alignment'], library), d['alignment'], library)
    write_json(source/'object_alignment.json', d['alignment'])
    write_json(source/'skill_plan_final.json', suite.build_result(d['alignment'], library, proposal))


def fake_retrieval(shared, digest='fixture_digest'):
    retrieval = read_json(shared/'skill_retrieval_semantic_1.json')
    retrieval.update(source='same_frozen_real_embedding_retrieval_for_both_formats',
                     embedding_model_sha256=digest, synthetic_unit_fixture=True)
    for task in retrieval['subtasks']:
        task['confident'] = False
    write_json(shared/'skill_retrieval_semantic_1.json', retrieval)
    write_json(shared.parent/'embedding_metrics.json', {'seconds_including_startup': .1})


class TypeSuiteTests(unittest.TestCase):
    def test_catalog_distinct_and_increasing_types(self):
        cases = suite.catalog(Path('scene'))
        self.assertEqual(len(cases), 7)
        self.assertEqual(len({c['case_id'] for c in cases}), 7)
        self.assertEqual([c['difficulty'] for c in cases], sorted(c['difficulty'] for c in cases))

    def test_null_assertion_only_removed_not_other_assertions(self):
        d = {'expected_support_id': None, 'expected_end': {'held_object_id': 67}}
        original = copy.deepcopy(d)
        result = suite.normalize_assertions(d)
        self.assertNotIn('expected_support_id', result)
        self.assertEqual(result['expected_end'], original['expected_end'])
        self.assertEqual(d, original)
        self.assertEqual(suite.normalize_assertions(dict(d, expected_support_id=13))['expected_support_id'], 13)

    def test_case_preflight_source_unchanged_and_live_placeholder_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, shared = Path(tmp)/'source', Path(tmp)/'case/shared_input'
            saved_fixture(source)
            before = suite.source_hashes(source)
            case = dict(case_id='fixture', label='单元测试', difficulty=1, source=source)
            result = suite.prepare_case(case, shared)
            self.assertEqual(before, suite.source_hashes(source))
            self.assertEqual(result['reference_action_count'], 3)
            self.assertEqual(result['parameter_contract_version'], 'serial_proxy_exact_parameters_v1')
            shared_library = read_json(shared/'execution_skill_library.json')
            self.assertEqual([p['name'] for p in shared_library['skills'][0]['parameters']], ['target'])
            self.assertEqual(read_json(shared.parent/'codebook.json')['values'], [])
            self.assertEqual(shared_library['execution_contract'], read_json(shared/'execution_contract.json'))
            with self.assertRaisesRegex(ValueError, 'Real cosine preparation missing'):
                suite.finalize_retrieval(result, shared)
            fake_retrieval(shared)
            suite.finalize_retrieval(result, shared)
            self.assertEqual(result['retrieval']['ambiguous_subtasks'], 2)
            self.assertTrue(result['reference_in_initial_candidates'])

    def test_wrong_expected_visit_order_fails_before_requests(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)/'source'
            saved_fixture(source)
            d = read_json(source/'round_input.json')
            d['expected_navigation_targets'] = [62]
            write_json(source/'round_input.json', d)
            case = dict(case_id='bad', label='测试', difficulty=1, source=source)
            with self.assertRaisesRegex(ValueError, 'visit order'):
                suite.prepare_case(case, Path(tmp)/'case/shared_input')

    def test_aggregation_separates_timeout_and_validation(self):
        trials = [dict(case_id='a', arm='matrix', status='skill_stage_symbolic_passed',
            first_proposal_valid=True, seconds=200., audit={'http_attempt_count': 2,
            'transport_failure_count': 1}, token_usage={'total_tokens': 500,
            'transport_failures_unknown_usage': 1}),
            dict(case_id='a', arm='matrix', status='failed', first_proposal_valid=False,
                 seconds=20., audit={'program_rejection_count': 2}, token_usage={'total_tokens': 800})]
        result = suite.aggregate(trials, [dict(case_id='a', label='测试')])[0]
        self.assertEqual(result['final_passes'], 1)
        self.assertEqual(result['first_proposal_valid'], 1)
        self.assertEqual(result['clean_first_passes'], 0)
        self.assertEqual(result['unknown_usage_transport_failures'], 1)
        self.assertEqual(result['median_seconds_without_transport_failure'], 20.)
        self.assertEqual(result['known_total_tokens'], 1300)

    def test_suite_one_shared_retrieval_then_continue_after_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)/'source'
            saved_fixture(source)
            case = dict(case_id='fixture', label='测试', difficulty=1, source=source)
            calls, retrieval_hashes = [], []
            def execute(command, log, timeout):
                if '--worker-shared-input' in command:
                    fake_retrieval(Path(command[command.index('--worker-shared-input')+1]))
                    calls.append('retrieval')
                    return
                arm = command[command.index('--arm')+1]
                shared = Path(command[command.index('--worker-input')+1])
                retrieval_hashes.append(suite.input_hashes(shared)['skill_retrieval_semantic_1.json'])
                folder = Path(command[command.index('--output-dir')+1])
                calls.append(arm)
                write_json(folder/'trial_summary.json', dict(status='failed' if len(calls) == 2 else 'skill_stage_symbolic_passed',
                    first_proposal_valid=len(calls) > 2, attempts=[], audit={}, token_usage={'total_tokens': 10}))
                if len(calls) == 2:
                    raise ValueError('Synthetic rejected trial')
            args = SimpleNamespace(scene_dir=Path(tmp), output_dir=Path(tmp)/'batch',
                cases=None, repeats=2, model='offline', base_url='https://invalid.test', preflight_only=False,
                embedding_model_dir=Path(tmp)/'model', embedding_python=Path(sys.executable))
            with patch.object(suite, 'catalog', return_value=[case]), patch.object(suite, 'execute_worker', side_effect=execute), \
                 patch.object(suite, 'model_fingerprint', return_value='fixture_digest'):
                self.assertEqual(suite.run(args), 2)
            summary = read_json(args.output_dir/'batch_summary.json')
            self.assertEqual(calls, ['retrieval', 'nested_json', 'matrix', 'matrix', 'nested_json'])
            self.assertEqual(len(set(retrieval_hashes)), 1)
            self.assertEqual(summary['status'], 'completed')
            self.assertFalse(summary['all_skill_stage_checks_passed'])
            self.assertTrue(summary['source_unchanged'])
            self.assertTrue(summary['frozen_engine_unchanged'])
            self.assertTrue(summary['embedding_model_unchanged'])

    def test_failed_cosine_preparation_never_substitutes_full_inventory(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)/'source'
            saved_fixture(source)
            case = dict(case_id='fixture', label='测试', difficulty=1, source=source)
            args = SimpleNamespace(scene_dir=Path(tmp), output_dir=Path(tmp)/'batch',
                cases=None, repeats=1, model='offline', base_url='https://invalid.test', preflight_only=False,
                embedding_model_dir=Path(tmp)/'model', embedding_python=Path(sys.executable))
            with patch.object(suite, 'catalog', return_value=[case]), \
                 patch.object(suite, 'model_fingerprint', return_value='fixture_digest'), \
                 patch.object(suite, 'execute_worker', side_effect=ValueError('Synthetic model error')) as execute:
                self.assertEqual(suite.run(args), 2)
                execute.assert_called_once()
            summary = read_json(args.output_dir/'batch_summary.json')
            self.assertFalse(summary['embedding_prepared'])
            self.assertEqual(summary['trials'], [])

    def test_missing_input_prevents_any_requests(self):
        with tempfile.TemporaryDirectory() as tmp:
            case = dict(case_id='missing', label='测试', difficulty=1, source=Path(tmp)/'missing')
            args = SimpleNamespace(scene_dir=Path(tmp), output_dir=Path(tmp)/'batch',
                cases=None, repeats=1, model='offline', base_url='https://invalid.test', preflight_only=False)
            with patch.object(suite, 'catalog', return_value=[case]), patch.object(suite, 'execute_worker') as execute:
                self.assertEqual(suite.run(args), 2)
                execute.assert_not_called()


if __name__ == '__main__':
    unittest.main()
