"""Offline synthetic responses only; these tests are not measured LLM accuracy."""
import contextlib
import copy
import io
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bounded_skill_planner as bounded
import plan_live_round as live
import run_combined_matrix_long as combined
import run_skill_cosine_ab as experiment
from run_navigation_demo import read_json, write_json
from run_json_long_horizon import initial_state, prepare_round
from test_json_long_horizon import nodes, row
from verify_plan import flatten_steps


class Encoder:
    def encode(self, texts): return [[1., 0.] for _ in texts]
    def stats(self): return {'seconds_including_startup': 0., 'synthetic_test': True}
    def close(self): pass


class CombinedTests(unittest.TestCase):
    def test_each_recursive_leaf_uses_matrix_and_inherits_validated_state(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)/'round_1/planning'
            descriptor = {'round': 1, 'instruction': 'Visit ID 13, then visit ID 62.',
                          'expected_end': {'held_object_id': None, 'lifted': False},
                          'expected_navigation_targets': [13, 62]}
            prepare_round(root, descriptor, initial_state(nodes()), nodes(), 'offline', 'https://invalid.test')
            args = SimpleNamespace(model='offline', base_url='https://invalid.test', skill_output_format='matrix',
                                   compile_navigation=False, top_k=2, cosine_threshold=.35, cosine_margin=.05)
            starts, stages, requests = [], [], []
            encoder = Encoder()
            def decision(a, folder, stage, system, payload, check):
                stages.append(stage)
                value = ({'decision': 'split', 'reason': 'synthetic fixture',
                          'parts': ['Visit ID 13,', ' then visit ID 62.']} if stage == 'hierarchy_split' else
                         dict(complete=True, order_preserved=True, constraints_preserved=True,
                              no_unrequested_actions=True, reason='synthetic fixture'))
                check(value)
                return value
            def decompose(a, child, summary, *unused):
                self.assertEqual(a.skill_output_format, 'matrix')
                self.assertIs(a.skill_retriever, encoder)
                self.assertEqual(a.skill_proposal_limit, 2)
                self.assertFalse(a.compile_navigation)
                starts.append(read_json(a.output_dir/'runtime_state.json'))
                fixture, _ = row([('navigate_to_object', 13 if '13' in child['instruction'] else 62)])
                fixture['alignment']['task_decomposition']['instruction'] = child['instruction']
                write_json(a.output_dir/'object_alignment.json', fixture['alignment'])
                return fixture['alignment']
            def matrix(alignment, library, book, *unused, **options):
                requests.append(options['audit_context']['skill_output_format'])
                goal = alignment['task_decomposition']['goal_state'][0]['object_role']
                obj = next(o['code'] for o in book['objects'] if o['role_id'] == goal)
                skill = next(s['code'] for s in book['skills'] if s['skill_id'] == 'navigate_to_object')
                return {'matrix': [[1, skill, obj, 0]]}
            with patch.object(bounded, 'request_decision', side_effect=decision), \
                 patch.object(live, 'decompose_and_bind', side_effect=decompose), \
                 patch.object(live, 'request_matrix', side_effect=matrix), \
                 patch.object(live, 'request_plan', side_effect=AssertionError('nested JSON must not run')), \
                 patch.object(live, 'request_order', side_effect=lambda plan, *a, **k:
                              {'ordered_step_ids': [s['step_id'] for s in flatten_steps(plan)], 'reason': 'retain'}):
                self.assertEqual(bounded.plan_bounded(args, root, encoder), 0)
            self.assertEqual(requests, ['matrix', 'matrix'])
            self.assertEqual(stages, ['hierarchy_split', 'hierarchy_coverage'])
            leaf = read_json(root/'hierarchy/node_002/validated_leaf.json')
            self.assertEqual(starts[1], leaf['end_state'])
            self.assertEqual(starts[1]['backend']['robot_position'], [0., .1, 0.])
            metrics = combined.hierarchy_metrics(Path(d))
            self.assertTrue(metrics['all_planned_leaves_matrix'])
            self.assertTrue(metrics['round_inputs_unchanged'])
            self.assertEqual(metrics['validated_leaves_including_failed_round_prefixes'], 2)
            self.assertFalse(read_json(root/'planning_summary.json')['simulation_executed'])

    def test_matrix_missing_encoder_fails_before_any_llm(self):
        with patch.object(bounded, 'plan_hierarchical') as planner:
            with self.assertRaisesRegex(ValueError, 'requires cosine encoder'):
                bounded.plan_bounded(SimpleNamespace(skill_output_format='matrix'), Path('unused'))
            planner.assert_not_called()

    def test_worker_routes_single_arm_and_restores_callback(self):
        from json_stability_cases import catalog
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            case = next(c for c in catalog(nodes()) if c['name'] == 'baseline_12')
            write_json(root/'case.json', case)
            args = SimpleNamespace(worker_case=root/'case.json', output_dir=root, arm=combined.ARM,
                                   embedding_model_dir=root/'model', embedding_model_sha='fixture',
                                   embedding_python=sys.executable, max_parts=4)
            original = experiment.baseline.plan_one_round
            def worker(a, c):
                return experiment.baseline.plan_one_round(a, root/'round_1/planning')
            with patch.object(experiment, 'LocalEncoder', return_value=Encoder()), \
                 patch.object(experiment, 'model_fingerprint', return_value='fixture'), \
                 patch.object(experiment.baseline, 'positive_worker', side_effect=worker), \
                 patch.object(experiment, 'plan_bounded', return_value=0) as planner:
                self.assertEqual(experiment.worker(args), 0)
                self.assertEqual(planner.call_args.args[0].skill_output_format, 'matrix')
            self.assertIs(experiment.baseline.plan_one_round, original)

    def test_descriptive_history_refuses_changed_case_and_counts_failures_cost(self):
        trial = {'case': 'fixture', 'arm': 'matrix', 'evaluation_outcome': 'final_failure',
                 'completed_round_count': 1, 'expected_round_count': 18,
                 'token_usage': {'prompt_tokens': 100, 'completion_tokens': 20, 'total_tokens': 120},
                 'duration_seconds': 5}
        history = {'status': 'completed', 'case_input_hashes': {'fixture': 'same'}, 'trials': [trial]}
        current = {'case_input_hashes': {'fixture': 'same'}, 'trials': [copy.deepcopy(trial)]}
        data = combined.results(current, history)
        a = data['methods']['new_combined']
        self.assertEqual(a['passed_trials'], 0)
        self.assertEqual(a['known_total_tokens'], 120)
        self.assertIsNone(a['tokens_per_success_including_failures'])
        self.assertAlmostEqual(a['known_fee_cny_no_discount'], .000134)
        self.assertFalse(data['simulation_executed'])
        current['case_input_hashes']['fixture'] = 'changed'
        with self.assertRaisesRegex(ValueError, 'do not match'):
            combined.results(current, history)

    def test_single_arm_preflight_has_no_api_or_embedding_calls(self):
        from json_stability_cases import catalog
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            case = next(c for c in catalog(nodes()) if c['name'] == 'baseline_12')
            source = root/'source'
            write_json(source/'cases/baseline_12.json', case)
            history = {'status': 'completed', 'case_input_hashes': {
                'baseline_12': experiment.baseline.sha(source/'cases/baseline_12.json')}, 'trials': []}
            write_json(source/'batch_summary.json', history)
            args = SimpleNamespace(source_batch=source, output_dir=root/'result', repeats=1,
                                   case=['baseline_12'], arms=[combined.ARM], max_parts=4,
                                   model='offline', base_url='https://invalid.test', preflight_only=True,
                                   top_k=2, cosine_threshold=.35, cosine_margin=.05,
                                   embedding_model_dir=None, embedding_python=Path(sys.executable))
            reporter = combined.make_report(history, experiment.baseline.sha(source/'batch_summary.json'))
            with patch.object(experiment, 'report', reporter), \
                 patch.object(experiment, 'LocalEncoder', side_effect=AssertionError('no model')), \
                 patch.object(experiment.baseline, 'execute_worker', side_effect=AssertionError('no API')), \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(experiment.run(args), 0)
            summary = read_json(root/'result/batch_summary.json')
            self.assertEqual(summary['arms'], [combined.ARM])
            self.assertEqual(summary['new_nested_json_trials'], 0)
            self.assertEqual(summary['recursive_split_budget'], 1)
            self.assertEqual(summary['trials'], [])
            self.assertTrue(read_json(root/'result/comparison_summary.json')['evaluation_valid'])


if __name__ == '__main__':
    unittest.main()
