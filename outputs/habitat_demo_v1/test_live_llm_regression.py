import copy
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_live_llm_regression as suite
from run_live_llm_sequence import validate_instructions
from run_navigation_demo import read_json, write_json
from proxy_placement import POLICY


def state(held, version, center, support):
    return {'held_object_id': held, 'lifted': held is not None, 'backend': {
        'robot_position': [0., .1, 0.], 'memory_version': version,
        'objects': {'67': {'center': list(center), 'support_id': support}}}}


def observed_report(case):
    # Synthetic diagnostic report: audit unit tests only; not simulated success.
    result = dict(status='passed', seed=case['seed'], simulation_executed=True, simulator_creation_count=1,
        initial_start_sampling_count=1, uninterrupted_simulator_execution=True, base_graph_unchanged=True,
        cross_round_held_object_continuations=case['held_continuations'], training_eligible=False, rounds=[])
    previous = state(None, 0, [1., 1., 1.], None)
    placement_rounds = case['placement_rounds']
    for descriptor in case['instructions']['rounds']:
        i = descriptor['round']
        moved = i > placement_rounds[0]
        targets = descriptor.get('expected_navigation_targets', [67 if moved else 13])
        actions = [dict(skill_id='navigate_to_object', status='arrived', target_id=t,
            target_center=[3., 1., 3.] if moved and t == 67 else [1., 1., 1.],
            target_source='current_memory_confirmed_by_live_instance_readback', memory_target_error_m=0.) for t in targets]
        if i in placement_rounds:
            actions.append(dict(skill_id='place_object_on_surface', status='proxy_instance_support_passed', validation={
                'passed': True, 'checks': {k: True for k in ('horizontal_error_m', 'com_height_error_m', 'tail_motion_m', 'direct_support_contact_frames')},
                'metrics': dict(horizontal_error_m=.05, com_height_error_m=.001, tail_motion_m=0., direct_support_contact_frames=60),
                'thresholds': dict(horizontal_error_m=.35, com_height_error_m=.2, tail_motion_m=.02, direct_support_contact_frames=20)},
                placement_preparation={'placement_policy': POLICY, 'release_rotation_wxyz': [1., 0., 0., 0.]},
                at_release={'rotation_wxyz': [1., 0., 0., 0.], 'linear_velocity': [0., 0., 0.], 'angular_velocity': [0., 0., 0.]},
                surface_candidates=[{'valid': True, 'volume_probes': [{'blocking_contacts': []}]}]))
        end = state(descriptor['expected_end']['held_object_id'], i,
                    [3., 1., 3.] if i >= placement_rounds[0] else [1., 1., 1.],
                    13 if i in placement_rounds else None)
        result['rounds'].append(dict(round=i, instruction=descriptor['instruction'], status='passed',
            training_eligible=False, begin_state=copy.deepcopy(previous), end_state=end,
            continuity_with_previous_round=True, expected_runtime_state_passed=True,
            planning_evidence={'fresh_llm': True, 'input_memory_version': previous['backend']['memory_version']},
            goal_checks=[{'required': True, 'status': 'observed_in_proxy'}], actions=actions))
        previous = end
    result['final_state'] = previous
    result['attempted_action_count'] = result['executed_action_count'] = sum(len(r['actions']) for r in result['rounds'])
    return result


class LiveRegressionTests(unittest.TestCase):
    def test_three_cases_use_instructions_only_and_no_hardcoded_action_count(self):
        for case in suite.cases():
            validate_instructions(case['instructions'])
            self.assertFalse(any('plan' in row for row in case['instructions']['rounds']))
            result = observed_report(case)
            checks = suite.evaluate_case(result, 0, case)
            self.assertTrue(all(checks.values()), [k for k, v in checks.items() if not v])
            self.assertNotEqual(result['executed_action_count'], 16)

    def test_passed_string_cannot_mask_bad_evidence(self):
        case = suite.cases()[0]
        for field, value, check in [('status', 'failed', 'reported_complete'),
            ('seed', 203, 'requested_seed_recorded'), ('simulation_executed', False, 'actually_simulated'),
            ('simulator_creation_count', 2, 'one_simulator'), ('initial_start_sampling_count', 2, 'one_start'),
            ('base_graph_unchanged', False, 'source_graph_unchanged'),
            ('training_eligible', True, 'scope_not_overclaimed'),
            ('executed_action_count', 999, 'all_actions_completed')]:
            result = observed_report(case)
            result[field] = value
            with self.subTest(field=field):
                self.assertFalse(suite.evaluate_case(result, 0, case)[check])
        self.assertFalse(suite.evaluate_case(observed_report(case), 2, case)['reported_complete'])

    def test_stale_memory_and_changed_cross_round_state_rejected(self):
        case = suite.cases()[0]
        result = observed_report(case)
        result['rounds'][2]['planning_evidence']['input_memory_version'] = 0
        self.assertFalse(suite.evaluate_case(result, 0, case)['current_memory_versions'])
        result['rounds'][2]['begin_state']['held_object_id'] = None
        self.assertFalse(suite.evaluate_case(result, 0, case)['cross_round_state_equal'])

    def test_route_reordering_and_old_target_position_rejected(self):
        case = suite.cases()[0]
        result = observed_report(case)
        result['rounds'][0]['actions'].reverse()
        # Palindromic default route: alter an ID instead of reversing it.
        result['rounds'][0]['actions'][0]['target_id'] = 13
        self.assertFalse(suite.evaluate_case(result, 0, case)['explicit_visit_order_preserved'])
        result = observed_report(case)
        result['rounds'][3]['actions'][0]['target_center'] = [1., 1., 1.]
        self.assertFalse(suite.evaluate_case(result, 0, case)['moved_kettle_navigation_uses_updated_position'])

    def test_placement_metrics_not_trusted_from_boolean_alone(self):
        case = suite.cases()[0]
        for field, value in [('tail_motion_m', .03), ('direct_support_contact_frames', 19),
                             ('com_height_error_m', float('nan')), ('horizontal_error_m', -.1)]:
            result = observed_report(case)
            result['rounds'][2]['actions'][-1]['validation']['metrics'][field] = value
            with self.subTest(field=field):
                self.assertFalse(suite.evaluate_case(result, 0, case)['placement_metrics_independently_pass'])

    def test_relaxed_placement_threshold_and_missing_probe_rejected(self):
        case = suite.cases()[0]
        result = observed_report(case)
        place = result['rounds'][2]['actions'][-1]
        place['validation']['thresholds']['horizontal_error_m'] = 1.
        self.assertFalse(suite.evaluate_case(result, 0, case)['placement_metrics_independently_pass'])
        place['surface_candidates'][0]['volume_probes'] = []
        self.assertFalse(suite.evaluate_case(result, 0, case)['placement_volume_and_upright_release_evidenced'])

    def test_log_statistics_count_one_log_and_keep_rejections(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            planning = root / 'round_1/planning'
            write_json(planning / 'planning_summary.json', {'semantic_redecomposition_count': 1})
            write_json(planning / 'skill_plan_final.json', {'subtask_plans': [{'steps': [{}, {}]}]})
            # No secrets or network; fixture text comes from mock subprocess output.
            log_text = 'LLM HTTP attempt=1/2\nRejected LLM decomposition attempt 1\nLLM HTTP request attempt 2/2\nTransient LLM failure\n'
            with patch.object(Path, 'read_text', return_value=log_text), patch.object(suite, 'read_json',
                side_effect=[{'status': 'planning_passed', 'semantic_redecomposition_count': 1,
                    'input_memory_version': 0}, {'subtask_plans': [{'steps': [{}, {}]}]}]), \
                patch.object(Path, 'is_file', return_value=True):
                stats = suite.planning_statistics(root, [{'round': 1, 'begin_state': {'backend': {'memory_version': 0}}}])
            self.assertEqual(stats['http_attempt_log_count'], 2)
            self.assertEqual(stats['decomposition_rejections'], 1)
            self.assertEqual(stats['transport_retry_warnings'], 1)
            self.assertEqual(stats['semantic_redecompositions'], 1)
            self.assertEqual(stats['rounds_with_multiple_actions_in_one_subtask'], [1])

    def test_selected_preflight_is_not_reported_as_simulation_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(case=['combined_seed_203'], output_dir=Path(directory) / 'run',
                preflight_only=True, scene_dir=Path(directory), planning_python='not-installed', model='test', base_url='https://invalid.test')
            def mock_child(command, log):
                self.assertIn('--preflight-only', command)
                output = Path(command[command.index('--output-dir') + 1])
                write_json(output / 'pipeline_summary.json', {'status': 'preflight_passed',
                    'simulation_executed': False, 'simulator_creation_count': 0})
                return 0
            with patch.object(suite, 'stream_case', side_effect=mock_child):
                self.assertEqual(suite.run_suite(args), 0)
            report = read_json(args.output_dir / 'regression_summary.json')
            self.assertEqual(report['status'], 'preflight_passed')
            self.assertFalse(report['simulation_executed'])
            self.assertEqual(report['selected_cases'], ['combined_seed_203'])


if __name__ == '__main__':
    unittest.main()
