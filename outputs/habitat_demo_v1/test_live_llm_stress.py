import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_live_llm_stress as stress
import run_live_llm_regression as suite
from run_navigation_demo import write_json, read_json
from run_live_llm_sequence import validate_instructions
from test_live_llm_regression import observed_report


class StressTests(unittest.TestCase):
    def test_cases_have_outcome_assertions_not_skill_plans(self):
        catalog = stress.stress_cases()
        self.assertEqual([len(c['instructions']['rounds']) for c in catalog], [12, 18])
        for case in catalog:
            validate_instructions(case['instructions'])
            previous = None
            continuations = []
            for row in case['instructions']['rounds']:
                self.assertNotIn('plan', row)
                self.assertNotIn('alignment', row)
                self.assertNotIn('steps', row)
                if previous == 67:
                    continuations.append(row['round'])
                previous = row['expected_end']['held_object_id']
            self.assertEqual(continuations, case['held_continuations'])
            self.assertIsNone(previous)
            self.assertEqual(len(case['placement_rounds']), case['expected_placements'])
            self.assertTrue(all(case['instructions']['rounds'][i-1]['expected_end']['held_object_id'] is None
                                for i in case['placement_rounds']))

    def test_profiles_are_independent(self):
        catalog = stress.stress_cases()
        catalog[0]['instructions']['rounds'][0]['instruction'] = 'Changed test input.'
        self.assertNotEqual(catalog[0]['instructions']['rounds'][0]['instruction'],
                            catalog[1]['instructions']['rounds'][0]['instruction'])

    def test_inherited_result_checks_support_both_long_profiles(self):
        for case in stress.stress_cases():
            checks = suite.evaluate_case(observed_report(case), 0, case)
            self.assertTrue(all(checks.values()), [k for k, v in checks.items() if not v])

    def test_custom_catalog_preflight_never_claims_actual_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(case=['mixed_12_seed_202'], output_dir=Path(directory) / 'run',
                preflight_only=True, scene_dir=Path(directory), planning_python='not-installed',
                model='test', base_url='https://invalid.test')
            def child(command, log):
                instructions = read_json(Path(command[command.index('--instructions-json')+1]))
                self.assertEqual(len(instructions['rounds']), 12)
                output = Path(command[command.index('--output-dir')+1])
                write_json(output / 'pipeline_summary.json', {'status': 'preflight_passed',
                    'simulation_executed': False, 'simulator_creation_count': 0})
                return 0
            with patch.object(suite, 'stream_case', side_effect=child), patch.object(stress, 'stress_checks') as check:
                self.assertEqual(suite.run_suite(args, stress.stress_cases(), check), 0)
                check.assert_not_called()
            report = read_json(args.output_dir / 'regression_summary.json')
            self.assertFalse(report['simulation_executed'])
            self.assertEqual(report['selected_cases'], ['mixed_12_seed_202'])

    def fixture(self, root):
        case = {'instructions': {'rounds': [{'round': 1}]}}
        result = {'fresh_llm_calls': True, 'planning_origin': 'fresh_llm_each_round_live_memory',
            'stage_collision_configuration': {'status': 'loaded_parameters_verified',
                'loaded_stage_collidable': True, 'loaded_stage_margin_m': .001},
            'rounds': [{'actions': [
                {'skill_id': 'navigate_to_object', 'motion_safety_checks': []},
                {'skill_id': 'grasp_object', 'motion_safety_checks': [{'status': 'sampled_clear'}]}]}],
            'visual_audit': {'status': 'saved', 'frame_count': 20, 'render_error_count': 0,
                             'frame_limit_reached': False}}
        directory = root / 'round_1/planning'
        report = {k: 0 for k in ('logical_request_count', 'http_attempt_count', 'http_response_count',
            'transport_failure_count', 'http_error_count', 'program_rejection_count',
            'content_parse_rejection_count', 'requests_without_transport_result_count')}
        report.update(logical_request_count=4, http_attempt_count=4, http_response_count=4)
        write_json(directory / 'llm_audit_report/llm_error_summary.json', report)
        for i in range(4):
            raw = directory / ('llm_response_audit' if i < 3 else 'semantic_replan_2/llm_response_audit')
            write_json(raw / f'test_{i}_request.json', {'synthetic_test': True})
            write_json(raw / f'test_{i}_http_1.json', {'synthetic_test': True})
        return result, case

    def test_stationary_navigation_does_not_require_motion_but_raw_audit_does(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result, case = self.fixture(root)
            extra = stress.stress_checks(result, root, case)
            self.assertTrue(all(extra['checks'].values()))
            self.assertEqual(extra['statistics']['logical_request_count'], 4)
            self.assertEqual(extra['statistics']['sampled_motion_check_count'], 1)

    def test_missing_audit_and_incomplete_visual_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result, case = self.fixture(root)
            # Include another requested round without files: partial success cannot pass.
            case['instructions']['rounds'].append({'round': 2})
            result['visual_audit']['frame_limit_reached'] = True
            checks = stress.stress_checks(result, root, case)['checks']
            self.assertFalse(checks['every_round_llm_raw_audit_present'])
            self.assertFalse(checks['visual_audit_saved_without_render_errors'])

    def test_requested_visual_audit_cannot_be_absent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result, case = self.fixture(root)
            case['visual_audit_required'] = True
            result.pop('visual_audit')
            checks = stress.stress_checks(result, root, case)['checks']
            self.assertFalse(checks['visual_audit_saved_without_render_errors'])

    def test_collider_disabled_blocked_motion_and_missing_transfer_evidence_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result, case = self.fixture(root)
            result['stage_collision_configuration']['loaded_stage_collidable'] = False
            result['rounds'][0]['actions'][1]['motion_safety_checks'][0]['status'] = 'blocked'
            result['rounds'][0]['actions'].append({'skill_id': 'place_object_on_surface'})
            checks = stress.stress_checks(result, root, case)['checks']
            self.assertFalse(checks['static_collision_enabled_and_margin_verified'])
            self.assertFalse(checks['motion_evidence_all_sampled_clear'])
            self.assertFalse(checks['motion_evidence_for_each_moving_action'])

    def test_missing_raw_response_cannot_be_hidden_by_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result, case = self.fixture(root)
            write_json(root / 'round_1/planning/llm_audit_report/llm_error_summary.json',
                {'logical_request_count': 4, 'http_attempt_count': 4, 'http_response_count': 3,
                 'transport_failure_count': 0, 'http_error_count': 0, 'program_rejection_count': 0,
                 'content_parse_rejection_count': 0, 'requests_without_transport_result_count': 0})
            self.assertFalse(stress.stress_checks(result, root, case)['checks']['every_round_llm_raw_audit_present'])


if __name__ == '__main__':
    unittest.main()
