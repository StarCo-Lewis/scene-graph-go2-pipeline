import unittest
from run_visual_stress_suite import case_metrics


class StressTests(unittest.TestCase):
    def test_stale_success_not_hidden(self):
        move = {'status': 'passed', 'translation_error_m': .01, 'stale_position_error_m': .5,
                'repaired_position_error_m': .01, 'source_memory_unchanged': True}
        nav = {'status': 'passed', 'repaired_navigation_success_count': 3,
               'stale_navigation_success_count': 3, 'source_memories_unchanged': True}
        self.assertFalse(case_metrics(move, nav)['navigation_success_gain_observed'])
        nav['stale_navigation_success_count'] = 2
        self.assertTrue(case_metrics(move, nav)['navigation_success_gain_observed'])

    def test_failed_stage_not_counted_as_success(self):
        with self.assertRaises(ValueError):
            case_metrics({'status': 'failed'}, {'status': 'passed'})
