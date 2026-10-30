import argparse
import copy
import tempfile
import unittest
from pathlib import Path
from capture_visual_memory import same_physical_checkpoint
from run_chain_suite import rejection_checks, latest_robustness, validate_multiround_checkpoint
from update_memory import initialize, write


class ChainSuiteTests(unittest.TestCase):
    def setUp(self):
        self.state = initialize([{'id': 67, 'bbox_center': [1, 2, 3]}], [], 'fixture')

    def test_all_synthetic_guard_cases_reject(self):
        rows = [{'translation_cg': [.25, 0, 0], 'symmetric_trimmed_rms_m': .01},
                {'translation_cg': [.251, 0, 0], 'symmetric_trimmed_rms_m': .01}]
        reports = rejection_checks({'views': rows}, self.state)
        self.assertEqual(len(reports), 5)
        self.assertTrue(all(row['passed'] for row in reports))
        self.assertEqual(self.state['version'], 0)

    def test_visual_overlay_not_a_physical_change(self):
        revised = copy.deepcopy(self.state)
        revised['visual_revision'] = 3
        revised['entities']['67']['node']['visual_memory'] = {'voxel_count': 100}
        self.assertTrue(same_physical_checkpoint(self.state, revised))
        revised['entities']['67']['node']['bbox_center'][0] += 1
        self.assertFalse(same_physical_checkpoint(self.state, revised))

    def test_long_requires_passed_versioned_report(self):
        with tempfile.TemporaryDirectory() as folder:
            report = Path(folder) / 'summary.json'
            args = argparse.Namespace(robustness_summary=report)
            write(report, {'status': 'needs_review', 'suite_version': '1.0'})
            with self.assertRaises(ValueError):
                latest_robustness(args)
            write(report, {'status': 'passed', 'suite_version': '1.0'})
            self.assertEqual(latest_robustness(args), report)

    def test_missing_reference_views_not_a_success(self):
        with self.assertRaises(ValueError):
            rejection_checks({'views': []}, self.state)

    def test_resume_requires_matching_successful_memory(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            memory = root / 'memory'
            memory.mkdir()
            write(memory / 'memory_state.json', self.state)
            execution = root / 'round_3/execution/followup_execution.json'
            execution.parent.mkdir(parents=True)
            write(execution, {'status': 'followup_proxy_passed', 'memory_source': str(memory / 'memory_state.json')})
            instructions = ['Move the electric kettle to the sideboard.', 'Go to the electric kettle.', 'Go to the sideboard.']
            summary = {'status': 'passed', 'base_graph_unchanged': True,
                'rounds': [{'status': 'passed', 'instruction': text, 'memory': str(memory), 'memory_version': 0} for text in instructions]}
            write(root / 'pipeline_summary.json', summary)
            self.assertEqual(validate_multiround_checkpoint(root), summary)
            summary['rounds'][-1]['memory_version'] = 1
            write(root / 'pipeline_summary.json', summary)
            with self.assertRaisesRegex(ValueError, 'version mismatch'):
                validate_multiround_checkpoint(root)
            summary['status'] = 'failed'
            write(root / 'pipeline_summary.json', summary)
            with self.assertRaisesRegex(ValueError, 'passed'):
                validate_multiround_checkpoint(root)
