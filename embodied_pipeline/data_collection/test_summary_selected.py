"""Offline tests: selected scope, timing, source preservation, failure evidence."""
import tempfile
import unittest
from pathlib import Path

from .batch import write, jsonl, sha
from .summarize_selected import summarize_selected


class SelectedReportTest(unittest.TestCase):
    def test_scope_timing_and_immutable_sources(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            write(root / 'collection_config.json', {'scenes': {'scene200': {'count': 2}, 'scene201': {'count': 999}}})
            base = root / 'cases/scene200'
            tasks = []
            for number in (1, 2):
                rid = f'scene200_{number:08d}'
                task = dict(record_id=rid, scene_id='scene200', instruction='Go to object',
                            initial_state={}, constraints={}, goal={}, grounded_object_ids=[1])
                tasks.append(task)
                run = base / 'tasks' / rid / 'model_run'
                write(run.parent / 'reference.json', {'steps': [], 'dependencies': []})
                write(run / 'planning_summary.json', {
                    'status': 'engineering_json_generated_ungraded' if number == 1 else 'collection_stopped_at_stage_ungraded',
                    'collection_started_at_utc': f'2026-10-23T13:0{number}:00+00:00',
                    'collection_finished_at_utc': f'2026-10-23T13:0{number}:30+00:00', 'duration_seconds': 30})
                if number == 1:
                    write(run / 'engineering_plan.json', {})
            jsonl(base / 'tasks.jsonl', tasks)
            write(base / 'graph/scene_graph.json', [{'id': 1}])
            before = {str(p): sha(p) for p in root.rglob('*') if p.is_file()}
            report = summarize_selected(root, ['scene200'], ['scene200_00000001'])
            self.assertEqual(report['excluded_scenes'], ['scene201'])
            self.assertEqual(report['finished_records'], 2)
            self.assertEqual(report['engineering_json_count_ungraded'], 1)
            self.assertIsNone(report['valid_candidate_records'])
            self.assertFalse(report['snapshot_not_final'])
            self.assertEqual(report['timing']['all_attempts_including_gaps']['elapsed_seconds'], 90)
            self.assertEqual(report['timing']['continuation_excluding_explicit_pilots']['elapsed_seconds'], 30)
            self.assertEqual(before, {p: sha(p) for p in before})
            with self.assertRaises(ValueError):
                summarize_selected(root, ['../scene200'])
            with self.assertRaises(ValueError):
                summarize_selected(root, ['scene200', 'scene200'])
            write(base / 'tasks/scene200_00000002/model_run/planning_summary.json', [])
            report = summarize_selected(root, ['scene200'])
            self.assertTrue(report['snapshot_not_final'])
            self.assertEqual(len(report['read_issues']), 1)
            self.assertEqual(report['finished_records'], 1)


if __name__ == '__main__':
    unittest.main()
