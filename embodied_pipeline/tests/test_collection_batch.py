import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from embodied_pipeline.data_collection import batch


class CollectionBatchTests(unittest.TestCase):
    def test_pending_never_replays_partial_or_failed(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            scene = root / 'cases/scene200'
            scene.mkdir(parents=True)
            records = [{'record_id': f'scene200_{i:08d}', 'scene_id': 'scene200', 'instruction': f'task{i}'}
                       for i in range(1, 4)]
            batch.jsonl(scene / 'tasks.jsonl', records)
            batch.write(root / 'collection_config.json', {'scenes': {'scene200': {
                'tasks_sha256': batch.sha(scene / 'tasks.jsonl')}}})
            (scene / 'tasks/scene200_00000001/model_run').mkdir(parents=True)
            (scene / 'tasks/scene200_00000002/model_run').mkdir(parents=True)
            batch.write(scene / 'tasks/scene200_00000002/model_run/planning_summary.json', {'status': 'failed'})
            self.assertEqual([r['record_id'] for r in batch.pending_tasks(root, 'scene200', None)], ['scene200_00000003'])
            dataset = batch.summarize(root)
            self.assertIsNone(dataset[0]['valid_candidate'])
            report = batch.read(root / 'quality_report.json')
            self.assertEqual(report['started_records'], 2)
            self.assertEqual(report['confirmed_valid_candidate_records'], 0)
            self.assertEqual(report['records_with_http_response'], 0)

    def test_catalog_changes_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            p = root / 'cases/scene200/tasks.jsonl'
            batch.jsonl(p, [])
            batch.write(root / 'collection_config.json', {'scenes': {'scene200': {'tasks_sha256': 'wrong'}}})
            with self.assertRaisesRegex(ValueError, 'changed'):
                batch.tasks_for(root, 'scene200')

    def test_launch_requires_explicit_budget_without_calls(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            batch.write(root / 'collection_config.json', {'scenes': {}})
            args = SimpleNamespace(output=root, approve_new_api_calls=False)
            with patch.object(batch.subprocess, 'Popen') as popen:
                with self.assertRaisesRegex(ValueError, 'Explicit'):
                    batch.launch(args)
                popen.assert_not_called()

    def test_code_version_frozen(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            batch.write(root / 'code_manifest.json', {'embodied_pipeline/data_collection/batch.py': 'wrong'})
            with self.assertRaisesRegex(ValueError, 'changed'):
                batch.require_code_unchanged(root)

    def test_pilot_covers_horizons_without_new_records(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            path = root / 'cases/scene200/tasks.jsonl'
            records = [{'record_id': 'scene200_%08d' % i, 'horizon': 'long' if i == 3 else 'short',
                        'task_family': 'ordered_visit' if i == 3 else 'navigate'} for i in range(1, 4)]
            batch.jsonl(path, records)
            batch.write(root / 'collection_config.json', {'scenes': {'scene200': {'tasks_sha256': batch.sha(path)}}})
            self.assertEqual([r['record_id'] for r in batch.pending_tasks(root, 'scene200', 2)],
                             ['scene200_00000001', 'scene200_00000003'])

    def test_runtime_upgrade_rejects_active_or_incomplete_attempt(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            lock = root / 'cases/scene200/worker.lock'
            batch.write(lock, {})
            with self.assertRaisesRegex(ValueError, 'running'):
                batch.register_runtime_revision(root)
            lock.unlink()
            (root / 'cases/scene200/tasks/scene200_00000001/model_run').mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, 'terminal'):
                batch.register_runtime_revision(root)

    def test_runtime_upgrade_preserves_original_manifest_and_attempt(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            original = root / 'code_manifest.json'
            batch.write(original, {'old_fixture': 'original'})
            batch.write(root / 'collection_config.json', {'scenes': {}})
            summary = root / 'cases/scene200/tasks/scene200_00000001/model_run/planning_summary.json'
            batch.write(summary, {'status': 'failed', 'collection_finished_at_utc': '2026-10-23T00:00:00Z'})
            before = original.read_bytes(), summary.read_bytes()
            def snapshot(destination):
                batch.write(destination / 'code_manifest.json', {})
            with patch.object(batch, 'code_snapshot', side_effect=snapshot):
                batch.register_runtime_revision(root)
            self.assertEqual(before, (original.read_bytes(), summary.read_bytes()))
            self.assertTrue((root / 'runtime_revision.json').is_file())
            batch.require_code_unchanged(root)


if __name__ == '__main__':
    unittest.main()
