"""Offline lifecycle checks: cancellation, ownership, and damaged audit files."""

import os
from pathlib import Path
import signal
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from embodied_pipeline._legacy import load
from embodied_pipeline.data_collection import batch
from embodied_pipeline.orchestration import task_collection_worker as worker


class CollectionLifecycleTests(unittest.TestCase):
    def test_exclusive_lock_never_automatically_removed(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "worker.lock"
            owner = batch.claim_worker_lock(path)
            with self.assertRaisesRegex(RuntimeError, "manual"):
                batch.claim_worker_lock(path)
            self.assertTrue(path.exists())
            batch.release_worker_lock(path, {"token": "not-owner"})
            self.assertTrue(path.exists())
            batch.release_worker_lock(path, owner)
            self.assertFalse(path.exists())

    def test_cross_host_lock_requires_manual_review(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "worker.lock"
            batch.write(path, {"pid": 123, "hostname": "some-other-host", "platform": "other"})
            self.assertEqual(batch.lock_owner_state(path), "other_host_or_platform_manual_review_required")
            self.assertTrue(path.exists())

    def test_stop_targets_owner_without_deleting_another_request(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            batch.request_stop(root, [("scene200", SimpleNamespace(pid=os.getpid()))])
            self.assertTrue(batch.stop_requested_for_current(root, "scene200"))
            batch.request_stop(root, [("scene200", SimpleNamespace(pid=os.getpid() + 1))])
            self.assertFalse(batch.stop_requested_for_current(root, "scene200"))
            self.assertTrue(batch.stop_path(root, "scene200").exists())

    def test_worker_finishes_current_task_and_skips_next_cleans_owned_lock(self):
        with tempfile.TemporaryDirectory() as temp:
            root, scene_id = Path(temp), "scene200"
            scene = root / "cases" / scene_id
            graph = scene / "graph"
            batch.write(graph / "scene_graph.json", [{"id": 1}])
            batch.write(graph / "scene_graph_edges.json", [])
            records = [{"record_id": f"scene200_{i:08d}", "scene_id": scene_id,
                        "instruction": "Offline fixture"} for i in (1, 2)]
            batch.jsonl(scene / "tasks.jsonl", records)
            batch.write(root / "collection_config.json", {"model": "fixture", "base_url": "https://invalid.example",
                "scenes": {scene_id: {"graph_sha256": batch.sha(graph / "scene_graph.json"),
                    "edges_sha256": batch.sha(graph / "scene_graph_edges.json"),
                    "tasks_sha256": batch.sha(scene / "tasks.jsonl")}}})
            batch.write(root / "code_manifest.json", {})
            args = SimpleNamespace(output=root, scene=scene_id, approve_new_api_calls=True,
                                   limit_per_scene=None, max_api_calls=6,
                                   embedding_python=Path("fixture"), embedding_model=Path("fixture"))
            encoder = SimpleNamespace(environment={}, close=lambda: None)
            def finish_then_stop(*args):
                batch.request_stop(root, [(scene_id, SimpleNamespace(pid=os.getpid()))])
                return {"status": "fixture_completed"}
            previous = signal.getsignal(signal.SIGINT)
            retrieval = load("skill_retrieval")
            with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "offline-fixture"}), \
                    patch.object(retrieval, "LocalEncoder", return_value=encoder), \
                    patch.object(retrieval, "model_fingerprint", return_value="fixture"), \
                    patch.object(worker, "run_task", side_effect=finish_then_stop) as run:
                self.assertEqual(batch.run_scene(args), 0)
            self.assertEqual(run.call_count, 1)
            self.assertFalse((scene / "worker.lock").exists())
            self.assertIs(signal.getsignal(signal.SIGINT), previous)

    def test_audit_corruption_is_evidence_not_finalizer_exception(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            directory = root / "llm_response_audit"
            directory.mkdir()
            (directory / "stage_a_request.json").write_text("{broken", encoding="utf-8")
            (directory / "stage_a_http_1.json").write_text("{broken", encoding="utf-8")
            evidence = worker.response_evidence(root)
            self.assertFalse(evidence["usage_complete"])
            self.assertEqual(len(evidence["malformed_audit_files"]), 2)

    def test_request_response_timestamps_explicit_local_timezone(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            batch.write(root / "llm_response_audit/stage_a_request.json", {
                "stage": "decomposition", "recorded_at": "2026-10-23T12:00:00+00:00"})
            entry = worker.response_evidence(root)["request_response_timestamps"][0]
            self.assertEqual(entry["recorded_at_local"], "2026-10-23T20:00:00+08:00")


if __name__ == "__main__":
    unittest.main()
