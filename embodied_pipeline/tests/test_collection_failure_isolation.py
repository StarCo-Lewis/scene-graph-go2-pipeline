"""Offline failures stay scoped to one task/scene without repeat API attempts."""

from contextlib import ExitStack
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from embodied_pipeline._legacy import load
from embodied_pipeline.data_collection import batch
from embodied_pipeline.orchestration import task_collection_worker as worker


class CollectionFailureIsolationTests(unittest.TestCase):
    def fixture(self, root, scene_ids=("scene200",), task_count=2):
        scenes = {}
        for scene_id in scene_ids:
            scene = root / "cases" / scene_id
            graph = scene / "graph"
            batch.write(graph / "scene_graph.json", [{"id": 1}])
            batch.write(graph / "scene_graph_edges.json", [])
            records = [{"record_id": f"{scene_id}_{i:08d}", "scene_id": scene_id,
                        "instruction": "Offline fixture; never send."}
                       for i in range(1, task_count + 1)]
            batch.jsonl(scene / "tasks.jsonl", records)
            scenes[scene_id] = {
                "graph_sha256": batch.sha(graph / "scene_graph.json"),
                "edges_sha256": batch.sha(graph / "scene_graph_edges.json"),
                "tasks_sha256": batch.sha(scene / "tasks.jsonl"),
            }
        batch.write(root / "collection_config.json", {
            "model": "offline-fixture", "base_url": "https://invalid.example", "scenes": scenes})
        batch.write(root / "code_manifest.json", {})
        python = root / "embedding_python.fixture"
        batch.write(python, {})
        model = root / "embedding_model"
        model.mkdir()
        return SimpleNamespace(output=root, scene=scene_ids[0], approve_new_api_calls=True,
                               limit_per_scene=None, max_api_calls=3 * task_count * len(scene_ids),
                               embedding_python=python, embedding_model=model, request_timeout=17)

    def worker_patches(self, stack, run_task):
        encoder = SimpleNamespace(environment={}, close=Mock())
        retrieval = load("skill_retrieval")
        stack.enter_context(patch.dict(os.environ, {"DASHSCOPE_API_KEY": "offline-fixture"}))
        stack.enter_context(patch.object(retrieval, "LocalEncoder", return_value=encoder))
        stack.enter_context(patch.object(retrieval, "model_fingerprint", return_value="fixture"))
        run = stack.enter_context(patch.object(worker, "run_task", side_effect=run_task))
        return run, encoder

    def launch_patches(self, stack, popen):
        stack.enter_context(patch.dict(os.environ, {"DASHSCOPE_API_KEY": "offline-fixture"}))
        stack.enter_context(patch.object(batch, "safe_summarize", return_value=True))
        spawned = stack.enter_context(patch.object(batch.subprocess, "Popen", side_effect=popen))
        stop = stack.enter_context(patch.object(batch, "request_stop"))
        return spawned, stop

    def errors(self, root, scene=None):
        folder = root / "cases" / scene if scene else root
        return [batch.read(path) for path in (folder / "errors").glob("*.json")]

    def process(self):
        return SimpleNamespace(pid=43210, returncode=0, poll=lambda: 0)

    def test_unhandled_task_runtime_error_preserves_evidence_and_runs_next(self):
        with tempfile.TemporaryDirectory() as temp, ExitStack() as stack:
            root = Path(temp)
            args = self.fixture(root)
            calls = []
            def task_run(options, task, scene, destination, encoder):
                calls.append(task["record_id"])
                if len(calls) == 1:
                    raise RuntimeError("Offline failure before normal finalization")
                return {"status": "fixture_completed"}
            run, encoder = self.worker_patches(stack, task_run)
            self.assertEqual(batch.run_scene(args), 0)
            self.assertEqual(run.call_count, 2)
            self.assertEqual(calls, ["scene200_00000001", "scene200_00000002"])
            error = batch.read(root / "cases/scene200/tasks/scene200_00000001/model_run/harness_error.json")
            self.assertEqual(error["status"], "task_harness_error_ungraded")
            self.assertEqual(error["error_type"], "RuntimeError")
            self.assertFalse(error["automatic_retry"])
            self.assertTrue(error["collection_finished_at_utc"].endswith("+00:00"))
            self.assertTrue(error["collection_finished_at_local"].endswith("+08:00"))
            self.assertEqual(len(self.errors(root, "scene200")), 1)
            encoder.close.assert_not_called()
            self.assertFalse((root / "cases/scene200/worker.lock").exists())

    def test_claim_collision_does_not_overwrite_owned_output_or_retry(self):
        with tempfile.TemporaryDirectory() as temp, ExitStack() as stack:
            root = Path(temp)
            args = self.fixture(root)
            existing = root / "cases/scene200/tasks/scene200_00000001/model_run"
            def task_run(options, task, scene, destination, encoder):
                if task["record_id"].endswith("00000001"):
                    batch.write(destination / "planning_summary.json", {"status": "prior_owner_summary"})
                    batch.write(destination / "raw_response.json", {"owned": "prior attempt"})
                    raise FileExistsError("Claim was obtained concurrently")
                return {"status": "fixture_completed"}
            run, _ = self.worker_patches(stack, task_run)
            self.assertEqual(batch.run_scene(args), 0)
            self.assertEqual(run.call_count, 2)
            self.assertEqual(batch.read(existing / "planning_summary.json"), {"status": "prior_owner_summary"})
            self.assertEqual(batch.read(existing / "raw_response.json"), {"owned": "prior attempt"})
            self.assertFalse((existing / "harness_error.json").exists())
            event, = self.errors(root, "scene200")
            self.assertEqual(event["stage"], "already_claimed_skipped")
            self.assertEqual([row["record_id"] for row in batch.pending_tasks(root, "scene200", None)],
                             ["scene200_00000002"])

    def test_catalog_preflight_failure_does_not_block_other_scene(self):
        with tempfile.TemporaryDirectory() as temp, ExitStack() as stack:
            root = Path(temp)
            args = self.fixture(root, ("scene200", "scene201"), task_count=1)
            original = batch.pending_tasks
            def pending(output, scene, limit):
                if scene == "scene200":
                    raise ValueError("Task catalogue changed after preparation")
                return original(output, scene, limit)
            stack.enter_context(patch.object(batch, "pending_tasks", side_effect=pending))
            spawned, stop = self.launch_patches(stack, [self.process()])
            self.assertEqual(batch.launch(args), 2)
            spawned.assert_called_once()
            self.assertIn("scene201", spawned.call_args.args[0])
            self.assertEqual(self.errors(root, "scene200")[0]["stage"], "scene_preflight")
            stop.assert_not_called()

    def test_graph_preflight_failure_does_not_block_other_scene(self):
        with tempfile.TemporaryDirectory() as temp, ExitStack() as stack:
            root = Path(temp)
            args = self.fixture(root, ("scene200", "scene201"), task_count=1)
            batch.write(root / "cases/scene200/graph/scene_graph.json", [{"id": 2}])
            spawned, stop = self.launch_patches(stack, [self.process()])
            self.assertEqual(batch.launch(args), 2)
            spawned.assert_called_once()
            self.assertIn("scene201", spawned.call_args.args[0])
            self.assertIn("Frozen graph", self.errors(root, "scene200")[0]["error"])
            stop.assert_not_called()

    def test_first_spawn_failure_does_not_block_other_scene(self):
        with tempfile.TemporaryDirectory() as temp, ExitStack() as stack:
            root = Path(temp)
            args = self.fixture(root, ("scene200", "scene201"), task_count=1)
            spawned, stop = self.launch_patches(stack, [OSError("Offline spawn failure"), self.process()])
            self.assertEqual(batch.launch(args), 2)
            self.assertEqual(spawned.call_count, 2)
            self.assertIn("scene200", spawned.call_args_list[0].args[0])
            self.assertIn("scene201", spawned.call_args_list[1].args[0])
            self.assertEqual(self.errors(root, "scene200")[0]["stage"], "scene_launch")
            self.assertTrue(all(call.kwargs["stdout"].closed for call in spawned.call_args_list))
            stop.assert_not_called()

    def test_summary_error_returns_false_and_never_stops_workers(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with patch.object(batch, "summarize", side_effect=RuntimeError("Offline corrupt summary")), \
                    patch.object(batch, "request_stop") as stop:
                self.assertIs(batch.safe_summarize(root), False)
            stop.assert_not_called()
            event, = self.errors(root)
            self.assertEqual(event["stage"], "summary_update")
            self.assertEqual(event["error_type"], "RuntimeError")

    def test_runtime_graph_change_stops_only_its_scene_before_next_task(self):
        with tempfile.TemporaryDirectory() as temp, ExitStack() as stack:
            root = Path(temp)
            args = self.fixture(root)
            def task_run(options, task, scene, destination, encoder):
                batch.write(scene / "scene_graph.json", [{"id": 999}])
                return {"status": "fixture_completed"}
            run, encoder = self.worker_patches(stack, task_run)
            self.assertEqual(batch.run_scene(args), 2)
            self.assertEqual(run.call_count, 1)
            self.assertEqual(self.errors(root, "scene200")[0]["stage"], "input_integrity_scene_blocked")
            encoder.close.assert_not_called()
            self.assertFalse((root / "cases/scene200/worker.lock").exists())


if __name__ == "__main__":
    unittest.main()
