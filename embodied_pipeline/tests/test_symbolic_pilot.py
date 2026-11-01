"""Offline contracts: source coordinates, metadata binding and bounded pilot isolation."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from embodied_pipeline.data_collection import symbolic_pilot as pilot
from embodied_pipeline.orchestration import task_collection_worker as worker


class SymbolicPilotTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.net = patch("urllib.request.urlopen", side_effect=AssertionError("No network"))
        self.net.start()
        self.addCleanup(self.net.stop)

    def scene_task(self):
        scene = self.root / "camera_scene"
        pilot.save(scene / "scene_graph.json", [{"id": 1, "object_tag": "chair", "caption": "",
            "bbox_center": [0, 0, 1], "bbox_extent": [1, 1, 1], "active": True}])
        pilot.save(scene / "scene_graph_edges.json", [])
        pilot.save(scene / "scene_metadata.json", {"scene_id": scene.name,
            "coordinate_frame": "camera_optical", "length_unit": "m"})
        return scene, {"record_id": "camera_91000001", "scene_id": scene.name,
            "instruction": "Navigate to chair ID 1.", "input_scope": "symbolic_source_frame_v1",
            "prepared_scene_sha256": {n: pilot.digest(scene / n) for n in pilot.NAMES}}

    def test_source_coordinates_preserved_without_world_registration(self):
        scene, task = self.scene_task()
        (scene / "fixture.bin").write_bytes(b"offline fixture")
        pilot.save(scene / "modality_index.json", {"point_files": ["fixture.bin"], "rgb_images": []})
        metadata, nodes = pilot.validate_scene(scene)
        self.assertEqual(nodes[0]["id"], 1)
        self.assertEqual(metadata["coordinate_frame"], "camera_optical")
        dest = self.root / "task"
        worker.claim(dest, task)
        library, metadata, hashes = worker.prepare_inputs(task, scene, dest)
        self.assertEqual(metadata["coordinate_frame"], "camera_optical")
        self.assertEqual(hashes, task["prepared_scene_sha256"])
        contract = library["execution_contract"]
        self.assertFalse(contract["execution_ready"])
        self.assertNotIn("current_runtime_state", contract)
        self.assertIn("only the final destination", contract["factual_boundary"])

    def test_changed_metadata_fails_before_input_copy(self):
        scene, task = self.scene_task()
        metadata = pilot.read(scene / pilot.NAMES[2])
        metadata["scene_id"] = "changed"
        pilot.save(scene / pilot.NAMES[2], metadata)
        dest = self.root / "task"
        with self.assertRaisesRegex(ValueError, "hashes changed"):
            worker.prepare_inputs(task, scene, dest)
        self.assertFalse((dest / "input_memory").exists())

    def test_scene_identity_mismatch_is_rejected(self):
        scene, task = self.scene_task()
        task["scene_id"] = "another_scene"
        with self.assertRaisesRegex(ValueError, "scene_id differ"):
            worker.prepare_inputs(task, scene, self.root / "task")

    def test_run_continues_after_one_failure_without_reading_references(self):
        tasks = [{"record_id": "fixture_"+str(i), "scene_id": "scene"} for i in range(12)]
        pilot.save(self.root / "tasks.json", tasks)
        (self.root / "references_not_model_input.json").write_text("INVALID_REFERENCE_MUST_NOT_BE_READ", encoding="utf-8")
        calls = []
        def fake(args, task, scene, destination, encoder):
            calls.append(task["record_id"])
            if len(calls) == 1:
                raise RuntimeError("offline fixture failure")
            return {"record_id": task["record_id"], "status": "fixture_only"}
        with patch.object(pilot, "check"), patch.object(worker, "run_task", side_effect=fake), \
                patch.dict("os.environ", {"DASHSCOPE_API_KEY": "offline-fixture"}):
            pilot.run(self.root, True)
            with self.assertRaises(FileExistsError):
                pilot.run(self.root, True)
        summary = pilot.read(self.root / "results/pilot_summary.json")
        self.assertEqual(len(calls), 12)
        self.assertEqual(summary["finished_tasks"], 12)
        self.assertEqual(summary["rows"][0]["status"], "worker_exception")
        self.assertEqual(summary["outcome_counts"]["complete_plans"], 0)

    def test_failure_envelopes_are_not_counted_as_complete_plans(self):
        tasks = [{"record_id": "fixture_"+str(i), "scene_id": "scene"} for i in range(4)]
        pilot.save(self.root / "tasks.json", tasks)
        def fake(args, task, scene, destination, encoder):
            number = int(task["record_id"].split("_")[1])
            Path(destination).mkdir(parents=True)
            complete = number == 0
            # Task 2 claims success in its row but the delivered file is incomplete: not counted.
            pilot.save(Path(destination) / "engineering_plan.json", {
                "plan": {"subtask_plans": []} if complete else None,
                "artifact_completeness": "complete" if complete else "incomplete",
                "planning_completed": complete})
            if number == 3:
                raise RuntimeError("offline fixture failure after delivery")
            return {"record_id": task["record_id"], "status": "fixture",
                    "planning_completed": complete or number == 2,
                    "engineering_json_generated": complete}
        with patch.object(pilot, "check"), patch.object(worker, "run_task", side_effect=fake), \
                patch.dict("os.environ", {"DASHSCOPE_API_KEY": "offline-fixture"}):
            result = pilot.run(self.root, True)
        counts = pilot.read(self.root / "results/pilot_summary.json")["outcome_counts"]
        self.assertEqual(counts, result["outcome_counts"])
        self.assertEqual(counts["engineering_plan_files"], 4)
        self.assertEqual(counts["complete_plans"], 1)
        self.assertEqual(counts["not_complete"], 3)
        self.assertEqual(counts["worker_exceptions"], 1)
        rows = pilot.read(self.root / "results/pilot_summary.json")["rows"]
        self.assertTrue(rows[3]["engineering_plan_file_written"])
        self.assertFalse(rows[3]["planning_completed"])

    def test_no_paid_calls_without_acknowledgment(self):
        with patch.object(pilot, "check"), patch.object(worker, "run_task") as run:
            with self.assertRaisesRegex(ValueError, "acknowledge"):
                pilot.run(self.root, False)
            run.assert_not_called()

    def test_hash_checker_and_path_escape(self):
        file = self.root / "file.json"
        file.write_text("{}", encoding="utf-8")
        entries = pilot.inventory(self.root)
        pilot.verify_files(self.root, entries)
        file.write_text("[]", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "mismatch"):
            pilot.verify_files(self.root, entries)
        with self.assertRaises(ValueError):
            pilot.within(self.root, "../outside.json")
