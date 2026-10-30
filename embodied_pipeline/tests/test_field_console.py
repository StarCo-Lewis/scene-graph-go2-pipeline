"""Offline GUI backend tests, no API/WSL/robot connection."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from embodied_pipeline.ui import field_backend as b
from embodied_pipeline.ui.field_worker import validate_job


class BackendTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.scene_dir = self.root / "scene"
        self.scene_dir.mkdir()
        b.write(self.scene_dir / "scene_graph.json", [{"id": 179, "object_tag": "table", "bbox_center": [1, 2, 3], "bbox_extent": [1, 1, 1]}])
        b.write(self.scene_dir / "scene_graph_edges.json", [])
        b.write(self.scene_dir / "scene_metadata.json", {"scene_id": "test", "unit": "metre", "up_axis": "z", "coordinate_frame": "test_scene"})
        self.scene = b.load_scene(self.scene_dir)

    def test_create_task_no_dispatch(self):
        with patch.object(b, "ROOT", self.root):
            folder = b.create_task(self.scene, "Navigate to the table.")
        task = b.read(folder / "task.json")
        self.assertNotIn("target_element_id", task)
        self.assertEqual(task["schema_version"], "real_scene_navigation_instruction_v1")
        self.assertEqual(task["scene_dir"], "scene")
        self.assertFalse((folder / "planning").exists())

    def test_no_silent_translation_or_empty_instruction(self):
        with self.assertRaises(ValueError):
            b.create_task(self.scene, "去桌子那里")
        with self.assertRaises(ValueError):
            b.create_task(self.scene, "")

    def test_instruction_is_not_rewritten_to_list_reference(self):
        with patch.object(b, "ROOT", self.root):
            folder = b.create_task(self.scene, "navigate to office machine(id13)")
        task = b.read(folder / "task.json")
        self.assertEqual(task["instruction"], "navigate to office machine(id13)")
        self.assertNotIn("target_element_id", task)
        # Existence/ambiguity is left to graph binding, not an optional UI selection.
        self.assertNotIn("selected", task)

    def test_changed_graph_rejected(self):
        (self.scene_dir / "scene_graph_edges.json").write_text("[1]", encoding="utf-8")
        with self.assertRaises(ValueError):
            b.create_task(self.scene, "Navigate there")

    def test_paths_and_command_quoting(self):
        self.assertEqual(b.wsl_path("C:\\My Project\\file.json"), "/mnt/c/My Project/file.json")
        value = b.command(self.root, "/a b/python", "/embed/python", "/models/$(bad command)")
        self.assertIn("'/models/$(bad command)'", value)
        self.assertNotIn("--execute", value)

    def test_feedback_requires_confirmed_frame_and_normal_quaternion(self):
        row = {"event": "feedback", "data": {"current_pose": {
            "header": {"frame_id": "scene"}, "pose": {
                "position": {"x": 1, "y": 2}, "orientation": {"x": 0, "y": 0, "z": 0, "w": 1}}}}}
        self.assertIsNone(b.feedback_pose(row, None))
        self.assertIsNone(b.feedback_pose(row, "odom"))
        self.assertEqual(b.feedback_pose(row, "scene"), (1, 2, 0))
        row["data"]["current_pose"]["pose"]["orientation"]["w"] = 0
        self.assertIsNone(b.feedback_pose(row, "scene"))

    def test_partial_log_lines_are_not_lost(self):
        path = self.root / "events.jsonl"
        path.write_bytes(b'{"event":"feedback"')
        tail = b.JsonlTail(path)
        self.assertEqual(tail.poll(), [])
        with path.open("ab") as f:
            f.write(b'}\n')
        self.assertEqual(tail.poll(), [{"event": "feedback"}])
        self.assertEqual(tail.poll(), [])

    def test_log_truncation_and_bad_json(self):
        path = self.root / "events.jsonl"
        path.write_bytes(b'{"event":"feedback"}\n')
        tail = b.JsonlTail(path)
        tail.poll()
        path.write_bytes(b'bad\n')
        self.assertEqual(tail.poll(), [{"event": "invalid_log_record"}])

    def test_plan_from_wrong_scene_rejected(self):
        path = self.root / "plan.json"
        b.write(path, {"schema_version": "wrong"})
        with self.assertRaises(ValueError):
            b.inspect_plan(path, self.scene)

    def test_worker_requires_hash_and_new_run(self):
        with patch.object(b, "ROOT", self.root):
            folder = b.create_task(self.scene, "Navigate there")
        record = {"schema_version": "field_planning_job_v1", "folder": folder.relative_to(self.root).as_posix(), "task_sha256": b.sha(folder / "task.json")}
        with patch("embodied_pipeline.ui.field_worker.ROOT", self.root):
            self.assertEqual(validate_job(record), folder)
            (folder / "planning").mkdir()
            with self.assertRaises(ValueError):
                validate_job(record)

    def test_worker_path_escape(self):
        record = {"schema_version": "field_planning_job_v1", "folder": "../escape", "task_sha256": "none"}
        with self.assertRaises(ValueError):
            validate_job(record)

    def test_queue_requires_online_worker(self):
        with patch.object(b, "worker_status", return_value=None):
            with self.assertRaises(ValueError):
                b.submit(self.root)

    def test_queue_no_duplicates(self):
        queue = self.root / "queue"
        queue.mkdir()
        with patch.object(b, "ROOT", self.root), patch.object(b, "QUEUE", queue), patch.object(b, "worker_status", return_value={"status": "ready"}):
            folder = b.create_task(self.scene, "Navigate there")
            self.assertTrue(b.submit(folder).exists())
            with self.assertRaises(ValueError):
                b.submit(folder)


if __name__ == "__main__":
    unittest.main()
