"""Delivery metadata tests: no API, ROS, WSL or model inference."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from embodied_pipeline.data_collection import delivery


class DeliveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "package"
        self.root.mkdir()

    def test_manifest_hashes_bytes_not_just_size(self):
        (self.root / "data.txt").write_bytes(b"abc")
        manifest = delivery.write_file_manifest(self.root)
        self.assertEqual(manifest["count"], 1)
        self.assertEqual(delivery.verify_file_manifest(self.root)["status"], "verified")
        (self.root / "data.txt").write_bytes(b"xyz")
        self.assertEqual(delivery.verify_file_manifest(self.root)["changed"], ["data.txt"])

    def test_manifest_missing_changed_extra_are_separate(self):
        (self.root / "missing.txt").write_text("a")
        delivery.write_file_manifest(self.root)
        (self.root / "missing.txt").unlink()
        (self.root / "new.txt").write_text("b")
        result = delivery.verify_file_manifest(self.root)
        self.assertEqual(result["missing"], ["missing.txt"])
        self.assertEqual(result["extra"], ["new.txt"])
        self.assertEqual(result["count"], 1)

    def test_pending_and_manifest_itself_excluded(self):
        (self.root / "x.pending").write_text("partial")
        (self.root / "a.json").write_text("{}")
        first = delivery.write_file_manifest(self.root)
        second = delivery.write_file_manifest(self.root)
        self.assertEqual(first["files"], second["files"])
        self.assertEqual(first["count"], 1)

    def test_active_worker_freeze_refused_without_deleting_lock(self):
        lock = self.root / "worker.lock"
        lock.touch()
        with self.assertRaisesRegex(ValueError, "Workers may still be active"):
            delivery.write_file_manifest(self.root)
        self.assertTrue(lock.exists())

    def test_manifest_traversal_entry_rejected(self):
        delivery._write(self.root / delivery.MANIFEST_NAME, {"count": 1, "files": [
            {"path": "../secret.txt", "sha256": "0" * 64, "bytes": 1}]})
        result = delivery.verify_file_manifest(self.root)
        self.assertEqual(result["invalid_entries"], [0])

    def test_missing_manifest(self):
        result = delivery.verify_file_manifest(self.root)
        self.assertEqual(result["status"], "manifest_missing")

    def test_docs_are_honest_and_create_no_model_outputs(self):
        config = {"scenes": {s: {} for s in ("scene200", "scene201", "scene202", "d3_619")},
                  "requested_per_scene": 200, "model": "qwen3.8-flash",
                  "base_url": "https://example.invalid/v1", "max_full_collection_requests": 2400}
        with patch("subprocess.Popen", side_effect=AssertionError("must not start processes")):
            inventory = delivery.write_delivery_docs(self.root, config)
        self.assertEqual(inventory["actual_http_response_files"], 0)
        self.assertEqual(inventory["actual_task_files"], 0)
        self.assertIsNone(inventory["valid_sample_count"])
        self.assertEqual(list(self.root.rglob("*_http_*.json")), [])
        readme = (self.root / "README.txt").read_text(encoding="utf-8")
        self.assertIn("--limit-per-scene 2", readme)
        self.assertIn("不自动重试", readme)
        self.assertIn("2400", readme)
        self.assertIn("source_scene_capture_time=null", readme)
        library = delivery._read(self.root / "shared/actual_collection_skill_library.json")
        self.assertEqual({s["skill_id"] for s in library["skills"]}, set(delivery.SKILL_IDS))
        self.assertFalse(library["delivery_boundary"]["robot_execution_authorized"])
        model = delivery._read(self.root / "shared/model_versions.json")
        self.assertIsNone(model["provider_response_model"])
        delivery.write_file_manifest(self.root)
        self.assertEqual(delivery.verify_file_manifest(self.root)["status"], "verified")

    def test_schemas_support_catalog_and_aggregate_rows(self):
        from embodied_pipeline.data_collection.catalog import build_scene
        graph_dir = self.root / "input"
        graph_dir.mkdir()
        nodes = [{"id": 1, "object_tag": "table", "caption": "A table.",
                  "bbox_center": [0, 0, 0], "bbox_extent": [1, 1, 1]}]
        delivery._write(graph_dir / "scene_graph.json", nodes)
        delivery._write(graph_dir / "scene_graph_edges.json", [])
        records, _, references = build_scene("scene201", graph_dir, count=1)
        self.assertEqual(len(records), 1)
        self.assertTrue(set(delivery.dataset_schema()["required"]).issubset(records[0]))
        self.assertTrue(set(delivery.reference_schema()["required"]).issubset(references[0]))
        aggregate = dict(records[0], model_run=None, model_result=None, semantic_evaluation="pending", valid_candidate=None)
        self.assertEqual(delivery.dataset_schema()["properties"]["valid_candidate"]["type"], "null")
        try:
            import jsonschema
        except ImportError:
            # Required-field compatibility is always checked above. Full JSON
            # Schema validation is optional; quality.py performs data checks.
            return
        jsonschema.Draft202012Validator.check_schema(delivery.dataset_schema())
        jsonschema.validate(records[0], delivery.dataset_schema())
        jsonschema.validate(aggregate, delivery.dataset_schema())
        jsonschema.validate(references[0], delivery.reference_schema())


if __name__ == "__main__":
    unittest.main()
