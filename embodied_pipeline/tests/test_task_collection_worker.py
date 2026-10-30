"""Offline fixtures only: no model/provider/ROS/WSL calls are permitted."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from embodied_pipeline._legacy import load
from embodied_pipeline.orchestration import task_collection_worker as worker


class Encoder:
    def encode(self, texts):
        return [[1., 0.] for _ in texts]


class CollectionWorkerTests(unittest.TestCase):
    def setUp(self):
        block = patch("urllib.request.urlopen", side_effect=AssertionError("Network forbidden"))
        block.start()
        self.addCleanup(block.stop)
        key = patch.dict("os.environ", {"DASHSCOPE_API_KEY": "offline-fixture-not-a-key"})
        key.start()
        self.addCleanup(key.stop)
        self.args = SimpleNamespace(model="offline-fixture", base_url="https://invalid.example",
                                    request_timeout=17)
        self.task = {"record_id": "fixture_001", "scene_id": "fixture_scene",
                     "instruction": "Navigate to table ID 179, then visit it again."}

    def source(self, root):
        source = Path(root) / "source"
        worker.write(source / "scene_graph.json", [
            {"id": 179, "object_tag": "table", "caption": "A table.",
             "bbox_center": [1., 2., .8], "bbox_extent": [1., 1., .8]},
            {"id": 180, "object_tag": "unknown", "active": False}])
        worker.write(source / "scene_graph_edges.json", [])
        return source

    def alignment(self):
        issued = load("typed_matrix_ids").allocation("offline-collection", 5)
        return dict(status="success", context_bindings={"table": {"element_ids": [179]}},
            task_decomposition=dict(instruction=self.task["instruction"],
                subtasks=[dict(subtask_id="s1", description="Navigate to table and repeat.", dependencies=[])],
                initial_scene_constraints=[],
                goal_state=[dict(constraint_id="g1", subject_role="robot", relation="near",
                                 object_role="table", required=True)],
                identifier_context=dict(version="typed5_ids_v1", plan_id=issued["plan_id"],
                    allocation_sha256=issued["sha256"], role_codes={"table": 20001}, task_codes={"s1": 50001})))

    def matrix(self, alignment, library, book, *args, **kwargs):
        self.assertEqual(kwargs["attempts"], 1)
        self.assertEqual(kwargs["timeout"], 17)
        worker.skills.matrix_payload(alignment, library, book, "offline", kwargs["retrieval"], False,
                                     matrix_contract="typed5_schema")
        row = [book["subtasks"][0]["code"],
               next(s["code"] for s in book["skills"] if s["skill_id"] == "navigate_to_object"),
               book["objects"][0]["code"], 0]
        return dict(schema_version="action_matrix_typed5_v1", codebook_sha256=book["sha256"],
                    matrix=[row, list(row)])

    def test_full_snapshot_unknown_state_graph_extension(self):
        with tempfile.TemporaryDirectory() as temp:
            source = self.source(temp)
            root = Path(temp) / "task"
            worker.claim(root, self.task)
            library, metadata, hashes = worker.prepare_inputs(self.task, source, root)
            contract = library["execution_contract"]
            self.assertNotIn("current_runtime_state", contract)
            self.assertNotIn("runtime_state", contract)
            self.assertFalse((root / "runtime_state.json").exists())
            self.assertEqual(contract["scene_graph_context"]["nodes"], worker.read(source / "scene_graph.json"))
            self.assertFalse(contract["scene_graph_context"]["nodes"][1]["active"])
            self.assertEqual(metadata, {})
            self.assertTrue(contract["decomposition_graph_context_extension"])
            self.assertEqual(len(library["skills"]), 5)
            original = {s["skill_id"]: s for s in worker.read(worker.LIBRARY)["skills"]}
            for skill in library["skills"]:
                self.assertEqual(skill["parameters"], original[skill["skill_id"]]["parameters"])
            for name, digest in hashes.items():
                self.assertEqual(worker.sha(source / name), digest)
                self.assertEqual(worker.sha(root / "input_memory/scene" / name), digest)
            projected = load("structural_matrix").matrix_contract_context(contract, "structural_typed5_v1")
            self.assertEqual(projected["scene_graph_context"], contract["scene_graph_context"])

    def test_alignment_flags_single_http_and_no_repair(self):
        command = worker.alignment_command(self.args, self.task, Path("fixture"))
        for flag, value in [("--request-attempts", "1"), ("--validation-attempts", "1"),
                            ("--request-timeout", "17"), ("--matrix-contract", "typed5_schema"),
                            ("--prompt-arm", "intent_constraints_prompt")]:
            self.assertEqual(command[command.index(flag) + 1], value)
        self.assertIn("--process-constraints", command)
        self.assertNotIn("--decomposition-json", command)
        self.assertNotIn("offline-fixture-not-a-key", command)

    def test_duplicate_actions_preserved_without_grading_or_execution(self):
        with tempfile.TemporaryDirectory() as temp:
            source = self.source(temp)
            root = Path(temp) / "task"
            with patch.object(worker, "collect_alignment", return_value=self.alignment()) as alignment, \
                    patch.object(worker.skills, "request_matrix", side_effect=self.matrix) as request:
                result = worker.run_task(self.args, self.task, source, root, Encoder())
            self.assertEqual(alignment.call_count, 1)
            self.assertEqual(request.call_count, 1)
            self.assertEqual(result["status"], "engineering_json_generated_ungraded")
            self.assertEqual(result["action_count"], 2)
            self.assertTrue(result["collection_started_at_local"].endswith("+08:00"))
            self.assertTrue(result["collection_finished_at_local"].endswith("+08:00"))
            self.assertEqual(result["collection_started_at_utc"], worker.read(root / "attempt_started.json")["started_at_utc"])
            self.assertFalse(result["semantic_correctness_evaluated"])
            envelope = worker.read(root / "engineering_plan.json")
            self.assertIsNone(envelope["coordinate_frame"])
            self.assertFalse(envelope["execution_ready"])
            self.assertEqual(envelope["plan"]["action_count"], 2)
            for name in ("static_review.json", "evaluation.json", "execution_feedback.json", "runtime_state.json"):
                self.assertFalse((root / name).exists())
            with patch.object(worker, "collect_alignment") as alignment:
                with self.assertRaises(FileExistsError):
                    worker.run_task(self.args, self.task, source, root, Encoder())
                alignment.assert_not_called()

    def test_malformed_skill_output_kept_without_retry(self):
        def malformed(*args, **kwargs):
            value = self.matrix(*args, **kwargs)
            value["matrix"][0][0] = str(value["matrix"][0][0])
            return value
        with tempfile.TemporaryDirectory() as temp:
            source, root = self.source(temp), Path(temp) / "task"
            with patch.object(worker, "collect_alignment", return_value=self.alignment()), \
                    patch.object(worker.skills, "request_matrix", side_effect=malformed) as request:
                result = worker.run_task(self.args, self.task, source, root, Encoder())
            self.assertEqual(request.call_count, 1)
            self.assertEqual(result["status"], "collection_stopped_at_stage_ungraded")
            self.assertTrue((root / "llm_matrix_proposal.json").is_file())
            self.assertFalse((root / "engineering_plan.json").exists())
            self.assertFalse(result["failure_is_correctness_grade"])

    def test_transport_failure_does_not_call_downstream(self):
        with tempfile.TemporaryDirectory() as temp:
            source, root = self.source(temp), Path(temp) / "task"
            with patch.object(worker, "collect_alignment", side_effect=TimeoutError("offline timeout")), \
                    patch.object(worker.skills, "request_matrix") as request:
                result = worker.run_task(self.args, self.task, source, root, Encoder())
            request.assert_not_called()
            self.assertEqual(result["failed_stage"], "decomposition_and_object_binding")
            self.assertFalse(result["engineering_json_generated"])
            self.assertTrue((root / "attempt_started.json").is_file())

    def test_missing_key_is_recorded_no_requests(self):
        with tempfile.TemporaryDirectory() as temp:
            source, root = self.source(temp), Path(temp) / "task"
            with patch.dict("os.environ", {"DASHSCOPE_API_KEY": ""}), \
                    patch.object(worker, "collect_alignment") as alignment:
                result = worker.run_task(self.args, self.task, source, root, Encoder())
            alignment.assert_not_called()
            self.assertIn("no request made", result["error"])

    def test_claim_alone_prevents_unknown_outcome_resubmission(self):
        with tempfile.TemporaryDirectory() as temp:
            source, root = self.source(temp), Path(temp) / "task"
            worker.claim(root, self.task)
            with patch.object(worker, "collect_alignment") as alignment:
                with self.assertRaises(FileExistsError):
                    worker.run_task(self.args, self.task, source, root, Encoder())
                alignment.assert_not_called()

    def test_http_and_transport_audit_usage(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp) / "llm_response_audit"
            worker.write(directory / "stage_a_request.json", {"stage": "decomposition"})
            worker.write(directory / "stage_a_http_1.json", {"response_body": json.dumps(
                {"model": "provider-version", "usage": {"prompt_tokens": 12, "completion_tokens": 4}})})
            worker.write(directory / "stage_b_request.json", {"stage": "object_binding"})
            worker.write(directory / "stage_b_transport_1.json", {"response_received": False})
            evidence = worker.response_evidence(temp)
            self.assertEqual(evidence["local_request_records"], 2)
            self.assertEqual(evidence["reported_token_usage"]["prompt_tokens"], 12)
            self.assertEqual(evidence["response_models"], ["provider-version"])
            self.assertFalse(evidence["usage_complete"])


if __name__ == "__main__":
    unittest.main()
