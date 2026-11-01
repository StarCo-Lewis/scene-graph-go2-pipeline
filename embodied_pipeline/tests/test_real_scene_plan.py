"""Offline wiring tests. Fixture matrices are NOT live model results."""

import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from embodied_pipeline._legacy import load
from embodied_pipeline.orchestration import real_scene_plan as smoke
from embodied_pipeline.perception.real_rgbd import WORLD, sha, write


class Encoder:
    def encode(self, texts):
        return [[1., 0.] for _ in texts]


class PlanningOnlyTests(unittest.TestCase):
    def setUp(self):
        self.network = patch("urllib.request.urlopen", side_effect=AssertionError("Network forbidden"))
        self.network.start()
        self.addCleanup(self.network.stop)

    def prepare(self, temp, instruction_led=False):
        source = Path(temp) / "source"
        write(source / "scene_graph.json", [dict(id=179, object_tag="table", caption="A round table.",
            bbox_center=[1., 2., .8], bbox_extent=[1., 1., .8])])
        write(source / "scene_graph_edges.json", [])
        write(source / "scene_metadata.json", dict(schema_version="real_scene_import_v1",
            coordinate_frame=WORLD, node_count=1, edge_count=0))
        task = dict(schema_version="real_scene_navigation_smoke_v1", task_id="fixture_only",
            scene_dir=str(source), scene_graph_sha256=sha(source / "scene_graph.json"),
            scene_edges_sha256=sha(source / "scene_graph_edges.json"), target_element_id=179,
            instruction="Navigate to table ID 179; do not manipulate objects.")
        if instruction_led:
            task["schema_version"] = smoke.INSTRUCTION_SCHEMA
            task["scene_metadata_sha256"] = sha(source / "scene_metadata.json")
            del task["target_element_id"]
        path = Path(temp) / "task.json"
        write(path, task)
        root = Path(temp) / "run"
        return root, smoke.prepare(path, root)

    def alignment(self, task):
        wire = load("typed_matrix_ids")
        issued = wire.allocation("offline-smoke", 5)
        return dict(status="success", context_bindings={"table": {"element_ids": [179]}},
            task_decomposition=dict(instruction=task["instruction"],
                subtasks=[dict(subtask_id="s1", description="Navigate to the table", dependencies=[])],
                initial_scene_constraints=[],
                goal_state=[dict(constraint_id="g1", subject_role="robot", relation="near", object_role="table", required=True)],
                identifier_context=dict(version="typed5_ids_v1", plan_id=issued["plan_id"],
                    allocation_sha256=issued["sha256"], role_codes={"table": 20001}, task_codes={"s1": 50001})))

    def test_prepare_preserves_graph_and_unknown_state(self):
        with tempfile.TemporaryDirectory() as temp:
            root, (task, bundle, library, source, hashes) = self.prepare(temp)
            for name, value in hashes.items():
                self.assertEqual(sha(root / "input_memory/scene" / name), value)
                self.assertEqual(sha(source / name), value)
            self.assertFalse((root / "runtime_state.json").exists())
            self.assertNotIn("current_runtime_state", library["execution_contract"])
            self.assertFalse(library["execution_contract"]["execution_ready"])
            self.assertEqual(len(library["skills"]), 5)
            self.assertEqual(library["skills"][0]["parameters"][1]["default"], .6)

    def test_changed_scene_revision_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            self.prepare(temp)
            write(Path(temp) / "source/scene_graph_edges.json", [{"unexpected": True}])
            with self.assertRaisesRegex(ValueError, "different graph revision"):
                smoke.prepare(Path(temp) / "task.json", Path(temp) / "other")

    def exercise(self, duplicate=False, invalid_type=False, instruction_led=False):
        with tempfile.TemporaryDirectory() as temp:
            root, (task, bundle, library, source, hashes) = self.prepare(temp, instruction_led=instruction_led)
            alignment = self.alignment(task)
            def matrix(a, lib, book, *args, **kw):
                smoke.skills.matrix_payload(a, lib, book, "offline", kw["retrieval"], False,
                                            matrix_contract=kw["matrix_contract"])
                row = [book["subtasks"][0]["code"],
                       next(s["code"] for s in book["skills"] if s["skill_id"] == "navigate_to_object"),
                       book["objects"][0]["code"], 0]
                if invalid_type:
                    row[0] = str(row[0])
                return dict(schema_version="action_matrix_typed5_v1", codebook_sha256=book["sha256"],
                            matrix=[row, row.copy()] if duplicate else [row])
            args = SimpleNamespace(model="offline", base_url="https://invalid.example")
            summary = {"stages": []}
            with patch.object(smoke.planning, "decompose_and_bind", return_value=alignment) as decomp, \
                    patch.object(smoke.skills, "request_matrix", side_effect=matrix) as request:
                if invalid_type:
                    with self.assertRaises(ValueError):
                        smoke.generate(args, root, task, bundle, library, summary, Encoder())
                    self.assertTrue((root / "llm_matrix_proposal.json").exists())
                    self.assertFalse((root / "engineering_plan.json").exists())
                    return
                smoke.generate(args, root, task, bundle, library, summary, Encoder())
            self.assertEqual((decomp.call_count, request.call_count), (1, 1))
            result = smoke.read(root / "engineering_plan.json")
            self.assertEqual(summary["status"], "planning_json_generated_not_executed")
            self.assertFalse(result["execution_ready"])
            self.assertFalse(result["robot_executed"])
            self.assertFalse(result["memory_committed"])
            self.assertIsNone(result["robot_goal_pose"])
            self.assertEqual(result["plan"]["action_count"], 2 if duplicate else 1)
            review = smoke.read(root / "static_review.json")
            self.assertEqual(review["physical_feasibility"], "unknown")
            self.assertTrue(any(c["predicate"] == "navigable" and c["status"] == "unknown"
                for c in review["symbolic_trace"][0]["preconditions"]))
            self.assertEqual(result["task_output_check"]["repeated_navigation"], duplicate)
            for name, value in hashes.items():
                self.assertEqual(sha(source / name), value)
            self.assertFalse((root / "runtime_state.json").exists())

    def test_full_typed5_offline_wiring_without_execution(self):
        self.exercise()

    def test_duplicate_actions_preserved_not_replaced_by_review(self):
        self.exercise(duplicate=True)

    def test_invalid_matrix_preserved_not_repaired(self):
        self.exercise(invalid_type=True)

    def test_instruction_led_full_pipeline_without_reference_target(self):
        self.exercise(instruction_led=True)

    def test_instruction_schema_rejects_leftover_reference(self):
        with tempfile.TemporaryDirectory() as temp:
            self.prepare(temp, instruction_led=True)
            path = Path(temp) / "task.json"
            task = smoke.read(path)
            task["target_element_id"] = 183
            write(path, task)
            with self.assertRaisesRegex(ValueError, "preselected reference"):
                smoke.prepare(path, Path(temp) / "other")

    def test_metadata_mutation_rejected_before_output(self):
        with tempfile.TemporaryDirectory() as temp:
            self.prepare(temp, instruction_led=True)
            metadata = Path(temp) / "source/scene_metadata.json"
            value = smoke.read(metadata)
            value["scene_id"] = "changed_after_preparation"
            write(metadata, value)
            output = Path(temp) / "blocked"
            with self.assertRaisesRegex(ValueError, "metadata hash mismatch"):
                smoke.prepare(Path(temp) / "task.json", output)
            self.assertFalse(output.exists())

    def test_v1_and_missing_v2_hash_require_reprepare(self):
        with tempfile.TemporaryDirectory() as temp:
            self.prepare(temp, instruction_led=True)
            path = Path(temp) / "task.json"
            original = smoke.read(path)
            for schema in ("real_scene_navigation_instruction_v1", smoke.INSTRUCTION_SCHEMA):
                task = dict(original, schema_version=schema)
                task.pop("scene_metadata_sha256")
                write(path, task)
                output = Path(temp) / "blocked"
                with self.assertRaisesRegex(ValueError, "re-prepare"):
                    smoke.prepare(path, output)
                self.assertFalse(output.exists())

    def test_instruction_checks_grounding_and_single_navigation(self):
        task = {"schema_version": smoke.INSTRUCTION_SCHEMA, "instruction": "navigate to office machine(id13)"}
        plan = {"subtask_plans": [{"steps": [{"step_id": "s1", "skill_id": "navigate_to_object",
                 "arguments": {"target": {"role_id": "machine", "element_id": 13}}}]}]}
        bundle = {"nodes": [{"id": 13}, {"id": 183}, {"id": 14, "active": False}]}
        alignment = {"context_bindings": {"machine": {"element_ids": [13]}}}
        good = smoke.check_navigation(plan, task, bundle, alignment)
        self.assertTrue(good["passed"])
        self.assertFalse(good["reference_target_comparison_performed"])
        for skill, target in [("grasp_object", 13), ("navigate_to_object", 999),
                               ("navigate_to_object", 14), ("navigate_to_object", 183)]:
            bad = copy.deepcopy(plan)
            step = bad["subtask_plans"][0]["steps"][0]
            step["skill_id"] = skill
            step["arguments"]["target"]["element_id"] = target
            self.assertFalse(smoke.check_navigation(bad, task, bundle, alignment)["passed"])
        for steps in [[], plan["subtask_plans"][0]["steps"]*2]:
            self.assertFalse(smoke.check_navigation({"subtask_plans": [{"steps": steps}]}, task, bundle, alignment)["passed"])

    def test_task_check_rejects_wrong_target_or_manipulation(self):
        task = {"target_element_id": 179}
        plan = {"subtask_plans": [{"steps": [{"step_id": "s1_01", "skill_id": "navigate_to_object",
                  "arguments": {"target": {"role_id": "table", "element_id": 179}}}]}]}
        self.assertTrue(smoke.check_navigation(plan, task)["passed"])
        for skill, target in (("grasp_object", 179), ("navigate_to_object", 3)):
            bad = copy.deepcopy(plan)
            bad["subtask_plans"][0]["steps"][0].update(skill_id=skill,
                arguments={"target": {"role_id": "table", "element_id": target}})
            self.assertFalse(smoke.check_navigation(bad, task)["passed"])


if __name__ == "__main__":
    unittest.main()
