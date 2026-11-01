"""Offline fixtures only: no model/provider/ROS/WSL calls are permitted."""

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from embodied_pipeline._legacy import load
from embodied_pipeline.orchestration import task_collection_worker as worker
from embodied_pipeline.validation import plan_semantics


def goal(constraint_id, role, relation="near", required=True):
    return {"constraint_id": constraint_id, "subject_role": "robot",
            "relation": relation, "object_role": role, "required": required}


def decomposition(goals, instruction="Visit chair ID 1, then go to table ID 2.",
                  subtasks=2, process_constraints=None, contract=True):
    value = {"instruction": instruction,
             "subtasks": [{"subtask_id": "st_%03d" % (i + 1), "description": instruction,
                           "dependencies": [] if i == 0 else ["st_%03d" % i]}
                          for i in range(subtasks)],
             "initial_scene_constraints": [], "goal_state": goals}
    if contract:
        value["process_constraint_contract"] = "process_constraints_v1"
        value["process_constraints"] = process_constraints or []
    return value


def navigation_plan(pairs):
    return {"subtask_plans": [
        {"subtask_id": "st_%03d" % (i + 1), "steps": [
            {"step_id": "st_%03d_01" % (i + 1), "skill_id": "navigate_to_object",
             "arguments": {"target": {"role_id": role, "element_id": element}}}]}
        for i, (role, element) in enumerate(pairs)], "unplanned_subtasks": []}


class TerminalGoalTests(unittest.TestCase):
    def test_single_terminal_goal_with_ordered_visits_passes(self):
        result = plan_semantics.audit(
            decomposition([goal("c1", "role_002")]),
            bindings={"role_001": {"element_ids": [1]}, "role_002": {"element_ids": [2]}},
            plan=navigation_plan([("role_001", 1), ("role_002", 2)]))
        self.assertTrue(result["passed"])
        self.assertFalse(result["has_defect"])
        self.assertTrue(result["inputs_available"]["navigation_plan"])

    def test_intermediate_visit_kept_as_terminal_goal_is_flagged(self):
        result = plan_semantics.audit(
            decomposition([goal("c1", "role_001"), goal("c2", "role_002")]),
            bindings={"role_001": {"element_ids": [1]}, "role_002": {"element_ids": [2]}},
            plan=navigation_plan([("role_001", 1), ("role_002", 2)]))
        self.assertEqual(result["defect_codes"], ["mutually_exclusive_terminal_robot_location"])
        detail = result["issues"][0]["detail"]
        self.assertEqual(detail["binding_evidence"], "bound_element_ids_disjoint")
        self.assertEqual(detail["distinct_target_groups"], 2)
        self.assertEqual(result["suggested_terminal_goal"],
                         {"source": "final_navigation_target", "role_id": "role_002", "element_id": 2})

    def test_two_roles_on_one_object_are_not_a_conflict(self):
        result = plan_semantics.audit(
            decomposition([goal("c1", "role_001"), goal("c2", "role_002")]),
            bindings={"role_001": {"element_ids": [7]}, "role_002": {"element_ids": [7]}})
        self.assertTrue(result["passed"])

    def test_optional_and_non_location_goals_are_ignored(self):
        result = plan_semantics.audit(decomposition([
            goal("c1", "role_002"),
            goal("c2", "role_001", required=False),
            {"constraint_id": "c3", "subject_role": "cup", "relation": "on", "object_role": "role_003"}]))
        self.assertTrue(result["passed"])
        self.assertEqual(result["terminal_location_goal_count"], 1)

    def test_unresolved_roles_still_report_the_conflict(self):
        result = plan_semantics.audit(
            decomposition([goal("c1", "role_001"), goal("c2", "role_002")]), bindings={})
        self.assertTrue(result["has_defect"])
        self.assertEqual(result["issues"][0]["detail"]["binding_evidence"],
                         "unbound_roles_assumed_distinct")

    def test_terminal_goal_must_be_the_last_destination(self):
        result = plan_semantics.audit(
            decomposition([goal("c1", "role_001")]),
            bindings={"role_001": {"element_ids": [1]}, "role_002": {"element_ids": [2]}},
            plan=navigation_plan([("role_001", 1), ("role_002", 2)]))
        self.assertEqual(result["defect_codes"], ["terminal_goal_not_final_navigation_target"])


class PreservationTests(unittest.TestCase):
    text = "Visit sink ID 14 and counter ID 12, in that order. Do not move any objects."

    def test_text_only_protection_is_flagged(self):
        result = plan_semantics.audit(
            decomposition([goal("c1", "role_002")], instruction=self.text))
        self.assertEqual(result["defect_codes"], ["preservation_requirement_not_structured"])
        detail = result["issues"][0]["detail"]
        self.assertTrue(detail["scene_wide_scope"])
        self.assertEqual(detail["reason"], "contract_returned_empty")

    def test_missing_contract_is_named_separately(self):
        result = plan_semantics.audit(
            decomposition([goal("c1", "role_002")], instruction=self.text, contract=False))
        self.assertEqual(result["issues"][0]["detail"]["reason"], "contract_absent")

    def test_structured_protection_with_wide_scope_warns_only(self):
        result = plan_semantics.audit(decomposition(
            [goal("c1", "role_002")], instruction=self.text,
            process_constraints=[{"role_id": "role_001", "restriction": "do_not_manipulate"}]))
        self.assertFalse(result["has_defect"])
        self.assertEqual(result["issue_codes"], ["preservation_scope_wider_than_contract"])

    def test_ordinary_navigation_text_is_not_flagged(self):
        result = plan_semantics.audit(decomposition([goal("c1", "role_002")]))
        self.assertTrue(result["passed"])


class WorkerAnnotationTests(unittest.TestCase):
    def setUp(self):
        block = patch("urllib.request.urlopen", side_effect=AssertionError("Network forbidden"))
        block.start()
        self.addCleanup(block.stop)
        key = patch.dict("os.environ", {"DASHSCOPE_API_KEY": "offline-fixture-not-a-key"})
        key.start()
        self.addCleanup(key.stop)
        self.args = SimpleNamespace(model="offline-fixture", base_url="https://invalid.example",
                                    request_timeout=17)
        self.task = {"record_id": "fixture_gate_001", "scene_id": "fixture_scene",
                     "instruction": "First visit table ID 179, then go to table ID 180."}

    def source(self, root):
        source = Path(root) / "source"
        worker.write(source / "scene_graph.json", [
            {"id": 179, "object_tag": "table", "caption": "A table.",
             "bbox_center": [1., 2., .8], "bbox_extent": [1., 1., .8]},
            {"id": 180, "object_tag": "table", "caption": "Another table.",
             "bbox_center": [3., 2., .8], "bbox_extent": [1., 1., .8]}])
        worker.write(source / "scene_graph_edges.json", [])
        return source

    def alignment(self):
        issued = load("typed_matrix_ids").allocation("offline-gate", 5)
        value = decomposition([goal("c1", "role_001"), goal("c2", "role_002")],
                              instruction=self.task["instruction"])
        for index, subtask in enumerate(value["subtasks"]):
            subtask["required_roles"] = [{
                "role_id": "role_%03d" % (index + 1), "name": "table_%d" % (179 + index),
                "semantic_query_en": "table", "semantic_labels": ["table"],
                "quantity": 1, "required": True}]
        value["identifier_context"] = dict(
            version="typed5_ids_v1", plan_id=issued["plan_id"], allocation_sha256=issued["sha256"],
            role_codes={"role_001": 20001, "role_002": 20002},
            task_codes={"st_001": 50001, "st_002": 50002})
        return dict(status="success",
                    context_bindings={"role_001": {"element_ids": [179]},
                                      "role_002": {"element_ids": [180]}},
                    task_decomposition=value)

    def matrix(self, alignment, library, book, *args, **kwargs):
        navigate = next(s["code"] for s in book["skills"] if s["skill_id"] == "navigate_to_object")
        rows = [[subtask["code"], navigate, book["objects"][index]["code"], 0]
                for index, subtask in enumerate(book["subtasks"])]
        return dict(schema_version="action_matrix_typed5_v1", codebook_sha256=book["sha256"],
                    matrix=rows)

    def test_flagged_round_still_delivers_the_engineering_json(self):
        with tempfile.TemporaryDirectory() as temp:
            source = self.source(temp)
            root = Path(temp) / "task"
            with patch.object(worker, "collect_alignment", return_value=self.alignment()), \
                    patch.object(worker.skills, "request_matrix", side_effect=self.matrix) as request:
                result = worker.run_task(self.args, self.task, source, root, None)
            self.assertEqual(request.call_count, 1)
            self.assertTrue(result["engineering_json_generated"])
            self.assertEqual(result["status"],
                             "engineering_json_generated_requires_semantic_review_ungraded")
            envelope = worker.read(root / "engineering_plan.json")
            self.assertEqual(envelope["quality_flags"],
                             ["mutually_exclusive_terminal_robot_location"])
            self.assertTrue(envelope["semantic_review_required"])
            self.assertEqual(envelope["status"], "ungraded_proposal_requires_semantic_review")
            # Mirrors the 91000003 case: intermediate + final near goals. JSON kept, complete, not usable.
            self.assertEqual(envelope["artifact_completeness"], "complete")
            self.assertFalse(envelope["usable_as_plan"])
            self.assertFalse(result["usable_as_plan"])
            self.assertFalse(envelope["execution_ready"])
            self.assertTrue(worker.plan_artifact.has_complete(root / "engineering_plan.json"))
            audit = worker.read(root / "decomposition_audit.json")
            self.assertEqual(audit["mode"], "annotate")
            self.assertFalse(audit["enforced"])
            self.assertIn("suggested_terminal_goal", audit)

    def test_opt_in_block_mode_skips_the_paid_skill_request(self):
        with tempfile.TemporaryDirectory() as temp:
            source = self.source(temp)
            root = Path(temp) / "task"
            args = SimpleNamespace(**dict(vars(self.args), semantic_audit_mode="block"))
            with patch.object(worker, "collect_alignment", return_value=self.alignment()), \
                    patch.object(worker.skills, "request_matrix",
                                 side_effect=AssertionError("Paid request after block")) as request:
                result = worker.run_task(args, self.task, source, root, None)
            self.assertEqual(request.call_count, 0)
            self.assertEqual(result["status"], "decomposition_semantics_blocked_ungraded")
            # The blocked round is still delivered, explicitly incomplete and without steps.
            self.assertFalse(result["engineering_json_generated"])
            self.assertTrue(result["engineering_plan_file_written"])
            envelope = worker.read(root / "engineering_plan.json")
            self.assertEqual(envelope["artifact_completeness"], "incomplete")
            self.assertIsNone(envelope["plan"])
            self.assertEqual(envelope["failed_stage"], "decomposition_semantics_audit")
            self.assertTrue(envelope["decomposition_audit"]["enforced"])
            self.assertIn("mutually_exclusive_terminal_robot_location", envelope["quality_flags"])
            self.assertEqual(envelope["partial_evidence"]["object_bindings"],
                             self.alignment()["context_bindings"])

    def test_stopped_binding_still_audits_the_decomposition(self):
        with tempfile.TemporaryDirectory() as temp:
            source = self.source(temp)
            root = Path(temp) / "task"

            def stopped(args, task, destination):
                worker.write(Path(destination) / "task_decomposition.json",
                             self.alignment()["task_decomposition"])
                raise ValueError("Existing alignment contract blocked continuation")

            with patch.object(worker, "collect_alignment", side_effect=stopped):
                result = worker.run_task(self.args, self.task, source, root, None)
            self.assertEqual(result["status"], "collection_stopped_at_stage_ungraded")
            digest = result["decomposition_audit"]
            self.assertTrue(digest["has_defect"])
            self.assertFalse(digest["enforced"])
            self.assertFalse(digest["object_binding_completed"])
            envelope = worker.read(root / "engineering_plan.json")
            self.assertEqual(envelope["artifact_completeness"], "incomplete")
            self.assertEqual(envelope["partial_evidence"]["task_decomposition"]["instruction"],
                             self.task["instruction"])
            self.assertIn("mutually_exclusive_terminal_robot_location", envelope["quality_flags"])


class RobustnessTests(unittest.TestCase):
    def test_unplanned_subtasks_reported_for_non_navigation_plan_without_goals(self):
        plan = {"subtask_plans": [{"steps": [{"skill_id": "grasp_object", "arguments": {}}]}],
                "unplanned_subtasks": ["st_002"]}
        result = plan_semantics.audit(decomposition([]), plan=plan)
        self.assertEqual(result["defect_codes"], ["unplanned_subtasks_remain"])
        self.assertFalse(result["inputs_available"]["navigation_plan"])

    def test_unplanned_subtasks_reported_when_navigation_plan_has_no_goal(self):
        plan = navigation_plan([("role_001", 1)])
        plan["unplanned_subtasks"] = ["st_002"]
        result = plan_semantics.audit(decomposition([]), plan=plan)
        self.assertIn("unplanned_subtasks_remain", result["defect_codes"])

    def test_none_role_never_matches_none_target_role(self):
        result = plan_semantics.audit(
            decomposition([goal("c1", None)]), bindings={},
            plan=navigation_plan([(None, 5)]))
        self.assertIn("terminal_goal_missing_object_role", result["defect_codes"])
        self.assertIn("terminal_goal_not_final_navigation_target", result["defect_codes"])

    def test_two_missing_roles_are_not_merged_into_one_object(self):
        result = plan_semantics.audit(decomposition([goal("c1", None), goal("c2", None)]))
        missing = [i for i in result["issues"] if i["code"] == "terminal_goal_missing_object_role"]
        self.assertEqual(missing[0]["detail"]["constraint_ids"], ["c1", "c2"])
        self.assertNotIn("mutually_exclusive_terminal_robot_location", result["issue_codes"])

    def test_null_arguments_and_target_do_not_crash(self):
        plan = {"subtask_plans": [
            {"steps": [{"skill_id": "navigate_to_object", "arguments": None}]},
            {"steps": [{"skill_id": "navigate_to_object", "arguments": {"target": None}}]}]}
        result = plan_semantics.audit(decomposition([goal("c1", "role_001")]),
                                      bindings={"role_001": {"element_ids": [1]}}, plan=plan)
        self.assertEqual(result["issue_codes"], ["final_navigation_target_unresolved"])
        self.assertFalse(result["has_defect"])

    def test_null_containers_and_wrong_types_are_tolerated(self):
        value = {"instruction": "Go to the chair.", "goal_state": [None, "x", goal("c1", "role_001")],
                 "subtasks": [None, {"description": None, "required_roles": None}],
                 "process_constraints": None}
        plan = {"subtask_plans": [None, {"steps": None}], "unplanned_subtasks": None}
        result = plan_semantics.audit(value, bindings=["not", "a", "dict"], plan=plan)
        self.assertTrue(result["passed"])
        self.assertEqual(result["terminal_location_goal_count"], 1)
        self.assertEqual(result["subtask_count"], 2)
        self.assertEqual(result["version"], "plan_semantics_audit_v2")
        self.assertTrue(plan_semantics.audit({"subtask_plans": None, "goal_state": None})["passed"])

    def test_unhashable_ids_from_malformed_output_are_unresolved(self):
        result = plan_semantics.audit(
            decomposition([goal("c1", "role_001")]),
            bindings={"role_001": {"element_ids": [{"id": 1}]}},
            plan=navigation_plan([("role_002", {"id": 1})]))
        self.assertEqual(result["defect_codes"], ["terminal_goal_not_final_navigation_target"])


if __name__ == "__main__":
    unittest.main()
