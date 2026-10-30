"""Structure-only quality tests; these are not model or robot experiments."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

from embodied_pipeline.data_collection import quality


class QualityTests(unittest.TestCase):
    def setUp(self):
        self.nodes = [{"id": 1}, {"id": 2}, {"id": 3}]
        self.task = {"record_id": "scene200_00000001", "scene_id": "scene200",
            "instruction": "Move object ID 1 to surface ID 2, then visit ID 3.",
            "grounded_object_ids": [1, 2, 3],
            "initial_state": {"held_object_id": None, "support_relations": []},
            "constraints": {"preserve_objects": []},
            "goal": {"final_predicates": [{"predicate": "on", "object_id": 1, "support_id": 2}],
                     "ordered_milestones": [{"predicate": "visit", "object_id": 3}]}}
        self.reference = {"steps": [
            {"step_id": "r01", "intent": "deliver", "object_id": 1, "destination_id": 2, "depends_on": []},
            {"step_id": "r02", "intent": "visit", "object_id": 3, "destination_id": None, "depends_on": ["r01"]}],
            "dependencies": [["r01", "r02"]]}

    def check(self, **kwargs):
        return quality.validate_record(self.task, self.reference, self.nodes, **kwargs)

    def test_complete_record_not_claimed_semantically_valid(self):
        result = self.check()
        self.assertTrue(result["format_valid"])
        self.assertTrue(result["reference_valid"])
        self.assertEqual(result["semantic_evaluation"], "not_performed")
        self.assertIsNone(result["valid_candidate"])

    def test_id_requires_exact_scene_and_eight_digits(self):
        for bad in ("scene200_0000001", "scene201_00000001", "scene200_000000001", "scene200_abcdefgh"):
            self.task["record_id"] = bad
            self.assertFalse(self.check()["format_valid"])

    def test_graph_references_in_task_reference_and_plan(self):
        self.task["grounded_object_ids"] = [1, 999]
        self.reference["steps"][0]["object_id"] = 998
        plan = {"subtask_plans": [{"steps": [{"arguments": {"target": {"element_id": 997}}}]}]}
        codes = {row["code"] for row in self.check(skill_plan=plan)["reference_issues"]}
        self.assertIn("unknown_task_object_reference", codes)
        self.assertIn("unknown_reference_object", codes)
        self.assertIn("unknown_plan_object_reference", codes)

    def test_dependency_cycle_forward_and_unknown_ref(self):
        decomposition = {"subtasks": [
            {"subtask_id": "s1", "dependencies": ["s2", "absent"]},
            {"subtask_id": "s2", "dependencies": ["s1"]}]}
        codes = {row["code"] for row in self.check(decomposition=decomposition)["reference_issues"]}
        self.assertIn("dependency_cycle", codes)
        self.assertIn("dependency_not_previous_step", codes)
        self.assertIn("unknown_dependency_reference", codes)

    def test_reference_depends_on_checked_not_only_external_edges(self):
        self.reference["dependencies"] = []
        self.reference["steps"][0]["depends_on"] = ["r02"]
        codes = {row["code"] for row in self.check()["reference_issues"]}
        self.assertIn("dependency_cycle", codes)

    def test_decoded_files_and_invalid_json(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "decomposition.json"
            path.write_text(json.dumps({"subtasks": [{"subtask_id": "s1", "dependencies": []}]}), encoding="utf-8")
            self.assertTrue(self.check(decomposition=path)["reference_valid"])
            path.write_text("not JSON", encoding="utf-8")
            self.assertFalse(self.check(decomposition=path)["format_valid"])

    def test_paraphrases_group_without_instruction_or_id(self):
        second = copy.deepcopy(self.task)
        second.update(instruction="Please deliver item 1 to 2 and next approach 3.", record_id="scene200_00000002")
        self.assertEqual(quality.near_duplicate_key(self.task, self.reference),
                         quality.near_duplicate_key(second, self.reference))
        refs = {self.task["record_id"]: self.reference, second["record_id"]: self.reference}
        groups = quality.duplicate_groups([self.task, second], refs)
        self.assertEqual(len(groups), 1)
        self.assertTrue(groups[0]["requires_review"])

    def test_order_changes_signature_goal_conjunction_order_does_not(self):
        second = copy.deepcopy(self.task)
        extra = {"predicate": "held", "object_id": 1, "expected": False}
        self.task["goal"]["final_predicates"].append(extra)
        second["goal"]["final_predicates"].insert(0, extra)
        self.assertEqual(quality.near_duplicate_key(self.task, self.reference),
                         quality.near_duplicate_key(second, self.reference))
        reordered = copy.deepcopy(self.reference)
        reordered["steps"].reverse()
        self.assertNotEqual(quality.near_duplicate_key(self.task, self.reference),
                            quality.near_duplicate_key(self.task, reordered))

    def test_missing_signature_not_silently_grouped(self):
        with self.assertRaises(ValueError):
            quality.near_duplicate_key({"scene_id": "s"}, {})

    def test_no_answer_comparison_even_if_model_different_from_reference(self):
        # Existing node but a different action than the authored reference: this
        # helper must not label it incorrect; semantic review is a later phase.
        plan = {"subtask_plans": [{"steps": [{"skill_id": "navigate_to_object",
                                             "arguments": {"target": {"element_id": 1}}}]}]}
        self.assertTrue(self.check(skill_plan=plan)["reference_valid"])


if __name__ == "__main__":
    unittest.main()
