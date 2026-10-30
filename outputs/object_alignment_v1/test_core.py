#!/usr/bin/env python3
import importlib.util
import json
import sys
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("align_objects.py")
SPEC = importlib.util.spec_from_file_location("align_objects", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class CoreTests(unittest.TestCase):
    def test_example_decomposition_is_valid(self):
        value = json.loads(Path(__file__).with_name("example_decomposition.scene201.json").read_text(encoding="utf-8"))
        result = MODULE.validate_decomposition(value)
        self.assertEqual(len(result["task_roles"]), 4)

    def test_quantity_shortage_is_strict_error(self):
        roles = [{
            "role_id": "cups", "name": "cups", "quantity": 2,
            "required": True, "relation_hints": [],
        }]
        candidates = {"cups": [{"element_id": 7, "score": 0.8}]}
        bindings, errors = MODULE.top_down_prune(roles, candidates, [])
        self.assertEqual(bindings, {})
        self.assertEqual(errors[0]["reason"], "insufficient_scene_graph_candidates")

    def test_category_gate_rejects_high_scoring_wrong_class(self):
        desk = {"object_tag": "desk", "possible_tags": ["table", "furniture"]}
        kettle = {"object_tag": "electric kettle", "possible_tags": ["kettle", "jug"]}
        self.assertFalse(MODULE.category_matches(desk, ["cup", "tumbler"]))
        self.assertTrue(MODULE.category_matches(kettle, ["kettle", "water pitcher"]))

    def test_string_role_has_clear_validation_error(self):
        malformed = {
            "subtasks": [{
                "subtask_id": "st_01", "dependencies": [],
                "required_roles": ["cup"],
            }]
        }
        with self.assertRaisesRegex(MODULE.AlignmentError, "must be an object"):
            MODULE.validate_decomposition(malformed)

    def test_shared_role_queries_and_labels_are_merged(self):
        def role(query, labels):
            return {
                "role_id": "cup", "name": "cup", "semantic_query_en": query,
                "semantic_labels": labels, "quantity": 1, "required": True,
                "relation_hints": [],
            }
        value = {"subtasks": [
            {"subtask_id": "a", "dependencies": [], "required_roles": [role("a cup on a table", ["cup"])]},
            {"subtask_id": "b", "dependencies": ["a"], "required_roles": [role("a graspable drinking cup", ["drinking cup"])]},
        ], "initial_scene_constraints": [], "goal_state": []}
        result = MODULE.validate_decomposition(value)
        merged = result["task_roles"][0]
        self.assertEqual(len(merged["semantic_queries_en"]), 2)
        self.assertEqual(merged["semantic_labels"], ["cup", "drinking cup"])

    def test_error_contract_uses_exact_message(self):
        args = type("Args", (), {
            "top_k": 8, "min_score": 0.24, "min_metadata_score": 0.20,
            "visual_weight": 0.65, "metadata_weight": 0.35,
        })()
        result = MODULE.build_result(
            Path("scene"), {
                "instruction": "x", "initial_scene_constraints": [], "goal_state": []
            }, {}, {},
            [{"reason": "missing"}], args,
        )
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error"]["message"], "NO object can be used in this task")

    def test_example_is_ascii_serializable(self):
        value = json.loads(Path(__file__).with_name("example_decomposition.scene201.json").read_text(encoding="utf-8"))
        json.dumps(value, ensure_ascii=True).encode("ascii")


if __name__ == "__main__":
    unittest.main()
