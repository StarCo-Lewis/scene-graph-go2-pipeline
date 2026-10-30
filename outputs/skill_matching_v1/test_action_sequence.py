import copy
import unittest

from action_sequence import refresh_sequence, validate_sequence, validate_subtask_dependencies
from match_skills import build_result, PlanError, read_json
from optimize_plan import candidate_plan
from run_multi_action_regression import fixtures, HERE
from verify_plan import review

LIBRARY = read_json(HERE.parent / "quadruped_atomic_skill_library.v0.1.json")


class ActionSequenceTests(unittest.TestCase):
    def build(self, index=0):
        _, alignment, proposal = list(fixtures())[index]
        return alignment, build_result(alignment, LIBRARY, proposal)

    def test_one_subtask_many_actions(self):
        _, result = self.build()
        self.assertEqual(result["action_count"], 7)
        self.assertEqual(result["subtask_action_mapping"][0]["action_count"], 7)
        self.assertEqual(result["action_sequence"][1]["depends_on"], ["st_01_01"])
        self.assertTrue(all(s["execution_status"] == "not_executed" for s in result["subtask_plans"][0]["steps"]))

    def test_cross_subtask_sequence_preserved(self):
        _, result = self.build(1)
        self.assertEqual([g["action_count"] for g in result["subtask_action_mapping"]], [2, 2, 3])
        self.assertEqual(result["subtask_plans"][1]["dependencies"], ["st_01"])
        self.assertEqual(result["action_sequence"][2]["depends_on"], ["st_01_02"])

    def test_unknown_forward_self_and_duplicate_dependencies_rejected(self):
        for deps in (["missing"], ["b"], ["a"], ["a", "a"]):
            with self.subTest(deps=deps), self.assertRaises(ValueError):
                validate_subtask_dependencies([{"subtask_id": "a", "dependencies": deps}, {"subtask_id": "b"}])

    def test_duplicate_subtask_ids_rejected_in_matcher(self):
        _, alignment, proposal = list(fixtures())[1]
        alignment["task_decomposition"]["subtasks"][1]["subtask_id"] = "st_01"
        with self.assertRaises(PlanError):
            build_result(alignment, LIBRARY, proposal)

    def test_stale_sequence_rejected_by_symbolic_review(self):
        alignment, plan = self.build()
        plan["action_sequence"].reverse()
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            review(alignment, plan, LIBRARY, [], [])

    def test_return_visit_protected_and_reorder_rejected(self):
        _, result = self.build(2)
        ids = [s["step_id"] for s in result["subtask_plans"][0]["steps"]]
        with self.assertRaisesRegex(ValueError, "intent protected"):
            candidate_plan(result, {"ordered_step_ids": ids[:2]})
        with self.assertRaisesRegex(ValueError, "order may not change"):
            candidate_plan(result, {"ordered_step_ids": list(reversed(ids))})

    def test_local_duplicate_navigation_removal_rebuilds_index(self):
        _, result = self.build(2)
        duplicate = copy.deepcopy(result["subtask_plans"][0]["steps"][0])
        duplicate["step_id"] = "duplicate"
        result["subtask_plans"][0]["steps"].insert(1, duplicate)
        result = refresh_sequence(result)
        ids = [r["step_id"] for r in result["action_sequence"] if r["step_id"] != "duplicate"]
        candidate = candidate_plan(result, {"ordered_step_ids": ids})
        self.assertEqual(candidate["action_count"], 3)
        self.assertEqual(candidate["action_sequence"][1]["depends_on"], [ids[0]])
        validate_sequence(candidate)

    def test_adjacent_calls_across_subtasks_are_protected(self):
        _, result = self.build(2)
        step = copy.deepcopy(result["subtask_plans"][0]["steps"][-1])
        step["step_id"] = "round_two_visit"
        result["subtask_plans"].append({"subtask_id": "st_02", "dependencies": ["st_01"], "steps": [step]})
        result = refresh_sequence(result)
        with self.assertRaisesRegex(ValueError, "intent protected"):
            candidate_plan(result, {"ordered_step_ids": [s["step_id"] for s in result["action_sequence"][:-1]]})

    def test_unresolved_subtask_is_not_silently_dropped(self):
        _, alignment, proposal = list(fixtures())[1]
        proposal["subtask_plans"][1]["steps"] = []
        result = build_result(alignment, LIBRARY, proposal)
        self.assertEqual(result["status"], "unresolved")
        self.assertEqual(result["unplanned_subtasks"], ["st_02"])


if __name__ == "__main__":
    unittest.main()
