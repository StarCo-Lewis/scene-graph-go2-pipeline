import json
import unittest
from pathlib import Path
from verify_plan import review

LIBRARY = json.loads((Path(__file__).parent.parent / "quadruped_atomic_skill_library.v0.1.json").read_text(encoding="utf-8"))


class NavigationGoalTests(unittest.TestCase):
    def setUp(self):
        self.a = {"status": "success", "context_bindings": {"sideboard": {"element_ids": [13]}},
            "task_decomposition": {"initial_scene_constraints": [], "goal_state": [
                {"constraint_id": "g1", "subject_role": "robot", "relation": "near",
                 "object_role": "sideboard", "required": True}]}}
        self.p = {"status": "proposal_needs_environment_validation", "subtask_plans": [{"steps": [
            {"step_id": "nav1", "skill_id": "navigate_to_object",
             "arguments": {"target": {"role_id": "sideboard", "element_id": 13}}}]}]}
        self.nodes = [{"id": 13, "bbox_center": [0, 0, 1]}]

    def test_navigation_covers_robot_near_goal_conditionally(self):
        result = review(self.a, self.p, LIBRARY, self.nodes, [])
        self.assertEqual(result["status"], "conditionally_valid_symbolic_plan")
        self.assertEqual(result["goal_skill_alignment"][0]["status"], "covered")
        self.assertEqual(result["goal_check"][0]["status"], "predicted_only")
        self.assertEqual(result["physical_feasibility"], "unknown")

    def test_no_navigation_does_not_cover_goal(self):
        self.p["subtask_plans"][0]["steps"] = []
        self.assertEqual(review(self.a, self.p, LIBRARY, self.nodes, [])["status"], "invalid_symbolic_plan")

    def test_empty_goal_still_invalid(self):
        self.a["task_decomposition"]["goal_state"] = []
        self.assertEqual(review(self.a, self.p, LIBRARY, self.nodes, [])["status"], "invalid_symbolic_plan")
