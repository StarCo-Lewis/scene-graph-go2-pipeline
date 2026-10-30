import unittest
from run_sideboard_full_pipeline import validate_adapter, validate_resume, validate_memory_target, INSTRUCTION
from pathlib import Path


class AdapterTests(unittest.TestCase):
    def test_memory_target_must_match_updated_snapshot(self):
        node = {"id": 67, "bbox_center": [4, 5, 6]}
        state = {"memory_type": "incremental_object_state", "version": 1,
                 "entities": {"67": {"active": True, "node": node}},
                 "event_log": [{"event": {"object_id": 67, "operation": "move"},
                                "before": {"node": {"bbox_center": [1, 2, 3]}}}]}
        result = validate_memory_target(state, [node], 67)
        self.assertEqual(result["current_bbox_center"], [4, 5, 6])
        with self.assertRaises(ValueError):
            validate_memory_target(state, [{"id": 67, "bbox_center": [1, 2, 3]}], 67)
        state["entities"]["67"]["active"] = False
        with self.assertRaisesRegex(ValueError, "NO object"):
            validate_memory_target(state, [node], 67)

    def test_resume_validates_instruction_scene_and_goal(self):
        scene = Path("test_scene")
        decomposition = {"instruction": INSTRUCTION, "goal_state": [{"subject_role": "robot"}]}
        alignment = {"status": "success", "scene_reference": str(scene),
                     "task_decomposition": decomposition}
        validate_resume(alignment, decomposition, scene)
        with self.assertRaises(ValueError):
            validate_resume(alignment, decomposition, Path("wrong_scene"))
        decomposition["goal_state"] = []
        with self.assertRaises(ValueError):
            validate_resume(alignment, decomposition, scene)

    def setUp(self):
        self.a = {"status": "success", "context_bindings": {"destination": {"element_ids": [13]}}}
        self.p = {"instruction": INSTRUCTION, "subtask_plans": [{"steps": [
            {"step_id": "nav_01", "skill_id": "navigate_to_object",
             "arguments": {"target": {"role_id": "destination", "element_id": 13}}}]}]}
        self.n = [{"id": 13, "bbox_center": [0, 1, 2]}]

    def test_dynamic_llm_role_name_allowed(self):
        self.assertEqual(validate_adapter(self.a, self.p, self.n)["element_id"], 13)

    def test_unsupported_action_not_silently_ignored(self):
        self.p["subtask_plans"][0]["steps"].append({"skill_id": "grasp_object"})
        with self.assertRaises(ValueError):
            validate_adapter(self.a, self.p, self.n)

    def test_missing_object_rejected(self):
        with self.assertRaises(ValueError):
            validate_adapter(self.a, self.p, [])

    def test_old_task_instruction_rejected(self):
        self.p["instruction"] = "Move the kettle."
        with self.assertRaises(ValueError):
            validate_adapter(self.a, self.p, self.n)
