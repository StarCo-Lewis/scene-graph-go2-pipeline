import importlib.util
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).parent
SPEC = importlib.util.spec_from_file_location("run_navigation_demo", HERE / "run_navigation_demo.py")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class NavigationDemoTests(unittest.TestCase):
    def test_coordinate_modes(self):
        self.assertEqual(MODULE.cg_to_habitat([-2.7, 8.0, 1.4], "x_z_minus_y"),
                         [-2.7, 1.4, -8.0])
        self.assertEqual(MODULE.cg_to_habitat([-2.7, 8.0, 1.4], "identity"),
                         [-2.7, 8.0, 1.4])

    def test_targets_follow_current_alignment_not_hardcoded_ids(self):
        alignment = {"context_bindings": {
            "kettle": {"element_ids": [67]}, "sideboard": {"element_ids": [13]}}}
        plan = {"subtask_plans": [{"steps": [
            {"step_id": "a", "skill_id": "navigate_to_object", "arguments": {
                "target": {"role_id": "kettle", "element_id": 67}}},
            {"step_id": "b", "skill_id": "grasp_object", "arguments": {}},
            {"step_id": "c", "skill_id": "navigate_to_object", "arguments": {
                "target": {"role_id": "sideboard", "element_id": 13}}},
        ]}]}
        nodes = [{"id": 67, "bbox_center": [-2.7, 8.0, 1.4]},
                 {"id": 13, "bbox_center": [-5.8, 1.5, 0.5]}]
        self.assertEqual([x["element_id"] for x in MODULE.targets(alignment, plan, nodes)], [67, 13])

    def test_rejects_ungrounded_target(self):
        alignment = {"context_bindings": {"sideboard": {"element_ids": [13]}}}
        plan = {"subtask_plans": [{"steps": [{"step_id": "x", "skill_id": "navigate_to_object",
                   "arguments": {"target": {"role_id": "sideboard", "element_id": 113}}}]}]}
        with self.assertRaisesRegex(ValueError, "ungrounded"):
            MODULE.targets(alignment, plan, [{"id": 113, "bbox_center": [0, 0, 0]}])

    def test_snap_must_remain_near_mapped_target(self):
        class FarPathfinder:
            def snap_point(self, point):
                return [100.0, 0.0, 100.0]

        with self.assertRaisesRegex(ValueError, "cannot be snapped"):
            MODULE.snap_targets(FarPathfinder(), [{"element_id": 67,
                "cg_center": [-2.7, 8.0, 1.4]}], "x_z_minus_y", 2.0)


if __name__ == "__main__":
    unittest.main()
