"""Local Tcl variables only: no network, API, WSL or robot dispatch."""
import tkinter as tk
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from embodied_pipeline._legacy import ROOT
from embodied_pipeline.execution.navigation_contract import sha, CONFIRMATIONS
from embodied_pipeline.execution.navigation_points import bbox_distance
from embodied_pipeline.ui.field_backend import load_scene, DEFAULT_SCENE
from embodied_pipeline.ui.navigation_controls import NavigationPanel
from embodied_pipeline.execution.navigation_session import NavigationSession


class AutoMappingTests(unittest.TestCase):
    def setUp(self):
        # UI state-machine tests use an explicitly synthetic empty occupancy map.
        # Real obstacle rejection is covered separately in test_navigation_free_space.
        from embodied_pipeline.tests.test_navigation_free_space import empty_grid
        self.enterContext(patch('embodied_pipeline.ui.navigation_controls.scene_grid', return_value=empty_grid()))
        self.tcl = tk.Tcl()
        p = self.panel = object.__new__(NavigationPanel)
        scene = load_scene(DEFAULT_SCENE)
        path = ROOT / "outputs/real_scene_runs/d3_619_plan_20261022_132214_282023/engineering_plan.json"
        p.console = SimpleNamespace(scene=scene, loaded_plan_path=path, loaded_plan_hash=sha(path),
                                    api=SimpleNamespace(busy=False), log=Mock(),
                                    selected=Mock(side_effect=AssertionError("List selection must never be used")))
        p.active = p.uncertain = p.auto_suppressed = p.setting_goal = False
        p.current_request_id=None
        p.request=None
        p.outcomes={}
        p.clearance_m=.4
        p.clearance_input=tk.StringVar(master=self.tcl,value='.4')
        p.outcome_state=tk.StringVar(master=self.tcl)
        p.queue_running = p.setting_start = p.sequence_complete = False
        p.step_index = 0
        p.sequence_key = None
        p.navigation_session = NavigationSession()
        p.segment_state = tk.StringVar(master=self.tcl)
        p.auto_goal, p.auto_key = False, None
        p.start = {k: tk.StringVar(master=self.tcl) for k in ("x", "y", "yaw")}
        p.goal = {k: tk.StringVar(master=self.tcl) for k in ("x", "y", "yaw")}
        p.confirmations = {k: tk.BooleanVar(master=self.tcl, value=False) for k in CONFIRMATIONS}
        p.hardware = tk.BooleanVar(master=self.tcl, value=False)
        p.mapping_note = tk.StringVar(master=self.tcl)
        for var in p.start.values():
            var.trace_add("write", p.start_changed)
        for var in p.goal.values():
            var.trace_add("write", p.goal_changed)
        self.node = next(n for n in scene["nodes"] if n["id"] == 179)

    def fill_start(self):
        # Synthetic offline pose, not a site recommendation.
        self.panel.start["x"].set(str(self.node["bbox_center"][0]-self.node["bbox_extent"][0]-2))
        self.panel.start["y"].set(str(self.node["bbox_center"][1]))
        self.panel.start["yaw"].set("0")

    def test_empty_start_not_zero_filled(self):
        self.panel.maybe_auto_map()
        self.assertTrue(all(v.get() == "" for v in self.panel.goal.values()))
        self.assertTrue(all(v.get() == "" for v in self.panel.start.values()))

    def test_auto_fills_unconfirmed_point_not_list_target(self):
        self.fill_start()
        self.panel.maybe_auto_map()
        point = {k: float(v.get()) for k, v in self.panel.goal.items()}
        self.assertAlmostEqual(bbox_distance(self.node, point), .4)
        self.assertFalse(any(v.get() for v in self.panel.confirmations.values()))
        self.panel.console.selected.assert_not_called()
        self.panel.maybe_auto_map()
        self.panel.console.log.assert_called_once()

    def test_busy_or_failed_planning_does_not_map(self):
        self.fill_start()
        self.panel.console.api.busy = True
        self.panel.maybe_auto_map()
        self.assertEqual(self.panel.goal["x"].get(), "")
        self.panel.console.api.busy = False
        with patch("embodied_pipeline.ui.navigation_controls.read", return_value={"status": "planning_output_requires_review"}):
            self.panel.maybe_auto_map()
        self.assertEqual(self.panel.goal["x"].get(), "")

    def test_start_changes_clear_old_auto_point_and_confirmation(self):
        self.fill_start()
        self.panel.maybe_auto_map()
        self.panel.confirmations["approach_pose_verified"].set(True)
        self.panel.start["x"].set(str(self.node["bbox_center"][0]+self.node["bbox_extent"][0]+2))
        self.assertEqual(self.panel.goal["x"].get(), "")
        self.assertFalse(self.panel.confirmations["approach_pose_verified"].get())
        self.panel.maybe_auto_map()
        self.assertGreater(float(self.panel.goal["x"].get()), self.node["bbox_center"][0])

    def test_manual_edit_never_overwritten(self):
        self.fill_start()
        self.panel.maybe_auto_map()
        self.panel.goal["x"].set("123")
        self.panel.start["x"].set("100")
        self.panel.maybe_auto_map()
        self.assertEqual(self.panel.goal["x"].get(), "123")
        self.assertTrue(self.panel.auto_suppressed)

    def test_new_plan_preserves_entered_start_but_terminal_clears_it(self):
        self.fill_start()
        before = {k: v.get() for k, v in self.panel.start.items()}
        self.panel.reset_for_plan()
        self.assertEqual({k: v.get() for k, v in self.panel.start.items()}, before)
        self.panel.maybe_auto_map()
        self.assertNotEqual(self.panel.goal["x"].get(), "")
        self.panel.reset_confirmations()
        self.assertTrue(all(v.get() == "" for v in self.panel.start.values()))
        self.assertTrue(all(v.get() == "" for v in self.panel.goal.values()))


if __name__ == "__main__":
    unittest.main()
