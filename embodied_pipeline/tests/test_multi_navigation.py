"""Offline multi-goal tests with Tcl variables and mocked transport; no robot/API."""
import ast
import copy
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from embodied_pipeline._legacy import load
from embodied_pipeline.execution.navigation_session import arrived_pose, terminal_pose, NavigationSession
from embodied_pipeline.execution.navigation_contract import read, sha, prepare, CONFIRMATIONS
from embodied_pipeline.execution.tcp_protocol import FIELD_LIMITS
from embodied_pipeline.ui.navigation_controls import build_config
from embodied_pipeline.ui.navigation_controls import candidate_for_plan
from embodied_pipeline.deployment.configure_first_alignment_only import patched_source
from embodied_pipeline.orchestration.real_scene_plan import check_navigation, INSTRUCTION_SCHEMA
from embodied_pipeline.tests.test_auto_navigation import AutoMappingTests


def terminal(x=2.03, y=-1.02, yaw=.5):
    return {"ros_goal_status": 4, "result": {"success": True, "reason": "arrived",
        "final_pose": {"header": {"frame_id": "scene"}, "pose": {
            "position": {"x": x, "y": y, "z": 0},
            "orientation": {"x": 0, "y": 0, "z": math.sin(yaw/2), "w": math.cos(yaw/2)}}}}}


class SessionTests(unittest.TestCase):
    def test_actual_pose_not_target_or_origin(self):
        session = NavigationSession(); key = ("graph", "frame")
        initial = {"x": 0., "y": 0., "yaw": 0.}
        session.sent(initial, key)
        first = session.finish(terminal(), "A", key)
        self.assertEqual(first["x"], 2.03)
        self.assertEqual(session.provenance(first)["start_pose_source"], "previous_server_final_pose")
        session.sent(first, key)
        session.finish(terminal(-3, 4, -.7), "B", key)
        self.assertEqual(session.initial_pose, initial)
        self.assertEqual(session.completed, 2)
        self.assertEqual(session.current_pose["x"], -3)

    def test_invalid_or_failed_terminal_has_no_pose(self):
        cases = []
        for status in (5, 6):
            d = terminal(); d["ros_goal_status"] = status; cases.append(d)
        for field, val in (("success", False), ("reason", "blocked"), ("final_pose", None)):
            d = terminal(); d["result"][field] = val; cases.append(d)
        d = terminal(); d["result"]["final_pose"]["header"]["frame_id"] = "odom"; cases.append(d)
        d = terminal(); d["result"]["final_pose"]["pose"]["position"]["x"] = float("nan"); cases.append(d)
        d = terminal(); d["result"]["final_pose"]["pose"]["orientation"]["w"] = 0; cases.append(d)
        for data in cases:
            self.assertIsNone(arrived_pose(data))

    def test_scene_and_connection_invalidation(self):
        s = NavigationSession(); s.sent({"x": 0, "y": 0, "yaw": 0}, ("a", "b"))
        self.assertIsNone(s.finish(terminal(), "A", ("other", "b")))
        s.finish(terminal(), "A", ("a", "b")); s.invalidate_pose()
        self.assertIsNone(s.current_pose)
        self.assertIsNotNone(s.initial_pose)

    def test_failure_and_cancel_carry_pose_but_do_not_count_arrivals(self):
        for status, reason in ((6, "position_out_of_tol"), (5, "cancelled"), (6, "blocked")):
            s = NavigationSession(); key = ("graph", "scene")
            s.sent({"x": 0, "y": 0, "yaw": 0}, key)
            d = terminal(); d["ros_goal_status"] = status
            d["result"].update(success=False, reason=reason)
            self.assertIsNone(arrived_pose(d))
            self.assertEqual(s.finish(d, "A", key)["x"], 2.03)
            self.assertEqual(s.completed, 0)
            self.assertEqual(s.provenance(s.current_pose)["start_pose_source"], "previous_server_final_pose")

    def test_no_pose_or_nonterminal_cannot_supply_start(self):
        d = terminal(); d["result"]["reason"] = "no_pose"
        self.assertIsNone(terminal_pose(d))
        d = terminal(); d["ros_goal_status"] = 2
        self.assertIsNone(terminal_pose(d))


class PanelTests(unittest.TestCase):
    def setUp(self):
        AutoMappingTests.setUp(self)
        p = self.panel
        p.run_dir = None
        p.queue_route = []; p.queue_next_deadline = p.queue_ready_at = None
        p.dispatch_scene_key = p.scene_key()
        p.request = {"coordinate_frame": p.console.scene["metadata"]["coordinate_frame"], "navigation_step_count": 2}
        p.navigation_session.sent({"x": 0., "y": 0., "yaw": 0.}, p.scene_key())
        p.connection_id = "connection"
        p.client = SimpleNamespace(closed=SimpleNamespace(is_set=lambda: False), status={"busy": False})

    def row(self, data=None):
        return {"type": "result", "request_id": "A", "coordinate_frame": self.panel.request["coordinate_frame"],
                "data": data if data is not None else terminal()}

    def arm_queue(self):
        p = self.panel
        p.queue_route = [{"step_index": 0, "goal": {"x": 2., "y": -1., "yaw": .5}},
                         {"step_index": 1, "goal": {"x": 5., "y": 2., "yaw": 0.}}]
        p.queue_running = True
        p.queue_key = p.mapping_plan_key(); p.queue_connection = p.connection_id
        p.queue_confirmations = dict.fromkeys(CONFIRMATIONS, True)

    def test_system_one_carries_pose_without_dispatch(self):
        p = self.panel; p.dispatch = Mock()
        p.request["navigation_step_count"] = 1
        p.finish_segment(self.row())
        self.assertEqual(float(p.start["x"].get()), 2.03)
        self.assertTrue(p.sequence_complete)
        p.console.loaded_plan_hash = "next-plan"
        p.reset_for_plan()
        self.assertEqual(float(p.start["x"].get()), 2.03)
        self.assertEqual(p.step_index, 0)
        self.assertFalse(p.sequence_complete)
        p.dispatch.assert_not_called()

    def test_failed_goal_keeps_actual_pose_through_replanning_without_queue_advance(self):
        p = self.panel; self.arm_queue(); p.dispatch = Mock()
        data = terminal(-1.6034813814278637, -.018862912640625895, 2.95639058)
        data["ros_goal_status"] = 6
        data["result"].update(success=False, reason="position_out_of_tol")
        p.finish_segment(self.row(data))
        self.assertFalse(p.queue_running)
        self.assertEqual(p.step_index, 0)
        self.assertEqual(p.navigation_session.completed, 0)
        p.console.loaded_plan_hash = "new-sofa-plan"
        p.reset_for_plan()
        self.assertAlmostEqual(float(p.start["x"].get()), -1.6034813814278637)
        self.assertAlmostEqual(float(p.start["y"].get()), -.018862912640625895)
        self.assertFalse(any(v.get() for v in p.confirmations.values()))
        p.continue_queue(); p.dispatch.assert_not_called()

    def test_system_two_waits_ten_seconds_then_dispatches_exactly_once(self):
        p = self.panel; self.arm_queue(); p.dispatch = Mock()
        with patch("embodied_pipeline.ui.navigation_controls.time.monotonic", return_value=100):
            p.finish_segment(self.row()); p.continue_queue()
        self.assertEqual(float(p.start["x"].get()), 2.03)
        self.assertEqual(float(p.goal["x"].get()), 5.)
        self.assertEqual(p.queue_ready_at, 110)
        with patch("embodied_pipeline.ui.navigation_controls.time.monotonic", return_value=109.999):
            p.continue_queue()
        p.dispatch.assert_not_called()
        with patch("embodied_pipeline.ui.navigation_controls.time.monotonic", return_value=110):
            p.continue_queue(); p.continue_queue()
        p.dispatch.assert_called_once_with(queue_authorized=True)

    def test_cancel_during_dwell_stops_next_goal(self):
        p = self.panel; self.arm_queue(); p.dispatch = Mock(); p.client.cancel = Mock()
        p.state = p.segment_state
        p.finish_segment(self.row()); p.cancel(); p.continue_queue()
        p.dispatch.assert_not_called(); self.assertFalse(p.queue_running)

    def test_failure_and_missing_pose_stop_queue(self):
        for data in (terminal(), terminal()):
            self.setUp(); self.arm_queue(); p = self.panel
            data["result"]["final_pose"] = None
            p.finish_segment(self.row(data))
            self.assertFalse(p.queue_running)
            self.assertEqual(p.start["x"].get(), "")
        self.setUp(); self.arm_queue(); p = self.panel
        data = terminal(); data["ros_goal_status"] = 6; data["result"]["success"] = False
        p.finish_segment(self.row(data))
        self.assertEqual(p.step_index, 0); self.assertFalse(p.queue_running)

    def test_manual_edit_and_changed_connection_disarm_queue(self):
        p = self.panel; self.arm_queue(); p.start["x"].set("0")
        self.assertFalse(p.queue_running)
        self.arm_queue(); p.finish_segment(self.row()); p.connection_id = "other"
        p.dispatch = Mock(); p.continue_queue()
        self.assertFalse(p.queue_running); p.dispatch.assert_not_called()

    def test_second_goal_contract_preserves_limits_and_actual_start(self):
        p = self.panel
        original = read(p.console.loaded_plan_path)
        data = copy.deepcopy(original)
        group = data["plan"]["subtask_plans"][0]
        second = copy.deepcopy(group["steps"][0]); second["step_id"] += "_second"
        second["arguments"]["target"]["element_id"] = 189
        group["steps"].append(second)
        data["plan"] = load("action_sequence").refresh_sequence(data["plan"])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"engineering_plan.json"; path.write_text(json.dumps(data))
            (Path(folder)/"planning_summary.json").write_text(json.dumps({"status": "planning_json_generated_not_executed", "input_files_unchanged": True}))
            start = arrived_pose(terminal())
            node = next(n for n in p.console.scene["nodes"] if n["id"] == 189)
            from embodied_pipeline.execution.navigation_points import bbox_approach
            goal = bbox_approach(node, start)
            cfg = build_config(path, sha(path), p.console.scene, start, goal, dict.fromkeys(CONFIRMATIONS, True), 1)
            request = prepare(cfg)
            self.assertEqual(request["blockers"], [])
            self.assertEqual(request["goal"]["s_x"], 2.03)
            self.assertEqual(request["target_element_id"], 189)
            self.assertEqual({k: request["goal"][k] for k in FIELD_LIMITS}, FIELD_LIMITS)

    def test_authorized_queue_sends_a_then_b_using_actual_a_result(self):
        p = self.panel
        data = read(p.console.loaded_plan_path)
        group = data["plan"]["subtask_plans"][0]
        second = copy.deepcopy(group["steps"][0]); second["step_id"] += "_second"
        second["arguments"]["target"]["element_id"] = 189; group["steps"].append(second)
        data["plan"] = load("action_sequence").refresh_sequence(data["plan"])
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder); path = folder/"engineering_plan.json"; path.write_text(json.dumps(data))
            (folder/"planning_summary.json").write_text(json.dumps({"status": "planning_json_generated_not_executed", "input_files_unchanged": True}))
            p.console.loaded_plan_path, p.console.loaded_plan_hash = path, sha(path)
            p.console.set_text = Mock(); p.preview_box = Mock(); p.state = p.segment_state
            p.navigation_session = NavigationSession()
            AutoMappingTests.fill_start(self)
            start = {k: float(v.get()) for k, v in p.start.items()}
            route = []; candidate_start = start
            for index in (0, 1):
                target, goal = candidate_for_plan(path, sha(path), p.console.scene, candidate_start, index)
                route.append({"step_index": index, "target_element_id": target, "goal": goal}); candidate_start = goal
            for var in p.confirmations.values():var.set(True)
            p.hardware.set(True)
            p.client.status = {"coordinate_frame": p.scene_key()[1], "busy": False, "motion_enabled": True, "action_available": True}
            p.client.navigate = Mock(side_effect=lambda request, request_id: request_id)
            pending = folder/"pending.json"
            with patch("embodied_pipeline.ui.navigation_controls.ROOT", folder), patch("embodied_pipeline.ui.navigation_controls.PENDING", pending), patch("embodied_pipeline.ui.navigation_controls.messagebox.askyesno", return_value=True):
                p.start_queue(route)
                self.assertEqual(p.client.navigate.call_count, 1)
                self.assertEqual(p.client.navigate.call_args.args[0]["goal"]["s_x"], start["x"])
                pending.unlink(); p.active = p.uncertain = False
                actual = {**route[0]["goal"], "x": route[0]["goal"]["x"]+.03}
                with patch("embodied_pipeline.ui.navigation_controls.time.monotonic", return_value=100):
                    p.finish_segment(self.row(terminal(**actual))); p.continue_queue()
                self.assertEqual(p.client.navigate.call_count, 1)
                with patch("embodied_pipeline.ui.navigation_controls.time.monotonic", return_value=110):
                    p.continue_queue()
                self.assertEqual(p.client.navigate.call_count, 2)
                goal = p.client.navigate.call_args.args[0]["goal"]
                self.assertEqual(goal["s_x"], actual["x"])
                self.assertEqual(goal["g_x"], route[1]["goal"]["x"])
                self.assertEqual({k: goal[k] for k in FIELD_LIMITS}, FIELD_LIMITS)
                pending.unlink(); p.active = p.uncertain = False
                p.finish_segment(self.row(terminal(**route[1]["goal"]))); p.continue_queue()
                self.assertTrue(p.sequence_complete); self.assertFalse(p.queue_running)
                self.assertEqual(p.client.navigate.call_count, 2)


class PlannerAndServerTests(unittest.TestCase):
    def test_multi_plan_check_keeps_order_and_bindings(self):
        steps = [{"step_id": str(i), "skill_id": "navigate_to_object", "arguments": {"target": {"element_id": i, "role_id": str(i)}}} for i in (1, 2, 1)]
        plan = {"subtask_plans": [{"steps": steps}]}
        bundle = {"nodes": [{"id": 1}, {"id": 2}]}
        alignment = {"context_bindings": {str(i): {"element_ids": [i]} for i in (1, 2)}}
        task = {"schema_version": INSTRUCTION_SCHEMA, "navigation_mode": "ordered_goals"}
        checked = check_navigation(plan, task, bundle, alignment)
        self.assertTrue(checked["passed"])
        self.assertEqual(checked["bound_target_element_ids"], [1, 2, 1])
        task["navigation_mode"] = "single_goal"
        self.assertFalse(check_navigation(plan, task, bundle, alignment)["passed"])

    def test_server_patch_preserves_first_alignment_and_disables_realign(self):
        source = '''class RosActionServer:
    def _execute(self, start):
        return self.nav.ensure_aligned(start, realign_on_mismatch=self.realign_on_mismatch)
'''
        patched = patched_source(source)
        self.assertEqual(patched_source(patched), patched)
        scope = {}; exec(compile(patched, "test_server", "exec"), scope)
        server = scope["RosActionServer"](); server.nav = Mock(); server.realign_on_mismatch = True
        server._execute("first"); server._execute("second")
        self.assertEqual(server.nav.ensure_aligned.call_count, 2)
        for call in server.nav.ensure_aligned.call_args_list:
            self.assertIs(call.kwargs["realign_on_mismatch"], False)
        with self.assertRaises(ValueError):
            patched_source("class Unknown: pass")


if __name__ == "__main__":
    unittest.main()
