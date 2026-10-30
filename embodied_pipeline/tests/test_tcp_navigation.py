"""Offline tests: fake ROS driver and sockets; never connect to a robot."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import uuid

from embodied_pipeline.execution.navigation_points import bbox_approach, bbox_distance, validate_clearance
from embodied_pipeline.execution.tcp_protocol import PROTOCOL, GOAL_TYPES, FIELD_LIMITS, encode, decode, validate_goal
from embodied_pipeline.execution.tcp_bridge import BridgeCore, Journal
from embodied_pipeline.execution.tcp_client import NavigationClient
from embodied_pipeline.ui.navigation_controls import build_config, mapped_point
from embodied_pipeline.ui.field_backend import load_scene, DEFAULT_SCENE
from embodied_pipeline.execution.navigation_contract import sha, prepare, CONFIRMATIONS
from embodied_pipeline._legacy import ROOT


def packet():
    return {"protocol": PROTOCOL, "type": "navigate", "request_id": str(uuid.uuid4()),
            "operator_confirmed": True, "coordinate_frame": "test_scene", "source_action_id": "a1",
            "goal": {**{key: 0.0 for key in GOAL_TYPES}, **FIELD_LIMITS, "g_x": 2.0}}


class MappingTests(unittest.TestCase):
    node = {"bbox_center": [2, 3, 1], "bbox_extent": [2, 4, 1]}

    def test_full_extent_and_boundary_not_center(self):
        p = bbox_approach(self.node, {"x": -2, "y": 3, "yaw": 0})
        self.assertAlmostEqual(p["x"], .6)
        self.assertEqual(p["y"], 3)
        self.assertEqual(p["yaw"], 0)
        self.assertAlmostEqual(bbox_distance(self.node, p), .4)

    def test_all_sides_and_corners(self):
        for x, y in [(0, 3), (4, 3), (2, 0), (2, 6), (0, 0), (4, 6)]:
            with self.subTest(x=x, y=y):
                p = bbox_approach(self.node, {"x": x, "y": y, "yaw": 0})
                self.assertAlmostEqual(bbox_distance(self.node, p), .4)

    def test_inside_and_on_edge_rejected(self):
        for x, y in [(2, 3), (1, 3), (3, 5)]:
            with self.assertRaises(ValueError):
                bbox_approach(self.node, {"x": x, "y": y, "yaw": 0})

    def test_ground_region_inside_uses_nearest_face(self):
        floor = {**self.node, "object_tag": "floorboard"}
        for start, expected in [((1.1, 3), (.6, 3)), ((2.9, 3), (3.4, 3)),
                                ((2, 1.1), (2, .6)), ((2, 4.9), (2, 5.4)),
                                ((3, 3), (3.4, 3)), ((2, 3), (3.4, 3))]:
            with self.subTest(start=start):
                result = bbox_approach(floor, {"x": start[0], "y": start[1], "yaw": 0})
                self.assertAlmostEqual(result["x"], expected[0])
                self.assertAlmostEqual(result["y"], expected[1])
                self.assertAlmostEqual(validate_clearance(floor, result), .4)

    def test_real_floorboard_364_origin_regression(self):
        scene = load_scene(DEFAULT_SCENE)
        node = next(n for n in scene["nodes"] if n["id"] == 364)
        goal = bbox_approach(node, {"x": 0, "y": 0, "yaw": 0})
        self.assertAlmostEqual(goal["x"], .8310933649539948)
        self.assertEqual(goal["y"], 0)
        self.assertAlmostEqual(bbox_distance(node, goal), .4)

    def test_ground_region_outside_uses_original_mapping(self):
        start = {"x": -2, "y": 3, "yaw": 0}
        self.assertEqual(bbox_approach(self.node, start),
                         bbox_approach({**self.node, "object_tag": "floorboard"}, start))

    def test_floor_tag_does_not_make_unknown_or_furniture_interiors_valid(self):
        for tag in ("table", "wall", "wood panel", "floor lamp", ""):
            with self.assertRaises(ValueError):
                bbox_approach({**self.node, "object_tag": tag}, {"x": 2, "y": 3, "yaw": 0})

    def test_center_and_too_close_rejected(self):
        for p in [{"x": 2, "y": 3, "yaw": 0}, {"x": .8, "y": 3, "yaw": 0}]:
            with self.assertRaises(ValueError):
                validate_clearance(self.node, p)

    def test_bad_geometry_and_start(self):
        for v in [float("nan"), True, "2", None]:
            n = copy.deepcopy(self.node)
            n["bbox_extent"][0] = v
            with self.assertRaises(ValueError):
                bbox_approach(n, {"x": -2, "y": 3, "yaw": 0})
        with self.assertRaises(ValueError):
            bbox_approach(self.node, {"x": float("nan"), "y": 0, "yaw": 0})

    def test_existing_plan_mapping_preserves_source_and_limits(self):
        from unittest.mock import patch
        from embodied_pipeline.tests.test_navigation_free_space import empty_grid
        self.enterContext(patch('embodied_pipeline.ui.navigation_controls.scene_grid',return_value=empty_grid()))
        path = ROOT / "outputs/real_scene_runs/d3_619_plan_20261022_132214_282023/engineering_plan.json"
        scene = load_scene(DEFAULT_SCENE)
        digest = sha(path)
        node = next(n for n in scene["nodes"] if n["id"] == 179)
        # Synthetic offline start, NOT a suggested field pose.
        start = {"x": node["bbox_center"][0]-node["bbox_extent"][0]-2,
                 "y": node["bbox_center"][1], "yaw": 0}
        goal = bbox_approach(node, start)
        cfg = build_config(path, digest, scene, start, goal, dict.fromkeys(CONFIRMATIONS, True))
        result = prepare(cfg)
        self.assertEqual(result["blockers"], [])
        self.assertEqual({k: result["goal"][k] for k in FIELD_LIMITS}, FIELD_LIMITS)
        self.assertAlmostEqual(cfg["navigation_point_policy"]["actual_xy_reference_point_clearance_m"], .4)
        self.assertEqual(sha(path), digest)
        self.assertEqual(result["target_element_id"], 179)
        with self.assertRaises(ValueError):
            build_config(path, "changed", scene, start, goal, {})

    def test_mapping_file_identity(self):
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "points.json"
            scene = {"metadata": {"coordinate_frame": "test"}, "graph_hash": "hash"}
            table = {"schema_version": "navigation_points_v1", "coordinate_frame": "test",
                     "source_graph_sha256": "hash", "points": {"1": {"x": 3, "y": 2, "yaw": 0}}}
            f.write_text(json.dumps(table))
            self.assertEqual(mapped_point(f, scene, 1), table["points"]["1"])
            table["coordinate_frame"] = "wrong"
            f.write_text(json.dumps(table))
            with self.assertRaises(ValueError):
                mapped_point(f, scene, 1)


class ProtocolTests(unittest.TestCase):
    def test_operator_timeout_120_seconds(self):
        p = packet()
        self.assertEqual(FIELD_LIMITS["timeout"], 120.0)
        self.assertEqual(validate_goal(p)["timeout"], 120.0)
        p["goal"]["timeout"] = 10.0
        with self.assertRaises(ValueError):
            validate_goal(p)

    def test_exact_goal_pass_through(self):
        p = packet()
        self.assertIs(validate_goal(p), p["goal"])
        self.assertEqual(decode(encode(p)), p)

    def test_reject_parameter_changes_and_invalid_numbers(self):
        for key in FIELD_LIMITS:
            p = packet()
            p["goal"][key] *= 2
            with self.assertRaises(ValueError):
                validate_goal(p)
        for value in [True, "1", float("nan"), float("inf"), None]:
            p = packet()
            p["goal"]["g_x"] = value
            with self.assertRaises(ValueError):
                validate_goal(p)

    def test_auth_identity_and_fixed_pose_fields(self):
        for changes in [{"operator_confirmed": False}, {"request_id": "1"}, {"coordinate_frame": ""}]:
            with self.assertRaises(ValueError):
                validate_goal({**packet(), **changes})
        for key, value in [("g_z", 1), ("s_roll", 1), ("g_yaw", 4)]:
            p = packet()
            p["goal"][key] = value
            with self.assertRaises(ValueError):
                validate_goal(p)


class BridgeTests(unittest.TestCase):
    def setUp(self):
        self.driver = Mock()
        self.driver.ready.return_value = True
        self.journal = Mock()
        self.now = 0
        self.core = BridgeCore(self.driver, "test_scene", True, self.journal, lambda: self.now)
        self.sent = []
        self.p = packet()

    def start(self):
        self.core.navigate("owner", self.p, self.sent.append)
        return self.driver.start.call_args.args[1:]

    def test_reservation_and_unchanged_request(self):
        self.driver.start.side_effect = lambda *a: self.journal.reserve.assert_called_once_with(self.p)
        self.start()
        self.assertEqual(self.driver.start.call_args.args[0], self.p["goal"])
        with self.assertRaises(ValueError):
            self.core.navigate("owner", packet(), self.sent.append)

    def test_rejection_and_duplicate(self):
        accepted, _, _, _ = self.start()
        accepted(SimpleNamespace(accepted=False))
        self.assertIsNone(self.core.active)
        self.journal.finish.assert_called_once()
        with self.assertRaises(ValueError):
            self.core.navigate("owner", self.p, self.sent.append)

    def test_late_acceptance_after_disconnect_cancel_not_stop(self):
        accepted, _, result, _ = self.start()
        self.core.disconnected("owner")
        self.driver.cancel.assert_not_called()
        accepted(SimpleNamespace(accepted=True))
        self.driver.cancel.assert_called_once()
        self.driver.cancel.call_args.args[1](True)
        self.assertIsNotNone(self.core.active)
        self.journal.finish.assert_not_called()
        result(5, {"success": False, "reason": "cancelled"})
        self.assertIsNone(self.core.active)

    def test_old_goal_callbacks_do_not_affect_next_goal(self):
        accepted, feedback, result, error = self.start()
        accepted(SimpleNamespace(accepted=False))
        self.p = packet()
        self.start()
        count = len(self.sent)
        feedback({"stage": "old"})
        result(4, {"success": True})
        error(ValueError("old"))
        self.assertEqual(len(self.sent), count)
        self.assertEqual(self.core.active["id"], self.p["request_id"])

    def test_watchdog_retains_unknown_no_retry(self):
        accepted, _, _, _ = self.start()
        accepted(SimpleNamespace(accepted=True))
        self.now = 30
        self.core.tick()
        self.core.tick()
        self.driver.cancel.assert_called_once()
        self.driver.start.assert_called_once()
        self.journal.finish.assert_not_called()
        self.assertEqual(sum(x["type"] == "execution_state_unknown" for x in self.sent), 1)

    def test_transport_error_cannot_release(self):
        self.driver.start.side_effect = OSError("offline")
        self.start()
        self.assertIsNotNone(self.core.active)
        self.journal.finish.assert_not_called()

    def test_frame_readonly_unavailable_and_foreign_cancel(self):
        for field, value in [("coordinate_frame", "other"), ("operator_confirmed", False)]:
            with self.assertRaises(ValueError):
                self.core.navigate("owner", {**self.p, field: value}, self.sent.append)
        self.core.allow_motion = False
        with self.assertRaises(ValueError):
            self.start()
        self.core.allow_motion = True
        self.driver.ready.return_value = False
        with self.assertRaises(ValueError):
            self.start()
        self.driver.ready.return_value = True
        self.start()
        with self.assertRaises(ValueError):
            self.core.cancel("other")
        with self.assertRaises(ValueError):
            self.core.cancel("owner", request_id=str(uuid.uuid4()))

    def test_disk_journal_blocks_unresolved_restart(self):
        with tempfile.TemporaryDirectory() as d:
            journal = Journal(d)
            journal.reserve(self.p)
            with self.assertRaises(RuntimeError):
                Journal(d)
            journal.finish()
            Journal(d)


class ClientTests(unittest.TestCase):
    def setUp(self):
        self.events = []
        self.client = NavigationClient(self.events.append)
        self.client.sock = Mock()
        self.client.status = {"motion_enabled": True, "action_available": True, "busy": False, "coordinate_frame": "test_scene"}

    def test_send_failure_keeps_identity_no_retry(self):
        self.client.sock.sendall.side_effect = OSError("offline")
        p = packet()
        with self.assertRaises(OSError):
            self.client.navigate(p, p["request_id"])
        self.assertEqual(self.client.active, p["request_id"])
        self.client.sock.sendall.assert_called_once()
        self.assertTrue(self.client.closed.is_set())

    def test_return_id_even_if_instant_terminal(self):
        p = packet()
        self.client.send = lambda _: setattr(self.client, "active", None)
        self.assertEqual(self.client.navigate(p, p["request_id"]), p["request_id"])

    def test_split_frames_and_invalid_terminal(self):
        p = packet()
        self.client.active = p["request_id"]
        frame = encode({"protocol": PROTOCOL, "type": "result", "request_id": p["request_id"], "data": {"ros_goal_status": 2, "result": {}}})
        self.client.sock.recv.side_effect = [frame[:9], frame[9:]]
        self.client.receive()
        self.assertEqual(self.client.active, p["request_id"])
        self.assertEqual(self.events[-1]["type"], "connection_lost")

    def test_valid_terminal_and_foreign_id(self):
        p = packet()
        self.client.active = p["request_id"]
        rows = [{"protocol": PROTOCOL, "type": "result", "request_id": rid,
                 "data": {"ros_goal_status": 4, "result": {"success": True}}}
                for rid in [str(uuid.uuid4()), p["request_id"]]]
        self.client.sock.recv.side_effect = [b"".join(encode(r) for r in rows), b""]
        self.client.receive()
        self.assertIsNone(self.client.active)
        self.assertEqual(sum(x["type"] == "result" for x in self.events), 1)


if __name__ == "__main__":
    unittest.main()
