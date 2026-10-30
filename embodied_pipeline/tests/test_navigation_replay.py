"""Offline contract and fake-transport tests: never connect to a robot."""
import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from embodied_pipeline._legacy import ROOT
from embodied_pipeline.execution.navigation_contract import prepare, read, sha, CONFIRMATIONS, GOAL_TYPES
from embodied_pipeline.execution.ros2_navigation import run_transaction
from embodied_pipeline.orchestration.navigation_replay import clean, write


def configured():
    config = read(ROOT / "embodied_pipeline/execution/d3_619_navigation_replay.json")
    # Synthetic coordinates only. These are NOT measured D3-619 robot poses.
    config["start_pose"] = {"x": 1, "y": 2, "yaw": 0.3}
    config["approach_pose"] = {"x": 3, "y": 4, "yaw": -0.7}
    config["limits"] = {"pos_tol": .05, "yaw_tol": .1, "timeout": 30, "max_speed": .1}
    config["confirmations"] = {k: True for k in CONFIRMATIONS}
    return config


class ContractTests(unittest.TestCase):
    def test_default_is_blocked(self):
        config = read(ROOT / "embodied_pipeline/execution/d3_619_navigation_replay.json")
        report = prepare(config)
        self.assertIsNone(report["goal"])
        self.assertEqual(len(report["blockers"]), 12)
        self.assertFalse(report["robot_execution_attempted"])

    def test_goal_mapping_and_no_ros(self):
        result = prepare(configured())
        self.assertEqual(set(result["goal"]), set(GOAL_TYPES))
        self.assertEqual(result["goal"]["s_yaw"], .3)
        self.assertEqual(result["goal"]["s_z"], 0)
        self.assertEqual(result["goal"]["g_yaw"], -.7)
        self.assertEqual(result["goal"]["g_x"], 3)
        self.assertEqual(result["requested_standoff_m"], .6)
        self.assertEqual(result["goal"]["pos_tol"], .05)
        self.assertEqual(result["target_element_id"], 179)

    def test_source_hash_and_frame_fail_closed(self):
        for field, value in (("engineering_plan_sha256", "wrong"),
                             ("coordinate_frame", "odom"), ("angle_unit", "degree"),
                             ("target_element_id", 180), ("target_element_id", True)):
            with self.subTest(field=field):
                config = configured()
                config[field] = value
                with self.assertRaises(ValueError):
                    prepare(config)

    def test_invalid_coordinates_and_limits(self):
        for value in (None, True, "0", float("nan"), float("inf")):
            config = configured()
            config["start_pose"]["x"] = value
            self.assertIsNone(prepare(config)["goal"])
        for value in (0, -1, None, True, "0.1", 1e39):
            config = configured()
            config["limits"]["max_speed"] = value
            self.assertIsNone(prepare(config)["goal"])
        config = configured()
        config["start_pose"]["yaw"] = 90
        self.assertIsNone(prepare(config)["goal"])

    def test_confirmation_requires_boolean(self):
        config = configured()
        config["confirmations"]["operator_and_estop_ready"] = "true"
        self.assertIsNone(prepare(config)["goal"])

    def test_unsupported_and_multiple_steps_not_dropped(self):
        config = configured()
        envelope = read(ROOT / config["engineering_plan"])
        for mode in ("unsupported", "multiple"):
            data = copy.deepcopy(envelope)
            steps = data["plan"]["subtask_plans"][0]["steps"]
            if mode == "unsupported":
                steps[0]["skill_id"] = "grasp_object"
            else:
                steps.append(copy.deepcopy(steps[0]))
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "proposal.json"
                write(path, data)
                config["engineering_plan"] = str(path)
                config["engineering_plan_sha256"] = sha(path)
                with self.assertRaises(ValueError):
                    prepare(config)

    def test_nonfinite_feedback_is_unknown(self):
        self.assertEqual(clean({"error": float("nan")}), {"error": None})


class Future:
    def __init__(self, value=None, done=True):
        self.value, self.ready = value, done
    def done(self):
        return self.ready
    def result(self):
        return self.value


class Handle:
    def __init__(self, accepted=True, ready=True, success=True, reason="arrived", status=4):
        self.accepted = accepted
        self.future = Future(NS(status=status, result={"success": success, "reason": reason}), ready)
        self.cancel_count = 0
        self.finish_on_cancel = False
    def get_result_async(self):
        return self.future
    def cancel_goal_async(self):
        self.cancel_count += 1
        if self.finish_on_cancel:
            self.future.ready = True
            self.future.value = NS(status=5, result={"success": False, "reason": "aborted"})
        return Future(NS(goals_canceling=[1]))


class Client:
    def __init__(self, handle, acknowledged=True):
        self.pending, self.sends = Future(handle, acknowledged), 0
    def send_goal_async(self, goal, feedback_callback):
        self.sends += 1
        return self.pending


def transaction(client):
    return run_transaction(client, {}, lambda seconds: None, lambda *args: None,
                           lambda value: value, 0, ack_timeout=0, cancel_timeout=0)


class TransportTests(unittest.TestCase):
    def test_arrival(self):
        self.assertEqual(transaction(Client(Handle()))["status"], "server_reported_arrival")
    def test_failure_never_retries(self):
        client = Client(Handle(success=False, reason="blocked", status=6))
        self.assertEqual(transaction(client)["status"], "server_reported_failure")
        self.assertEqual(client.sends, 1)
    def test_status_inconsistency_not_success(self):
        self.assertEqual(transaction(Client(Handle(status=6)))["status"], "server_reported_failure")
    def test_rejection(self):
        self.assertEqual(transaction(Client(Handle(accepted=False)))["status"], "goal_rejected")
    def test_cancel_ack_is_not_stop(self):
        handle = Handle(ready=False)
        result = transaction(Client(handle))
        self.assertTrue(result["cancel_acknowledged"])
        self.assertFalse(result["terminal_result_received"])
        self.assertEqual(result["status"], "execution_state_unknown_operator_check_required")
    def test_cancel_with_terminal(self):
        handle = Handle(ready=False)
        handle.finish_on_cancel = True
        result = transaction(Client(handle))
        self.assertTrue(result["terminal_result_received"])
        self.assertEqual(result["status"], "interrupted_with_server_terminal_result")
    def test_acceptance_timeout_never_resends(self):
        client = Client(Handle(), acknowledged=False)
        result = transaction(client)
        self.assertEqual(result["status"], "execution_state_unknown_operator_check_required")
        self.assertEqual(client.sends, 1)
    def test_late_acceptance_can_cancel(self):
        client = Client(Handle(ready=False), acknowledged=False)
        calls = iter([False, True, True, False])
        with patch("embodied_pipeline.execution.ros2_navigation.wait_future", side_effect=lambda *a: next(calls)):
            result = transaction(client)
        self.assertTrue(result["cancel_requested"])
        self.assertEqual(client.sends, 1)


if __name__ == "__main__":
    unittest.main()
