"""Native ROS transport contract tests using fake drivers only; no DDS/robot."""
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from embodied_pipeline.execution.ros2_client import Ros2NavigationClient
from embodied_pipeline.execution.tcp_protocol import FIELD_LIMITS
from embodied_pipeline.tests.test_tcp_navigation import packet


class DirectClientTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        env = patch.dict(os.environ, {"ROS_DOMAIN_ID": "0"})
        env.start()
        self.addCleanup(env.stop)
        self.driver = Mock()
        self.driver.ready.return_value = True
        self.factory = Mock(return_value=self.driver)
        self.events = []
        self.client = Ros2NavigationClient(self.events.append, Path(self.temp.name)/"runtime", self.factory)

    def connect(self):
        self.client.connect("test_scene", start_monitor=False)

    def test_connect_only_discovers_no_goal_and_no_token(self):
        self.connect()
        self.driver.start.assert_not_called()
        self.assertEqual(self.events[0]["transport"], "ros2_direct")
        self.assertEqual(self.events[0]["coordinate_frame_verification"], "operator_required_not_reported_by_action")
        self.assertFalse(self.client.closed.is_set())
        self.assertFalse(self.client.core.journal.pending.exists())

    def test_domain_explicit_no_driver_start(self):
        with patch.dict(os.environ, {"ROS_DOMAIN_ID": "42"}):
            with self.assertRaises(ValueError):
                self.connect()
        self.factory.assert_not_called()

    def test_missing_runtime_no_fallback(self):
        self.factory.side_effect = RuntimeError("ROS missing")
        with self.assertRaisesRegex(RuntimeError, "ROS missing"):
            self.connect()
        self.assertTrue(self.client.closed.is_set())
        self.assertIsNone(self.client.status)

    def test_discovery_timeout_cleanup(self):
        self.driver.ready.return_value = False
        with self.assertRaises(TimeoutError):
            self.client.connect("test_scene", timeout=0, start_monitor=False)
        self.driver.start.assert_not_called()
        self.driver.close.assert_called_once()

    def test_exact_goal_with_durable_reservation(self):
        self.connect()
        p = packet()
        self.driver.start.side_effect = lambda *args: self.assertTrue(self.client.core.journal.pending.exists())
        rid = self.client.navigate(p, request_id=p["request_id"])
        self.assertEqual(rid, p["request_id"])
        self.assertIs(self.driver.start.call_args.args[0], p["goal"])
        self.assertEqual({k: p["goal"][k] for k in FIELD_LIMITS}, FIELD_LIMITS)
        with self.assertRaises(ValueError):
            self.client.navigate(p)
        self.driver.start.call_args.args[3](4, {"success": True, "reason": "arrived"})
        self.assertIsNone(self.client.active)
        self.assertFalse(self.client.core.journal.pending.exists())
        self.assertEqual(self.events[-1]["type"], "result")

    def test_invalid_goal_and_frame_never_send(self):
        self.connect()
        p = packet()
        p["goal"]["max_speed"] = .3
        with self.assertRaises(ValueError):
            self.client.navigate(p)
        p = packet()
        p["coordinate_frame"] = "different_scene"
        with self.assertRaises(ValueError):
            self.client.navigate(p)
        self.driver.start.assert_not_called()

    def test_rejected_goal_releases_reservation(self):
        self.connect()
        self.client.navigate(packet())
        self.driver.start.call_args.args[1](SimpleNamespace(accepted=False))
        self.assertIsNone(self.client.active)
        self.assertFalse(self.client.core.journal.pending.exists())

    def test_send_exception_retains_unknown_and_no_replay(self):
        self.connect()
        self.driver.start.side_effect = RuntimeError("uncertain send")
        self.client.navigate(packet())
        self.assertIsNotNone(self.client.active)
        self.assertTrue(self.client.core.journal.pending.exists())
        self.assertEqual(self.events[-1]["type"], "execution_state_unknown")
        another = Ros2NavigationClient(Mock(), Path(self.temp.name)/"runtime", self.factory)
        with self.assertRaisesRegex(RuntimeError, "Unresolved"):
            another.connect("test_scene", start_monitor=False)
        self.assertEqual(self.factory.call_count, 1)

    def test_late_accept_after_cancel(self):
        self.connect()
        self.client.navigate(packet())
        accepted, feedback, result, error = self.driver.start.call_args.args[1:]
        self.client.cancel()
        self.driver.cancel.assert_not_called()
        handle = SimpleNamespace(accepted=True)
        accepted(handle)
        self.driver.cancel.assert_called_once()
        self.driver.cancel.call_args.args[1](True)
        self.assertIsNotNone(self.client.active)  # cancel ACK is not terminal
        result(5, {"success": False, "reason": "cancelled"})
        self.assertIsNone(self.client.active)

    def test_stalled_gui_watchdog_cancels(self):
        self.connect()
        now = [0.]
        self.client.core.clock = lambda: now[0]
        self.client.navigate(packet())
        accepted = self.driver.start.call_args.args[1]
        accepted(SimpleNamespace(accepted=True))
        now[0] = 4.
        self.client.touch()
        now[0] = 8.
        self.client.core.tick()
        self.driver.cancel.assert_not_called()
        now[0] = 10.
        self.client.core.tick()
        self.driver.cancel.assert_called_once()
        self.assertTrue(self.client.core.journal.pending.exists())

    def test_close_idle_never_sends(self):
        self.connect()
        self.client.close()
        self.assertTrue(self.client.closed.is_set())
        self.driver.start.assert_not_called()
        self.assertEqual(self.events[-1]["type"], "connection_lost")

    def test_nonterminal_result_not_success(self):
        self.connect()
        self.client.navigate(packet())
        self.driver.start.call_args.args[3](2, {"success": True})
        self.assertIsNotNone(self.client.active)
        self.assertTrue(self.client.core.journal.pending.exists())
        self.assertEqual(self.events[-1]["type"], "execution_state_unknown")


if __name__ == "__main__":
    unittest.main()
