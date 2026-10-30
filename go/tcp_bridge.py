"""Dog-side TCP -> existing ROS2 NavigateTo. Does not start or modify navigation."""
import argparse
from datetime import datetime, timezone
import hmac
import json
import os
from pathlib import Path
import socket
import socketserver
import threading
import time
import uuid

try:
    from .tcp_protocol import PROTOCOL, MAX_LINE, GOAL_TYPES, FIELD_LIMITS, encode, decode, clean, validate_goal
except ImportError:  # Standalone dog deployment: copy these two Python files together.
    from tcp_protocol import PROTOCOL, MAX_LINE, GOAL_TYPES, FIELD_LIMITS, encode, decode, clean, validate_goal


class Journal:
    def __init__(self, directory):
        self.folder = Path(directory)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.pending = self.folder / "pending_goal.json"
        if self.pending.exists():
            raise RuntimeError("Unresolved prior goal in pending_goal.json. Operator must confirm stop and archive it before restarting. Never auto-replay.")
    def reserve(self, packet):
        with self.pending.open("x", encoding="utf-8") as stream:
            json.dump(clean(packet), stream, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
    def finish(self):
        self.pending.unlink(missing_ok=True)
    def record(self, row):
        with (self.folder / "bridge_events.jsonl").open("ab") as stream:
            stream.write(encode(row))


class BridgeCore:
    """One active goal; injectable ROS driver for offline tests."""
    def __init__(self, driver, frame, allow_motion, journal, clock=time.monotonic):
        self.driver, self.frame, self.allow_motion = driver, frame, allow_motion
        self.journal, self.clock = journal, clock
        self.lock = threading.RLock()
        self.active, self.seen = None, set()

    def status(self):
        with self.lock:
            return {"protocol": PROTOCOL, "type": "status", "action_name": "/nav/navigate_to",
                    "ros_domain_id": 0, "coordinate_frame": self.frame,
                    "motion_enabled": self.allow_motion, "action_available": self.driver.ready(),
                    "busy": self.active is not None, "active_request_id": self.active["id"] if self.active else None,
                    "motion_request_parameters": FIELD_LIMITS,
                    "hardware_sink": "not_observed_by_bridge", "actual_speed_cap_verified": False,
                    "fallback": False}

    def emit(self, event, data=None):
        row = {"protocol": PROTOCOL, "type": event, "utc": datetime.now(timezone.utc).isoformat(),
               "request_id": self.active["id"] if self.active else None,
               "coordinate_frame": self.frame, "data": clean(data or {})}
        self.journal.record(row)
        if self.active:
            try:
                self.active["send"](row)
            except OSError:
                # Do not treat a broken TCP socket as a stopped robot.
                self.active["cancel_requested"] = True
        return row

    def navigate(self, owner, packet, send):
        with self.lock:
            goal = validate_goal(packet)
            if not self.allow_motion or not self.driver.ready():
                raise ValueError("Bridge not motion-enabled or ROS action unavailable")
            if packet["coordinate_frame"] != self.frame:
                raise ValueError("Scene frame mismatch; temporary-origin test is not D3-619 registration")
            if self.active is not None:
                raise ValueError("A goal is active or unresolved; concurrent goals are forbidden")
            if packet["request_id"] in self.seen:
                raise ValueError("Duplicate request ID; automatic replay forbidden")
            self.journal.reserve(packet)  # Durable reservation before any ROS side effect.
            self.seen.add(packet["request_id"])
            now = self.clock()
            self.active = {"id": packet["request_id"], "owner": owner, "send": send,
                           "handle": None, "cancel_requested": False, "cancel_sent": False,
                           "heartbeat": now, "ack_deadline": now+10,
                           "result_deadline": now+goal["timeout"]+15}
            try:
                self.emit("goal_reserved")
                rid = packet["request_id"]
                self.driver.start(goal, *(self.callback(rid, fn) for fn in
                                         (self.accepted, self.feedback, self.result, self.error)))
            except Exception as exc:
                self.error(exc)  # Delivery can be uncertain; never release reservation on a transport exception.

    def callback(self, request_id, fn):
        def checked(*args):
            with self.lock:
                if self.active and self.active["id"] == request_id:
                    fn(*args)
        return checked

    def touch(self, owner):
        with self.lock:
            if self.active and self.active["owner"] == owner:
                self.active["heartbeat"] = self.clock()

    def cancel(self, owner=None, reason="operator_cancel", request_id=None):
        with self.lock:
            if not self.active:
                return
            if owner is not None and self.active["owner"] != owner:
                raise ValueError("Only the owning connection can cancel this goal")
            if request_id is not None and self.active["id"] != request_id:
                raise ValueError("Cancellation request does not match active goal")
            self.active["cancel_requested"] = True
            if self.active["handle"] is not None and not self.active["cancel_sent"]:
                self.active["cancel_sent"] = True
                self.emit("cancel_requested", {"reason": reason, "stop_confirmed": False})
                try:
                    rid = self.active["id"]
                    self.driver.cancel(self.active["handle"], self.callback(rid, self.cancel_ack), self.callback(rid, self.error))
                except Exception as exc:
                    self.error(exc)

    def disconnected(self, owner):
        with self.lock:
            if self.active and self.active["owner"] == owner:
                self.cancel(owner, "owner_disconnected")

    def accepted(self, handle):
        with self.lock:
            if not self.active:
                return
            if not handle.accepted:
                self.emit("goal_rejected", {"ros_goal_accepted": False})
                self.journal.finish()
                self.active = None
                return
            self.active["handle"] = handle
            self.emit("goal_accepted")
            if self.active["cancel_requested"]:
                self.cancel(reason="cancel_after_late_acceptance")

    def feedback(self, value):
        with self.lock:
            if self.active:
                self.emit("feedback", value)

    def visualization(self, value):
        """Optional uncorrelated ROS topic observation; never a control/result event."""
        with self.lock:
            if not self.active:
                return
            row = {"protocol": PROTOCOL, "type": "planned_path",
                   "utc": datetime.now(timezone.utc).isoformat(),
                   "request_id": self.active["id"], "coordinate_frame": self.frame,
                   "data": clean(value), "source_topic": "/nav/planned_path",
                   "association": "observed_during_request_not_goal_uuid_verified"}
            try:
                encode(row)  # Preserve the existing bounded wire contract.
                self.journal.record(row)
                self.active["send"](row)
            except Exception:
                # Optional plotting must not alter navigation state or cause a retry.
                pass

    def result(self, status, value):
        with self.lock:
            if self.active:
                if status not in (4, 5, 6):
                    self.error(ValueError("Unexpected nonterminal ROS result status"))
                    return
                self.emit("result", {"ros_goal_status": status, "result": value,
                    "server_reported_arrival": status == 4 and value.get("success") is True and value.get("reason") == "arrived",
                    "physical_arrival_independently_verified": False})
                self.journal.finish()
                self.active = None

    def cancel_ack(self, accepted):
        with self.lock:
            if self.active:
                self.emit("cancel_ack", {"accepted": bool(accepted), "stop_confirmed": False})

    def error(self, exc):
        with self.lock:
            if self.active:
                self.active["cancel_requested"] = True
                self.emit("execution_state_unknown", {"error": type(exc).__name__ + ": " + str(exc), "operator_check_required": True})

    def tick(self):
        with self.lock:
            if not self.active:
                return
            a, now = self.active, self.clock()
            if now-a["heartbeat"] > 5 or (a["handle"] is None and now > a["ack_deadline"]) or now > a["result_deadline"]:
                if not a.get("watchdog_reported"):
                    a["watchdog_reported"] = True
                    self.emit("execution_state_unknown", {"error": "Transport watchdog or missing terminal result", "operator_check_required": True})
                self.cancel(reason="transport_watchdog_or_missing_terminal_result")
            if a["cancel_requested"]:
                self.cancel(reason="pending_cancel")


class RosDriver:
    def observe_paths(self, callback):
        from std_msgs.msg import String
        from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
        def receive(msg):
            try:
                value = json.loads(msg.data)
                if isinstance(value, dict) and isinstance(value.get("waypoints"), list):
                    callback(value)
            except (ValueError, TypeError):
                pass
        self.path_subscription = self.node.create_subscription(String, "/nav/planned_path", receive,
            QoSProfile(depth=100, reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST))

    def __init__(self):
        import rclpy
        from rclpy.action import ActionClient
        from nav_interfaces.action import NavigateTo
        from rosidl_runtime_py.convert import message_to_ordereddict
        if NavigateTo.Goal.get_fields_and_field_types() != GOAL_TYPES:
            raise RuntimeError("NavigateTo schema mismatch; no sandbox fallback")
        self.rclpy, self.message, self.serialize = rclpy, NavigateTo, message_to_ordereddict
        rclpy.init(args=[])
        self.node = rclpy.create_node("nebula_field_tcp_bridge")
        self.client = ActionClient(self.node, NavigateTo, "/nav/navigate_to")
        self.thread = threading.Thread(target=lambda: rclpy.spin(self.node), daemon=True)
        self.thread.start()
    def ready(self):
        return self.client.server_is_ready()
    def start(self, fields, accepted, feedback, result, error):
        goal = self.message.Goal()
        for key, value in fields.items():
            setattr(goal, key, float(value))
        def got_result(future):
            try:
                wrapped = future.result()
                result(wrapped.status, self.serialize(wrapped.result))
            except Exception as exc:
                error(exc)
        def got_handle(future):
            try:
                handle = future.result()
                accepted(handle)
                if handle.accepted:
                    handle.get_result_async().add_done_callback(got_result)
            except Exception as exc:
                error(exc)
        self.client.send_goal_async(goal, feedback_callback=lambda msg: feedback(self.serialize(msg.feedback))).add_done_callback(got_handle)
    def cancel(self, handle, acknowledged, error):
        def done(future):
            try:
                acknowledged(bool(future.result().goals_canceling))
            except Exception as exc:
                error(exc)
        handle.cancel_goal_async().add_done_callback(done)


class Handler(socketserver.StreamRequestHandler):
    def setup(self):
        super().setup()
        self.send_lock = threading.Lock()
        self.owner = uuid.uuid4().hex
        self.authenticated = False
        self.request.settimeout(8)
    def send_packet(self, row):
        with self.send_lock:
            self.request.sendall(encode(row))
    def handle(self):
        core = self.server.core
        try:
            raw = self.rfile.readline(MAX_LINE+1)
            hello = decode(raw)
            token = hello.get("token")
            if hello.get("type") != "hello" or not isinstance(token, str) or not hmac.compare_digest(token, self.server.token):
                return  # No credential values in logs/responses.
            self.authenticated = True
            self.send_packet({**core.status(), "type": "hello"})
            while True:
                raw = self.rfile.readline(MAX_LINE+1)
                if not raw:
                    break
                packet = decode(raw)
                core.touch(self.owner)
                kind = packet.get("type")
                if kind == "status":
                    self.send_packet(core.status())
                elif kind == "navigate":
                    try:
                        core.navigate(self.owner, packet, self.send_packet)
                    except ValueError as exc:
                        self.send_packet({"protocol": PROTOCOL, "type": "dispatch_rejected", "request_id": packet.get("request_id"), "data": {"error": str(exc), "not_sent_to_ros": True}})
                elif kind == "cancel":
                    if not isinstance(packet.get("request_id"), str):
                        raise ValueError("Cancellation requires request ID")
                    core.cancel(self.owner, request_id=packet["request_id"])
                else:
                    raise ValueError("Unknown operation")
        except (OSError, ValueError, TypeError):
            pass
        finally:
            if self.authenticated:
                core.disconnected(self.owner)


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = False
    daemon_threads = True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bind", default="192.168.123.18")
    parser.add_argument("--port", type=int, default=8888)
    parser.add_argument("--scene-frame", required=True, help="Operator-registered navigation scene frame, not merely a label change")
    parser.add_argument("--allow-motion", action="store_true", help="Authorize forwarding to the existing navigation action; no automatic goal")
    parser.add_argument("--runtime-dir", type=Path, default=Path("field_bridge_runtime"))
    args = parser.parse_args(argv)
    token = os.environ.get("FIELD_BRIDGE_TOKEN", "")
    if len(token) < 24:
        raise ValueError("Set FIELD_BRIDGE_TOKEN to a shared random secret of at least 24 characters")
    if os.environ.get("ROS_DOMAIN_ID") != "0":
        raise ValueError("The reviewed field node uses ROS_DOMAIN_ID=0; configure it explicitly")
    journal = Journal(args.runtime_dir)
    driver = RosDriver()  # Import failure is fatal; never fabricate a sandbox success.
    core = BridgeCore(driver, args.scene_frame, args.allow_motion, journal)
    try:
        driver.observe_paths(core.visualization)
    except Exception as exc:
        print("Optional planned-path observer unavailable: " + type(exc).__name__, flush=True)
    with Server((args.bind, args.port), Handler) as server:
        server.core, server.token = core, token
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        print(f"{PROTOCOL} listening {args.bind}:{args.port}; motion_enabled={args.allow_motion}; no goal sent", flush=True)
        try:
            while True:
                core.tick()
                time.sleep(.2)
        except KeyboardInterrupt:
            core.cancel(reason="bridge_shutdown")
            # Keep ROS spinning briefly for cancellation and a terminal result.
            deadline = time.monotonic()+3
            while core.active is not None and time.monotonic() < deadline:
                time.sleep(.1)
            print("Bridge shutting down. If no terminal result, robot state is UNKNOWN; use physical stop.", flush=True)
        finally:
            server.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
