"""Optional native ROS2 transport for the field console; no TCP/SSH/WSL.

ROS imports and discovery occur only on explicit connect. Keep the existing
single-goal state machine, durable reservation and cancellation semantics.
"""
import os
import threading
import time
import uuid

from .tcp_bridge import BridgeCore, Journal, RosDriver
from .tcp_protocol import GOAL_TYPES, validate_goal


class DirectRosDriver(RosDriver):
    """Reuse async send/result/cancel; own context/executor, not the global one."""
    def __init__(self):
        try:
            import rclpy
            from rclpy.context import Context
            from rclpy.executors import SingleThreadedExecutor
            from rclpy.action import ActionClient
            from nav_interfaces.action import NavigateTo
            from rosidl_runtime_py.convert import message_to_ordereddict
        except (ImportError, OSError) as exc:
            raise RuntimeError("ROS2 直连环境不可用。请用已配置 rclpy、nav_interfaces 和 Tk 的 ROS2 Python 启动窗口；"
                               "普通 Windows Python 不会因为狗上安装了 ROS2 就自动获得这些包。未发送导航。") from exc
        if NavigateTo.Goal.get_fields_and_field_types() != GOAL_TYPES:
            raise ValueError("NavigateTo Goal 类型不一致；请使用导航组提供的配套消息包。")
        self.message, self.serialize = NavigateTo, message_to_ordereddict
        self.context, self.node, self.executor, self.client = Context(), None, None, None
        self.stopping, self.spin_error = threading.Event(), None
        self.close_lock, self.destroyed = threading.Lock(), False
        self.thread = None
        try:
            rclpy.init(args=[], context=self.context)
            self.node = rclpy.create_node("nebula_field_direct_"+uuid.uuid4().hex[:12], context=self.context)
            self.client = ActionClient(self.node, NavigateTo, "/nav/navigate_to")
            self.executor = SingleThreadedExecutor(context=self.context)
            self.executor.add_node(self.node)
            self.thread = threading.Thread(target=self.spin, daemon=True)
            self.thread.start()
        except Exception:
            self.close()
            raise

    def spin(self):
        try:
            while not self.stopping.is_set() and self.context.ok():
                self.executor.spin_once(timeout_sec=.1)
        except Exception as exc:
            if not self.stopping.is_set():
                self.spin_error = str(exc)

    def ready(self):
        return (not self.stopping.is_set() and self.spin_error is None
                and self.context.ok() and self.client.server_is_ready())

    def close(self):
        with self.close_lock:
            self._close()

    def _close(self):
        if self.destroyed:
            return
        self.stopping.set()
        if self.thread and self.thread is not threading.current_thread():
            self.thread.join(timeout=2)
            if self.thread.is_alive():
                return  # Do not destroy handles while a ROS callback still uses them.
        if self.executor:
            self.executor.shutdown(timeout_sec=1)
        if self.client:
            self.client.destroy()
        if self.node:
            self.node.destroy_node()
        if self.context.ok():
            self.context.shutdown()
        self.destroyed = True


class Ros2NavigationClient:
    transport = "ros2_direct"

    def __init__(self, callback, runtime_dir, driver_factory=DirectRosDriver):
        self.callback, self.runtime_dir, self.driver_factory = callback, runtime_dir, driver_factory
        self.driver, self.core, self.status, self.active = None, None, None, None
        self.closed = threading.Event()
        self.owner = uuid.uuid4().hex
        self.frame = None

    def connect(self, frame, timeout=8., start_monitor=True):
        if self.closed.is_set() or self.core or self.driver:
            raise ValueError("ROS2 client already connected/closed")
        if not isinstance(frame, str) or not frame:
            raise ValueError("需要先加载明确坐标系的场景。")
        if os.environ.get("ROS_DOMAIN_ID") != "0":
            raise ValueError("请在启动窗口的 ROS2 环境显式设置 ROS_DOMAIN_ID=0；不会自动更改网络配置。")
        journal = Journal(self.runtime_dir)  # Refuse unresolved prior direct goals.
        self.frame = frame
        try:
            self.driver = self.driver_factory()
            deadline = time.monotonic()+timeout
            while not self.driver.ready():
                if self.closed.wait(.1):
                    raise ConnectionError("ROS2 connection cancelled; no goal sent")
                if time.monotonic() >= deadline:
                    raise TimeoutError("未发现可用的 /nav/navigate_to；检查 ROS2 环境、DDS 网络、DOMAIN 和导航节点。未发送目标。")
            if self.closed.is_set():
                raise ConnectionError("ROS2 connection cancelled; no goal sent")
            self.core = BridgeCore(self.driver, frame, True, journal)
            try:
                self.driver.observe_paths(self.core.visualization)
            except Exception:
                pass  # Plot subscription is optional; never substitute control feedback.
            self.status = self.snapshot()
            self.callback({**self.status, "type": "hello"})
            if start_monitor:
                threading.Thread(target=self.monitor, daemon=True).start()
        except Exception:
            self.closed.set()
            if self.driver:
                self.driver.close()
            raise

    def snapshot(self):
        return {**self.core.status(), "transport": self.transport,
                "coordinate_frame_verification": "operator_required_not_reported_by_action",
                "busy_scope": "this_client_only_external_goals_not_observed"}

    def touch(self):
        # Called by GUI poll, not the background monitor. A stalled GUI causes
        # the existing 5-second watchdog to request cancellation, never replay.
        if self.core and not self.closed.is_set():
            self.core.touch(self.owner)

    def on_event(self, row):
        if row.get("type") in ("result", "goal_rejected"):
            self.active = None
        self.callback({**row, "transport": self.transport})

    def navigate(self, request, request_id=None):
        if self.closed.is_set() or self.core is None or self.active is not None:
            raise ValueError("ROS2 未连接或已有未决目标；不会重试。")
        packet = {"request_id": request_id or str(uuid.uuid4()), "operator_confirmed": True,
                  "coordinate_frame": request["coordinate_frame"], "source_action_id": request["source_action_id"],
                  "goal": request["goal"], "transport": self.transport}
        validate_goal(packet)  # Preserve the existing exact 16 fields/limits.
        if packet["coordinate_frame"] != self.frame or not self.driver.ready():
            raise ValueError("ROS2 Action 不可用或所选场景已变化。")
        self.active = packet["request_id"]
        try:
            self.core.navigate(self.owner, packet, self.on_event)
        except Exception:
            # Core reserves durably before ROS; never clear an uncertain send.
            if self.core.active is None:
                self.active = None
            raise
        return packet["request_id"]

    def cancel(self):
        if self.core:
            self.core.cancel(self.owner, request_id=self.active)

    def monitor(self):
        while not self.closed.wait(.2):
            try:
                self.core.tick()
                self.status = self.snapshot()
                self.callback(self.status)
                if not self.status["action_available"]:
                    self.close("ROS2 Action discovery/executor lost; physical state not confirmed")
            except Exception as exc:
                self.close("ROS2 observer error: "+str(exc))

    def close(self, reason="ROS2 client closed; cancellation is not a physical stop"):
        if self.closed.is_set():
            return
        self.closed.set()
        try:
            self.cancel()
        except Exception:
            pass
        self.callback({"type": "connection_lost", "request_id": self.active, "transport": self.transport,
                       "data": {"reason": reason, "execution_state_unknown": self.active is not None}})
        def cleanup():
            deadline = time.monotonic()+3
            while self.core and self.core.active is not None and time.monotonic() < deadline:
                time.sleep(.05)
            if self.driver:
                self.driver.close()
        if self.driver and self.core:
            threading.Thread(target=cleanup, daemon=True).start()
