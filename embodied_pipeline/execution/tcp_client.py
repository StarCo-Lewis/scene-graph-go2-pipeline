"""Authenticated private-LAN client. No reconnect or navigation retry."""
import socket
import threading
import time
import uuid
from .tcp_protocol import PROTOCOL, MAX_LINE, FIELD_LIMITS, encode, decode, validate_goal


class NavigationClient:
    def __init__(self, callback):
        self.callback, self.sock, self.status = callback, None, None
        self.lock = threading.Lock()
        self.active, self.last_received, self.closed = None, 0, threading.Event()
    def connect(self, host, port, token):
        if self.sock is not None:
            raise ValueError("Client already connected")
        sock = socket.create_connection((host, port), timeout=3)
        try:
            sock.sendall(encode({"protocol": PROTOCOL, "type": "hello", "token": token}))
            raw = b""
            deadline = time.monotonic() + 5
            while not raw.endswith(b"\n"):
                if self.closed.is_set() or time.monotonic() >= deadline:
                    raise TimeoutError("Bridge greeting cancelled or expired; no goal sent")
                byte = sock.recv(1)
                if not byte or len(raw) >= MAX_LINE:
                    raise ValueError("No valid authenticated bridge greeting")
                raw += byte
            hello = decode(raw)
            if (hello.get("type") != "hello" or hello.get("fallback") is not False or hello.get("ros_domain_id") != 0
                    or hello.get("action_name") != "/nav/navigate_to"
                    or hello.get("motion_request_parameters") != FIELD_LIMITS):
                raise ValueError("Unreviewed bridge or sandbox response; no goal sent")
            sock.settimeout(.5)
            if self.closed.is_set():
                raise ConnectionError("Connection was cancelled; no goal sent")
            self.sock, self.status = sock, hello
            self.last_received = time.monotonic()
            self.callback(hello)
            threading.Thread(target=self.receive, daemon=True).start()
            threading.Thread(target=self.heartbeat, daemon=True).start()
        except Exception:
            sock.close()
            raise
    def send(self, packet):
        with self.lock:
            if self.sock is None or self.closed.is_set():
                raise ConnectionError("Bridge disconnected; state may be unknown")
            self.sock.sendall(encode(packet))
    def navigate(self, request, request_id=None):
        if self.active is not None:
            raise ValueError("Previous goal unresolved")
        packet = {"protocol": PROTOCOL, "type": "navigate", "request_id": request_id or str(uuid.uuid4()),
                  "operator_confirmed": True, "goal": request["goal"],
                  "coordinate_frame": request["coordinate_frame"], "source_action_id": request["source_action_id"]}
        validate_goal(packet)
        if not self.status or not self.status.get("motion_enabled") or not self.status.get("action_available") or self.status.get("busy"):
            raise ValueError("Bridge not ready for one navigation goal")
        if self.status.get("coordinate_frame") != packet["coordinate_frame"]:
            raise ValueError("Bridge and request scene frames do not match")
        self.active = packet["request_id"]  # Reserve before send, including an uncertain partial send.
        try:
            self.send(packet)
        except Exception:
            self.lost("Goal send failed; delivery is unknown")
            raise
        return packet["request_id"]
    def cancel(self):
        if self.active:
            self.send({"protocol": PROTOCOL, "type": "cancel", "request_id": self.active})
    def receive(self):
        buffer = b""
        try:
            while not self.closed.is_set():
                try:
                    data = self.sock.recv(8192)
                except socket.timeout:
                    if time.monotonic()-self.last_received > 6:
                        raise TimeoutError("Bridge heartbeat expired")
                    continue
                if not data:
                    raise ConnectionError("Bridge closed connection")
                buffer += data
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    row = decode(line)
                    self.last_received = time.monotonic()
                    if row.get("type") == "status":
                        self.status = row
                    elif row.get("request_id") != self.active:
                        continue
                    kind, payload = row.get("type"), row.get("data", {})
                    if kind == "result" and (payload.get("ros_goal_status") not in (4, 5, 6) or not isinstance(payload.get("result"), dict)):
                        raise ValueError("Invalid terminal result; execution state remains unknown")
                    if kind == "goal_rejected" and payload.get("ros_goal_accepted") is not False:
                        raise ValueError("Invalid goal rejection")
                    if kind == "dispatch_rejected" and payload.get("not_sent_to_ros") is not True:
                        raise ValueError("Invalid dispatch rejection")
                    if kind in ("result", "goal_rejected", "dispatch_rejected"):
                        self.active = None
                    self.callback(row)
                if len(buffer) > MAX_LINE:
                    raise ValueError("Oversized bridge frame")
        except (OSError, ValueError, TypeError, AttributeError) as exc:
            self.lost(str(exc))
    def heartbeat(self):
        while not self.closed.wait(1):
            try:
                self.send({"protocol": PROTOCOL, "type": "status"})
            except OSError as exc:
                self.lost(str(exc))
                return
    def lost(self, reason):
        if not self.closed.is_set():
            self.closed.set()
            self.callback({"type": "connection_lost", "request_id": self.active,
                           "data": {"reason": reason, "execution_state_unknown": self.active is not None}})
            if self.sock:
                self.sock.close()
    def close(self):
        try:
            self.cancel()
        except OSError:
            pass
        self.lost("Client closed; cancellation is not a physical emergency stop")
