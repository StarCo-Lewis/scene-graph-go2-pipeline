"""Navigation-point mapping and explicit dispatch; existing motion parameters unchanged."""
from datetime import datetime, timezone
import json
from pathlib import Path
import queue
import threading
import time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import uuid

from .._legacy import ROOT
from ..execution.navigation_contract import CONFIRMATIONS, prepare, read, sha, navigation_steps, selected_navigation
from ..execution.navigation_session import NavigationSession
from ..execution.navigation_outcome import outcome
from ..execution.tcp_protocol import PROTOCOL, FIELD_LIMITS, clean
from ..execution.tcp_client import NavigationClient
from ..execution.navigation_points import bbox_approach, validate_clearance, clearance_value, BBOX_CLEARANCE_M, is_ground_region, bbox_distance
from ..execution.navigation_free_space import scene_grid, free_approach, validate_endpoint
from .field_backend import feedback_pose

PENDING = ROOT / "outputs/field_control_state/pending_client.json"


def build_config(plan_path, reviewed_hash, scene, start, goal, confirmations, step_index=0, clearance_m=BBOX_CLEARANCE_M):
    if sha(plan_path) != reviewed_hash:
        raise ValueError("工程 JSON 在审阅后发生变化，请重新加载检查。")
    envelope = read(plan_path)
    summary_path = Path(plan_path).with_name("planning_summary.json")
    summary = read(summary_path)
    if (summary.get("status") != "planning_json_generated_not_executed"
            or summary.get("input_files_unchanged") is not True):
        raise ValueError("规划尚未完整通过或输入完整性未确认，不能发送导航。")
    if (envelope.get("task_output_check", {}).get("passed") is not True or
            envelope.get("static_review_status") != "conditionally_valid_symbolic_plan"):
        raise ValueError("工程计划检查未通过，不能发送导航。")
    step = selected_navigation(envelope, step_index)
    target = step["arguments"]["target"]["element_id"]
    node = next((n for n in scene["nodes"] if n["id"] == target), None)
    if node is None or node.get("active", True) is False:
        raise ValueError("当前场景没有可用的计划目标。")
    clearance = validate_clearance(node, goal, clearance_m)
    endpoint_check = validate_endpoint(scene_grid(scene), goal)
    return {"schema_version": "single_navigation_replay_v1",
            "engineering_plan": str(Path(plan_path).resolve()), "engineering_plan_sha256": reviewed_hash,
            "scene_graph": str(scene["folder"] / "scene_graph.json"),
            "scene_edges": str(scene["folder"] / "scene_graph_edges.json"),
            "coordinate_frame": scene["metadata"]["coordinate_frame"], "angle_unit": "radian",
            "target_element_id": target, "navigation_step_index": step_index,
            "navigation_step_count": len(navigation_steps(envelope)),
            "start_pose": start, "approach_pose": goal, "limits": dict(FIELD_LIMITS),
            "confirmations": confirmations,
            "ros_environment": {"ROS_DOMAIN_ID": "0"},
            "transport": PROTOCOL,
            "navigation_point_policy": {"bbox_clearance_m": clearance_m,
                "actual_xy_reference_point_clearance_m": clearance,
                "source_plan_standoff_m": step["arguments"].get("standoff_m"),
                "ground_region_candidate_rule_available": is_ground_region(node),
                "navigation_grid_screening": endpoint_check,
                "override_authority": "Operator-selected BBOX boundary clearance in the client; default 0.4 m",
                "robot_body_clearance_verified": False, "collision_free_path_verified": False},
            "parameter_policy": "Forward the 2026-10-23 field request parameters unchanged; not a physical speed guarantee"}


def mapped_point(path, scene, target):
    table = read(path)
    if (table.get("schema_version") != "navigation_points_v1" or
            table.get("coordinate_frame") != scene["metadata"]["coordinate_frame"] or
            table.get("source_graph_sha256") != scene["graph_hash"]):
        raise ValueError("导航点表与当前场景版本/坐标系不一致。")
    value = table["points"][str(target)]
    from ..execution.navigation_contract import pose
    pose(value, "navigation_point")
    return value


def candidate_for_plan(path, digest, scene, start, step_index=0, clearance_m=BBOX_CLEARANCE_M):
    if sha(path) != digest:
        raise ValueError("工程 JSON 已变化，请重新加载。")
    step = selected_navigation(read(path), step_index)
    target = step["arguments"]["target"]["element_id"]
    node = next((n for n in scene["nodes"] if n["id"] == target), None)
    if node is None:
        raise ValueError("计划目标不在当前场景。")
    value = free_approach(node, start, clearance_m, scene_grid(scene))
    build_config(path, digest, scene, start, value, {}, step_index, clearance_m)  # Require a complete reviewed plan.
    return target, value


class NavigationPanel(ttk.Frame):
    def __init__(self, parent, console):
        super().__init__(parent, padding=12)
        self.console = console
        self.events = queue.Queue()
        self.client, self.run_dir, self.request = None, None, None
        self.uncertain = PENDING.exists()
        self.active = False
        self.current_request_id = None
        self.clearance_m = BBOX_CLEARANCE_M
        self.clearance_input = tk.StringVar(value=str(BBOX_CLEARANCE_M))
        self.outcome_state = tk.StringVar(value="执行结果：尚未执行；规划通过不等于执行成功。")
        self.outcomes = {}
        self.connection_id = None
        self.last_pose_time = None
        self.navigation_session = NavigationSession()
        self.step_index = 0
        self.sequence_complete = False
        self.sequence_key = None
        self.dispatch_scene_key = None
        self.segment_state = tk.StringVar(value="首次填写场景起点；到达后自动接续实际终态位姿，坐标系不归零。")
        self.queue_running = False
        self.queue_route = []
        self.queue_key = None
        self.queue_connection = None
        self.queue_confirmations = {}
        self.queue_next_deadline = None
        self.queue_ready_at = None
        self.setting_start = False
        self.auto_key = None
        self.auto_goal = False
        self.auto_suppressed = False
        self.setting_goal = False
        self.mapping_note = tk.StringVar(value="规划通过且已填当前起点后，自动填写候选点；不会自动发送。")
        self.token = tk.StringVar()
        self.transport_mode = tk.StringVar(value="TCP 网桥")
        self.connected_mode = None
        self.start = {k: tk.StringVar() for k in ("x", "y", "yaw")}
        self.goal = {k: tk.StringVar() for k in ("x", "y", "yaw")}
        self.confirmations = {k: tk.BooleanVar(value=False) for k in CONFIRMATIONS}
        for value in self.start.values():
            value.trace_add("write", self.start_changed)
        for value in self.goal.values():
            value.trace_add("write", self.goal_changed)
        self.hardware = tk.BooleanVar(value=False)
        self.state = tk.StringVar(value="尚未连接。TCP 需配套网桥；ROS2 直连需当前 Python 已配置 ROS2 和消息包。")
        self.build()
        self.after(200, self.poll)

    def build(self):
        ttk.Label(self, text="工程 JSON → 机器人导航点 → 既有 /nav/navigate_to；不改控制器参数", font=("Microsoft YaHei UI", 12, "bold")).pack(anchor="w")
        row = ttk.Frame(self)
        row.pack(fill="x", pady=7)
        self.transport_selector = ttk.Combobox(row, textvariable=self.transport_mode,
            values=("TCP 网桥", "ROS2 直连"), state="readonly", width=13)
        self.transport_selector.pack(side="left", padx=(0, 8))
        self.transport_selector.bind("<<ComboboxSelected>>", self.mode_changed)
        ttk.Label(row, text="TCP 口令（直连不填）").pack(side="left")
        self.token_entry = ttk.Entry(row, textvariable=self.token, show="*", width=24)
        self.token_entry.pack(side="left", padx=8)
        ttk.Button(row, text="连接 / 检查（不发送导航）", command=lambda: self.console.guarded(self.connect)).pack(side="left")
        ttk.Button(row, text="断开连接", command=self.disconnect).pack(side="left", padx=5)
        ttk.Label(self, textvariable=self.state, wraplength=1000).pack(anchor="w")
        ttk.Label(self, textvariable=self.outcome_state, foreground="#174c75", font=("Microsoft YaHei UI",11,"bold"),wraplength=1050).pack(anchor="w")
        ttk.Label(self, text="TCP 使用“连接”页 IP/端口；ROS2 直连不使用 8888/口令。DOMAIN=0；场景对齐、真机出口仍需现场确认。", foreground="#7b4500").pack(anchor="w", pady=5)
        ttk.Label(self, textvariable=self.segment_state, wraplength=1050).pack(anchor="w", pady=4)
        for label, values in (("本段起点（首段测量，后续接续）", self.start), ("目标停靠点（不是物体中心）", self.goal)):
            row = ttk.Frame(self)
            row.pack(fill="x", pady=3)
            ttk.Label(row, text=label, width=32).pack(side="left")
            for key, variable in values.items():
                ttk.Label(row, text=key).pack(side="left", padx=(10, 3))
                ttk.Entry(row, textvariable=variable, width=12).pack(side="left")
        mapping = ttk.Frame(self)
        mapping.pack(fill="x", pady=4)
        ttk.Label(mapping,text="BBOX 外侧距离 / m").pack(side="left")
        ttk.Entry(mapping,textvariable=self.clearance_input,width=7).pack(side="left",padx=4)
        ttk.Button(mapping,text="应用距离",command=lambda:self.console.guarded(self.apply_clearance)).pack(side="left")
        ttk.Button(mapping, text="按当前距离生成候选点", command=lambda: self.console.guarded(self.generate_point)).pack(side="left")
        ttk.Button(mapping, text="从导航点 JSON 映射当前计划目标", command=lambda: self.console.guarded(self.load_points)).pack(side="left", padx=8)
        ttk.Label(self, textvariable=self.mapping_note, wraplength=1050).pack(anchor="w", pady=3)
        ttk.Label(self, text="参数锁定为当前指定请求值：pos_tol=0.05 m · yaw_tol=0.17 rad · timeout=120 s · max_speed=0.5 m/s\n不改 CAPO 比例/末端接近/跟随器/避障；max_speed 字段不能保证实际限速。", justify="left").pack(anchor="w", pady=3)
        labels = {
            "initial_pose_measured": "已核对本段起点：首段测量；后续终态位姿仍有效，期间未搬动或遥控移动狗",
            "scene_frame_verified": "已核对语义图、619 导航栅格与狗端场景系的实际对齐；不只是 frame 同名",
            "approach_pose_verified": "目标是机器人可停靠导航点，路线与周围净空已人工确认",
            "navigation_limits_verified": "保留上述现场请求参数，已知实际限速不能仅由 max_speed 保证",
            "operator_and_estop_ready": "现场有人全程看护，遥控器/物理急停可用",
            "server_alignment_policy_reviewed": "已核对服务端首次对齐及后续起点校验策略；无其他客户端发目标，未重启或重置原点",
        }
        for key in CONFIRMATIONS:
            ttk.Checkbutton(self, text=labels[key], variable=self.confirmations[key]).pack(anchor="w", pady=1)
        ttk.Checkbutton(self, text="已确认现场导航是 SportSink，CAPO 与 ActionServer 就绪；不是 dry-run 或沙箱", variable=self.hardware).pack(anchor="w", pady=2)
        buttons = ttk.Frame(self)
        buttons.pack(fill="x", pady=5)
        ttk.Button(buttons, text="预览映射（不执行）", command=lambda: self.console.guarded(self.preview)).pack(side="left")
        self.send_button = ttk.Button(buttons, text="确认后发送一次导航", state="disabled", command=lambda: self.console.guarded(self.dispatch))
        self.send_button.pack(side="left", padx=7)
        self.cancel_button = ttk.Button(buttons, text="请求取消（不是物理急停）", command=self.cancel, state="disabled")
        self.cancel_button.pack(side="left")
        ttk.Button(buttons, text="人工核对后解除本地未知状态", command=self.clear_unknown).pack(side="right")
        self.preview_box = tk.Text(self, width=1, height=8, wrap="word", bg="#edf2fa", relief="flat")
        self.preview_box.pack(fill="both", expand=True)

    def mode_changed(self, *_):
        if self.client and not self.client.closed.is_set():
            self.transport_mode.set(self.connected_mode)
            self.state.set("请先断开当前连接；不能在连接/执行期间切换控制方式。")
        direct = self.transport_mode.get() == "ROS2 直连"
        self.token_entry.configure(state="disabled" if direct else "normal")

    def connect(self):
        if self.active:
            raise ValueError("已有活动任务；不能更换连接。")
        if self.client and not self.client.closed.is_set():
            raise ValueError("已有控制连接，请先断开再重新连接。")
        mode = self.transport_mode.get()
        if mode not in ("TCP 网桥", "ROS2 直连"):
            raise ValueError("未知控制方式；不自动降级。")
        direct = mode == "ROS2 直连"
        frame = self.console.scene["metadata"]["coordinate_frame"]
        if not direct:
            host, port, token = self.console.host.get().strip(), int(self.console.port.get()), self.token.get()
            if len(token) < 24:
                raise ValueError("请填写狗端 FIELD_BRIDGE_TOKEN 对应的至少 24 字符口令。")
        connection_id = self.connection_id = uuid.uuid4().hex
        callback = lambda packet: self.events.put({**packet, "client_connection_id": connection_id})
        if direct:
            from ..execution.ros2_client import Ros2NavigationClient
            self.client = Ros2NavigationClient(callback, ROOT / "outputs/field_control_state/ros2_direct")
        else:
            self.client = NavigationClient(callback)
        self.connected_mode = mode
        client = self.client
        self.state.set("正在检查 " + mode + "，不发送导航…")
        def worker():
            try:
                if direct:
                    client.connect(frame)
                else:
                    client.connect(host, port, token)
            except Exception as exc:
                client.close()
                self.events.put({"type": "connect_failed", "client_connection_id": connection_id, "data": {"error": str(exc)}})
        threading.Thread(target=worker, daemon=True).start()

    def get_config(self):
        c = self.console
        if not c.loaded_plan_path:
            raise ValueError("先加载或生成要执行的工程 JSON。")
        start = {k: float(v.get()) for k, v in self.start.items()}
        goal = {k: float(v.get()) for k, v in self.goal.items()}
        config = build_config(c.loaded_plan_path, c.loaded_plan_hash, c.scene, start, goal,
                              {k: v.get() for k, v in self.confirmations.items()}, self.step_index, self.clearance())
        if self.navigation_session.scene_key not in (None, self.scene_key()):
            raise ValueError("当前场景与本次导航会话不同；请重新加载场景并核对服务端地图。")
        config["navigation_session"] = self.navigation_session.provenance(start)
        if self.queue_running:
            config["authorized_navigation_queue"] = self.queue_route
        config["transport"] = getattr(self.client, "transport", PROTOCOL)
        return config

    def preview(self):
        config = self.get_config()
        request = prepare(config)
        request["transport"] = config["transport"]
        self.console.set_text(self.preview_box, json.dumps({"target_element_id": config["target_element_id"],
            "start_pose": config["start_pose"], "approach_pose": config["approach_pose"], "limits_unchanged": FIELD_LIMITS,
            "navigation_point_policy": config["navigation_point_policy"],
            "navigation_step_index": self.step_index, "navigation_step_count": config["navigation_step_count"],
            "navigation_session": config["navigation_session"],
            "blockers": request["blockers"], "goal": request["goal"]}, ensure_ascii=False, indent=2))
        if hasattr(self.console,"navigation_view"):
            self.console.navigation_view.preview_goals([{"step_index":self.step_index,
                "target_element_id":config["target_element_id"],"goal":config["approach_pose"]}],self.clearance())
        return config, request

    def load_points(self):
        c = self.console
        if not c.loaded_plan_path:
            raise ValueError("请先加载当前工程计划。")
        envelope = read(c.loaded_plan_path)
        target = selected_navigation(envelope, self.step_index)["arguments"]["target"]["element_id"]
        path = filedialog.askopenfilename(title="选择 navigation_points.json", filetypes=[("JSON", "*.json")])
        if path:
            value = mapped_point(path, c.scene, target)
            self.set_goal(value, automatic=False)
            self.state.set(f"已映射 ID {target} 的导航点，未改变运动参数；请核对位置。")

    def generate_point(self):
        if self.active:
            raise ValueError("任务活动时不能重新映射。")
        c = self.console
        if not c.loaded_plan_path:
            raise ValueError("请先加载并检查当前工程 JSON。")
        start = {k: float(v.get()) for k, v in self.start.items()}
        target, value = candidate_for_plan(c.loaded_plan_path, c.loaded_plan_hash, c.scene, start, self.step_index,self.clearance())
        self.set_goal(value, automatic=True)
        self.auto_key = self.mapping_key()
        self.mapping_note.set(f"已填入 ID {target} 的 BBOX 外侧 {self.clearance():g} m 环绕候选点，已通过导航栅格/0.35 m 圆形净空筛选；仍需核对地图对齐和路线。")
        node = next(n for n in c.scene["nodes"] if n["id"] == target)
        if is_ground_region(node) and bbox_distance(node, start) <= 1e-9:
            self.mapping_note.set(f"ID {target} 为地面类目标：已在 BBOX 外侧 {self.clearance():g} m 环绕采样并通过栅格净空筛选；仍需核对地图对齐和路线。")
        c.log("导航点已填写（未发送）：ID " + str(target) + " " + json.dumps(value))
        if hasattr(c,"navigation_view"):
            c.navigation_view.preview_goals([{"step_index":self.step_index,"target_element_id":target,"goal":value}],self.clearance())

    def clearance(self):
        return self.__dict__.get("clearance_m",BBOX_CLEARANCE_M)

    def apply_clearance(self):
        if self.active or self.queue_running or self.uncertain:
            raise ValueError("活动导航、队列或未知执行状态期间不能修改停靠距离。")
        value=clearance_value(float(self.clearance_input.get()))
        if value==self.clearance():return
        self.clearance_m=value
        self.reset_confirmations(clear_start=False)
        self.set_goal(None,automatic=True);self.auto_key=None;self.auto_suppressed=False
        self.mapping_note.set(f"已应用 {value:g} m；候选终点需重新核对，多目标队列需重新生成。速度/超时不变。")
        if hasattr(self.console,"navigation_view"):self.console.navigation_view.preview_goals([],value)
        self.console.log(f"BBOX 停靠距离改为 {value:g} m；运动参数未修改。")
        self.maybe_auto_map()

    def mapping_key(self):
        return (str(self.console.loaded_plan_path), self.console.loaded_plan_hash,
                self.step_index, self.clearance(),
                *(self.start[k].get() for k in ("x", "y", "yaw")))

    def scene_key(self):
        scene = self.console.scene
        return (scene["graph_hash"], scene["metadata"]["coordinate_frame"])

    def set_goal(self, value, automatic):
        self.setting_goal = True
        try:
            for key, variable in self.goal.items():
                variable.set(repr(value[key]) if value is not None else "")
        finally:
            self.setting_goal = False
        self.auto_goal, self.auto_suppressed = automatic, not automatic
        self.confirmations["approach_pose_verified"].set(False)

    def start_changed(self, *_):
        if self.queue_running and not self.setting_start:
            self.stop_queue("起点被编辑，已暂停后续队列。不会重置场景坐标系。")
        self.confirmations["initial_pose_measured"].set(False)
        self.confirmations["approach_pose_verified"].set(False)
        self.auto_key = None
        if self.auto_goal:
            self.set_goal(None, automatic=True)

    def goal_changed(self, *_):
        self.confirmations["approach_pose_verified"].set(False)
        if not self.setting_goal:
            if self.queue_running:
                self.stop_queue("终点被编辑，已暂停后续队列，请重新核对。")
            self.auto_goal, self.auto_suppressed = False, True
            self.mapping_note.set("已手动编辑导航点；不会自动覆盖。需要重算可点击生成候选点。")

    def reset_for_plan(self, keep_start=True):
        if self.queue_running:
            self.stop_queue("计划或场景切换，队列授权已失效。")
        key = (str(self.console.loaded_plan_path), self.console.loaded_plan_hash)
        if not keep_start or self.sequence_key != key:
            self.step_index, self.sequence_complete = 0, False
            self.sequence_key = key
        if not keep_start:
            self.navigation_session = NavigationSession()
            self.segment_state.set("场景已加载；请测量首次起点。此操作不会重置狗端坐标系。")
        self.reset_confirmations(clear_start=not keep_start)
        self.set_goal(None, automatic=True)
        self.auto_key = None
        self.mapping_note.set("等待规划通过与当前起点；自动填写后仍需人工核对。")

    def maybe_auto_map(self):
        c = self.console
        if self.active or self.uncertain or self.sequence_complete or self.auto_suppressed or not c.loaded_plan_path or c.api.busy:
            return
        key = self.mapping_key()
        if self.auto_key == key:
            return
        try:
            summary = read(Path(c.loaded_plan_path).with_name("planning_summary.json"))
        except (OSError, ValueError):
            return  # May still be writing; never use an incomplete result.
        if summary.get("status") != "planning_json_generated_not_executed":
            self.mapping_note.set("规划尚未通过，不自动生成导航点。")
            return
        if "multi_navigation" in c.__dict__:
            try:
                if len(navigation_steps(read(c.loaded_plan_path))) > 1:
                    self.mapping_note.set("多目标计划：全部候选终点自动显示在“系统 2 · 多目标导航”，请在该页核对整队。")
                    return
            except (OSError, ValueError, KeyError, TypeError):
                return
        if any(not v.get().strip() for v in self.start.values()):
            self.mapping_note.set("规划已通过；请填写当前 x、y、yaw，随后自动填入候选点（不默认零点）。")
            return
        self.auto_key = key
        try:
            self.generate_point()
        except (ValueError, KeyError, OSError, TypeError) as exc:
            self.mapping_note.set("未自动填写，请核对起点/计划：" + str(exc))

    def dispatch(self, queue_authorized=False):
        if self.queue_running:
            if not queue_authorized:
                raise ValueError("队列已启动；不能同时手动发送另一段。")
            if (self.queue_key != self.mapping_plan_key() or self.connection_id != self.queue_connection):
                raise ValueError("队列计划、场景或连接已变化，需要重新核对。")
            item = next((r for r in self.queue_route if r["step_index"] == self.step_index), None)
            if item is None or {k: float(v.get()) for k, v in self.goal.items()} != item["goal"]:
                raise ValueError("当前终点不在已核对的队列中。")
        elif queue_authorized:
            raise ValueError("没有有效的整队授权。")
        if self.sequence_complete:
            raise ValueError("当前工程计划已完成，请加载下一份计划；不会重复发送最后一段。")
        if hasattr(self.console, "api") and self.console.api.busy:
            raise ValueError("API 规划尚未结束，不能发送导航。")
        if self.uncertain or PENDING.exists():
            raise ValueError("存在尚未核对的执行状态，禁止重发。先现场确认机器人停止。")
        if not self.client or self.client.closed.is_set() or self.active:
            raise ValueError("需要有效控制连接，且不能有活动任务。")
        if not self.hardware.get():
            raise ValueError("必须确认真实控制出口、定位和操作人员就绪。")
        config, request = self.preview()
        if request["blockers"]:
            raise ValueError("检查项未完成：\n" + "\n".join(request["blockers"]))
        status = self.client.status or {}
        if status.get("coordinate_frame") != request["coordinate_frame"] or status.get("busy") or not status.get("motion_enabled") or not status.get("action_available"):
            raise ValueError("控制接口未就绪、正忙或场景系不一致。")
        text = f"将驱动真实机器人，仅发送本段。\n第 {self.step_index+1}/{config['navigation_step_count']} 段，目标 ID {request['target_element_id']}\n场景系起点 {config['start_pose']}\n场景系终点 {config['approach_pose']}\n后续沿用场景坐标系，不归零。参数保持现场值。\n确认现场净空、对齐正确且遥控器在手？"
        if not queue_authorized and not messagebox.askyesno("真实运动确认", text):
            return
        self.run_dir = ROOT / "outputs/real_scene_runs" / ("tcp_navigation_" + datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8])
        self.run_dir.mkdir(parents=True, exist_ok=False)
        for name, data in (("dispatch_config.json", config), ("navigation_request.json", request)):
            (self.run_dir / name).write_text(json.dumps(data, indent=2, ensure_ascii=True, allow_nan=False), encoding="utf-8")
        PENDING.parent.mkdir(parents=True, exist_ok=True)
        self.current_request_id = str(uuid.uuid4())
        with PENDING.open("x", encoding="utf-8") as stream:
            json.dump({"request_id": self.current_request_id, "run_dir": str(self.run_dir), "source_action_id": request["source_action_id"], "status": "send_reserved_result_unknown"}, stream)
        self.request, self.active, self.uncertain = request, True, True
        self.dispatch_scene_key = self.scene_key()
        self.navigation_session.sent(config["start_pose"], self.dispatch_scene_key)
        self.console.tail = None
        self.last_pose_time = None
        self.record({"type": "dispatch_reserved", "data": request})
        try:
            request_id = self.client.navigate(request, request_id=self.current_request_id)
            self.state.set("已发送一次导航，等待 ROS 接收/结果；请求 " + str(request_id))
        except Exception as exc:
            self.state.set("发送状态未知，请现场检查；不会重试：" + str(exc))
            raise

    def cancel(self):
        self.stop_queue("后续队列已停止；当前段取消仍需等待服务端终态。")
        if self.client:
            try:
                self.client.cancel()
                self.state.set("已请求取消，尚未证明停止；必要时立即使用物理急停。")
            except Exception as exc:
                self.state.set("取消未送达，使用物理急停并核对状态：" + str(exc))

    def disconnect(self):
        if self.active and not messagebox.askyesno("断开不等于停止", "将尝试取消，但断线不能保证机器人停止。确认已做好物理停止准备并断开？"):
            return
        if self.client:
            self.stop_queue("连接断开，队列授权失效。")
            self.client.close()

    def mapping_plan_key(self):
        return (str(self.console.loaded_plan_path), self.console.loaded_plan_hash, self.scene_key(),self.clearance())

    def stop_queue(self, reason):
        self.queue_running = False
        self.queue_next_deadline = None
        self.queue_ready_at = None
        self.segment_state.set(reason)

    def start_queue(self, route):
        if self.queue_running or self.active or self.uncertain or PENDING.exists():
            raise ValueError("已有队列、活动目标或未决状态。")
        if not self.client or self.client.closed.is_set() or not self.hardware.get():
            raise ValueError("请先在控制页连接并确认真实控制出口。")
        if self.console.api.busy:
            raise ValueError("规划尚未完成。")
        c = self.console
        steps = navigation_steps(read(c.loaded_plan_path))
        if [r["step_index"] for r in route] != list(range(self.step_index, len(steps))):
            raise ValueError("队列必须完整包含工程计划剩余各段，不能跳过或调换顺序。")
        confirmations = {k: v.get() for k, v in self.confirmations.items()}
        confirmations["approach_pose_verified"] = True  # Explicit whole-route review checkbox + dialog.
        start = {k: float(v.get()) for k, v in self.start.items()}
        for item in route:
            cfg = build_config(c.loaded_plan_path, c.loaded_plan_hash, c.scene, start,
                               item["goal"], confirmations, item["step_index"],self.clearance())
            req = prepare(cfg)
            if req["blockers"] or item["target_element_id"] != req["target_element_id"]:
                raise ValueError("请完成控制页检查项并核对队列：" + str(req["blockers"]))
        description = "\n".join(f"{r['step_index']+1}. ID {r['target_element_id']} → {r['goal']}" for r in route)
        if not messagebox.askyesno("真实运动：整条队列授权",
                "将依次执行下列已核对目标。每段成功并返回有效实际位姿后停留 10 秒，再继续；失败即暂停。\n"
                "场景坐标系沿用首次对齐，运动参数不变。\n" + description + "\n确认执行整条队列？"):
            return
        self.queue_route = json.loads(json.dumps(route, allow_nan=False))
        self.queue_key, self.queue_connection = self.mapping_plan_key(), self.connection_id
        self.queue_confirmations = confirmations
        self.set_goal(self.queue_route[0]["goal"], automatic=False)
        self.confirmations["approach_pose_verified"].set(True)
        self.queue_running = True
        try:
            self.dispatch(queue_authorized=True)
        except Exception:
            self.stop_queue("队列启动未完成，请核对日志和当前执行状态；不自动重试。")
            raise

    def continue_queue(self):
        if not self.queue_running or self.queue_next_deadline is None or self.active or self.uncertain:
            return
        try:
            if (not self.client or self.client.closed.is_set() or self.connection_id != self.queue_connection
                    or self.mapping_plan_key() != self.queue_key):
                raise ValueError("连接、计划或场景变化")
            if self.queue_ready_at is not None and time.monotonic() < self.queue_ready_at:
                remaining = max(0, int(self.queue_ready_at-time.monotonic()+.999))
                self.segment_state.set(f"已到达上一目标，停留中：约 {remaining} 秒后继续；坐标系保持不变。")
                return
            if (self.client.status or {}).get("busy"):
                if time.monotonic() < self.queue_next_deadline:
                    return  # Wait for the existing heartbeat to report idle; no navigation retry.
                raise ValueError("上一段终态后，服务端仍未报告空闲")
            self.queue_next_deadline = None
            self.queue_ready_at = None
            self.dispatch(queue_authorized=True)
        except Exception as exc:
            self.stop_queue("队列已暂停：" + str(exc))
            self.console.log("多目标队列暂停：" + str(exc))

    def clear_unknown(self):
        if self.client and not self.client.closed.is_set() and (self.client.active or (self.client.status or {}).get("busy")):
            messagebox.showerror("不能解除", "网桥仍报告任务活动或状态未决。先处理现场停止与终态。")
            return
        if messagebox.askyesno("人工确认，不是停止命令", "你是否已现场确认机器人停止、核对服务端状态，并准备重新测量实际起点？此操作只解除本地阻止，不停止狗，也不清除狗端锁。"):
            if PENDING.exists():
                PENDING.rename(PENDING.with_name("operator_cleared_" + uuid.uuid4().hex + ".json"))
            self.uncertain, self.active = False, False
            self.reset_confirmations()
            self.navigation_session.invalidate_pose()
            self.state.set("本地未知状态已由人工确认解除；请重新核对起点和全部检查项。")

    def reset_confirmations(self, clear_start=True):
        for value in self.confirmations.values():
            value.set(False)
        self.hardware.set(False)
        if clear_start:
            self.set_start(None)

    def set_start(self, pose):
        self.setting_start = True
        try:
            for key, value in self.start.items():
                value.set(repr(pose[key]) if pose is not None else "")
        finally:
            self.setting_start = False

    def record(self, row):
        row = {"request_id": self.current_request_id,
               "coordinate_frame": self.request.get("coordinate_frame") if self.request else None,
               **row}
        if row.get("type") == "dispatch_reserved":
            row["audit_context"] = {"engineering_plan": str(self.console.loaded_plan_path),
                "engineering_plan_sha256": self.console.loaded_plan_hash,
                "instruction": read(self.console.loaded_plan_path).get("instruction"),
                "navigation_step_index": self.step_index, "run_directory": str(self.run_dir)}
        if hasattr(self.console, "navigation_view"):
            try:
                self.console.navigation_view.observe({"request_id": self.current_request_id,
                    "coordinate_frame": self.request.get("coordinate_frame") if self.request else None, **row}, self.run_dir)
            except Exception as exc:
                self.console.log("可视化记录失败（控制状态不变）："+str(exc))
        if self.run_dir:
            value = {"received_utc": datetime.now(timezone.utc).isoformat(), **clean(row)}
            with (self.run_dir / "tcp_events.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(value, ensure_ascii=True, allow_nan=False) + "\n")

    def show_outcome(self,row):
        value=outcome(row)
        key=(str(self.console.loaded_plan_path),self.console.loaded_plan_hash,self.step_index)
        self.__dict__.setdefault("outcomes",{})[key]=value
        variable=self.__dict__.get("outcome_state")
        if variable is not None:
            target=(self.request or {}).get("target_element_id","?")
            variable.set(f"GO 执行结果：第 {self.step_index+1} 段 / ID {target} — {value['label_zh']}；{value['reason']}")
        if value['code'] not in ('pending','running'):
            self.console.log("GO 任务终态："+json.dumps(value,ensure_ascii=False))
        return value

    def finish_segment(self, row):
        """Carry actual arrival forward, but never dispatch the next goal automatically."""
        self.show_outcome(row)
        kind, data = row["type"], row.get("data", {})
        if kind != "result":
            # Rejection is not motion; retain the measured start and current index.
            self.reset_confirmations(clear_start=False)
            self.stop_queue("目标被拒绝，队列已暂停；不会跳过或重试。")
            return
        value = None
        if (self.dispatch_scene_key == self.scene_key() and self.request
                and row.get("coordinate_frame") == self.request["coordinate_frame"]):
            value = self.navigation_session.finish(data, row.get("request_id"), self.scene_key())
        arrived = (self.dispatch_scene_key == self.scene_key() and self.request
                   and row.get("coordinate_frame") == self.request["coordinate_frame"]
                   and data.get("ros_goal_status") == 4 and data.get("result", {}).get("success") is True
                   and data.get("result", {}).get("reason") == "arrived")
        self.reset_confirmations()
        if value is not None:
            self.set_start(value)
            self.segment_state.set(("已到达；" if arrived else "上一段未成功到达；") +
                "已接续服务端实际终态位姿（场景系）。核对停止且期间未移动狗，再规划/发送下一任务，不归零。")
        else:
            self.navigation_session.invalidate_pose()
            self.segment_state.set("终态未提供有效场景位姿：请在原场景系重新测量当前位置，不重置原点。")
        if arrived:
            self.step_index += 1
            count = self.request["navigation_step_count"]
            self.sequence_complete = self.step_index >= count
            self.set_goal(None, automatic=True)
            self.auto_key = None
            if self.sequence_complete:
                self.mapping_note.set("本计划全部导航段已完成；加载下一任务时保留已接续的实际位姿。")
            else:
                self.mapping_note.set(f"准备第 {self.step_index+1}/{count} 段；沿用场景系，核对候选点后手动发送。")
        if self.queue_running:
            if not arrived or value is None:
                self.stop_queue("本段未成功到达或缺少有效位姿，队列已暂停；不会跳过或重试。" +
                                ("已保留实际终态起点，可核对后重新规划。" if value is not None else "需要核对当前位置。"))
            elif self.sequence_complete:
                self.stop_queue("整条导航队列已完成；保持场景坐标系和最后实际位姿。")
            else:
                item = next(r for r in self.queue_route if r["step_index"] == self.step_index)
                self.set_goal(item["goal"], automatic=False)
                for key, var in self.confirmations.items():
                    var.set(self.queue_confirmations[key])
                self.hardware.set(True)
                self.queue_ready_at = time.monotonic() + 10
                self.queue_next_deadline = self.queue_ready_at + 6
                self.mapping_note.set(f"已到达并停留 10 秒，随后继续第 {self.step_index+1}/{count} 段。")
                self.record({"type": "queue_dwell_started", "data": {"duration_seconds": 10, "next_step_index": self.step_index}})
        self.record({"type": "segment_completed", "data": {
            "next_step_index": self.step_index, "sequence_complete": self.sequence_complete,
            "carried_start_pose": value, "origin_reset_requested": False}})

    def poll(self):
        try:
            if self.client and hasattr(self.client, "touch"):
                self.client.touch()
            while not self.events.empty():
                row = self.events.get_nowait()
                if row.get("client_connection_id") != self.connection_id:
                    continue
                kind, data = row.get("type"), row.get("data", {})
                if kind not in ("hello", "status", "connect_failed", "connection_lost") and row.get("request_id") != self.current_request_id:
                    continue
                self.record(row)
                if kind == "goal_accepted" or (kind in ("connection_lost","execution_state_unknown") and self.active):
                    self.show_outcome(row)
                if kind == "hello" or (kind == "status" and self.request is None):
                    direct = row.get("transport") == "ros2_direct"
                    label = "ROS2 直连（无 TCP 网桥）" if direct else "TCP 协议已验证"
                    boundary = "场景名是本地声明；不验证狗端地图/对齐，也不检测其他客户端任务。" if direct else "不代表 SportSink/定位/实际限速已验证。"
                    self.state.set(f"{label} · Action={'就绪' if row.get('action_available') else '未发现'} · 允许发送={row.get('motion_enabled')} · 本连接忙={row.get('busy')}\n{boundary}")
                elif kind == "feedback":
                    if self.request and row.get("coordinate_frame") == self.request["coordinate_frame"]:
                        pose = feedback_pose({"event": "feedback", "data": data}, "scene")
                        if pose:
                            self.console.trajectory.append(pose)
                            self.console.trajectory = self.console.trajectory[-5000:]
                            self.console.draw()
                            self.last_pose_time = time.monotonic()
                    self.console.trace_status.set(f"导航实时反馈：{data.get('stage')} · 剩余 {data.get('remaining_dist')} m（里程计估计，不是独立真值）")
                elif kind in ("result", "goal_rejected", "dispatch_rejected"):
                    self.active, self.uncertain = False, False
                    PENDING.unlink(missing_ok=True)
                    self.finish_segment(row)
                    if kind == "result":
                        result = data.get("result", {})
                        self.state.set(f"服务端终态：success={result.get('success')} reason={result.get('reason')}；不会重试，请核对实际停止与位置。")
                    else:
                        self.state.set("目标被拒绝：" + str(data))
                    if self.run_dir:
                        (self.run_dir / "run_summary.json").write_text(json.dumps({"status": kind, "data": clean(data), "execution_outcome":outcome(row), "physical_arrival_independently_verified": False}, indent=2), encoding="utf-8")
                    self.current_request_id = None  # Ignore duplicate/late terminal packets.
                elif kind in ("execution_state_unknown", "connection_lost"):
                    self.stop_queue("连接或执行状态未核实，队列已暂停。")
                    self.navigation_session.invalidate_pose()
                    self.reset_confirmations()
                    self.segment_state.set("连接或执行状态未核实；请测量原场景系当前位置，不会自动归零。")
                    if self.active:
                        self.uncertain = True
                    self.state.set("连接/执行状态需核对：" + str(data) + "；不自动重发。")
                elif kind == "connect_failed":
                    self.state.set("连接失败，未发送导航：" + str(data))
                elif kind == "cancel_ack":
                    self.state.set("取消请求已应答，但仍需终态/现场确认；不能当作已经停止。")
                if kind not in ("status", "feedback"):
                    self.console.log("控制接口 " + str(kind) + " " + str(data))
            self.continue_queue()
            self.maybe_auto_map()
            online = self.client is not None and not self.client.closed.is_set() and self.client.status is not None
            planning_busy = hasattr(self.console, "api") and self.console.api.busy
            ready = online and self.console.loaded_plan_path is not None and not self.sequence_complete and not self.queue_running and not self.active and not self.uncertain and not planning_busy
            for button in (self.send_button, self.console.send_button):
                button.configure(state="normal" if ready else "disabled")
            self.cancel_button.configure(state="normal" if online and self.active else "disabled")
            if self.last_pose_time and time.monotonic()-self.last_pose_time > 3:
                self.console.trace_status.set("位姿反馈已过期；图中是最后已知位置，不能据此认定机器人停止。")
        except Exception as exc:
            self.stop_queue("控制观察异常，后续队列已暂停。")
            self.uncertain = self.uncertain or self.active
            self.state.set("控制观察错误，需人工核对：" + str(exc))
        self.after(200, self.poll)
