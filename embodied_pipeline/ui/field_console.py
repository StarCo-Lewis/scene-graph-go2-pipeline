"""Chinese Tk console with explicit authenticated single-goal navigation dispatch."""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import queue
import socket
import threading
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from .._legacy import ROOT
from . import field_backend as backend

BG, INK, BLUE = "#f3f6fc", "#17253e", "#2563eb"


def configure_dpi():
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except (OSError, AttributeError):
            pass


class Console:
    def __init__(self, root, scene_path):
        self.root, self.scene, self.session, self.tail = root, None, None, None
        self.events = queue.Queue()
        self.plan_seen, self.summary_seen, self.trajectory = None, None, []
        self.last_feedback = None
        self.log_offsets = {}
        self.loaded_plan_path, self.loaded_plan_hash = None, None
        self.navigation_mode = tk.StringVar(value="single_goal")
        self.status = tk.StringVar(value="本地查看模式 · 尚未连接机器人")
        self.connection = tk.StringVar(value="尚未检查 TCP；端口连通不等于导航就绪")
        self.worker_state = tk.StringVar(value="WSL 规划后端：未连接")
        self.trace_status = tk.StringVar(value="暂无反馈；不模拟机器人位置")
        self.scene_title = tk.StringVar(value="场景未加载")
        self.filter = tk.StringVar()
        self.host = tk.StringVar(value="192.168.123.18")
        self.port = tk.StringVar(value="8888")
        self.frame = tk.StringVar()
        self.frame_confirmed = tk.BooleanVar(value=False)
        self.python = tk.StringVar(value="/home/lewis/miniforge3/envs/conceptgraph/bin/python")
        self.embed_python = tk.StringVar(value="/home/lewis/miniforge3/envs/skill_embed/bin/python")
        self.embed_model = tk.StringVar(value="/mnt/d/NebulaAI/models/skill_embedding_minilm_v1")
        self.build()
        self.load_scene(scene_path)
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.after(500, self.poll)

    def build(self):
        r = self.root
        # Keep controls reachable on high-DPI laptops; this affects only this Tk app.
        r.tk.call("tk", "scaling", 1.4)
        r.title("D3-619 · 现场任务与导航观察台")
        r.geometry(f"{min(1380, r.winfo_screenwidth()-60)}x{min(900, r.winfo_screenheight()-80)}")
        r.minsize(1040, 840)
        r.configure(bg=BG)
        r.option_add("*Font", ("Microsoft YaHei UI", 10))
        style = ttk.Style(r)
        style.theme_use("clam")
        style.configure("TButton", padding=(10, 7))
        style.configure("Primary.TButton", background=BLUE, foreground="white")
        header = tk.Frame(r, bg="#17253e", padx=20, pady=12)
        header.pack(fill="x")
        tk.Label(header, text="现场任务工作台", fg="white", bg="#17253e", font=("Microsoft YaHei UI", 19, "bold")).pack(side="left")
        tk.Label(header, textvariable=self.scene_title, fg="#cbd5e1", bg="#17253e").pack(side="right")
        tk.Label(r, text="仅导航任务｜发送前须握手、映射导航点并人工确认｜运动参数保持现场值｜关闭窗口不能急停", bg="#fff1d6", fg="#8a4b0f", pady=8).pack(fill="x")
        tk.Label(r, textvariable=self.status, bg=BG, fg=INK, anchor="w", padx=15, pady=7).pack(side="bottom", fill="x")
        tabs = ttk.Notebook(r)
        tabs.pack(fill="both", expand=True, padx=12, pady=10)
        task_tab, settings, logs_tab = (ttk.Frame(tabs, padding=10) for _ in range(3))
        tabs.add(task_tab, text="系统 1 · 逐次任务 / 场景")
        tabs.add(settings, text="连接 / 规划环境 / 反馈")
        from .api_controls import ApiPanel
        self.api = ApiPanel(tabs, self)
        tabs.add(self.api, text="API 规划")
        from .navigation_controls import NavigationPanel
        self.controls = NavigationPanel(tabs, self)
        tabs.add(self.controls, text="导航点映射 / 控制")
        from .multi_navigation import MultiNavigationPanel
        self.multi_navigation = MultiNavigationPanel(tabs, self)
        tabs.add(self.multi_navigation, text="系统 2 · 多目标导航")
        from .navigation_view import NavigationView
        self.navigation_view = NavigationView(tabs, self)
        tabs.add(self.navigation_view, text="导航地图 / 轨迹")
        tabs.add(logs_tab, text="详细日志")
        panes = ttk.Panedwindow(task_tab, orient="horizontal")
        panes.pack(fill="both", expand=True)
        left, right = ttk.Frame(panes, width=420), ttk.Frame(panes)
        panes.add(left, weight=2)
        panes.add(right, weight=4)
        self.root.after(100, lambda: panes.sashpos(0, max(390, int(panes.winfo_width()*.35))))
        row = ttk.Frame(left)
        row.pack(fill="x")
        ttk.Label(row, text="1 浏览场景（可选，不约束指令）").pack(side="left")
        ttk.Button(row, text="加载场景", command=self.browse_scene).pack(side="right")
        ttk.Entry(left, textvariable=self.filter).pack(fill="x", pady=5)
        self.filter.trace_add("write", lambda *a: self.fill_objects())
        objects_box = ttk.Frame(left)
        objects_box.pack(fill="both", expand=True)
        self.objects = ttk.Treeview(objects_box, columns=("id", "name"), show="headings", height=7, selectmode="browse")
        self.objects.heading("id", text="ID")
        self.objects.heading("name", text="场景标签（可搜索英文 / ID）")
        self.objects.column("id", width=50, stretch=False)
        self.objects.column("name", width=280)
        scroll = ttk.Scrollbar(objects_box, command=self.objects.yview)
        self.objects.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        self.objects.pack(fill="both", expand=True)
        self.objects.bind("<<TreeviewSelect>>", lambda e: self.selection_changed())
        self.detail = tk.Text(left, width=1, height=4, wrap="word", bg="#eaf0fa", relief="flat")
        self.detail.pack(fill="x", pady=5)
        ttk.Label(left, text="2 输入英文任务（中文界面，不自动翻译）").pack(anchor="w")
        self.instruction = tk.Text(left, width=1, height=4, wrap="word")
        self.instruction.pack(fill="x", pady=4)
        buttons = ttk.Frame(left)
        buttons.pack(fill="x")
        ttk.Button(buttons, text="用所选物体填写示例", command=self.example).pack(side="left")
        ttk.Button(buttons, text="准备任务 / 生成命令", style="Primary.TButton", command=self.prepare).pack(side="right")
        ttk.Label(left, text="3 在 API 页配置密钥，然后点击下方调用规划", wraplength=390).pack(anchor="w", pady=(10, 3))
        self.command_box = tk.Text(left, width=1, height=3, wrap="char", font=("Consolas", 9))
        self.command_box.pack(fill="x")
        ttk.Button(left, text="复制 WSL 规划命令（不会自动执行）", command=self.copy_command).pack(fill="x", pady=4)
        self.submit_button = ttk.Button(left, text="调用 API 规划当前指令（不执行导航）", command=lambda: self.guarded(self.api.start))
        self.submit_button.pack(fill="x")
        ttk.Label(left, textvariable=self.worker_state).pack(anchor="w", pady=3)
        ttk.Label(right, text="场景图 XY 投影（米）· 方框表示物体，不表示净空", wraplength=600).pack(anchor="w")
        self.canvas = tk.Canvas(right, bg="white", highlightthickness=0, height=380)
        self.canvas.pack(fill="both", expand=True, pady=6)
        self.canvas.bind("<Configure>", lambda e: self.draw())
        self.canvas.bind("<Button-1>", self.canvas_select)
        self.canvas_items = {}
        ttk.Label(right, textvariable=self.trace_status, wraplength=700).pack(anchor="w")
        action_bar = ttk.Frame(right)
        action_bar.pack(fill="x", pady=6)
        ttk.Button(action_bar, text="加载已有工程 JSON", command=self.browse_plan).pack(side="left")
        self.send_button = ttk.Button(action_bar, text="确认后发送导航", state="disabled", command=lambda: self.guarded(self.controls.dispatch))
        self.send_button.pack(side="right")
        self.plan_box = tk.Text(right, width=1, height=9, wrap="word", bg="#edf2fa", relief="flat")
        self.plan_box.pack(fill="x")
        self.set_text(self.plan_box, "暂无当前任务的规划结果。\n选择目标、输入任务，或显式加载已有工程 JSON。\n不会把历史成功当作本次成功，也不会用动画模拟实跑。")
        self.build_settings(settings)
        self.logs = tk.Text(logs_tab, wrap="word", bg="#132136", fg="#d7e4fb", font=("Consolas", 10))
        self.logs.pack(fill="both", expand=True)

    def build_settings(self, frame):
        def entry(label, variable):
            row = ttk.Frame(frame)
            row.pack(fill="x", pady=5)
            ttk.Label(row, text=label, width=24).pack(side="left")
            ttk.Entry(row, textvariable=variable).pack(side="left", fill="x", expand=True)
        entry("狗的 IP", self.host)
        entry("TCP 端口", self.port)
        ttk.Button(frame, text="仅检查 TCP 连通（不发送数据）", command=self.check_tcp).pack(anchor="w", pady=5)
        ttk.Label(frame, textvariable=self.connection).pack(anchor="w")
        ttk.Separator(frame).pack(fill="x", pady=16)
        entry("WSL 规划 Python", self.python)
        ttk.Label(frame, text="推荐在 API 页输入密钥并点击规划：窗口会按你的点击启动 WSL，密钥不落盘。\n以下手动命令/队列为备用模式；不要重复提交同一任务。窗口不会启动狗上节点。", justify="left").pack(anchor="w", pady=9)
        ttk.Button(frame, text="复制规划后端启动命令（在 WSL 只启动一次）", command=self.copy_worker).pack(anchor="w")
        ttk.Button(frame, text="备用：提交已准备任务给手动启动的 WSL 后端", command=self.submit).pack(anchor="w", pady=4)
        ttk.Separator(frame).pack(fill="x", pady=12)
        entry("已确认对应场景系的反馈 frame_id", self.frame)
        ttk.Checkbutton(frame, text="已核对反馈坐标与当前场景相同（否则不绘制机器人轨迹）", variable=self.frame_confirmed, command=self.clear_trace).pack(anchor="w")
        self.frame.trace_add("write", lambda *a: self.clear_trace())
        ttk.Button(frame, text="选择 events.jsonl 反馈日志（只读观察 / 回放）", command=self.browse_events).pack(anchor="w", pady=8)
        ttk.Label(frame, text="历史日志可在此回放；配套 TCP 网桥的连接、现场检查与发送在“导航点映射 / 控制”页。\n只显示实际反馈位姿；旧日志标注回放。急停请使用现场遥控器。", justify="left").pack(anchor="w", pady=6)

    def log(self, message):
        self.logs.insert("end", datetime.now().strftime("%H:%M:%S ") + message + "\n")
        if int(self.logs.index("end-1c").split(".")[0]) > 1800:
            self.logs.delete("1.0", "300.0")
        self.logs.see("end")

    def set_text(self, widget, value):
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", value)
        widget.configure(state="disabled")

    def guarded(self, fn):
        try:
            return fn()
        except Exception as exc:
            uncertain = hasattr(self, "controls") and (self.controls.active or self.controls.uncertain)
            self.status.set("操作未完成；执行状态需人工核对，不会重发" if uncertain else "操作未完成；未发送导航")
            self.log(type(exc).__name__ + ": " + str(exc))
            messagebox.showerror("检查输入或文件", str(exc), parent=self.root)

    def load_scene(self, path):
        if hasattr(self, "api") and self.api.busy:
            raise ValueError("API 规划期间不能切换场景。")
        if hasattr(self, "controls") and self.controls.active:
            raise ValueError("活动导航期间不能切换场景。")
        scene = backend.load_scene(path)
        self.loaded_plan_path, self.loaded_plan_hash = None, None
        self.scene = scene
        self.controls.reset_for_plan(keep_start=False)
        self.session, self.plan_seen, self.summary_seen, self.tail = None, None, None, None
        self.frame_confirmed.set(False)
        self.frame.set(scene["metadata"]["coordinate_frame"])
        self.scene_title.set(f"{scene['metadata'].get('scene_id', 'Unknown')} · {len(scene['nodes'])} 个物体 · 未连接真机")
        self.set_text(self.plan_box, "场景已加载，当前没有已确认的导航计划。")
        self.set_text(self.command_box, "")
        self.fill_objects()
        if self.objects.exists("179"):
            self.objects.selection_set("179")
            self.objects.see("179")
        self.draw()
        self.log("加载场景：" + str(path))

    def browse_scene(self):
        value = filedialog.askdirectory(title="选择含 scene_graph.json 的目录")
        if value:
            self.guarded(lambda: self.load_scene(value))

    def fill_objects(self):
        if not self.scene:
            return
        selected = self.objects.selection()
        for iid in self.objects.get_children():
            self.objects.delete(iid)
        term = self.filter.get().strip().lower()
        for node in self.scene["nodes"]:
            label = node.get("object_tag", "object")
            if term in f"{node['id']} {label} {node.get('caption', '')}".lower():
                self.objects.insert("", "end", iid=str(node["id"]), values=(node["id"], label))
        if selected and self.objects.exists(selected[0]):
            self.objects.selection_set(selected)

    def selected(self):
        selected = self.objects.selection()
        if not selected:
            raise ValueError("请先从左侧列表选择一个导航目标。")
        return next(n for n in self.scene["nodes"] if n["id"] == int(selected[0]))

    def selection_changed(self):
        if self.objects.selection():
            node = self.selected()
            self.set_text(self.detail, f"ID {node['id']} · {node.get('caption', '')}\n中心 {node['bbox_center']}（不是机器人停靠点）")
        self.draw()

    def example(self):
        def run():
            node = self.selected()
            text = f"Navigate near the {node.get('object_tag', 'object')} (ID {node['id']}) and stop. Do not pick up, lift, place, or move any object."
            self.instruction.delete("1.0", "end")
            self.instruction.insert("1.0", text)
        self.guarded(run)

    def prepare(self):
        def run():
            if self.api.busy:
                raise ValueError("API 规划正在运行，不重复准备或提交任务。")
            if self.controls.active or self.controls.queue_running:
                raise ValueError("活动导航期间不能准备另一条任务，请先核对本次结果。")
            if self.session and not (self.session / "planning/planning_summary.json").exists():
                if not messagebox.askyesno("切换观察任务", "上一条任务尚无结果。继续会切换窗口观察对象，但不会停止此前命令。是否继续？"):
                    return
            self.navigation_mode.set("single_goal")
            self.session = backend.create_task(self.scene, self.instruction.get("1.0", "end"), "single_goal")
            self.loaded_plan_path, self.loaded_plan_hash = None, None
            self.controls.reset_for_plan()
            cmd = backend.command(self.session, self.python.get(), self.embed_python.get(), self.embed_model.get())
            self.set_text(self.command_box, cmd)
            self.set_text(self.plan_box, "任务文件已写入，尚未调用模型。请复制命令到 WSL 执行。\n" + str(self.session))
            self.plan_seen = self.summary_seen = None
            self.status.set("任务已准备，等待你在 WSL 启动规划；未发送导航")
            self.log("新任务：" + str(self.session))
        self.guarded(run)

    def copy_command(self):
        value = self.command_box.get("1.0", "end").strip()
        if value:
            self.root.clipboard_clear()
            self.root.clipboard_append(value)
            self.status.set("命令已复制；在已设置 API key 的 WSL 终端执行，只生成计划")

    def copy_worker(self):
        def run():
            value = backend.worker_command(self.python.get(), self.embed_python.get(), self.embed_model.get())
            self.root.clipboard_clear()
            self.root.clipboard_append(value)
            self.status.set("后端启动命令已复制。请在已设置 API key 的 WSL 中执行一次，再回窗口提交任务。")
        self.guarded(run)

    def submit(self):
        def run():
            if self.api.busy or (self.session and (self.session / "api_launch.json").exists()):
                raise ValueError("本任务已从 API 入口启动，不再提交备用队列。")
            if self.session is None:
                raise ValueError("请先点击“准备任务 / 生成命令”。")
            saved = backend.read(self.session / "task.json")
            if saved["instruction"] != self.instruction.get("1.0", "end").strip():
                raise ValueError("指令已经改变，请重新准备任务，避免提交旧指令。")
            if not messagebox.askyesno("提交规划，不执行机器人", "将调用 LLM 生成计划，可能产生 API 费用；不会发送导航。确认提交？"):
                return
            backend.submit(self.session)
            self.status.set("任务已提交至 WSL 规划队列；等待结果，未发送导航")
            self.log("任务排队 " + self.session.name)
        self.guarded(run)

    def show_plan(self, path):
        value, steps, supported = backend.inspect_plan(path, self.scene)
        if self.controls.active:
            raise ValueError("活动导航期间不更换已加载的计划；可在完成后重新加载。")
        self.loaded_plan_path, self.loaded_plan_hash = Path(path), backend.sha(path)
        self.controls.reset_for_plan()
        lines = ["计划文件：" + str(path), "原始指令：" + value["instruction"],
                 "静态检查：" + str(value.get("static_review_status")),
                 "纯导航计划形状（逐段确认发送）：" + ("符合（不代表可执行）" if supported else "不符合，禁止发送"),
                 "执行状态：未由本窗口发送；物理可达性、停靠点和坐标对齐待确认。"]
        for i, step in enumerate(steps, 1):
            lines.append(f"{i}. {step['skill_id']}  {json.dumps(step.get('arguments'), ensure_ascii=False)}")
        self.set_text(self.plan_box, "\n".join(lines))
        self.log("加载工程计划（仅查看） " + str(path))
        if supported and len(steps) > 1:
            self.navigation_mode.set("ordered_goals")
            self.multi_navigation.instruction.delete("1.0", "end")
            self.multi_navigation.instruction.insert("1.0", value["instruction"])
            self.multi_navigation.auto_attempt = None
            self.multi_navigation.master.select(self.multi_navigation)

    def browse_plan(self):
        if self.api.busy:
            messagebox.showerror("规划进行中", "请等待本次 API 结果后再加载其它计划。")
            return
        path = filedialog.askopenfilename(title="选择 engineering_plan.json", filetypes=[("JSON", "*.json")])
        if path:
            self.guarded(lambda: self.show_plan(Path(path)))

    def check_tcp(self):
        def start():
            host, port = self.host.get().strip(), int(self.port.get())
            if not 1 <= port <= 65535:
                raise ValueError("端口必须在 1–65535 之间")
            if not messagebox.askyesno("仅建立 TCP 连接", f"检查 {host}:{port}，不发送应用数据、不调用导航。继续？"):
                return
            self.connection.set("正在检查 TCP…")
            def worker():
                try:
                    with socket.create_connection((host, port), timeout=3):
                        pass
                    result = "TCP 可连；未验证协议、节点、里程计或导航就绪"
                except OSError as exc:
                    result = "TCP 未连通：" + str(exc)
                self.events.put(result)
            threading.Thread(target=worker, daemon=True).start()
        self.guarded(start)

    def clear_trace(self):
        self.trajectory, self.last_feedback = [], None
        self.trace_status.set("轨迹已清空；等待坐标已确认的日志反馈")
        if hasattr(self, "canvas"):
            self.draw()

    def browse_events(self):
        if self.controls.active:
            messagebox.showerror("正在导航", "活动导航期间不能用历史日志覆盖实时轨迹。")
            return
        path = filedialog.askopenfilename(title="选择导航 events.jsonl", filetypes=[("JSONL", "*.jsonl")])
        if path:
            self.tail = backend.JsonlTail(path)
            self.clear_trace()
            self.log("观察日志（含历史回放）：" + path)

    def poll(self):
        try:
            worker = backend.worker_status()
            self.worker_state.set("API 规划：运行中" if self.api.busy else "备用队列后端：" + ("在线 / " + worker["status"] if worker else "未连接（不影响 API 按钮启动）"))
            while not self.events.empty():
                text = self.events.get_nowait()
                self.connection.set(text)
                self.log(text)
            if self.session:
                for log_path in (self.session / "worker.console.log", self.session / "planning/alignment.console.log"):
                    if log_path.exists():
                        offset = self.log_offsets.get(log_path, 0)
                        if log_path.stat().st_size < offset:
                            offset = 0
                        with log_path.open("rb") as stream:
                            stream.seek(offset)
                            block = stream.read(32768)
                            # Keep an incomplete final line for the next poll.
                            end = block.rfind(b"\n") + 1
                            if end:
                                self.log_offsets[log_path] = offset + end
                                for line in block[:end].decode("utf-8", errors="replace").splitlines():
                                    self.log(line)
                result_file = backend.QUEUE / (self.session.name + ".result.json")
                if result_file.exists():
                    result = backend.read(result_file)
                    if result.get("error"):
                        self.status.set("规划后端失败：" + result["error"])
                folder = self.session / "planning"
                path = folder / "engineering_plan.json"
                if path.exists() and self.plan_seen != path:
                    try:
                        self.show_plan(path)
                        self.plan_seen = path
                    except json.JSONDecodeError:
                        pass
                summary = folder / "planning_summary.json"
                if summary.exists():
                    try:
                        value = backend.read(summary)
                        state = str(value.get("status")) + " / " + str(value.get("stage"))
                        if state != self.summary_seen:
                            self.status.set("规划状态：" + state + "；未发送导航")
                            self.log(state + (" | " + value["error"] if value.get("error") else ""))
                            self.summary_seen = state
                    except json.JSONDecodeError:
                        pass
            if self.tail:
                changed = False
                for row in self.tail.poll():
                    self.log("日志事件 " + str(row.get("event")) + " " + str(row.get("utc", "")))
                    if row.get("event") == "feedback":
                        data = row.get("data", {})
                        self.trace_status.set(f"日志反馈：{data.get('stage', 'unknown')} · 剩余距离 {data.get('remaining_dist', 'unknown')} m；非独立定位真值")
                        pose = backend.feedback_pose(row, self.frame.get() if self.frame_confirmed.get() else None)
                        if pose:
                            self.trajectory.append(pose)
                            self.trajectory = self.trajectory[-5000:]
                            self.last_feedback = row.get("utc")
                            changed = True
                        else:
                            self.trace_status.set("反馈未绘图：坐标系未确认/不匹配，或位姿数据无效")
                if changed:
                    self.draw()
                if self.last_feedback:
                    try:
                        when = datetime.fromisoformat(self.last_feedback.replace("Z", "+00:00"))
                        age = (datetime.now(timezone.utc) - when).total_seconds()
                        self.trace_status.set(f"日志位姿 · {'历史回放/反馈已过期' if age > 3 or age < -3 else '近期记录（不等于直连订阅）'} · {max(0, age):.0f} 秒前 · 非独立定位真值")
                    except (ValueError, TypeError):
                        self.trace_status.set("日志位姿时间未知；不能声称实时")
        except Exception as exc:
            self.log("观察错误：" + str(exc))
            self.status.set("观察发生错误；未发送导航")
        self.root.after(600, self.poll)

    def draw(self):
        if not self.scene:
            return
        c = self.canvas
        c.delete("all")
        self.canvas_items = {}
        nodes = self.scene["nodes"]
        xmin = min(n["bbox_center"][0]-n["bbox_extent"][0]/2 for n in nodes)
        xmax = max(n["bbox_center"][0]+n["bbox_extent"][0]/2 for n in nodes)
        ymin = min(n["bbox_center"][1]-n["bbox_extent"][1]/2 for n in nodes)
        ymax = max(n["bbox_center"][1]+n["bbox_extent"][1]/2 for n in nodes)
        w, h = max(c.winfo_width(), 100), max(c.winfo_height(), 100)
        scale = min((w-60)/max(xmax-xmin, 1), (h-60)/max(ymax-ymin, 1))
        def xy(x, y):
            return w/2+(x-(xmin+xmax)/2)*scale, h/2-(y-(ymin+ymax)/2)*scale
        self.transform = xy
        selected = self.objects.selection()
        selected = int(selected[0]) if selected else None
        for node in nodes:
            x, y, _ = node["bbox_center"]
            dx, dy, _ = node["bbox_extent"]
            a, b = xy(x-dx/2, y+dy/2), xy(x+dx/2, y-dy/2)
            chosen = node["id"] == selected
            iid = c.create_rectangle(*a, *b, outline=BLUE if chosen else "#b9c7db", fill="#bfdbfe" if chosen else "", width=3 if chosen else 1)
            self.canvas_items[iid] = node["id"]
        if selected is not None:
            n = next(n for n in nodes if n["id"] == selected)
            x, y = xy(*n["bbox_center"][:2])
            c.create_oval(x-4, y-4, x+4, y+4, fill=BLUE, outline=BLUE)
            c.create_text(x+8, y-12, anchor="w", text=f"ID {selected} {n.get('object_tag', '')}", fill=BLUE)
        if len(self.trajectory) > 1:
            c.create_line(*[v for p in self.trajectory for v in xy(*p[:2])], fill="#069c82", width=3)
        if self.trajectory:
            x, y, yaw = self.trajectory[-1]
            p = xy(x, y)
            q = xy(x+.35*math.cos(yaw), y+.35*math.sin(yaw))
            c.create_oval(p[0]-6, p[1]-6, p[0]+6, p[1]+6, fill="#069c82", outline="white")
            c.create_line(*p, *q, arrow="last", fill="#069c82", width=3)
        c.create_text(12, 15, text="X →    Y ↑    物体投影 ≠ 障碍物栅格", anchor="w", fill="#64748b")
        metre = min(1, max(xmax-xmin, 1)/4)
        c.create_line(15, h-20, 15+metre*scale, h-20, width=2, fill=INK)
        c.create_text(15, h-30, text=f"{metre:g} m", anchor="w", fill=INK)

    def canvas_select(self, event):
        for iid in reversed(self.canvas.find_overlapping(event.x-3, event.y-3, event.x+3, event.y+3)):
            node = self.canvas_items.get(iid)
            if node is not None:
                self.filter.set("")
                self.objects.selection_set(str(node))
                self.objects.see(str(node))
                return

    def close(self):
        if self.api.busy and not messagebox.askyesno("规划仍在运行", "关闭窗口不保证停止 WSL 中已启动的 API 规划，仍可能计费；不会自动执行导航。确认关闭？"):
            return
        if self.controls.active and not messagebox.askyesno("关闭不是急停", "导航仍活动或状态未知。关闭窗口会尝试取消但不保证停止；请使用现场遥控器。确定关闭？"):
            return
        if self.controls.client:
            self.controls.client.close()
        try:
            self.navigation_view.close()
        except Exception as exc:
            self.log("可视化最终存档失败："+str(exc))
        self.root.destroy()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, default=backend.DEFAULT_SCENE)
    parser.add_argument("--smoke-test", action="store_true", help="Open, lay out, assert disabled motion, then close; no network")
    args = parser.parse_args(argv)
    configure_dpi()
    root = tk.Tk()
    try:
        app = Console(root, args.scene)
    except Exception as exc:
        if args.smoke_test:
            root.destroy()
            raise
        messagebox.showerror("窗口启动失败", str(exc), parent=root)
        root.destroy()
        return 2
    if args.smoke_test:
        root.update()
        assert app.send_button.instate(["disabled"])
        assert len(app.objects.get_children()) == len(app.scene["nodes"])
        assert app.session is None and app.tail is None
        assert app.submit_button.winfo_rooty() + app.submit_button.winfo_height() < root.winfo_rooty() + root.winfo_height()
        root.after(100, root.destroy)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
