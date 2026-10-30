"""API settings and one-click planning; explicit click is required to start WSL."""
import json
import queue
import threading
import tkinter as tk
from tkinter import ttk, messagebox

from . import api_planner, field_backend as backend


class ApiPanel(ttk.Frame):
    def __init__(self, parent, console):
        super().__init__(parent, padding=15)
        self.console, self.busy = console, False
        self.events = queue.Queue()
        self.key = tk.StringVar()
        self.model = tk.StringVar(value="qwen3.8-flash")
        self.url = tk.StringVar(value="https://dashscope.aliyuncs.com/compatible-mode/v1")
        self.distro = tk.StringVar()
        self.state = tk.StringVar(value="未调用 API；密钥仅保留在内存，不写入任务或命令行")
        ttk.Label(self, text="窗口输入任务 → API 完整规划 → 工程 JSON → 导航点映射 → 人工确认发送",
                  font=("Microsoft YaHei UI", 12, "bold")).pack(anchor="w", pady=8)
        for label, var, secret in (("API key", self.key, True), ("模型", self.model, False),
                                   ("API Base URL", self.url, False), ("WSL 发行版（空白为默认）", self.distro, False)):
            row = ttk.Frame(self)
            row.pack(fill="x", pady=8)
            ttk.Label(row, text=label, width=30).pack(side="left")
            ttk.Entry(row, textvariable=var, show="*" if secret else "").pack(side="left", fill="x", expand=True)
        ttk.Label(self, text="复用“连接 / 规划环境 / 反馈”页中的 WSL Python 路径（不再需要 embedding 环境）。\n"
            "点击规划才启动 WSL 和调用 API；打开窗口不会启动。保留分解、物体绑定、完整技能清单、技能矩阵和校验；不做 cosine 筛选。\n"
            "任务及场景内容会发送至填写的 API 地址；可能产生费用。自定义兼容服务/模型尚需验证。\n"
            "API 规划只生成提案，不会自动向狗发送；拔掉网线也能在有互联网时规划。", justify="left").pack(anchor="w", pady=12)
        row = ttk.Frame(self)
        row.pack(fill="x", pady=10)
        self.button = ttk.Button(row, text="为任务页当前指令调用 API 规划", command=lambda: console.guarded(self.start))
        self.button.pack(side="left")
        ttk.Button(row, text="清空窗口中的密钥", command=lambda: self.key.set("")).pack(side="left", padx=8)
        ttk.Label(self, textvariable=self.state, wraplength=1050).pack(anchor="w", pady=10)
        ttk.Label(self, text="规划失败请检查详细日志；不会生成假计划或自动重试整轮任务。\n"
            "API 调用沿用既有管线的超时/请求重试策略。关闭窗口不等于停止已启动的 WSL 规划。\n"
            "已有工程 JSON 可直接加载，不需要再调用模型。", justify="left").pack(anchor="w", pady=8)
        self.after(250, self.poll)

    def start(self, navigation_mode="single_goal"):
        c = self.console
        if self.busy or c.controls.active or c.controls.uncertain or c.controls.queue_running:
            raise ValueError("请先完成当前规划，或核对未决导航状态。")
        api_planner.validate_api(self.model.get(), self.url.get(), self.key.get())
        instruction = c.instruction.get("1.0", "end").strip()
        # Validate launch arguments before creating a task. No process is started here.
        command = api_planner.launch_command(c.python.get(), self.distro.get().strip())
        if not messagebox.askyesno("调用 API 规划，不执行机器人",
                f"服务地址：{self.url.get()}\n模型：{self.model.get()}\n目标由指令和场景绑定确定，列表选择不作为约束。\n"
                f"指令：{instruction}\n\n将启动本机 WSL，向上述服务发送任务和场景信息，可能计费。\n生成结果后仍需确认导航点与真实运动。继续？"):
            return
        folder = backend.create_task(c.scene, instruction, navigation_mode)
        c.navigation_mode.set(navigation_mode)
        config = api_planner.config_for(folder, self.model.get(), self.url.get(), self.key.get(),
                                        c.embed_python.get(), c.embed_model.get())
        # Exact secret is only sent down the pipe. Public metadata is allowlisted.
        payload = json.dumps(config, ensure_ascii=True).encode("utf-8")
        if len(payload) > api_planner.MAX_CONFIG_BYTES:
            raise ValueError("API 配置过长，尚未发起请求。")
        backend.write(folder / "api_launch.json", api_planner.public_config(config))
        c.session, c.plan_seen, c.summary_seen = folder, None, None
        c.loaded_plan_path, c.loaded_plan_hash = None, None
        c.controls.reset_for_plan()
        c.set_text(c.command_box, "本次通过窗口 API 入口启动，不再执行手动命令或重复提交队列。")
        c.set_text(c.plan_box, "正在调用完整 API 规划；结果尚未生成，禁止导航。\n" + str(folder))
        self.busy = True
        self.button.configure(state="disabled")
        self.state.set("正在启动 WSL / API 规划；详细阶段请查看“详细日志”。")
        c.status.set("API 规划中；不会自动发送导航")
        secret = self.key.get()
        def worker():
            try:
                self.events.put({"folder": folder, **api_planner.launch_wsl(command, config)})
            except Exception as exc:
                self.events.put({"folder": folder, "exit_code": -1,
                                 "message": (type(exc).__name__+": "+str(exc)).replace(secret, "[REDACTED]")})
        threading.Thread(target=worker, daemon=True).start()

    def poll(self):
        while not self.events.empty():
            row = self.events.get_nowait()
            self.busy = False
            self.button.configure(state="normal")
            summary = row["folder"] / "planning/planning_summary.json"
            try:
                report = backend.read(summary) if summary.exists() else {}
                outcome = report.get("status", "no_planning_summary")
            except (OSError, ValueError):
                report = {}
                outcome = "unreadable_planning_summary"
            passed = row["exit_code"] == 0 and outcome == "planning_json_generated_not_executed"
            message = ("API 规划已完成。已有当前起点时自动填写候选导航点；否则请到映射页填写起点。核对后再发送，尚未执行。"
                       if passed else f"API 规划未通过：{outcome}，退出码 {row['exit_code']}。请查看日志，不会自动补跑。")
            if passed and self.console.navigation_mode.get() == "ordered_goals":
                message = "API 多目标规划已完成。请到“系统 2 · 多目标导航”查看自动生成的全部导航段；起点为空时先填写起点。核对整队后启动，尚未执行。"
            if not passed:
                reasons = [i.get("message_zh") for i in report.get("review_issues", []) if i.get("message_zh")]
                if reasons:
                    message += " 原因：" + "；".join(dict.fromkeys(reasons))
            self.state.set(message)
            self.console.status.set(message)
            self.console.log(message)
            if row["message"]:
                self.console.log(row["message"])
        self.after(250, self.poll)
