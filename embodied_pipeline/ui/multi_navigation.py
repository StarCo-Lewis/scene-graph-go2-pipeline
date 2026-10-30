"""Ordered goal review UI; uses the existing planner and navigation transports."""
import tkinter as tk
from tkinter import ttk

from ..execution.navigation_contract import read, navigation_steps
from .navigation_controls import candidate_for_plan, build_config


class MultiNavigationPanel(ttk.Frame):
    def __init__(self, parent, console):
        super().__init__(parent, padding=15)
        self.console = console
        self.route = None
        self.route_key = None
        self.observed_plan = None
        self.auto_attempt = None
        self.route_start = None
        self.route_note = tk.StringVar(value="规划通过后自动在此列出全部导航段；核对后才启动。")
        self.reviewed = tk.BooleanVar(value=False)
        ttk.Label(self, text="系统 2：一次规划多目标 → 核对整条队列 → 到达后停留 10 秒 → 下一目标",
                  font=("Microsoft YaHei UI", 12, "bold")).pack(anchor="w", pady=8)
        ttk.Label(self, text="示例语法：Navigate to A, then navigate to B. 请把 A、B 换成场景中的物体名称或 ID。\n"
                  "首段建立场景对齐；后续用上一段实际终态位姿继续。不会回到原点，不修改运动参数。",
                  justify="left").pack(anchor="w")
        self.status_label = ttk.Label(self, textvariable=console.controls.segment_state,
                                      wraplength=1050, foreground="#185a76")
        self.status_label.pack(anchor="w", pady=4)
        ttk.Label(self,textvariable=console.controls.outcome_state,foreground="#174c75").pack(anchor="w")
        start_row = ttk.Frame(self); start_row.pack(fill="x", pady=7)
        ttk.Label(start_row, text="场景系当前起点（与控制页共享）：").pack(side="left")
        for key, variable in console.controls.start.items():
            ttk.Label(start_row, text=key).pack(side="left", padx=4)
            ttk.Entry(start_row, textvariable=variable, width=12).pack(side="left")
        ttk.Label(start_row,text="BBOX 距离 / m").pack(side="left",padx=8)
        ttk.Entry(start_row,textvariable=console.controls.clearance_input,width=7).pack(side="left")
        ttk.Button(start_row,text="应用距离",command=lambda:console.guarded(console.controls.apply_clearance)).pack(side="left")
        self.instruction = tk.Text(self, height=3, wrap="word")
        self.instruction.pack(fill="x", pady=8)
        row = ttk.Frame(self); row.pack(fill="x", pady=5)
        ttk.Button(row, text="调用 API 生成多目标计划", command=lambda: console.guarded(self.plan)).pack(side="left")
        ttk.Button(row, text="加载已有工程 JSON", command=console.browse_plan).pack(side="left", padx=6)
        ttk.Button(row, text="从当前工程计划生成队列", command=lambda: console.guarded(self.build_route)).pack(side="left")
        ttk.Label(self, textvariable=self.route_note, wraplength=1050, foreground="#185a76").pack(anchor="w")
        self.table = ttk.Treeview(self, columns=("index", "target", "x", "y", "yaw", "state"), show="headings", height=8)
        for key, label, width in (("index", "段", 45), ("target", "目标 ID / 名称", 230), ("x", "X / m", 100),
                                  ("y", "Y / m", 100), ("yaw", "Yaw / rad", 110), ("state", "执行状态", 150)):
            self.table.heading(key, text=label); self.table.column(key, width=width)
        self.table.pack(fill="both", expand=True, pady=8)
        self.table.bind("<<TreeviewSelect>>", self.select_point)
        edit = ttk.Frame(self); edit.pack(fill="x")
        self.point = {k: tk.StringVar() for k in ("x", "y", "yaw")}
        for k, var in self.point.items():
            ttk.Label(edit, text=k).pack(side="left", padx=4)
            ttk.Entry(edit, textvariable=var, width=14).pack(side="left")
        ttk.Button(edit, text="更新所选停靠点", command=lambda: console.guarded(self.update_point)).pack(side="left", padx=8)
        ttk.Label(self, text="队列在 BBOX 外侧环绕采样，按导航栅格和 0.35 m 圆形净空筛选；无合格点则拒绝生成。\n"
                  "先在控制页连接、填写首次起点并完成现场检查项。\n"
                  "下一段只在成功到达且 final_pose 有效后，停留 10 秒再发送；失败、取消、断线或位姿缺失会停止队列。",
                  justify="left").pack(anchor="w", pady=8)
        ttk.Checkbutton(self, text="已核对指令与目标顺序、所有停靠点及路线，并允许成功后自动执行下一段",
                        variable=self.reviewed).pack(anchor="w")
        row = ttk.Frame(self); row.pack(fill="x", pady=10)
        self.execute_button = ttk.Button(row, text="确认并启动整条队列", command=lambda: console.guarded(self.execute))
        self.execute_button.pack(side="left")
        ttk.Button(row, text="停止队列并请求取消当前段", command=console.controls.cancel).pack(side="left", padx=8)
        self.after(400, self.refresh)

    def plan(self):
        c = self.console
        if c.controls.active or c.controls.queue_running:
            raise ValueError("请先结束当前导航队列。")
        c.instruction.delete("1.0", "end")
        c.instruction.insert("1.0", self.instruction.get("1.0", "end").strip())
        c.navigation_mode.set("ordered_goals")
        c.api.start(navigation_mode="ordered_goals")

    def key(self):
        c = self.console
        return (str(c.loaded_plan_path), c.loaded_plan_hash, c.controls.scene_key(),c.controls.clearance())

    def build_route(self):
        c, panel = self.console, self.console.controls
        if panel.active or panel.queue_running or c.api.busy or panel.uncertain:
            raise ValueError("请先结束当前规划/导航并核对未决状态。")
        if not c.loaded_plan_path or panel.sequence_complete:
            raise ValueError("先加载未执行完的工程计划。")
        self.route = None
        self.reviewed.set(False)
        self.table.delete(*self.table.get_children())
        if hasattr(c,"navigation_view"):c.navigation_view.preview_goals([],panel.clearance())
        steps = navigation_steps(read(c.loaded_plan_path))
        start = {k: float(v.get()) for k, v in panel.start.items()}
        route = []
        for index in range(panel.step_index, len(steps)):
            target, goal = candidate_for_plan(c.loaded_plan_path, c.loaded_plan_hash, c.scene, start, index,panel.clearance())
            route.append({"step_index": index, "target_element_id": target, "goal": goal})
            # Geometry preview only. Actual subsequent s_* MUST come from final_pose.
            start = goal
        self.route, self.route_key = route, self.key()
        self.route_start = tuple(panel.start[k].get() for k in ("x", "y", "yaw"))
        self.observed_plan = self.route_key
        if hasattr(c,"navigation_view"):c.navigation_view.preview_goals(route,panel.clearance())
        self.reviewed.set(False)
        self.table.delete(*self.table.get_children())
        for item in route:
            node = next(n for n in c.scene["nodes"] if n["id"] == item["target_element_id"])
            self.table.insert("", "end", iid=str(item["step_index"]), values=(item["step_index"]+1,
                f"{node['id']} / {node.get('object_tag', 'object')}", *(f"{item['goal'][k]:.4f}" for k in ("x", "y", "yaw")), "待核对"))
        self.route_note.set(f"已生成 {len(route)} 段候选导航，按指令顺序列在下方；尚未发送。")

    def maybe_build_route(self):
        """Populate the review table only; never authorize or dispatch navigation."""
        c, panel = self.console, self.console.controls
        if not c.loaded_plan_path or c.api.busy or panel.active or panel.queue_running or panel.uncertain or panel.sequence_complete:
            return
        start = tuple(panel.start[k].get() for k in ("x", "y", "yaw"))
        if self.route and start != self.route_start and panel.step_index == self.route[0]["step_index"]:
            self.route = None
            self.reviewed.set(False)
            self.table.delete(*self.table.get_children())
        if self.route:
            return  # Preserve reviewed/manual edits until their inputs change.
        attempt = (self.key(), start, panel.step_index)
        if attempt == self.auto_attempt:
            return
        try:
            summary = read(c.loaded_plan_path.with_name("planning_summary.json"))
            if summary.get("status") != "planning_json_generated_not_executed":
                self.route_note.set("规划未通过，不能生成可执行队列。")
                return
            if len(navigation_steps(read(c.loaded_plan_path))) < 2:
                return
            if any(not value.strip() for value in start):
                self.route_note.set("多目标计划已加载；请填写当前起点，随后自动填入全部导航段。")
                return
            self.auto_attempt = attempt
            self.build_route()
            c.log(self.route_note.get())
        except (ValueError, KeyError, OSError, TypeError) as exc:
            self.auto_attempt = attempt
            self.route_note.set("多目标队列未生成，请核对起点/计划：" + str(exc))
            c.log(self.route_note.get())

    def select_point(self, *_):
        selected = self.table.selection()
        if selected and self.route:
            item = next(r for r in self.route if r["step_index"] == int(selected[0]))
            for k, var in self.point.items():
                var.set(repr(item["goal"][k]))

    def update_point(self):
        c, panel = self.console, self.console.controls
        if panel.active or panel.queue_running:
            raise ValueError("执行期间不能更改队列。")
        if not self.route or self.route_key != self.key() or not self.table.selection():
            raise ValueError("先生成当前计划队列并选择一段。")
        item = next(r for r in self.route if r["step_index"] == int(self.table.selection()[0]))
        goal = {k: float(v.get()) for k, v in self.point.items()}
        from ..execution.navigation_contract import pose
        pose(goal, "approach_pose")
        build_config(c.loaded_plan_path, c.loaded_plan_hash, c.scene,
                     {k: float(v.get()) for k, v in panel.start.items()}, goal, {}, item["step_index"],panel.clearance())
        item["goal"] = goal
        for k in ("x", "y", "yaw"):
            self.table.set(str(item["step_index"]), k, f"{goal[k]:.4f}")
        self.reviewed.set(False)

        if hasattr(c,"navigation_view"):c.navigation_view.preview_goals(self.route,panel.clearance())

    def execute(self):
        if not self.reviewed.get() or not self.route or self.route_key != self.key():
            raise ValueError("请为当前计划生成队列，核对所有导航点并勾选整队授权。")
        self.console.controls.start_queue(self.route)

    def refresh(self):
        panel = self.console.controls
        current = self.key()
        if self.console.loaded_plan_path and current != self.observed_plan:
            self.observed_plan = current
            self.reviewed.set(False)
            self.route = None
            self.auto_attempt = None
            self.table.delete(*self.table.get_children())
            if hasattr(self.console,"navigation_view"):
                self.console.navigation_view.preview_goals([],panel.clearance())
        self.maybe_build_route()
        self.execute_button.configure(state="disabled" if panel.active or panel.queue_running or self.console.api.busy else "normal")
        if self.route:
            valid = self.route_key == self.key()
            for item in self.route:
                index = item["step_index"]
                state = ("计划已切换，请重建" if not valid else "已到达" if index < panel.step_index
                         else "执行中" if index == panel.step_index and panel.active
                         else "已授权等待" if panel.queue_running else "待启动 / 已暂停")
                result=panel.outcomes.get((str(self.console.loaded_plan_path),self.console.loaded_plan_hash,index))
                if result and not panel.active:
                    state=result['label_zh']+(" / "+result['reason'] if result['reason'] else '')
                self.table.set(str(index), "state", state)
        self.after(400, self.refresh)
