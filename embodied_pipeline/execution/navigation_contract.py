"""Offline, fail-closed conversion of a single navigation proposal to ROS fields."""
import hashlib
import json
import math
from pathlib import Path

from .._legacy import ROOT, load

CONFIRMATIONS = (
    "initial_pose_measured", "scene_frame_verified", "approach_pose_verified",
    "navigation_limits_verified", "operator_and_estop_ready",
    "server_alignment_policy_reviewed",
)
POSE_FIELDS = ("x", "y", "z", "roll", "pitch", "yaw")
GOAL_TYPES = {**{p + k: "double" for p in ("s_", "g_") for k in POSE_FIELDS},
              **{k: "float" for k in ("pos_tol", "yaw_tol", "timeout", "max_speed")}}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def resolve(path):
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def number(value, name, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} requires a finite number, not null, boolean or string")
    if positive and value <= 0:
        raise ValueError(f"{name} must be positive")
    return float(value)


def pose(value, label):
    if not isinstance(value, dict) or set(value) != {"x", "y", "yaw"}:
        raise ValueError(f"{label} requires exactly x, y, yaw")
    result = {k: number(value[k], label + "." + k) for k in value}
    if abs(result["yaw"]) > math.pi:
        raise ValueError(f"{label}.yaw must be radians in [-pi, pi]")
    return {**result, "z": 0.0, "roll": 0.0, "pitch": 0.0}


def navigation_steps(envelope):
    steps = [s for t in envelope["plan"]["subtask_plans"] for s in t["steps"]]
    if not steps or any(s["skill_id"] != "navigate_to_object" for s in steps):
        raise ValueError("本入口仅接受全部由导航组成的计划；不跳过其他动作。")
    return steps


def selected_navigation(envelope, index):
    steps = navigation_steps(envelope)
    if type(index) is not int or not 0 <= index < len(steps):
        raise ValueError("导航段编号无效或全部导航段已完成。")
    return steps[index]


def prepare(config):
    """Never infer an approach pose from a table centre; never call ROS or LLM."""
    if config.get("schema_version") != "single_navigation_replay_v1":
        raise ValueError("Unsupported replay schema")
    source = resolve(config["engineering_plan"])
    if sha(source) != config["engineering_plan_sha256"]:
        raise ValueError("Engineering plan hash mismatch")
    envelope = read(source)
    for kind in ("graph", "edges"):
        if sha(resolve(config["scene_" + kind])) != envelope["source_" + kind + "_sha256"]:
            raise ValueError(f"Scene {kind} hash mismatch")
    if (config.get("coordinate_frame") != envelope["coordinate_frame"] or
            envelope["unit"] != "metre" or config.get("angle_unit") != "radian"):
        raise ValueError("Explicit matching scene frame, metre and radian units required")
    plan = envelope["plan"]
    load("action_sequence").validate_sequence(plan)
    steps = navigation_steps(envelope)
    if len(steps) > 1 and "navigation_step_index" not in config:
        raise ValueError("Multi-segment navigation requires an explicit segment index")
    if plan.get("unplanned_subtasks"):
        raise ValueError("Unplanned subtasks remain")
    step = selected_navigation(envelope, config.get("navigation_step_index", 0))
    target = step["arguments"]["target"]["element_id"]
    if type(target) is not int or type(config.get("target_element_id")) is not int or target != config["target_element_id"]:
        raise ValueError("Approach pose target does not match the bound scene object")
    if not step.get("action_id"):
        raise ValueError("Stable source action ID required")
    result = {
        "status": "configuration_incomplete_not_executed", "blockers": [],
        "source_plan_sha256": sha(source), "source_action_id": step["action_id"],
        "navigation_step_index": config.get("navigation_step_index", 0), "navigation_step_count": len(steps),
        "target_element_id": target, "coordinate_frame": envelope["coordinate_frame"],
        "action_name": "/nav/navigate_to", "action_type": "nav_interfaces/action/NavigateTo",
        "requested_standoff_m": step["arguments"].get("standoff_m"),
        "goal_pose_source": "operator_confirmed_approach_pose_not_object_centre",
        "goal": None, "robot_execution_attempted": False, "memory_committed": False,
    }
    goal = {}
    for prefix, key in (("s_", "start_pose"), ("g_", "approach_pose")):
        try:
            goal.update({prefix + k: v for k, v in pose(config.get(key), key).items()})
        except ValueError as exc:
            result["blockers"].append(str(exc))
    for key in ("pos_tol", "yaw_tol", "timeout", "max_speed"):
        try:
            goal[key] = number(config.get("limits", {}).get(key), key, positive=True)
            if goal[key] > 3.402823466e38:
                raise ValueError(f"{key} exceeds ROS float32 range")
            if key == "yaw_tol" and goal[key] > math.pi:
                raise ValueError("yaw_tol must not exceed pi radians")
        except ValueError as exc:
            result["blockers"].append(str(exc))
    for key in CONFIRMATIONS:
        if config.get("confirmations", {}).get(key) is not True:
            result["blockers"].append("Confirmation required: " + key)
    if not result["blockers"]:
        result["goal"] = goal
        result["status"] = "request_prepared_not_executed"
    return result
