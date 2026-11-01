"""Non-GUI helpers for the navigation-only field console."""
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import shlex
import uuid

from .._legacy import ROOT
from ..execution.navigation_contract import read, sha

DEFAULT_SCENE = ROOT / "outputs/real_scene_runs/d3_619_graph_20261022_094742_334872/scene"
QUEUE = ROOT / "outputs/field_console_queue"


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=True, allow_nan=False, indent=2)


def load_scene(folder):
    folder = Path(folder).resolve()
    metadata = read(folder / "scene_metadata.json")
    if metadata.get("unit") != "metre" or metadata.get("up_axis") != "z":
        raise ValueError("Scene requires metre units and Z-up coordinates")
    nodes = read(folder / "scene_graph.json")
    identifiers = set()
    for node in nodes:
        if type(node.get("id")) is not int or node["id"] in identifiers:
            raise ValueError("Scene IDs must be unique integers")
        identifiers.add(node["id"])
        for field in ("bbox_center", "bbox_extent"):
            values = node.get(field)
            if not isinstance(values, list) or len(values) != 3 or any(
                    isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in values):
                raise ValueError("Invalid object geometry")
        if any(v < 0 for v in node["bbox_extent"]):
            raise ValueError("Negative object extent")
    if not nodes:
        raise ValueError("Empty scene")
    return {"folder": folder, "metadata": metadata, "nodes": nodes,
            "graph_hash": sha(folder / "scene_graph.json"),
            "edges_hash": sha(folder / "scene_graph_edges.json"),
            "metadata_hash": sha(folder / "scene_metadata.json")}


def wsl_path(path):
    value = str(path).replace("\\", "/")
    match = re.match(r"^([A-Za-z]):/(.*)$", value)
    if match:
        return "/mnt/" + match[1].lower() + "/" + match[2]
    if value.startswith("/"):
        return value
    raise ValueError("Use an absolute Windows or Linux path")


def create_task(scene, instruction, navigation_mode="single_goal"):
    instruction = instruction.strip()
    if not instruction or len(instruction) > 8000:
        raise ValueError("Enter an instruction between 1 and 8000 characters")
    # Keep the established English experimental input contract; no silent translation.
    if re.search(r"[\u3400-\u9fff]", instruction):
        raise ValueError("Enter the task in English; this console does not silently translate instructions")
    if navigation_mode not in ("single_goal", "ordered_goals"):
        raise ValueError("Unsupported navigation mode")
    if (sha(scene["folder"] / "scene_graph.json") != scene["graph_hash"] or
            sha(scene["folder"] / "scene_graph_edges.json") != scene["edges_hash"] or
            sha(scene["folder"] / "scene_metadata.json") != scene.get("metadata_hash")):
        raise ValueError("Scene changed; reload it before preparing a task")
    folder = ROOT / "outputs/real_scene_runs" / ("field_ui_" + datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8])
    folder.mkdir(parents=True, exist_ok=False)
    try:
        scene_path = scene["folder"].relative_to(ROOT).as_posix()
    except ValueError:
        scene_path = wsl_path(scene["folder"])
    task = {"schema_version": "real_scene_navigation_instruction_v2", "task_id": folder.name,
            "scene_dir": scene_path, "scene_graph_sha256": scene["graph_hash"],
            "scene_edges_sha256": scene["edges_hash"],
            "scene_metadata_sha256": scene["metadata_hash"], "instruction": instruction,
            "target_authority": "user_instruction_via_llm_object_binding",
            "navigation_mode": navigation_mode,
            "evaluation_scope": "Instruction-led ordered navigation; proposal only, no dispatch.",
            "evaluation_rule": "Preserve the requested target order. Each navigation must target an active supplied scene node consistent with object binding and symbolic goals. No list selection is a reference answer; semantic correctness still needs operator review."}
    write(folder / "task.json", task)
    return folder


def command(folder, planning_python, embedding_python, embedding_model):
    # Every user-editable token is shell-quoted; no shell is invoked by this helper.
    tokens = [wsl_path(planning_python), "-B", "-m", "embodied_pipeline", "real-scene-plan",
              "--task-json", wsl_path(folder / "task.json"), "--output-dir", wsl_path(folder / "planning"),
              "--embedding-python", wsl_path(embedding_python), "--embedding-model", wsl_path(embedding_model)]
    return "cd " + shlex.quote(wsl_path(ROOT)) + " && " + " ".join(shlex.quote(v) for v in tokens)


def worker_command(planning_python, embedding_python, embedding_model):
    tokens = [wsl_path(planning_python), "-B", "-m", "embodied_pipeline.ui.field_worker",
              "--embedding-python", wsl_path(embedding_python), "--embedding-model", wsl_path(embedding_model)]
    return "cd " + shlex.quote(wsl_path(ROOT)) + " && " + " ".join(shlex.quote(v) for v in tokens)


def worker_status():
    try:
        value = read(QUEUE / "worker_status.json")
        timestamp = datetime.fromisoformat(value["utc"])
        age = (datetime.now(timezone.utc) - timestamp).total_seconds()
        return value if 0 <= age <= 6 and value.get("status") in ("ready", "planning") else None
    except (OSError, ValueError, KeyError, TypeError):
        return None


def submit(folder):
    if (folder / "api_launch.json").exists() or (folder / "planning_owner.lock").exists():
        raise ValueError("This planning run is already reserved; do not submit it again")
    if worker_status() is None:
        raise ValueError("WSL planning worker is not online; start it manually before submitting")
    payload = {"schema_version": "field_planning_job_v1", "folder": folder.relative_to(ROOT).as_posix(),
               "task_sha256": sha(folder / "task.json")}
    path = QUEUE / (folder.name + ".job.json")
    if any((QUEUE / (folder.name + suffix)).exists() for suffix in (".job.json", ".claimed.json", ".result.json")):
        raise ValueError("This task was already queued; do not submit it again")
    # Publish a complete record, never a partially written JSON job.
    temporary = QUEUE / (folder.name + ".pending.json")
    write(temporary, payload)
    temporary.rename(path)
    return path


def inspect_plan(path, scene):
    value = read(path)
    if (value.get("schema_version") != "real_robot_proposal_envelope_v1" or
            value.get("source_graph_sha256") != scene["graph_hash"] or
            value.get("source_edges_sha256") != scene["edges_hash"] or
            value.get("coordinate_frame") != scene["metadata"]["coordinate_frame"]):
        raise ValueError("Engineering plan does not belong to the loaded scene revision/frame")
    steps = [step for task in value["plan"]["subtask_plans"] for step in task["steps"]]
    supported = bool(steps) and all(s.get("skill_id") == "navigate_to_object" for s in steps)
    return value, steps, supported


def feedback_pose(row, confirmed_frame):
    if not confirmed_frame or row.get("event") != "feedback":
        return None
    pose = row.get("data", {}).get("current_pose", {})
    if pose.get("header", {}).get("frame_id") != confirmed_frame:
        return None
    value = pose.get("pose", {})
    p, q = value.get("position", {}), value.get("orientation", {})
    values = [p.get("x"), p.get("y")] + [q.get(k) for k in ("x", "y", "z", "w")]
    if any(isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v) for v in values):
        return None
    x, y, qx, qy, qz, qw = values
    if abs(sum(v*v for v in (qx, qy, qz, qw)) - 1) > .02:
        return None
    return x, y, math.atan2(2*(qw*qz+qx*qy), 1-2*(qy*qy+qz*qz))


class JsonlTail:
    """Read only complete new lines; retain partial writes and detect truncation."""
    def __init__(self, path):
        self.path, self.offset = Path(path), 0
    def poll(self):
        if not self.path.exists():
            return []
        if self.path.stat().st_size < self.offset:
            self.offset = 0
        rows = []
        with self.path.open("rb") as stream:
            stream.seek(self.offset)
            for _ in range(500):
                raw = stream.readline()
                if not raw.endswith(b"\n"):
                    break
                self.offset = stream.tell()
                try:
                    value = json.loads(raw)
                    rows.append(value if isinstance(value, dict) else {"event": "invalid_log_record"})
                except (ValueError, UnicodeError):
                    rows.append({"event": "invalid_log_record"})
        return rows
