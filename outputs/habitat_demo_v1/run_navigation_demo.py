#!/usr/bin/env python3
"""Scene201 ConceptGraphs -> Habitat navigation smoke test (no manipulation)."""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path

LOG = logging.getLogger("habitat_demo")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=True, indent=2) + "\n", encoding="ascii")


def rss_mb():
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return round(int(line.split()[1]) / 1024, 1)
    except OSError:
        pass
    return None


def peak_rss_mb():
    """Return the process resident-set high-water mark reported by Linux."""
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmHWM:"):
                return round(int(line.split()[1]) / 1024, 1)
    except OSError:
        pass
    return None


def system_memory_mb():
    try:
        values = {}
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, raw = line.split(":", 1)
            if key in {"MemTotal", "MemAvailable"}:
                values[key] = round(int(raw.split()[0]) / 1024, 1)
        return {"total_mb": values.get("MemTotal"),
                "available_mb": values.get("MemAvailable")}
    except (OSError, ValueError, IndexError):
        return {"total_mb": None, "available_mb": None}


def gpu_free_mb():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=True,
        )
        return int(result.stdout.splitlines()[0].strip())
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None


def cg_to_habitat(point, mode):
    x, y, z = map(float, point)
    if mode == "x_z_minus_y":
        return [x, z, -y]
    if mode == "x_z_y":
        return [x, z, y]
    if mode == "identity":
        return [x, y, z]
    raise ValueError(f"unsupported coordinate mode: {mode}")


def flatten_steps(plan):
    return [step for subtask in plan.get("subtask_plans", []) for step in subtask.get("steps", [])]


def targets(alignment, plan, nodes):
    by_id = {int(node["id"]): node for node in nodes}
    bindings = alignment.get("context_bindings", {})
    output = []
    for step in flatten_steps(plan):
        if step["skill_id"] != "navigate_to_object":
            continue
        target = step["arguments"]["target"]
        role_id, element_id = target["role_id"], int(target["element_id"])
        if element_id not in bindings.get(role_id, {}).get("element_ids", []):
            raise ValueError(f"ungrounded target: {role_id}/{element_id}")
        node = by_id.get(element_id)
        if node is None or node.get("bbox_center") is None:
            raise ValueError(f"missing or unlocalized scene node: {element_id}")
        output.append({"step_id": step["step_id"], "role_id": role_id,
                       "element_id": element_id, "cg_center": node["bbox_center"],
                       "object_tag": node.get("object_tag")})
    if not output:
        raise ValueError("final plan has no navigate_to_object steps")
    return output


def horizontal_distance(a, b):
    return math.hypot(float(a[0]) - float(b[0]), float(a[2]) - float(b[2]))


def habitat_vector(values):
    # Habitat's Python bindings normally use NumPy vectors; keep pure-Python
    # helpers testable when Habitat/NumPy are not installed on the host.
    try:
        import numpy as np
        return np.asarray(values, dtype=np.float32)
    except ImportError:
        return values


def snap_targets(pathfinder, items, axis_mode, max_snap_m):
    result = []
    for item in items:
        location = cg_to_habitat(item["cg_center"], axis_mode)
        # Node centres can be well above the floor; query the navmesh at floor height.
        query = [location[0], 0.0, location[2]]
        snapped = list(map(float, pathfinder.snap_point(habitat_vector(query))))
        offset = horizontal_distance(query, snapped)
        if not all(math.isfinite(x) for x in snapped) or offset > max_snap_m:
            raise ValueError(f"node {item['element_id']} cannot be snapped near its mapped position "
                             f"(horizontal_offset_m={offset:.3f}, axis_mode={axis_mode})")
        result.append({**item, "habitat_query": query, "nav_goal": snapped,
                       "horizontal_snap_offset_m": round(offset, 3)})
    return result


def select_coordinates(pathfinder, items, requested_mode, max_snap_m):
    modes = (["x_z_minus_y", "x_z_y", "identity"] if requested_mode == "auto"
             else [requested_mode])
    valid, errors = [], {}
    for mode in modes:
        try:
            points = snap_targets(pathfinder, items, mode, max_snap_m)
            valid.append((sum(x["horizontal_snap_offset_m"] for x in points), mode, points))
        except ValueError as exc:
            errors[mode] = str(exc)
    if not valid:
        raise ValueError(f"no coordinate mapping produced nearby navmesh goals: {errors}")
    valid.sort(key=lambda item: item[0])
    if len(valid) > 1 and valid[1][0] - valid[0][0] < 0.25:
        raise ValueError(f"coordinate mapping is ambiguous: {[(x[1], x[0]) for x in valid]}; "
                         "rerun with explicit --axis-mode after checking scene coordinates")
    return valid[0][1], valid[0][2]


def choose_start(pathfinder, first_goal, seed, selection="closest"):
    random.seed(seed)
    if selection not in ("closest", "first"):
        raise ValueError("Unknown start selection policy")
    # Habitat's sampler uses its own RNG, not Python's random module.
    pathfinder.seed(seed)
    best = None
    for _ in range(100):
        point = list(map(float, pathfinder.get_random_navigable_point()))
        if not all(map(math.isfinite, point)) or horizontal_distance(point, first_goal) < 1.5:
            continue
        path = __import__("habitat_sim").ShortestPath()
        path.requested_start = habitat_vector(point)
        path.requested_end = habitat_vector(first_goal)
        if pathfinder.find_path(path):
            if selection == "first":
                return point
            if best is None or path.geodesic_distance < best[0]:
                best = (path.geodesic_distance, point)
    if best is None:
        raise ValueError("could not sample a reachable start point for the first target")
    return best[1]


def navigate(sim, goal, max_actions, arrival_radius_m, log_every, on_action=None, before_action=None):
    import habitat_sim

    follower = sim.make_greedy_follower(goal_radius=arrival_radius_m)
    start = list(map(float, sim.get_agent(0).get_state().position))
    path = habitat_sim.ShortestPath()
    path.requested_start, path.requested_end = habitat_vector(start), habitat_vector(goal)
    if not sim.pathfinder.find_path(path):
        return {"status": "no_path", "start": start, "goal": goal, "actions": 0}
    started = time.monotonic()
    actions = 0
    for index in range(max_actions):
        position = list(map(float, sim.get_agent(0).get_state().position))
        remaining = horizontal_distance(position, goal)
        if remaining <= arrival_radius_m:
            return {"status": "arrived", "start": start, "goal": goal,
                    "end": position, "actions": actions,
                    "planned_geodesic_m": round(float(path.geodesic_distance), 3),
                    "final_horizontal_distance_m": round(remaining, 3),
                    "elapsed_s": round(time.monotonic() - started, 3)}
        action = follower.next_action_along(habitat_vector(goal))
        if action is None or action == "stop":
            break
        if before_action is not None:
            before_action(action)
        sim.step(action)
        if on_action is not None:
            on_action()
        actions += 1
        if actions % log_every == 0:
            LOG.info("navigation actions=%d remaining_m=%.2f rss_mb=%s gpu_free_mb=%s",
                     actions, remaining, rss_mb(), gpu_free_mb())
    end = list(map(float, sim.get_agent(0).get_state().position))
    return {"status": "not_arrived", "start": start, "goal": goal, "end": end,
            "actions": actions, "final_horizontal_distance_m": round(horizontal_distance(end, goal), 3),
            "elapsed_s": round(time.monotonic() - started, 3)}


def run(args):
    alignment = read_json(args.alignment_json)
    plan = read_json(args.skill_plan_json)
    nodes = read_json(args.scene_graph_json)
    if alignment.get("status") != "success" or plan.get("status") != "proposal_needs_environment_validation":
        raise ValueError("successful object alignment and valid final skill proposal are required")
    items = targets(alignment, plan, nodes)
    if not args.stage_config.is_file() or not args.navmesh.is_file():
        raise ValueError("Habitat stage config or navmesh is missing")

    # Preflight checks precede the expensive stage load.
    import habitat_sim
    from habitat_sim.agent import ActionSpec, ActuationSpec, AgentConfiguration

    LOG.info("preflight targets=%s", [(x["element_id"], x["role_id"]) for x in items])
    pathfinder = habitat_sim.PathFinder()
    if not pathfinder.load_nav_mesh(str(args.navmesh)):
        raise ValueError(f"failed to load navmesh: {args.navmesh}")
    axis_mode, snapped = select_coordinates(pathfinder, items, args.axis_mode, args.max_snap_m)
    start = choose_start(pathfinder, snapped[0]["nav_goal"], args.seed)
    LOG.info("navmesh loaded; start=%s; targets=%s", start,
             [(x["element_id"], x["nav_goal"]) for x in snapped])
    result = {"schema_version": "1.0", "status": "preflight_passed",
              "scope": "Habitat navigation proxy only; no quadruped dynamics or manipulation",
              "input": {"alignment": str(args.alignment_json), "plan": str(args.skill_plan_json),
                        "scene_graph": str(args.scene_graph_json), "stage_config": str(args.stage_config),
                        "navmesh": str(args.navmesh)},
              "coordinate_mode": axis_mode, "targets": snapped,
              "start": start, "resource_samples": [{"stage": "pre_scene_load", "rss_mb": rss_mb(),
                                                    "gpu_free_mb": gpu_free_mb()}],
              "navigation_results": [],
              "non_navigation_steps": [{"step_id": step["step_id"], "skill_id": step["skill_id"],
                                        "status": "not_executed_no_manipulation_adapter"}
                                       for step in flatten_steps(plan)
                                       if step["skill_id"] != "navigate_to_object"]}
    write_json(args.output, result)
    if args.preflight_only:
        return result

    cfg = habitat_sim.SimulatorConfiguration()
    cfg.scene_id = str(args.stage_config)
    cfg.create_renderer = False
    cfg.enable_physics = False
    agent_cfg = AgentConfiguration()
    agent_cfg.action_space = {
        "move_forward": ActionSpec("move_forward", ActuationSpec(amount=0.25)),
        "turn_left": ActionSpec("turn_left", ActuationSpec(amount=10.0)),
        "turn_right": ActionSpec("turn_right", ActuationSpec(amount=10.0)),
    }
    LOG.info("loading Habitat stage; rss_mb=%s gpu_free_mb=%s", rss_mb(), gpu_free_mb())
    with habitat_sim.Simulator(habitat_sim.Configuration(cfg, [agent_cfg])) as sim:
        if not sim.pathfinder.load_nav_mesh(str(args.navmesh)):
            raise ValueError("stage loaded but navmesh failed to load in simulator")
        state = sim.get_agent(0).get_state()
        state.position = habitat_vector(start)
        sim.get_agent(0).set_state(state)
        result["resource_samples"].append({"stage": "after_scene_load", "rss_mb": rss_mb(),
                                           "gpu_free_mb": gpu_free_mb()})
        for item in snapped:
            LOG.info("navigation step=%s object=%s goal=%s", item["step_id"],
                     item["element_id"], item["nav_goal"])
            nav = navigate(sim, item["nav_goal"], args.max_actions,
                           args.arrival_radius_m, args.log_every)
            result["navigation_results"].append({"step_id": item["step_id"],
                                                 "element_id": item["element_id"], **nav})
            result["resource_samples"].append({"stage": item["step_id"], "rss_mb": rss_mb(),
                                               "gpu_free_mb": gpu_free_mb()})
            write_json(args.output, result)
            if nav["status"] != "arrived":
                result["status"] = "navigation_failed"
                break
        else:
            result["status"] = "navigation_demo_passed"
    write_json(args.output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alignment-json", type=Path, required=True)
    parser.add_argument("--skill-plan-json", type=Path, required=True)
    parser.add_argument("--scene-graph-json", type=Path, required=True)
    parser.add_argument("--stage-config", type=Path, required=True)
    parser.add_argument("--navmesh", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--log-file", type=Path, required=True)
    parser.add_argument("--axis-mode", choices=["auto", "x_z_minus_y", "x_z_y", "identity"],
                        default="x_z_minus_y")
    parser.add_argument("--max-snap-m", type=float, default=2.0)
    parser.add_argument("--arrival-radius-m", type=float, default=0.6)
    parser.add_argument("--max-actions", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=201)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    args.log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(args.log_file, encoding="utf-8")])
    try:
        result = run(args)
        LOG.info("status=%s result=%s", result["status"], args.output)
        return 0 if result["status"] in {"preflight_passed", "navigation_demo_passed"} else 2
    except Exception as exc:
        LOG.exception("Habitat demo failed")
        write_json(args.output, {"schema_version": "1.0", "status": "error",
                                 "error": {"type": type(exc).__name__, "message": str(exc)}})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
