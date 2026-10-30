#!/usr/bin/env python3
"""Restore verified object translations and execute an existing memory follow-up plan."""
from __future__ import annotations
import argparse
import logging
import math
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "habitat_demo_v1"))
from run_navigation_demo import (read_json, write_json, habitat_vector, cg_to_habitat,
    horizontal_distance, select_coordinates, navigate, peak_rss_mb, system_memory_mb)
from run_sideboard_full_pipeline import validate_adapter, validate_memory_target
from update_memory import proxy_events, digest, path_key, vector

SCENE = Path("/mnt/d/NebulaAI/output/batches/scene201_vith_full_20260915_181750/seq_0001__hf_house_201")
LOG = logging.getLogger("memory_followup_navigation")


def choose_far_start(pathfinder, goal, object_position, seed, minimum_distance):
    if not math.isfinite(minimum_distance) or minimum_distance < 2.0:
        raise ValueError("Far-start distance must be finite and >=2 metres")
    import habitat_sim
    pathfinder.seed(seed)
    for _ in range(200):
        point = vector(list(map(float, pathfinder.get_random_navigable_point())))
        if horizontal_distance(point, object_position) < minimum_distance:
            continue
        path = habitat_sim.ShortestPath()
        path.requested_start, path.requested_end = habitat_vector(point), habitat_vector(goal)
        if pathfinder.find_path(path):
            return point
    raise ValueError("No sufficiently distant reachable start could be sampled")


def restore_contract(state, nodes, base_nodes, proxy):
    check = validate_memory_target(state, nodes, 67)
    if path_key(proxy.get("input", {}).get("scene_graph", "")) != path_key(Path(state["base_scene"]) / "scene_graph.json"):
        raise ValueError("Source execution and memory base graph differ")
    expected = proxy_events(proxy, base_nodes, "source-proxy", "2026-01-01T00:00:00Z")
    if len(expected) != 1 or math.dist(expected[0]["position"], check["current_bbox_center"]) > 1e-5:
        raise ValueError("Observed source placement does not match current memory position")
    moves = [row for row in state["event_log"] if row["event"].get("object_id") == 67
             and row["event"].get("operation") == "move"]
    if moves[-1]["event"].get("evidence", {}).get("proxy_digest") != digest(proxy):
        raise ValueError("Latest memory move does not originate from this source execution")
    steps = proxy["steps"]
    obj = next(s for s in steps if s.get("skill_id") == "instantiate_object")
    board = next(s for s in steps if s.get("skill_id") == "instantiate_sideboard")
    place = next(s for s in reversed(steps) if s.get("skill_id") == "place_object_on_surface")
    navs = [s for s in steps if s.get("skill_id") == "navigate_to_object"]
    if len(navs) != 2 or navs[-1].get("status") != "arrived":
        raise ValueError("Prior navigation endpoint unavailable")
    base = next(n for n in base_nodes if n["id"] == 67)
    centre_h = cg_to_habitat(vector(base["bbox_center"]), "x_z_minus_y")
    initial_com = vector(obj["position"])
    return {"kettle_com": vector(place["settled_position"]),
            "sideboard_com": vector(board["position"]),
            "robot_position": vector(navs[-1]["end"]),
            "bbox_minus_com_offset": [centre_h[i] - initial_com[i] for i in range(3)],
            "target_bbox_habitat": cg_to_habitat(check["current_bbox_center"], "x_z_minus_y"),
            "memory_check": check}


def run(args, output):
    instruction = getattr(args, "instruction", "Go to the electric kettle.")
    target_id = getattr(args, "expected_target_id", 67)
    if target_id not in (67, 13):
        raise ValueError("Restored proxy supports only known kettle/sideboard instances")
    plan_dir = args.plan_run
    planning = read_json(plan_dir / "pipeline_summary.json")
    if planning.get("status") != "planning_passed" or planning.get("instruction") != instruction:
        raise ValueError("A successful kettle memory-follow-up plan is required")
    target = read_json(plan_dir / "planning_target.json")
    state_path = Path(target["memory_source"])
    state = read_json(state_path)
    graph = Path(target["scene_reference"])
    if graph.resolve() != (state_path.parent / "scene").resolve():
        raise ValueError("Planning scene is not the supplied memory snapshot")
    nodes = read_json(graph / "scene_graph.json")
    plan = read_json(plan_dir / "skill_plan_final.json")
    alignment = read_json(plan_dir / "object_alignment.json")
    if path_key(alignment.get("scene_reference", "")) != path_key(graph):
        raise ValueError("Alignment used a different map")
    item = validate_adapter(alignment, plan, nodes, instruction, target_id)
    if target["target"]["cg_center"] != item["cg_center"]:
        raise ValueError("Stored planning target differs from current graph")
    moves = [row for row in state["event_log"] if row["event"].get("object_id") == 67
             and row["event"].get("operation") == "move"]
    source_path = Path(moves[-1]["event"]["evidence"]["reference"])
    proxy = read_json(source_path)
    base_nodes = read_json(Path(state["base_scene"]) / "scene_graph.json")
    contract = restore_contract(state, nodes, base_nodes, proxy)
    if planning["memory_target_check"]["memory_version"] != state["version"]:
        raise ValueError("Memory changed after planning; replan before execution")
    summary = {"schema_version": "1.0", "status": "preflight_passed",
        "instruction": plan["instruction"], "plan_run": str(plan_dir),
        "memory_source": str(state_path), "source_execution": str(source_path),
        "restore_contract": contract, "object_id": target_id, "training_eligible": False,
        "fresh_llm_call": False, "new_exploration": False, "source_files_modified": False,
        "scope": "Restored checkpoint translations and proxy navigation, not uninterrupted physical execution",
        "limitations": ["Object rotation and robot heading were not recorded and are not restored.",
                        "Kettle is held kinematically; previous support evidence is not re-tested.",
                        "Inherited navmesh is reused, not rebuilt for dynamic obstacles."]}
    if args.preflight_only:
        write_json(output / "followup_execution.json", summary)
        return summary
    import habitat_sim
    from habitat_sim.agent import AgentConfiguration, ActionSpec, ActuationSpec
    from habitat_sim.physics import MotionType
    navmesh = SCENE / "scene_package/hf_house_201/hf_house_201.navmesh"
    assets = SCENE / "habitat_split_assets_v2"
    for path in (navmesh, assets / "scene201_without_kettle.stage_config.json",
                 assets / "kettle_dynamic.object_config.json", assets / "sideboard_static.object_config.json"):
        if not path.is_file():
            raise ValueError(f"Missing simulation resource: {path}")
    cfg = habitat_sim.SimulatorConfiguration()
    cfg.scene_id = str(assets / "scene201_without_kettle.stage_config.json")
    cfg.create_renderer = False
    cfg.enable_physics = True
    agent = AgentConfiguration()
    agent.action_space = {"move_forward": ActionSpec("move_forward", ActuationSpec(amount=0.25)),
        "turn_left": ActionSpec("turn_left", ActuationSpec(amount=10)),
        "turn_right": ActionSpec("turn_right", ActuationSpec(amount=10))}
    with habitat_sim.Simulator(habitat_sim.Configuration(cfg, [agent])) as sim:
        if not sim.pathfinder.load_nav_mesh(str(navmesh)):
            raise ValueError("Navmesh load failed")
        mgr = sim.get_object_template_manager()
        instances = {}
        for role, config, position, motion in (
                ("sideboard", "sideboard_static.object_config.json", contract["sideboard_com"], MotionType.STATIC),
                ("kettle", "kettle_dynamic.object_config.json", contract["kettle_com"], MotionType.KINEMATIC)):
            ids = mgr.load_configs(str(assets / config))
            if not ids:
                raise ValueError(f"Cannot load {role} template")
            obj = sim.get_rigid_object_manager().add_object_by_template_handle(mgr.get_template_by_id(ids[0]).handle)
            if obj is None:
                raise ValueError(f"Cannot instantiate {role}")
            obj.motion_type = MotionType.KINEMATIC
            obj.translation = habitat_vector(position)
            obj.motion_type = motion
            instances[role] = obj
        actual_com = list(map(float, instances["kettle"].translation))
        actual_bbox = [actual_com[i] + contract["bbox_minus_com_offset"][i] for i in range(3)]
        pose_error = math.dist(actual_bbox, contract["target_bbox_habitat"])
        if pose_error > 1e-4:
            raise ValueError("Restored simulation target position differs from memory")
        summary["restored_instances"] = {role: {"habitat_object_id": int(obj.object_id),
            "position": list(map(float, obj.translation))} for role, obj in instances.items()}
        summary["restored_bbox_position_error_m"] = pose_error
        axis, snapped = select_coordinates(sim.pathfinder, [item], "x_z_minus_y", 2.0)
        actual_target = list(map(float, instances["kettle" if target_id == 67 else "sideboard"].translation))
        robot_position = (choose_far_start(sim.pathfinder, snapped[0]["nav_goal"], actual_target,
                            args.start_seed, args.min_start_distance_m)
                          if args.start_policy == "sampled_far" else contract["robot_position"])
        robot = habitat_vector(robot_position)
        if not sim.pathfinder.is_navigable(robot):
            raise ValueError("Previous robot endpoint is not navigable; cannot restore continuation")
        start = sim.get_agent(0).get_state()
        start.position = robot
        sim.get_agent(0).set_state(start)
        start_distance = horizontal_distance(robot_position, actual_target)
        old_position = cg_to_habitat(contract["memory_check"]["original_bbox_center"], "x_z_minus_y")
        LOG.info("Restored kettle at %s; start_policy=%s start=%s distance_to_new_object_m=%.3f",
                 actual_com, args.start_policy, robot_position, start_distance)
        nav = navigate(sim, snapped[0]["nav_goal"], args.max_actions, 0.6, 25)
        distance = horizontal_distance(sim.get_agent(0).get_state().position, actual_target)
        old_distance = horizontal_distance(sim.get_agent(0).get_state().position, old_position)
        passed = nav.get("status") == "arrived" and distance <= 1.5
        if args.start_policy == "sampled_far":
            passed = passed and nav.get("actions", 0) > 0 and (target_id != 67 or old_distance > 1.5)
        summary.update(status="followup_proxy_passed" if passed else "followup_proxy_failed",
                       training_eligible=passed, goal_observed_in_proxy=passed,
                       navigation=nav, distance_to_restored_object_m=distance,
                       start_policy=args.start_policy, start_seed=args.start_seed,
                       initial_distance_to_new_object_m=start_distance,
                       final_distance_to_old_position_m=old_distance,
                       min_start_distance_m=args.min_start_distance_m if args.start_policy == "sampled_far" else None,
                       memory_modified=False,
                       process_peak_rss_gib=round(peak_rss_mb() / 1024, 3) if peak_rss_mb() else None,
                       system_memory=system_memory_mb())
    write_json(output / "followup_execution.json", summary)
    write_json(output / "task_level_episode.json", dict(summary,
        record_type="memory_followup_restored_navigation_episode", physical_robot_execution_verified=False,
        uninterrupted_simulator_execution=False))
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan-run", type=Path, default=SCENE / "memory_followup_planning/scene201_20260926_140047_110543")
    p.add_argument("--instruction", default="Go to the electric kettle.")
    p.add_argument("--expected-target-id", type=int, choices=(67, 13), default=67)
    p.add_argument("--max-actions", type=int, default=500)
    p.add_argument("--preflight-only", action="store_true")
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--start-policy", choices=("previous_endpoint", "sampled_far"), default="previous_endpoint")
    p.add_argument("--start-seed", type=int, default=201)
    p.add_argument("--min-start-distance-m", type=float, default=3.0)
    args = p.parse_args()
    output = args.output_dir or SCENE / "memory_followup_execution" / datetime.now().strftime("scene201_%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(output / "execution.log", encoding="utf-8")])
    try:
        result = run(args, output)
    except (OSError, ValueError, KeyError, TypeError, IndexError, RuntimeError) as exc:
        LOG.exception("Memory follow-up execution failed")
        result = {"status": "error", "training_eligible": False, "error": str(exc)}
        write_json(output / "followup_execution.json", result)
    print(f"status={result['status']} result={output / 'followup_execution.json'}", flush=True)
    return 0 if result["status"] in ("followup_proxy_passed", "preflight_passed") else 2


if __name__ == "__main__":
    raise SystemExit(main())
