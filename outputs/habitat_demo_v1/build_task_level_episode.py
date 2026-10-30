#!/usr/bin/env python3
"""Build a task-level training record from grounded plan and proxy evidence.

Grasp and carry are policy assumptions once a safe approach is demonstrated.
This program never upgrades either assumption to a physical observation.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


SKILLS = [
    "navigate_to_object", "position_arm_for_object", "grasp_object",
    "lift_held_object", "navigate_to_object", "position_arm_for_object",
    "place_object_on_surface",
]


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def flatten_steps(plan):
    return [step for subtask in plan["subtask_plans"]
            for step in subtask["steps"]]


def build_episode(alignment, plan, approach, proxy, sources):
    if alignment.get("status") != "success":
        raise ValueError("object alignment did not succeed")
    if plan.get("instruction") != alignment.get("instruction"):
        raise ValueError("alignment and skill plan have different instructions")
    steps = flatten_steps(plan)
    if [step.get("skill_id") for step in steps] != SKILLS:
        raise ValueError("unsupported task skill sequence")
    roles = alignment.get("context_bindings", {})
    kettle_ids = roles.get("electric_kettle", {}).get("element_ids", [])
    board_ids = roles.get("sideboard", {}).get("element_ids", [])
    if len(kettle_ids) != 1 or len(board_ids) != 1:
        raise ValueError("the task needs one grounded kettle and one sideboard")
    kettle_id, board_id = int(kettle_ids[0]), int(board_ids[0])
    if (int(steps[2]["arguments"]["target"]["element_id"]) != kettle_id
            or int(steps[6]["arguments"]["surface"]["element_id"]) != board_id):
        raise ValueError("skill plan does not use the grounded object IDs")
    if (approach.get("status") != "spot_safe_base_pose_found"
            or approach.get("safe_base_pose_found") is not True
            or int(approach.get("target", {}).get("element_id", -1)) != kettle_id):
        raise ValueError("no validated approach for the grounded kettle")
    base = approach.get("selected_base_pose") or {}
    selected = base.get("selected_approach") or {}
    if (base.get("status") != "feasible"
            or selected.get("status") != "feasible"
            or selected.get("first_blocking_sample") is not None
            or float(selected.get("goal_habitat_fk_error_m", 1)) > 0.04):
        raise ValueError("selected approach is not a validated path")
    if (proxy.get("status") != "proxy_instance_support_passed"
            or proxy.get("scene_graph_ids") != {
                "kettle": kettle_id, "sideboard": board_id}):
        raise ValueError("proxy did not reach the grounded task goal")
    proxy_steps = [item for item in proxy.get("steps", [])
                   if item.get("skill_id") in SKILLS]
    if [item["skill_id"] for item in proxy_steps] != SKILLS:
        raise ValueError("proxy trace does not match the skill plan")
    placement = proxy_steps[-1]
    if (placement.get("status") != "proxy_instance_support_passed"
            or not placement.get("stable_last_second")
            or placement.get("direct_sideboard_contact_frames_last_second", 0) < 20):
        raise ValueError("sideboard goal lacks observed support evidence")
    if any(proxy_steps[index].get("status") != "arrived" for index in (0, 4)):
        raise ValueError("the proxy did not reach both navigation targets")
    proxy_kettle_nav_end = proxy_steps[0].get("end", [])
    if len(proxy_kettle_nav_end) != 3:
        raise ValueError("proxy navigation has no final position")
    base_nav_gap = math.hypot(
        float(proxy_kettle_nav_end[0]) - float(base["base_position"][0]),
        float(proxy_kettle_nav_end[2]) - float(base["base_position"][2]))
    if base_nav_gap > 0.6:
        raise ValueError("proxy navigation stopped too far from the validated base pose")

    evidence_types = [
        "proxy_navigation_observation", "kinematic_approach_validation",
        "assumed_after_validated_approach", "scripted_proxy_attachment",
        "proxy_navigation_observation", "scripted_proxy_positioning",
        "simulator_support_observation",
    ]
    trace = [{
        "step_id": step["step_id"], "skill_id": step["skill_id"],
        "arguments": step["arguments"], "evidence_type": evidence_type,
        "proxy_status": proxy_step["status"],
    } for step, proxy_step, evidence_type in zip(
        steps, proxy_steps, evidence_types)]
    return {
        "schema_version": "1.0",
        "record_type": "task_level_proxy_training_episode",
        "status": "goal_observed_in_proxy_with_assumed_grasp",
        "instruction": plan["instruction"],
        "scene_graph_reference": alignment["scene_reference"],
        "object_bindings": {
            "electric_kettle": kettle_id, "sideboard": board_id},
        "policy": {
            "grasp_rule": "assume_grasp_after_validated_approach",
            "physical_grasp_verified": False,
            "quadruped_locomotion_verified": False,
            "assumption_scope": "task_level_training_data_only",
            "combined_from_separate_simulation_runs": True,
            "continuous_robot_execution_verified": False,
        },
        "approach_evidence": {
            "base_position": base["base_position"],
            "base_rotation_rad": base["base_rotation_rad"],
            "target_world_position": selected["approach_world_target"],
            "object_standoff_m": selected["clearance_m"],
            "proxy_nav_to_spot_base_horizontal_gap_m": round(base_nav_gap, 4),
            "goal_fk_error_m": selected["goal_habitat_fk_error_m"],
            "collision_check": "passed_in_headless_habitat",
        },
        "execution_trace": trace,
        "goal": {
            "predicate": "on", "subject_id": kettle_id,
            "object_id": board_id,
            "status": "observed_in_habitat_proxy",
            "direct_support_contact_frames": placement[
                "direct_sideboard_contact_frames_last_second"],
            "stable_last_second": True,
        },
        "sources": {name: str(path) for name, path in sources.items()},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("alignment_json", "skill_plan_json", "base_search_json",
                "proxy_json", "output"):
        parser.add_argument("--" + key.replace("_", "-"), type=Path,
                            required=True)
    args = parser.parse_args()
    source_paths = {
        "alignment": args.alignment_json,
        "skill_plan": args.skill_plan_json,
        "approach": args.base_search_json,
        "proxy_execution": args.proxy_json,
    }
    result = build_episode(
        *(read_json(path) for path in source_paths.values()), source_paths)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=True) + "\n",
                           encoding="utf-8")
    print(f"status={result['status']}")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
