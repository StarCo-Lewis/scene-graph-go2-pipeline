#!/usr/bin/env python3
"""Check gripper/object contacts at a collision-free Scene201 Spot approach pose."""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from run_navigation_demo import (cg_to_habitat, gpu_free_mb, habitat_vector,
                                   peak_rss_mb, read_json, rss_mb,
                                   system_memory_mb, write_json)
from run_spot_approach_collision_check import classify_robot_contacts
from run_spot_ik_reachability import grounded_node, numeric_list

LOG = logging.getLogger("spot_grasp_contact_check")


def interpolate_vector(start, goal, steps):
    start, goal = numeric_list(start), numeric_list(goal)
    if len(start) != len(goal) or not start:
        raise ValueError("joint vectors must be equal-length and non-empty")
    if steps < 1:
        raise ValueError("steps must be positive")
    return [[a + (b - a) * index / steps for a, b in zip(start, goal)]
            for index in range(steps + 1)]


def contact_link_ids(contacts):
    return sorted({int(contact["robot_link_id"]) for contact in contacts})


def select_spot_grasp_contact_links(configured_links, link_names):
    """Include Spot's moving finger and the fixed opposing grasp surface."""
    selected = {int(link) for link in configured_links}
    for link_id, name in link_names.items():
        normalized = str(name).lower()
        if normalized.endswith("arm0.link_wr1") or normalized.endswith("arm0.link_fngr"):
            selected.add(int(link_id))
    return selected


def spot_grasp_contact_links(robot):
    configured = {int(link) for link in robot.params.gripper_joints}
    configured.add(int(robot.params.ee_link))
    all_links = ({int(link) for link in robot.params.arm_joints}
                 | configured)
    getter = getattr(robot.sim_obj, "get_link_name", None)
    names = {}
    for link_id in all_links:
        name = None
        if callable(getter):
            try:
                name = getter(link_id)
            except Exception:
                name = None
        names[link_id] = str(name or "unknown")
    return select_spot_grasp_contact_links(configured, names), names


def run(args):
    for path in (args.alignment_json, args.scene_graph_json, args.base_search_json,
                 args.stage_config, args.navmesh, args.spot_urdf, args.kettle_config):
        if not path.is_file():
            raise ValueError(f"required input is missing: {path}")
    alignment = read_json(args.alignment_json)
    nodes = read_json(args.scene_graph_json)
    search = read_json(args.base_search_json)
    node = grounded_node(alignment, nodes, args.role_id)
    if search.get("status") != "spot_safe_base_pose_found":
        raise ValueError("a successful Spot base-pose search is required")
    selected = search.get("selected_base_pose")
    approach = selected.get("selected_approach") if isinstance(selected, dict) else None
    if not isinstance(approach, dict) or approach.get("status") != "feasible":
        raise ValueError("base-pose search has no feasible selected approach")
    if int(search.get("target", {}).get("element_id", -1)) != int(node["id"]):
        raise ValueError("base-pose search and alignment reference different objects")

    import habitat_sim
    from habitat.robots.spot_robot import SpotRobot
    from habitat_sim.agent import AgentConfiguration
    from habitat_sim.physics import MotionType

    cfg = habitat_sim.SimulatorConfiguration()
    cfg.scene_id = str(args.stage_config)
    cfg.create_renderer = False
    cfg.enable_physics = True
    started = time.monotonic()
    result = {
        "schema_version": "1.0",
        "status": "starting",
        "scope": ("Kinematic gripper-closure contact diagnostic at a validated "
                  "position-only approach pose; no force closure, lift, carry, or "
                  "dynamic grasp success claim."),
        "target": {"role_id": args.role_id, "element_id": int(node["id"]),
                   "object_tag": node.get("object_tag")},
        "selected_base_pose": {"base_position": selected["base_position"],
                               "base_rotation_rad": selected["base_rotation_rad"]},
        "selected_approach": approach,
        "resource_samples": [{"stage": "pre_scene_load", "rss_mb": rss_mb(),
                              "gpu_free_mb": gpu_free_mb()}],
    }
    write_json(args.output, result)
    with habitat_sim.Simulator(habitat_sim.Configuration(cfg, [AgentConfiguration()])) as sim:
        if not sim.pathfinder.load_nav_mesh(str(args.navmesh)):
            raise ValueError("Scene201 navmesh failed to load")
        robot = SpotRobot(str(args.spot_urdf), sim, limit_robo_joints=True,
                          fixed_base=True)
        robot.reconfigure()
        robot.reset()
        robot.base_pos = habitat_vector(selected["base_position"])
        robot.base_rot = float(selected["base_rotation_rad"])
        robot.arm_joint_pos = numeric_list(approach["joint_solution"])
        open_joints = numeric_list(robot.params.gripper_open_state)
        if numeric_list(selected.get("gripper_joint_positions", [])) != open_joints:
            raise ValueError("selected base pose was not validated with the open gripper")

        manager = sim.get_object_template_manager()
        templates = manager.load_configs(str(args.kettle_config))
        if not templates:
            raise RuntimeError("Habitat could not load the kettle object template")
        template = manager.get_template_by_id(templates[0])
        kettle = sim.get_rigid_object_manager().add_object_by_template_handle(template.handle)
        if kettle is None:
            raise RuntimeError("Habitat could not instantiate the kettle")
        center = cg_to_habitat(node["bbox_center"], args.axis_mode)
        kettle.motion_type = MotionType.KINEMATIC
        kettle.translation = habitat_vector([center[0], args.kettle_origin_y, center[2]])

        gripper_links, link_names = spot_grasp_contact_links(robot)
        arm_links = set(int(link) for link in robot.params.arm_joints) | gripper_links
        robot_id, kettle_id = int(robot.sim_obj.object_id), int(kettle.object_id)
        closed_joints = numeric_list(robot.params.gripper_closed_state)
        samples = []
        any_environment_contacts = False
        any_non_gripper_kettle_contacts = False
        for index, joints in enumerate(interpolate_vector(
                open_joints, closed_joints, args.closure_steps)):
            robot.gripper_joint_pos = joints
            robot.update()
            sim.perform_discrete_collision_detection()
            contacts = classify_robot_contacts(sim.get_physics_contact_points(),
                                               robot_id, kettle_id, arm_links)
            gripper_kettle = [contact for contact in contacts["arm_kettle"]
                              if contact["robot_link_id"] in gripper_links]
            gripper_environment = [contact for contact in contacts["arm_environment"]
                                   if contact["robot_link_id"] in gripper_links]
            non_gripper_kettle = [contact for contact in contacts["arm_kettle"]
                                  if contact["robot_link_id"] not in gripper_links]
            any_environment_contacts = (any_environment_contacts or
                                        bool(contacts["arm_environment"]))
            any_non_gripper_kettle_contacts = (any_non_gripper_kettle_contacts or
                                               bool(non_gripper_kettle))
            samples.append({
                "sample_index": index,
                "closure_fraction": round(index / args.closure_steps, 4),
                "gripper_joint_positions": numeric_list(joints),
                "gripper_kettle_contact_count": len(gripper_kettle),
                "gripper_kettle_link_ids": contact_link_ids(gripper_kettle),
                "gripper_environment_contact_count": len(gripper_environment),
                "non_gripper_arm_kettle_contact_count": len(non_gripper_kettle),
                "arm_environment_contact_count": len(contacts["arm_environment"]),
                "non_arm_contacts_ignored": len(contacts["non_arm"]),
            })
            LOG.info("closure=%d/%d kettle=%d gripper_env=%d arm_env=%d",
                     index, args.closure_steps, len(gripper_kettle),
                     len(gripper_environment), len(contacts["arm_environment"]))

        final = samples[-1]
        final_links = final["gripper_kettle_link_ids"]
        contact_candidate = (bool(final_links) and not any_environment_contacts
                             and not any_non_gripper_kettle_contacts)
        multi_link_candidate = (len(final_links) >= 2 and not any_environment_contacts
                                and not any_non_gripper_kettle_contacts)
        status = ("spot_grasp_contact_candidate_found" if contact_candidate
                  else "spot_grasp_contact_candidate_not_found")
        result.update({
            "status": status,
            "robot_object_ids": {"spot": robot_id, "kettle": kettle_id},
            "gripper_link_ids": sorted(gripper_links),
            "robot_arm_link_names": {str(key): value
                                     for key, value in link_names.items()},
            "closure": {"open_joint_positions": open_joints,
                        "closed_joint_positions": closed_joints,
                        "samples": samples},
            "final_gripper_kettle_link_ids": final_links,
            "contact_candidate_found": contact_candidate,
            "multi_link_contact_candidate": multi_link_candidate,
            "arm_environment_collision_during_closure": any_environment_contacts,
            "non_gripper_arm_kettle_contact_during_closure":
                any_non_gripper_kettle_contacts,
            "skill_precondition_update": {
                "step_id": "st_01_03", "predicate": "graspable",
                "status": "unknown",
                "reason": ("multi_link_contact_candidate_requires_dynamic_lift_validation"
                           if multi_link_candidate else
                           "force_closure_and_dynamic_lift_not_validated")},
            "resource_summary": {"process_peak_rss_mb": peak_rss_mb(),
                                 "system_memory": system_memory_mb(),
                                 "elapsed_s": round(time.monotonic() - started, 3)},
        })
    write_json(args.output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("alignment_json", "scene_graph_json", "base_search_json",
                 "stage_config", "navmesh", "spot_urdf", "kettle_config",
                 "output", "log_file"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--role-id", default="electric_kettle")
    parser.add_argument("--axis-mode", choices=["x_z_minus_y", "x_z_y", "identity"],
                        default="x_z_minus_y")
    parser.add_argument("--kettle-origin-y", type=float, default=1.31381)
    parser.add_argument("--closure-steps", type=int, default=20)
    args = parser.parse_args()
    args.log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(args.log_file, encoding="utf-8")])
    try:
        result = run(args)
        LOG.info("status=%s contact_candidate=%s result=%s", result["status"],
                 result["contact_candidate_found"], args.output)
        return 0 if result["status"] == "spot_grasp_contact_candidate_found" else 2
    except Exception as exc:
        LOG.exception("Spot grasp contact check failed")
        write_json(args.output, {"schema_version": "1.0", "status": "error",
                                 "error": {"type": type(exc).__name__,
                                           "message": str(exc)},
                                 "resource_summary": {
                                     "process_peak_rss_mb": peak_rss_mb(),
                                     "system_memory": system_memory_mb()}})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
