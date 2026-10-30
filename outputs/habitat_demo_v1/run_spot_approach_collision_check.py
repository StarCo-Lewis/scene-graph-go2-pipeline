#!/usr/bin/env python3
"""Discrete collision check for the grounded Spot-to-kettle IK approach path."""
from __future__ import annotations

import argparse
import logging
import math
import sys
import time
from pathlib import Path

from run_navigation_demo import (
    cg_to_habitat,
    gpu_free_mb,
    habitat_vector,
    peak_rss_mb,
    read_json,
    rss_mb,
    system_memory_mb,
    write_json,
)
from run_spot_ik_reachability import grounded_node, numeric_list

LOG = logging.getLogger("spot_approach_collision")


def interpolate_joint_path(start, goal, max_joint_step_rad=0.08, minimum_steps=20):
    start, goal = numeric_list(start), numeric_list(goal)
    if len(start) != len(goal) or not start:
        raise ValueError("joint path endpoints must be equal-length non-empty vectors")
    max_delta = max(abs(a - b) for a, b in zip(start, goal))
    segments = max(int(minimum_steps), int(math.ceil(max_delta / max_joint_step_rad)))
    return [[a + (b - a) * index / segments for a, b in zip(start, goal)]
            for index in range(segments + 1)]


def classify_robot_contacts(contacts, robot_id, kettle_id, arm_links):
    arm_environment, arm_kettle, non_arm = [], [], []
    for contact in contacts:
        if int(contact.object_id_a) == robot_id:
            robot_link, other = int(contact.link_id_a), int(contact.object_id_b)
        elif int(contact.object_id_b) == robot_id:
            robot_link, other = int(contact.link_id_b), int(contact.object_id_a)
        else:
            continue
        record = {"robot_link_id": robot_link, "other_object_id": other,
                  "contact_distance": float(getattr(contact, "contact_distance", 0.0))}
        # Habitat-Sim contact-point fields differ slightly between releases.
        # Preserve whichever world-space diagnostics this build exposes.
        for output_name, candidates in (
                ("position_on_a_ws", ("position_on_a_in_ws", "position_on_a")),
                ("position_on_b_ws", ("position_on_b_in_ws", "position_on_b")),
                ("normal_on_b_ws", ("contact_normal_on_b_in_ws", "normal_on_b_in_ws"))):
            for field in candidates:
                value = getattr(contact, field, None)
                if value is None:
                    continue
                try:
                    record[output_name] = [float(item) for item in value]
                except (TypeError, ValueError):
                    pass
                break
        if robot_link not in arm_links:
            non_arm.append(record)
        elif other == kettle_id:
            arm_kettle.append(record)
        else:
            arm_environment.append(record)
    return {"arm_environment": arm_environment, "arm_kettle": arm_kettle,
            "non_arm": non_arm}


def run(args):
    for path in (args.alignment_json, args.scene_graph_json, args.ik_json,
                 args.stage_config, args.navmesh, args.spot_urdf, args.kettle_config):
        if not path.is_file():
            raise ValueError(f"required input is missing: {path}")
    alignment, nodes, ik_result = (read_json(args.alignment_json),
                                   read_json(args.scene_graph_json),
                                   read_json(args.ik_json))
    node = grounded_node(alignment, nodes, args.role_id)
    if ik_result.get("status") != "spot_ik_reachability_passed" or not ik_result.get("reachable"):
        raise ValueError("a successful grounded Spot IK result is required")
    if int(ik_result.get("target", {}).get("element_id", -1)) != int(node["id"]):
        raise ValueError("IK result and current object alignment reference different instances")
    selected = ik_result.get("selected_candidate")
    if not isinstance(selected, dict) or not selected.get("passed"):
        raise ValueError("IK result has no passing selected candidate")

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
        "schema_version": "1.0", "status": "starting",
        "scope": "Discrete position-only arm-path collision check; non-arm contacts are reported but ignored, and no grasp orientation, closure, force, or dynamic stability is validated.",
        "target": {"role_id": args.role_id, "element_id": int(node["id"]),
                   "object_tag": node.get("object_tag")},
        "selected_ik_candidate": selected,
        "inputs": {"alignment": str(args.alignment_json), "scene_graph": str(args.scene_graph_json),
                   "ik": str(args.ik_json), "stage_config": str(args.stage_config),
                   "spot_urdf": str(args.spot_urdf), "kettle_config": str(args.kettle_config)},
        "resource_samples": [{"stage": "pre_scene_load", "rss_mb": rss_mb(),
                              "gpu_free_mb": gpu_free_mb()}],
    }
    write_json(args.output, result)
    with habitat_sim.Simulator(habitat_sim.Configuration(cfg, [AgentConfiguration()])) as sim:
        if not sim.pathfinder.load_nav_mesh(str(args.navmesh)):
            raise ValueError("Scene201 navmesh failed to load")
        spot_pose = ik_result["spot_pose"]
        robot = SpotRobot(str(args.spot_urdf), sim, limit_robo_joints=True, fixed_base=True)
        robot.reconfigure()
        robot.reset()
        robot.base_pos = habitat_vector(spot_pose["base_position"])
        robot.base_rot = float(spot_pose["base_rotation_rad"])
        robot.update()

        template_manager = sim.get_object_template_manager()
        template_ids = template_manager.load_configs(str(args.kettle_config))
        if not template_ids:
            raise RuntimeError("Habitat could not load the kettle object template")
        template = template_manager.get_template_by_id(template_ids[0])
        kettle = sim.get_rigid_object_manager().add_object_by_template_handle(template.handle)
        if kettle is None:
            raise RuntimeError("Habitat could not instantiate the kettle")
        kettle.motion_type = MotionType.KINEMATIC
        center = cg_to_habitat(node["bbox_center"], args.axis_mode)
        kettle.translation = habitat_vector([center[0], args.kettle_origin_y, center[2]])

        initial = numeric_list(spot_pose["initial_arm_joints"])
        goal = numeric_list(selected["joint_solution"])
        path = interpolate_joint_path(initial, goal, args.max_joint_step_rad, args.minimum_steps)
        arm_links = set(int(link) for link in robot.params.arm_joints)
        arm_links.update(int(link) for link in robot.params.gripper_joints)
        arm_links.add(int(robot.params.ee_link))
        robot_id, kettle_id = int(robot.sim_obj.object_id), int(kettle.object_id)
        samples, blocking = [], []
        final_kettle_contacts = []
        max_non_arm_contacts = 0
        for index, joints in enumerate(path):
            robot.arm_joint_pos = joints
            robot.update()
            sim.perform_discrete_collision_detection()
            classified = classify_robot_contacts(sim.get_physics_contact_points(),
                                                  robot_id, kettle_id, arm_links)
            final = index == len(path) - 1
            frame_blocking = list(classified["arm_environment"])
            if not final:
                frame_blocking.extend(classified["arm_kettle"])
            if frame_blocking:
                blocking.append({"sample_index": index, "path_fraction": round(index / (len(path) - 1), 4),
                                 "contacts": frame_blocking[:20]})
            if final:
                final_kettle_contacts = classified["arm_kettle"]
            max_non_arm_contacts = max(max_non_arm_contacts, len(classified["non_arm"]))
            samples.append({"sample_index": index,
                            "path_fraction": round(index / (len(path) - 1), 4),
                            "arm_environment_contacts": len(classified["arm_environment"]),
                            "arm_kettle_contacts": len(classified["arm_kettle"]),
                            "non_arm_contacts_ignored": len(classified["non_arm"])})
            LOG.info("path=%d/%d arm_env=%d arm_kettle=%d non_arm=%d",
                     index, len(path) - 1, len(classified["arm_environment"]),
                     len(classified["arm_kettle"]), len(classified["non_arm"]))

        final_ee = numeric_list(robot.ee_transform.translation)
        final_error = math.dist(final_ee, numeric_list(selected["world_target"]))
        collision_free = not blocking
        result.update({
            "status": "spot_approach_collision_check_passed" if collision_free
                      else "spot_approach_collision_detected",
            "robot_object_ids": {"spot": robot_id, "kettle": kettle_id},
            "arm_link_ids": sorted(arm_links),
            "kettle_pose": numeric_list(kettle.translation),
            "trajectory": {"sample_count": len(path),
                           "max_joint_step_rad": args.max_joint_step_rad,
                           "samples": samples},
            "blocking_collisions": blocking,
            "final_arm_kettle_contacts": final_kettle_contacts,
            "max_non_arm_contacts_ignored": max_non_arm_contacts,
            "final_ee_world": final_ee,
            "final_ee_target_error_m": round(final_error, 6),
            "approach_path_collision_free": collision_free,
            "skill_precondition_updates": [
                {"step_id": "st_01_02", "predicate": "collision_free_approach",
                 "status": "true" if collision_free else "false",
                 "reason": "discrete_arm_path_check"},
                {"step_id": "st_01_03", "predicate": "graspable", "status": "unknown",
                 "reason": "grasp_orientation_and_closure_not_validated"},
            ],
            "resource_summary": {"process_peak_rss_mb": peak_rss_mb(),
                                 "system_memory": system_memory_mb(),
                                 "elapsed_s": round(time.monotonic() - started, 3)},
        })
    write_json(args.output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("alignment_json", "scene_graph_json", "ik_json", "stage_config", "navmesh",
                 "spot_urdf", "kettle_config", "output", "log_file"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--role-id", default="electric_kettle")
    parser.add_argument("--axis-mode", choices=["x_z_minus_y", "x_z_y", "identity"],
                        default="x_z_minus_y")
    parser.add_argument("--kettle-origin-y", type=float, default=1.31381)
    parser.add_argument("--max-joint-step-rad", type=float, default=0.08)
    parser.add_argument("--minimum-steps", type=int, default=20)
    args = parser.parse_args()
    args.log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(args.log_file, encoding="utf-8")])
    try:
        result = run(args)
        LOG.info("status=%s collision_free=%s result=%s", result["status"],
                 result["approach_path_collision_free"], args.output)
        return 0 if result["status"] == "spot_approach_collision_check_passed" else 2
    except Exception as exc:
        LOG.exception("Spot approach collision check failed")
        write_json(args.output, {"schema_version": "1.0", "status": "error",
                                 "error": {"type": type(exc).__name__, "message": str(exc)},
                                 "resource_summary": {"process_peak_rss_mb": peak_rss_mb(),
                                                      "system_memory": system_memory_mb()}})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
