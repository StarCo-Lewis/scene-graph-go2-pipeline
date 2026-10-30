#!/usr/bin/env python3
"""Search local pose-aware Spot grasp targets around a safe pre-grasp pose."""
from __future__ import annotations

import argparse
import logging
import math
import sys
import time
from pathlib import Path

from run_navigation_demo import (cg_to_habitat, gpu_free_mb, habitat_vector,
                                   peak_rss_mb, read_json, rss_mb,
                                   system_memory_mb, write_json)
from run_spot_approach_collision_check import (classify_robot_contacts,
                                                interpolate_joint_path)
from run_spot_grasp_contact_check import (contact_link_ids, interpolate_vector,
                                          spot_grasp_contact_links)
from run_spot_ik_reachability import (FullSpotIkHelper, distance, grounded_node,
                                      numeric_list, within_limits)

LOG = logging.getLogger("spot_grasp_target_search")


def local_grasp_targets(pregrasp, center, inward_offsets, downward_offsets,
                        wrist_pitch_offsets_deg=None,
                        wrist_roll_offsets_deg=None):
    pregrasp, center = numeric_list(pregrasp), numeric_list(center)
    dx, dz = center[0] - pregrasp[0], center[2] - pregrasp[2]
    norm = math.hypot(dx, dz)
    if norm == 0:
        raise ValueError("pre-grasp and object center have identical horizontal positions")
    ux, uz = dx / norm, dz / norm
    wrist_pitch_offsets_deg = ([0.0] if wrist_pitch_offsets_deg is None
                               else wrist_pitch_offsets_deg)
    wrist_roll_offsets_deg = ([0.0] if wrist_roll_offsets_deg is None
                              else wrist_roll_offsets_deg)
    return [{"inward_offset_m": float(inward),
             "downward_offset_m": float(down),
             "wrist_pitch_offset_deg": float(pitch),
             "wrist_roll_offset_deg": float(roll),
             "world_target": [pregrasp[0] + ux * inward,
                              pregrasp[1] - down,
                              pregrasp[2] + uz * inward]}
            for inward in inward_offsets for down in downward_offsets
            for pitch in wrist_pitch_offsets_deg
            for roll in wrist_roll_offsets_deg]


def inward_offsets_for_standoff(clearance_m):
    """Place local grasp probes relative to the selected pre-grasp standoff."""
    clearance = float(clearance_m)
    if not math.isfinite(clearance) or clearance <= 0.04:
        raise ValueError("selected pre-grasp standoff must exceed 0.04 m")
    return [round(clearance - delta, 4) for delta in (0.04, 0.02, 0.0)]


def _normalize3(value):
    value = numeric_list(value)
    if len(value) != 3:
        raise ValueError("3D vector required")
    norm = math.sqrt(sum(item * item for item in value))
    if norm <= 1e-9:
        raise ValueError("zero-length direction is invalid")
    return [item / norm for item in value]


def angle_degrees(a, b):
    a, b = _normalize3(a), _normalize3(b)
    cosine = max(-1.0, min(1.0, sum(x * y for x, y in zip(a, b))))
    return math.degrees(math.acos(cosine))


def grasp_candidate_rank(trial):
    """Rank contacts by bilateral support, shallow penetration, persistence."""
    final_links = trial.get("final_gripper_kettle_link_ids", [])
    contact_samples = int(trial.get("closure_contact_sample_count", 0))
    penetration = float(trial.get("final_gripper_max_penetration_m",
                                  float("inf")))
    return (len(final_links) >= 2, len(final_links), -penetration,
            contact_samples,
            -float(trial.get("goal_habitat_fk_error_m", float("inf"))))


def classify_contact_candidate(final_links, required_links, max_penetration_m,
                               penetration_limit_m, invalid_collision):
    """Require opposing grasp surfaces and shallow contact throughout closure."""
    if invalid_collision:
        return "invalid_collision"
    if max_penetration_m > penetration_limit_m:
        return "excessive_gripper_penetration"
    if not final_links:
        return "no_final_gripper_contact"
    if not set(required_links).issubset(set(final_links)):
        return "single_sided_contact"
    return "contact_candidate"


def pregrasp_joint_waypoints(pregrasp):
    """Read the validated direct or segmented route stored by base-pose search."""
    strategy = pregrasp.get("trajectory_strategy", "direct_joint_interpolation")
    if strategy == "lift_traverse_descend":
        segmented = pregrasp.get("segmented_approach")
        if not isinstance(segmented, dict) or segmented.get("status") != "feasible":
            raise ValueError("selected segmented pre-grasp route is not feasible")
        segments = segmented.get("segments")
        if not isinstance(segments, list) or not segments:
            raise ValueError("selected segmented pre-grasp route has no segments")
        if any(segment.get("status") != "feasible" for segment in segments):
            raise ValueError("selected segmented pre-grasp route contains a failed segment")
        return [{"name": segment.get("segment_name", f"segment_{index}"),
                 "joint_solution": numeric_list(segment["joint_solution"])}
                for index, segment in enumerate(segments)]
    return [{"name": "direct_pregrasp",
             "joint_solution": numeric_list(pregrasp["joint_solution"])}]


def run(args):
    for path in (args.alignment_json, args.scene_graph_json, args.base_search_json,
                 args.stage_config, args.navmesh, args.spot_urdf, args.kettle_config):
        if not path.is_file():
            raise ValueError(f"required input is missing: {path}")
    alignment, nodes, search = (read_json(args.alignment_json),
                                read_json(args.scene_graph_json),
                                read_json(args.base_search_json))
    node = grounded_node(alignment, nodes, args.role_id)
    selected = search.get("selected_base_pose")
    pregrasp = selected.get("selected_approach") if isinstance(selected, dict) else None
    if search.get("status") != "spot_safe_base_pose_found" or not isinstance(pregrasp, dict):
        raise ValueError("a successful Spot base-pose search is required")
    if int(search.get("target", {}).get("element_id", -1)) != int(node["id"]):
        raise ValueError("base-pose search and alignment reference different objects")
    cg_center = cg_to_habitat(node["bbox_center"], args.axis_mode)
    registration = search.get("object_registration")
    if not isinstance(registration, dict):
        raise ValueError("base-pose search has no dynamic object registration")
    center = numeric_list(search.get("target", {}).get("world_center", []))
    if len(center) != 3:
        raise ValueError("base-pose search has no registered object center")
    registered_origin = numeric_list(registration.get("registered_origin", []))
    if len(registered_origin) != 3:
        raise ValueError("base-pose search has no registered kettle origin")
    selected_clearance = float(pregrasp.get(
        "clearance_m", search.get("search_config", {}).get("clearance_m", 0.14)))
    inward_offsets = (args.inward_offsets_m if args.inward_offsets_m is not None
                      else inward_offsets_for_standoff(selected_clearance))
    max_pregrasp_penetration = float(search.get("search_config", {}).get(
        "max_gripper_penetration_m", 0.005))
    targets = local_grasp_targets(pregrasp["approach_world_target"], center,
                                  inward_offsets, args.downward_offsets_m,
                                  args.wrist_pitch_offsets_deg,
                                  args.wrist_roll_offsets_deg)

    import magnum as mn
    import numpy as np
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
        "scope": ("Bounded wrist-constrained local grasp-target search with discrete "
                  "path and kinematic gripper-contact checks; no force closure, "
                  "dynamic lift, or carry claim."),
        "target": {"role_id": args.role_id, "element_id": int(node["id"]),
                   "cg_world_center": cg_center,
                   "world_center": center},
        "object_registration": registration,
        "base_pose": {"position": selected["base_position"],
                      "rotation_rad": selected["base_rotation_rad"]},
        "pregrasp": pregrasp,
        "search_config": {"selected_pregrasp_clearance_m": selected_clearance,
                          "inward_offsets_m": inward_offsets,
                          "max_pregrasp_gripper_penetration_m":
                              max_pregrasp_penetration,
                          "downward_offsets_m": args.downward_offsets_m,
                          "wrist_pitch_offsets_deg": args.wrist_pitch_offsets_deg,
                          "wrist_roll_offsets_deg": args.wrist_roll_offsets_deg,
                          "candidate_count": len(targets)},
        "trials": [],
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
        open_joints = numeric_list(robot.params.gripper_open_state)
        closed_joints = numeric_list(robot.params.gripper_closed_state)
        if len(open_joints) != 1:
            raise ValueError("Spot IK adapter expects exactly one gripper joint")
        recorded_gripper = numeric_list(selected.get("gripper_joint_positions", []))
        if recorded_gripper != open_joints:
            raise ValueError("selected base pose was not validated with the open gripper")
        robot.gripper_joint_pos = open_joints
        robot.update()
        initial = np.asarray(robot.arm_joint_pos, dtype=np.float64)
        ik_seed = np.asarray(pregrasp["joint_solution"], dtype=np.float64)
        if ik_seed.shape != initial.shape:
            raise ValueError("selected pre-grasp joint solution has an invalid shape")
        lower, upper = robot.arm_joint_limits
        # Match the open-gripper Habitat frame used for grasp approach checks.
        ik = FullSpotIkHelper(args.spot_urdf, ik_seed, robot.ee_local_offset,
                              gripper_position=open_joints[0])

        manager = sim.get_object_template_manager()
        templates = manager.load_configs(str(args.kettle_config))
        if not templates:
            raise RuntimeError("Habitat could not load the kettle object template")
        template = manager.get_template_by_id(templates[0])
        kettle = sim.get_rigid_object_manager().add_object_by_template_handle(template.handle)
        if kettle is None:
            raise RuntimeError("Habitat could not instantiate the kettle")
        kettle.motion_type = MotionType.KINEMATIC
        kettle.translation = habitat_vector(registered_origin)

        gripper_links, grasp_link_names = spot_grasp_contact_links(robot)
        arm_links = set(int(link) for link in robot.params.arm_joints) | gripper_links
        result["grasp_contact_links"] = {
            "link_ids": sorted(gripper_links),
            "link_names": {str(key): grasp_link_names[key]
                           for key in sorted(gripper_links)},
            "note": ("Includes the movable finger and Spot's fixed opposing "
                     "grasp surface on the distal wrist link."),
        }
        robot_id, kettle_id = int(robot.sim_obj.object_id), int(kettle.object_id)
        base_inverse = robot.base_transformation.inverted()

        # Reproduce the route that the base-pose search actually validated.
        # Local candidates start at its endpoint rather than taking an
        # unvalidated shortcut from the default arm state.
        replay_segments = []
        replay_start = numeric_list(initial)
        replay_ok = True
        for waypoint in pregrasp_joint_waypoints(pregrasp):
            replay_goal = waypoint["joint_solution"]
            replay_path = interpolate_joint_path(
                replay_start, replay_goal, args.max_joint_step_rad,
                args.minimum_steps)
            replay_blocking = None
            for sample_index, joints in enumerate(replay_path):
                robot.arm_joint_pos = joints
                robot.gripper_joint_pos = open_joints
                robot.update()
                sim.perform_discrete_collision_detection()
                contacts = classify_robot_contacts(
                    sim.get_physics_contact_points(), robot_id, kettle_id,
                    arm_links)
                gripper_kettle = [
                    contact for contact in contacts["arm_kettle"]
                    if contact["robot_link_id"] in gripper_links]
                non_gripper_kettle = [
                    contact for contact in contacts["arm_kettle"]
                    if contact["robot_link_id"] not in gripper_links]
                gripper_penetration = max(
                    [max(0.0, -float(contact.get("contact_distance", 0.0)))
                     for contact in gripper_kettle] or [0.0])
                if (contacts["arm_environment"] or non_gripper_kettle or
                        gripper_penetration > max_pregrasp_penetration):
                    replay_blocking = {
                        "sample_index": sample_index,
                        "arm_environment_contacts": len(
                            contacts["arm_environment"]),
                        "non_gripper_arm_kettle_contacts": len(non_gripper_kettle),
                        "gripper_max_penetration_m": round(
                            gripper_penetration, 6),
                    }
                    break
            replay_segments.append({
                "segment_name": waypoint["name"],
                "trajectory_sample_count": len(replay_path),
                "first_blocking_sample": replay_blocking,
                "status": "feasible" if replay_blocking is None else "collision",
            })
            if replay_blocking is not None:
                replay_ok = False
                break
            replay_start = replay_goal
        result["pregrasp_replay"] = {
            "trajectory_strategy": pregrasp.get(
                "trajectory_strategy", "direct_joint_interpolation"),
            "segments": replay_segments,
            "status": "feasible" if replay_ok else "collision",
        }
        if not replay_ok:
            raise ValueError("selected pre-grasp route could not be reproduced")

        contact_candidates = []
        for index, target in enumerate(targets):
            world_target = target["world_target"]
            base_target = numeric_list(base_inverse.transform_point(mn.Vector3(world_target)))
            wrist_seed = np.asarray(ik_seed, dtype=np.float64).copy()
            wrist_seed[5] += math.radians(target["wrist_pitch_offset_deg"])
            wrist_seed[6] += math.radians(target["wrist_roll_offset_deg"])
            wrist_seed_ok = within_limits(wrist_seed, lower, upper)
            if not wrist_seed_ok:
                result["trials"].append({
                    "candidate_index": index, **target,
                    "wrist_seed": numeric_list(wrist_seed),
                    "within_joint_limits": False,
                    "orientation_strategy": "locked_wrist_joint_sweep",
                    "status": "wrist_seed_invalid"})
                continue
            ik.set_arm_state(wrist_seed)
            solution = numeric_list(ik.calc_ik(
                np.asarray(base_target, dtype=np.float64),
                locked_arm_joint_indices=[5, 6]))
            limits_ok = within_limits(solution, lower, upper)
            pybullet_error = distance(base_target, numeric_list(
                ik.calc_fk(np.asarray(solution))))
            trial = {"candidate_index": index, **target, "base_target": base_target,
                     "joint_solution": solution, "within_joint_limits": limits_ok,
                     "pybullet_fk_error_m": round(pybullet_error, 6),
                     "wrist_seed": numeric_list(wrist_seed),
                     "locked_arm_joint_indices": [5, 6],
                     "orientation_strategy": "locked_wrist_joint_sweep"}
            if not limits_ok or pybullet_error > args.max_fk_error_m:
                trial["status"] = "ik_invalid"
                result["trials"].append(trial)
                continue

            robot.gripper_joint_pos = open_joints
            robot.arm_joint_pos = numeric_list(ik_seed)
            robot.update()
            path = interpolate_joint_path(ik_seed, solution, args.max_joint_step_rad,
                                          args.minimum_steps)
            blocking = None
            first_gripper_kettle_contact = None
            max_approach_gripper_penetration = 0.0
            for sample_index, joints in enumerate(path):
                robot.arm_joint_pos = joints
                robot.update()
                sim.perform_discrete_collision_detection()
                contacts = classify_robot_contacts(sim.get_physics_contact_points(),
                                                   robot_id, kettle_id, arm_links)
                gripper_kettle = [contact for contact in contacts["arm_kettle"]
                                  if contact["robot_link_id"] in gripper_links]
                non_gripper_kettle = [contact for contact in contacts["arm_kettle"]
                                      if contact["robot_link_id"] not in gripper_links]
                gripper_penetration = max(
                    [max(0.0, -float(contact.get("contact_distance", 0.0)))
                     for contact in gripper_kettle] or [0.0])
                max_approach_gripper_penetration = max(
                    max_approach_gripper_penetration, gripper_penetration)
                if gripper_kettle and first_gripper_kettle_contact is None:
                    first_gripper_kettle_contact = {
                        "sample_index": sample_index,
                        "contact_count": len(gripper_kettle),
                        "gripper_kettle_link_ids": contact_link_ids(
                            gripper_kettle),
                    }
                # Contact by an open gripper is the event this search is meant
                # to find.  Only the rest of the arm touching the kettle, or
                # any arm link touching the stage, invalidates the approach.
                if (contacts["arm_environment"] or non_gripper_kettle or
                        gripper_penetration > max_pregrasp_penetration):
                    blocking = {"sample_index": sample_index,
                                "arm_environment_contacts": len(contacts["arm_environment"]),
                                "arm_kettle_contacts": len(contacts["arm_kettle"]),
                                "non_gripper_kettle_contacts": len(non_gripper_kettle),
                                "gripper_max_penetration_m": round(
                                    gripper_penetration, 6)}
                    break
            robot.arm_joint_pos = solution
            robot.update()
            habitat_error = distance(world_target,
                                     numeric_list(robot.ee_transform.translation))
            habitat_forward = numeric_list(robot.ee_transform.transform_vector(
                mn.Vector3(1.0, 0.0, 0.0)))
            desired_world_forward = [center[i] - world_target[i] for i in range(3)]
            object_alignment_error = angle_degrees(
                habitat_forward, desired_world_forward)
            trial.update({"trajectory_sample_count": len(path),
                          "first_blocking_sample": blocking,
                          "first_gripper_kettle_contact_sample":
                              first_gripper_kettle_contact,
                          "max_approach_gripper_penetration_m": round(
                              max_approach_gripper_penetration, 6),
                          "goal_habitat_fk_error_m": round(habitat_error, 6),
                          "habitat_tool_forward_world": habitat_forward,
                          "tool_to_object_angle_deg": round(
                              object_alignment_error, 3)})
            if blocking or habitat_error > args.max_fk_error_m:
                trial["status"] = "approach_invalid"
                result["trials"].append(trial)
                LOG.info("candidate=%d inward=%.3f down=%.3f wrist_pitch=%+.0f wrist_roll=%+.0f status=%s pos_err=%.3f object_angle=%.1f",
                         index, target["inward_offset_m"],
                         target["downward_offset_m"],
                         target["wrist_pitch_offset_deg"],
                         target["wrist_roll_offset_deg"], trial["status"],
                         habitat_error, object_alignment_error)
                continue

            closure_samples = []
            closure_invalid = False
            for closure_index, joints in enumerate(interpolate_vector(
                    open_joints, closed_joints, args.closure_steps)):
                robot.gripper_joint_pos = joints
                robot.update()
                sim.perform_discrete_collision_detection()
                contacts = classify_robot_contacts(sim.get_physics_contact_points(),
                                                   robot_id, kettle_id, arm_links)
                gripper_kettle = [contact for contact in contacts["arm_kettle"]
                                  if contact["robot_link_id"] in gripper_links]
                non_gripper_kettle = [contact for contact in contacts["arm_kettle"]
                                      if contact["robot_link_id"] not in gripper_links]
                closure_invalid = (closure_invalid or bool(contacts["arm_environment"])
                                   or bool(non_gripper_kettle))
                closure_samples.append({
                    "sample_index": closure_index,
                    "closure_fraction": round(closure_index / args.closure_steps, 4),
                    "gripper_kettle_contact_count": len(gripper_kettle),
                    "gripper_kettle_link_ids": contact_link_ids(gripper_kettle),
                    "gripper_contact_distances_m": [
                        round(float(contact.get("contact_distance", 0.0)), 6)
                        for contact in gripper_kettle],
                    "gripper_max_penetration_m": round(max(
                        [max(0.0, -float(contact.get("contact_distance", 0.0)))
                         for contact in gripper_kettle] or [0.0]), 6),
                    "arm_environment_contact_count": len(contacts["arm_environment"]),
                    "non_gripper_kettle_contact_count": len(non_gripper_kettle)})
            final_links = closure_samples[-1]["gripper_kettle_link_ids"]
            max_closure_penetration = max(
                sample["gripper_max_penetration_m"] for sample in closure_samples)
            candidate_status = classify_contact_candidate(
                final_links, gripper_links, max_closure_penetration,
                max_pregrasp_penetration, closure_invalid)
            contact_found = candidate_status == "contact_candidate"
            contact_sample_count = sum(
                bool(sample["gripper_kettle_link_ids"])
                for sample in closure_samples)
            trial.update({"closure_samples": closure_samples,
                          "final_gripper_kettle_link_ids": final_links,
                          "final_gripper_max_penetration_m":
                              closure_samples[-1]["gripper_max_penetration_m"],
                          "max_closure_gripper_penetration_m":
                              max_closure_penetration,
                          "closure_contact_sample_count": contact_sample_count,
                          "multi_link_contact_candidate": len(final_links) >= 2,
                          "closure_invalid_collision": closure_invalid,
                          "contact_candidate_found": contact_found,
                          "status": candidate_status})
            result["trials"].append(trial)
            LOG.info("candidate=%d inward=%.3f down=%.3f wrist_pitch=%+.0f wrist_roll=%+.0f status=%s final_links=%s",
                     index, target["inward_offset_m"], target["downward_offset_m"],
                     target["wrist_pitch_offset_deg"],
                     target["wrist_roll_offset_deg"], trial["status"], final_links)
            if contact_found:
                contact_candidates.append(trial)
        selected_trial = (max(contact_candidates, key=grasp_candidate_rank)
                          if contact_candidates else None)
        multi_link_count = sum(
            bool(trial.get("multi_link_contact_candidate"))
            for trial in contact_candidates)
        ik.close()
        result.update({
            "status": ("spot_grasp_target_candidate_found" if selected_trial
                       else "spot_grasp_target_candidate_not_found"),
            "selected_grasp_target": selected_trial,
            "contact_candidate_found": selected_trial is not None,
            "contact_candidate_count": len(contact_candidates),
            "multi_link_contact_candidate_count": multi_link_count,
            "candidate_selection": {
                "strategy": ("prefer_multi_link_then_link_count_then_shallow_"
                             "penetration_then_contact_persistence_then_fk_error"),
                "evaluated_all_candidates": True,
            },
            "skill_precondition_update": {
                "step_id": "st_01_03", "predicate": "graspable",
                "status": "unknown",
                "reason": ("contact_candidate_requires_dynamic_lift_validation"
                           if selected_trial else "no_collision_free_contact_target_found")},
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
    parser.add_argument("--inward-offsets-m", nargs="+", type=float,
                        default=None)
    parser.add_argument("--downward-offsets-m", nargs="+", type=float,
                        default=[0.08, 0.12, 0.15])
    parser.add_argument("--wrist-pitch-offsets-deg", nargs="+", type=float,
                        default=[-30.0, 0.0, 30.0])
    parser.add_argument("--wrist-roll-offsets-deg", nargs="+", type=float,
                        default=[0.0, 90.0, -90.0])
    parser.add_argument("--max-joint-step-rad", type=float, default=0.08)
    parser.add_argument("--minimum-steps", type=int, default=20)
    parser.add_argument("--closure-steps", type=int, default=12)
    parser.add_argument("--max-fk-error-m", type=float, default=0.04)
    args = parser.parse_args()
    args.log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(args.log_file, encoding="utf-8")])
    try:
        result = run(args)
        LOG.info("status=%s contact_candidate=%s result=%s", result["status"],
                 result["contact_candidate_found"], args.output)
        return 0 if result["status"] == "spot_grasp_target_candidate_found" else 2
    except Exception as exc:
        LOG.exception("Spot grasp-target search failed")
        write_json(args.output, {"schema_version": "1.0", "status": "error",
                                 "error": {"type": type(exc).__name__,
                                           "message": str(exc)},
                                 "resource_summary": {
                                     "process_peak_rss_mb": peak_rss_mb(),
                                     "system_memory": system_memory_mb()}})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
