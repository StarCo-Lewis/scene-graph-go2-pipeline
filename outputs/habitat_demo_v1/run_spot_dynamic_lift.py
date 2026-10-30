#!/usr/bin/env python3
"""Replay a validated Scene201 Spot grasp candidate and test a dynamic lift."""
from __future__ import annotations

import argparse
import logging
import math
import sys
import time
from pathlib import Path

from run_navigation_demo import (gpu_free_mb, habitat_vector, peak_rss_mb,
                                   read_json, rss_mb, system_memory_mb,
                                   write_json)
from run_spot_approach_collision_check import (classify_robot_contacts,
                                                interpolate_joint_path)
from run_spot_grasp_contact_check import (contact_link_ids, interpolate_vector,
                                          spot_grasp_contact_links)
from run_spot_ik_reachability import numeric_list, within_limits

LOG = logging.getLogger("spot_dynamic_lift")


def evaluate_dynamic_lift(object_lift_m, ee_lift_m, relative_drift_m,
                          gripper_contact_frames, invalid_collision,
                          min_object_lift_m=0.05, min_follow_ratio=0.5,
                          max_relative_drift_m=0.08):
    """Return an auditable dynamic-lift verdict from measured motion."""
    follow_ratio = (object_lift_m / ee_lift_m
                    if ee_lift_m > 1e-9 else 0.0)
    failures = []
    if invalid_collision:
        failures.append("invalid_arm_collision")
    if object_lift_m < min_object_lift_m:
        failures.append("object_did_not_rise_enough")
    if follow_ratio < min_follow_ratio:
        failures.append("object_did_not_follow_end_effector")
    if relative_drift_m > max_relative_drift_m:
        failures.append("object_separated_from_gripper")
    if gripper_contact_frames < 1:
        failures.append("no_gripper_contact_during_lift")
    return {
        "passed": not failures,
        "failures": failures,
        "object_lift_m": round(float(object_lift_m), 6),
        "end_effector_lift_m": round(float(ee_lift_m), 6),
        "follow_ratio": round(float(follow_ratio), 4),
        "relative_distance_drift_m": round(float(relative_drift_m), 6),
        "gripper_contact_frames": int(gripper_contact_frames),
        "invalid_collision": bool(invalid_collision),
        "thresholds": {
            "minimum_object_lift_m": float(min_object_lift_m),
            "minimum_follow_ratio": float(min_follow_ratio),
            "maximum_relative_distance_drift_m": float(max_relative_drift_m),
        },
    }


def vector3(value):
    return [float(value[0]), float(value[1]), float(value[2])]


def solve_habitat_position_ik(robot, start, target_world, lower, upper,
                              locked_indices=(5, 6), max_iterations=200):
    """Solve local position IK against Habitat's actual Spot EE transform."""
    import numpy as np

    state = np.asarray(start, dtype=np.float64).copy()
    target = np.asarray(target_world, dtype=np.float64)
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    locked = {int(index) for index in locked_indices}
    active = [index for index in range(len(state)) if index not in locked]
    locked_values = state.copy()

    def fk(joints):
        robot.arm_joint_pos = numeric_list(joints)
        robot.update()
        return np.asarray(vector3(robot.ee_transform.translation),
                          dtype=np.float64)

    best_state = state.copy()
    best_error = float(np.linalg.norm(target - fk(state)))
    finite_difference = 1e-4
    damping = 2e-3
    for _ in range(max_iterations):
        current = fk(state)
        error = target - current
        error_norm = float(np.linalg.norm(error))
        if error_norm < best_error:
            best_error, best_state = error_norm, state.copy()
        if error_norm <= 5e-5:
            break
        jacobian = np.zeros((3, len(active)), dtype=np.float64)
        for column, joint_index in enumerate(active):
            plus, minus = state.copy(), state.copy()
            plus[joint_index] = min(upper[joint_index],
                                    plus[joint_index] + finite_difference)
            minus[joint_index] = max(lower[joint_index],
                                     minus[joint_index] - finite_difference)
            span = plus[joint_index] - minus[joint_index]
            if span > 0.0:
                jacobian[:, column] = (fk(plus) - fk(minus)) / span
        system = (jacobian @ jacobian.T
                  + damping ** 2 * np.eye(3, dtype=np.float64))
        try:
            delta = jacobian.T @ np.linalg.solve(system, error)
        except np.linalg.LinAlgError:
            delta = np.linalg.pinv(jacobian) @ error
        delta_norm = float(np.linalg.norm(delta))
        if delta_norm > 0.12:
            delta *= 0.12 / delta_norm
        improved = False
        for scale in (1.0, 0.5, 0.25, 0.125, 0.0625):
            candidate = state.copy()
            candidate[active] += scale * delta
            candidate = np.minimum(np.maximum(candidate, lower), upper)
            for index in locked:
                candidate[index] = locked_values[index]
            if float(np.linalg.norm(target - fk(candidate))) + 1e-9 < error_norm:
                state = candidate
                improved = True
                break
        if not improved:
            break
    final_error = float(np.linalg.norm(target - fk(state)))
    if final_error < best_error:
        best_error, best_state = final_error, state.copy()
    fk(best_state)
    return numeric_list(best_state), best_error


def run(args):
    for path in (args.grasp_search_json, args.stage_config, args.navmesh,
                 args.spot_urdf, args.kettle_config):
        if not path.is_file():
            raise ValueError(f"required input is missing: {path}")
    search = read_json(args.grasp_search_json)
    candidate = search.get("selected_grasp_target")
    if (search.get("status") != "spot_grasp_target_candidate_found"
            or not isinstance(candidate, dict)
            or not candidate.get("contact_candidate_found")):
        raise ValueError("a successful grasp-target contact candidate is required")
    grasp_links = set(search.get("grasp_contact_links", {}).get("link_ids", []))
    final_links = set(candidate.get("final_gripper_kettle_link_ids", []))
    max_penetration = float(search.get("search_config", {}).get(
        "max_pregrasp_gripper_penetration_m", 0.005))
    closure_depth = candidate.get("max_closure_gripper_penetration_m")
    if (len(grasp_links) < 2 or not grasp_links.issubset(final_links)
            or closure_depth is None
            or float(closure_depth) > max_penetration):
        raise ValueError(
            "grasp candidate lacks bilateral shallow contact; rerun the "
            "updated grasp-target search before dynamic lift")

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
        "scope": ("Dynamic rigid-body lift test for one previously validated "
                  "collision-free Spot contact candidate; this is not a robust "
                  "force-closure or real-robot grasp claim."),
        "inputs": {"grasp_search": str(args.grasp_search_json),
                   "stage_config": str(args.stage_config),
                   "spot_urdf": str(args.spot_urdf),
                   "kettle_config": str(args.kettle_config)},
        "source_candidate": candidate,
        "resource_samples": [{"stage": "pre_scene_load", "rss_mb": rss_mb(),
                              "gpu_free_mb": gpu_free_mb()}],
    }
    write_json(args.output, result)

    with habitat_sim.Simulator(
            habitat_sim.Configuration(cfg, [AgentConfiguration()])) as sim:
        if not sim.pathfinder.load_nav_mesh(str(args.navmesh)):
            raise ValueError("Scene201 navmesh failed to load")
        base_pose = search["base_pose"]
        robot = SpotRobot(str(args.spot_urdf), sim, limit_robo_joints=True,
                          fixed_base=True)
        robot.reconfigure()
        robot.reset()
        robot.base_pos = habitat_vector(base_pose["position"])
        robot.base_rot = float(base_pose["rotation_rad"])
        open_joints = numeric_list(robot.params.gripper_open_state)
        closed_joints = numeric_list(robot.params.gripper_closed_state)
        candidate_joints = numeric_list(candidate["joint_solution"])
        lower, upper = robot.arm_joint_limits
        if not within_limits(candidate_joints, lower, upper):
            raise ValueError("source grasp candidate is outside Spot arm limits")
        robot.arm_joint_pos = candidate_joints
        robot.gripper_joint_pos = open_joints
        robot.update()

        manager = sim.get_object_template_manager()
        templates = manager.load_configs(str(args.kettle_config))
        if not templates:
            raise RuntimeError("Habitat could not load the kettle object template")
        template = manager.get_template_by_id(templates[0])
        kettle = sim.get_rigid_object_manager().add_object_by_template_handle(
            template.handle)
        if kettle is None:
            raise RuntimeError("Habitat could not instantiate the kettle")
        center = numeric_list(search["target"]["world_center"])
        registration = search.get("object_registration")
        registered_origin = (numeric_list(registration.get("registered_origin", []))
                             if isinstance(registration, dict) else [])
        if len(registered_origin) != 3:
            raise ValueError("grasp search has no registered kettle origin")
        kettle.motion_type = MotionType.KINEMATIC
        kettle.translation = habitat_vector(registered_origin)
        result["object_registration"] = registration

        gripper_links, grasp_link_names = spot_grasp_contact_links(robot)
        arm_links = set(int(link) for link in robot.params.arm_joints) | gripper_links
        robot_id, kettle_id = int(robot.sim_obj.object_id), int(kettle.object_id)
        result["grasp_contact_links"] = {
            "link_ids": sorted(gripper_links),
            "link_names": {str(key): grasp_link_names[key]
                           for key in sorted(gripper_links)},
        }

        # Reproduce the selected kinematic closure before releasing the object
        # to Bullet.  The source run already validated the incoming arm path.
        closure_samples = []
        for index, joints in enumerate(interpolate_vector(
                open_joints, closed_joints, args.closure_steps)):
            robot.gripper_joint_pos = joints
            robot.update()
            sim.perform_discrete_collision_detection()
            contacts = classify_robot_contacts(
                sim.get_physics_contact_points(), robot_id, kettle_id, arm_links)
            gripper_kettle = [contact for contact in contacts["arm_kettle"]
                              if contact["robot_link_id"] in gripper_links]
            non_gripper_kettle = [contact for contact in contacts["arm_kettle"]
                                  if contact["robot_link_id"] not in gripper_links]
            closure_samples.append({
                "sample_index": index,
                "gripper_kettle_link_ids": contact_link_ids(gripper_kettle),
                "arm_environment_contacts": len(contacts["arm_environment"]),
                "non_gripper_kettle_contacts": len(non_gripper_kettle),
            })
        final_closure = closure_samples[-1]
        if (not final_closure["gripper_kettle_link_ids"]
                or final_closure["arm_environment_contacts"]
                or final_closure["non_gripper_kettle_contacts"]):
            raise ValueError("source contact candidate could not be reproduced")

        # Solve directly against Habitat's actual closed-gripper EE transform.
        # The PyBullet adapter is useful for broad pose search, but its frame
        # mapping is not accurate enough to validate a 10 cm dynamic lift.
        habitat_lift_start = vector3(robot.ee_transform.translation)
        lift_target = habitat_lift_start.copy()
        lift_target[1] += args.lift_height_m
        lift_solution, lift_fk_error = solve_habitat_position_ik(
            robot, candidate_joints, lift_target, lower, upper,
            locked_indices=(5, 6))
        lift_ik_valid = (within_limits(lift_solution, lower, upper)
                         and lift_fk_error <= args.max_fk_error_m)
        result["lift_plan"] = {
            "requested_vertical_lift_m": args.lift_height_m,
            "solver": "habitat_local_damped_least_squares",
            "world_start": habitat_lift_start,
            "world_target": numeric_list(lift_target),
            "joint_solution": lift_solution,
            "within_joint_limits": within_limits(lift_solution, lower, upper),
            "habitat_fk_error_m": round(lift_fk_error, 6),
            "status": "feasible" if lift_ik_valid else "ik_invalid",
        }
        if not lift_ik_valid:
            result.update({
                "status": "spot_dynamic_lift_plan_failed",
                "closure_replay": closure_samples,
                "skill_precondition_update": {
                    "step_id": "st_01_03", "predicate": "graspable",
                    "status": "unknown", "reason": "dynamic_lift_ik_invalid"},
            })
            write_json(args.output, result)
            return result

        # The solver leaves Habitat at the lift pose; restore the reproduced
        # closed-gripper contact before releasing the kettle to dynamics.
        robot.arm_joint_pos = candidate_joints
        robot.gripper_joint_pos = closed_joints
        robot.update()
        object_start = vector3(kettle.translation)
        ee_start = vector3(robot.ee_transform.translation)
        initial_relative_distance = math.dist(object_start, ee_start)
        kettle.motion_type = MotionType.DYNAMIC
        kettle.linear_velocity = habitat_vector([0.0, 0.0, 0.0])
        kettle.angular_velocity = habitat_vector([0.0, 0.0, 0.0])

        path = interpolate_joint_path(candidate_joints, lift_solution,
                                      args.max_joint_step_rad,
                                      args.minimum_steps)
        samples = []
        gripper_contact_frames = 0
        invalid_collision = False
        for sample_index, joints in enumerate(path):
            robot.arm_joint_pos = joints
            robot.gripper_joint_pos = closed_joints
            robot.update()
            for _ in range(args.physics_frames_per_command):
                sim.step_physics(1.0 / 60.0)
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
                if gripper_kettle:
                    gripper_contact_frames += 1
                invalid_collision = (invalid_collision
                                     or bool(contacts["arm_environment"])
                                     or bool(non_gripper_kettle))
            samples.append({
                "sample_index": sample_index,
                "kettle_position": vector3(kettle.translation),
                "end_effector_position": vector3(
                    robot.ee_transform.translation),
                "gripper_kettle_link_ids": contact_link_ids(gripper_kettle),
                "arm_environment_contacts": len(contacts["arm_environment"]),
                "non_gripper_kettle_contacts": len(non_gripper_kettle),
            })

        for _ in range(args.hold_frames):
            sim.step_physics(1.0 / 60.0)
            robot.update()

        object_final = vector3(kettle.translation)
        ee_final = vector3(robot.ee_transform.translation)
        object_lift = object_final[1] - object_start[1]
        ee_lift = ee_final[1] - ee_start[1]
        final_relative_distance = math.dist(object_final, ee_final)
        initial_relative_vector = [object_start[index] - ee_start[index]
                                   for index in range(3)]
        final_relative_vector = [object_final[index] - ee_final[index]
                                 for index in range(3)]
        relative_drift = math.dist(initial_relative_vector,
                                   final_relative_vector)
        verdict = evaluate_dynamic_lift(
            object_lift, ee_lift, relative_drift, gripper_contact_frames,
            invalid_collision, args.min_object_lift_m, args.min_follow_ratio,
            args.max_relative_drift_m)
        passed = verdict["passed"]
        result.update({
            "status": ("spot_dynamic_lift_validated" if passed
                       else "spot_dynamic_lift_failed"),
            "closure_replay": closure_samples,
            "dynamic_execution": {
                "object_start_position": object_start,
                "object_final_position": object_final,
                "end_effector_start_position": ee_start,
                "end_effector_final_position": ee_final,
                "initial_object_ee_distance_m": round(
                    initial_relative_distance, 6),
                "final_object_ee_distance_m": round(
                    final_relative_distance, 6),
                "initial_object_ee_vector": initial_relative_vector,
                "final_object_ee_vector": final_relative_vector,
                "path_sample_count": len(path),
                "physics_frames_per_command": args.physics_frames_per_command,
                "post_lift_hold_frames": args.hold_frames,
                "samples": samples,
                "verdict": verdict,
            },
            "skill_precondition_update": {
                "step_id": "st_01_03", "predicate": "graspable",
                "status": "verified" if passed else "unknown",
                "reason": ("dynamic_lift_validated" if passed
                           else "dynamic_lift_failed"),
            },
            "resource_summary": {
                "process_peak_rss_mb": peak_rss_mb(),
                "system_memory": system_memory_mb(),
                "elapsed_s": round(time.monotonic() - started, 3),
            },
        })
    write_json(args.output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("grasp_search_json", "stage_config", "navmesh",
                 "spot_urdf", "kettle_config", "output", "log_file"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path,
                            required=True)
    parser.add_argument("--kettle-origin-y", type=float, default=1.31381)
    parser.add_argument("--lift-height-m", type=float, default=0.10)
    parser.add_argument("--closure-steps", type=int, default=12)
    parser.add_argument("--max-joint-step-rad", type=float, default=0.04)
    parser.add_argument("--minimum-steps", type=int, default=24)
    parser.add_argument("--physics-frames-per-command", type=int, default=2)
    parser.add_argument("--hold-frames", type=int, default=30)
    parser.add_argument("--max-fk-error-m", type=float, default=0.04)
    parser.add_argument("--min-object-lift-m", type=float, default=0.05)
    parser.add_argument("--min-follow-ratio", type=float, default=0.5)
    parser.add_argument("--max-relative-drift-m", type=float, default=0.08)
    args = parser.parse_args()
    args.log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout),
                  logging.FileHandler(args.log_file, encoding="utf-8")])
    try:
        result = run(args)
        LOG.info("status=%s result=%s", result["status"], args.output)
        return 0 if result["status"] == "spot_dynamic_lift_validated" else 2
    except Exception as exc:
        LOG.exception("Spot dynamic lift failed")
        write_json(args.output, {
            "schema_version": "1.0", "status": "error",
            "error": {"type": type(exc).__name__, "message": str(exc)},
            "resource_summary": {"process_peak_rss_mb": peak_rss_mb(),
                                 "system_memory": system_memory_mb()},
        })
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
