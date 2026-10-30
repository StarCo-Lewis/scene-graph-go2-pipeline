#!/usr/bin/env python3
"""Search navmesh-valid Spot base poses with reachable, collision-free kettle approach paths."""
from __future__ import annotations

import argparse
import logging
import math
import sys
import time
from pathlib import Path

from run_navigation_demo import cg_to_habitat, gpu_free_mb, habitat_vector, peak_rss_mb, read_json, rss_mb, system_memory_mb, write_json
from run_spot_approach_collision_check import classify_robot_contacts, interpolate_joint_path
from run_spot_grasp_contact_check import spot_grasp_contact_links
from run_spot_ik_reachability import FullSpotIkHelper, distance, grounded_node, numeric_list, within_limits

LOG = logging.getLogger("spot_base_pose_search")


def candidate_base_queries(center, reference_base, radii, angle_offsets_deg):
    center, reference_base = numeric_list(center), numeric_list(reference_base)
    dx, dz = reference_base[0] - center[0], reference_base[2] - center[2]
    base_angle = math.atan2(dz, dx)
    return [{"radius_m": float(radius), "angle_offset_deg": float(offset),
             "query": [center[0] + radius * math.cos(base_angle + math.radians(offset)),
                       0.0,
                       center[2] + radius * math.sin(base_angle + math.radians(offset))]}
            for radius in radii for offset in angle_offsets_deg]


def facing_rotation(base, target):
    dx, dz = float(target[0]) - float(base[0]), float(target[2]) - float(base[2])
    return math.atan2(-dz, dx)


def radial_approach(center, extent, base, clearance):
    dx, dz = float(base[0]) - float(center[0]), float(base[2]) - float(center[2])
    norm = math.hypot(dx, dz)
    if norm == 0:
        raise ValueError("base and target horizontal positions are identical")
    ux, uz = dx / norm, dz / norm
    # Axis-aligned graph boxes do not provide an oriented handle pose. This is
    # only the nearest exterior point in the horizontal approach direction.
    support = abs(ux) * float(extent[0]) / 2.0 + abs(uz) * float(extent[2]) / 2.0
    return [float(center[0]) + ux * (support + clearance), float(center[1]),
            float(center[2]) + uz * (support + clearance)]


def approach_targets(center, extent, base, clearance):
    """Generate position-only pre-grasp probes outside the observed object box.

    ``clearance`` is the tool/wrist standoff, rather than just a numerical
    epsilon around the object box.  The Spot distal wrist extends in front of
    the reported end-effector origin, so a short standoff can put the wrist
    through the object even while the requested end-effector point is outside.
    """
    radial = radial_approach(center, extent, base, clearance)
    top_y = float(center[1]) + float(extent[1]) / 2.0 + float(clearance)
    return [
        {"name": "radial_mid", "world_target": radial,
         "orientation_strategy": "preserve_measured_wrist_joints",
         "locked_arm_joint_indices": [5, 6]},
        {"name": "radial_high", "world_target": [radial[0], top_y, radial[2]],
         "orientation_strategy": "preserve_measured_wrist_joints",
         "locked_arm_joint_indices": [5, 6]},
        {"name": "above_center", "world_target": [float(center[0]), top_y,
                                                     float(center[2])],
         "orientation_strategy": "preserve_measured_wrist_joints",
         "locked_arm_joint_indices": [5, 6]},
    ]


def clearance_approaches(center, extent, base, clearances):
    """Try increasing object standoffs while retaining each candidate's source."""
    if not clearances or any(value <= 0 for value in clearances):
        raise ValueError("pre-grasp clearances must be positive and non-empty")
    for clearance in clearances:
        for approach in approach_targets(center, extent, base, clearance):
            yield {**approach, "clearance_m": float(clearance)}


def arm_contact_counts(sim, robot_id, kettle_id, arm_links):
    sim.perform_discrete_collision_detection()
    return classify_robot_contacts(sim.get_physics_contact_points(), robot_id, kettle_id, arm_links)


def clearance_waypoints(start_world, goal_world, clearance_m):
    """Return a lift, traverse, and descend route in Habitat world coordinates."""
    start, goal = numeric_list(start_world), numeric_list(goal_world)
    if len(start) != 3 or len(goal) != 3:
        raise ValueError("clearance waypoint endpoints must contain three values")
    safe_y = max(start[1], goal[1]) + float(clearance_m)
    return [
        {"name": "lift_from_start", "world_target": [start[0], safe_y, start[2]]},
        {"name": "traverse_above_goal", "world_target": [goal[0], safe_y, goal[2]]},
        {"name": "descend_to_goal", "world_target": goal},
    ]


def add_link_diagnostics(records, link_names):
    """Copy contact records and attach stable robot-link names when available."""
    enriched = []
    for record in records:
        item = dict(record)
        item["robot_link_name"] = link_names.get(int(item["robot_link_id"]), "unknown")
        enriched.append(item)
    return enriched


def max_contact_penetration(contacts):
    """Return the deepest negative Bullet contact distance in metres."""
    return max([max(0.0, -float(contact.get("contact_distance", 0.0)))
                for contact in contacts] or [0.0])


def robot_link_name_map(robot, link_ids):
    names = {}
    getter = getattr(robot.sim_obj, "get_link_name", None)
    for link_id in link_ids:
        name = None
        if callable(getter):
            try:
                name = getter(int(link_id))
            except Exception:
                name = None
        if not name:
            if int(link_id) == int(robot.params.ee_link):
                name = "end_effector"
            elif int(link_id) in {int(value) for value in robot.params.gripper_joints}:
                name = "gripper"
            elif int(link_id) in {int(value) for value in robot.params.arm_joints}:
                name = "arm_joint_link"
            else:
                name = "unknown"
        names[int(link_id)] = str(name)
    return names


def run(args):
    for path in (args.alignment_json, args.scene_graph_json, args.ik_json, args.stage_config,
                 args.navmesh, args.spot_urdf, args.kettle_config):
        if not path.is_file():
            raise ValueError(f"required input is missing: {path}")
    alignment, nodes, prior_ik = (read_json(args.alignment_json), read_json(args.scene_graph_json),
                                  read_json(args.ik_json))
    node = grounded_node(alignment, nodes, args.role_id)
    if prior_ik.get("status") != "spot_ik_reachability_passed":
        raise ValueError("a successful initial IK run is required")
    cg_center = cg_to_habitat(node["bbox_center"], args.axis_mode)
    center = numeric_list(cg_center)
    raw_extent = numeric_list(node["bbox_extent"])
    extent = [raw_extent[0], raw_extent[2], raw_extent[1]] if args.axis_mode != "identity" else raw_extent
    reference_base = prior_ik["spot_pose"]["base_position"]
    queries = candidate_base_queries(center, reference_base, args.radii, args.angle_offsets_deg)

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
    result = {"schema_version": "1.0", "status": "starting",
              "scope": "Navmesh base-pose search with measured-wrist-preserving contact-ready IK; environment and non-gripper arm contacts are rejected while registered open-gripper/object contact is allowed. No closure, force, locomotion, or dynamics claim.",
              "target": {"role_id": args.role_id, "element_id": int(node["id"]),
                         "cg_world_center": cg_center,
                         "world_center": center, "world_extent": extent},
              "search_config": {"radii_m": args.radii,
                                "angle_offsets_deg": args.angle_offsets_deg,
                                "max_poses": args.max_poses,
                                "clearance_m": args.clearance_m,
                                "additional_clearances_m": args.additional_clearances_m,
                                "lift_clearance_m": args.lift_clearance_m,
                                "max_gripper_penetration_m":
                                    args.max_gripper_penetration_m},
              "trials": [],
              "resource_samples": [{"stage": "pre_scene_load", "rss_mb": rss_mb(),
                                    "gpu_free_mb": gpu_free_mb()}]}
    write_json(args.output, result)
    with habitat_sim.Simulator(habitat_sim.Configuration(cfg, [AgentConfiguration()])) as sim:
        if not sim.pathfinder.load_nav_mesh(str(args.navmesh)):
            raise ValueError("Scene201 navmesh failed to load")
        robot = SpotRobot(str(args.spot_urdf), sim, limit_robo_joints=True, fixed_base=True)
        robot.reconfigure()
        robot.reset()
        open_gripper = numeric_list(robot.params.gripper_open_state)
        if len(open_gripper) != 1:
            raise ValueError("Spot IK adapter expects exactly one gripper joint")
        result["gripper_configuration"] = {
            "state": "open", "joint_positions": open_gripper}
        robot.gripper_joint_pos = open_gripper
        robot.update()
        initial = np.asarray(robot.arm_joint_pos, dtype=np.float64)
        lower, upper = robot.arm_joint_limits
        ik = FullSpotIkHelper(args.spot_urdf, initial, robot.ee_local_offset,
                              gripper_position=open_gripper[0])
        templates = sim.get_object_template_manager().load_configs(str(args.kettle_config))
        if not templates:
            raise RuntimeError("Habitat could not load the kettle object template")
        template = sim.get_object_template_manager().get_template_by_id(templates[0])
        kettle = sim.get_rigid_object_manager().add_object_by_template_handle(template.handle)
        if kettle is None:
            raise RuntimeError("Habitat could not instantiate the kettle")
        # Register the separately exported kettle collision body to the static
        # stage.  Its configured origin can intersect the counter even though
        # the ConceptGraph center is correct. Let Bullet resolve the vertical
        # rest pose without robot contact, then plan against that measured
        # offset and return the object to an upright kinematic state.
        requested_origin = [center[0], args.kettle_origin_y, center[2]]
        initial_rotation = kettle.rotation
        kettle.translation = habitat_vector(requested_origin)
        kettle.motion_type = MotionType.DYNAMIC
        kettle.linear_velocity = habitat_vector([0.0, 0.0, 0.0])
        kettle.angular_velocity = habitat_vector([0.0, 0.0, 0.0])
        settle_positions = []
        for frame in range(args.object_settle_frames):
            sim.step_physics(1.0 / 60.0)
            if frame >= max(0, args.object_settle_frames - 30):
                settle_positions.append(numeric_list(kettle.translation))
        settled_origin = numeric_list(kettle.translation)
        horizontal_drift = math.hypot(settled_origin[0] - requested_origin[0],
                                      settled_origin[2] - requested_origin[2])
        tail_vertical_motion = (max(item[1] for item in settle_positions)
                                - min(item[1] for item in settle_positions)
                                if settle_positions else float("inf"))
        vertical_offset = settled_origin[1] - requested_origin[1]
        if (horizontal_drift > args.max_registration_horizontal_drift_m
                or abs(vertical_offset) > args.max_registration_vertical_offset_m
                or tail_vertical_motion > args.max_registration_tail_motion_m):
            raise ValueError(
                "kettle-to-stage dynamic registration did not settle safely: "
                f"horizontal_drift_m={horizontal_drift:.6f}, "
                f"vertical_offset_m={vertical_offset:.6f}, "
                f"tail_vertical_motion_m={tail_vertical_motion:.6f}, "
                f"settled_origin={settled_origin}")
        registration_offset = [settled_origin[index] - requested_origin[index]
                               for index in range(3)]
        registered_origin = numeric_list(settled_origin)
        center = [center[index] + registration_offset[index]
                  for index in range(3)]
        kettle.motion_type = MotionType.KINEMATIC
        kettle.rotation = initial_rotation
        kettle.translation = habitat_vector(registered_origin)
        kettle.linear_velocity = habitat_vector([0.0, 0.0, 0.0])
        kettle.angular_velocity = habitat_vector([0.0, 0.0, 0.0])
        result["object_registration"] = {
            "method": "dynamic_vertical_rest_registration",
            "requested_origin": requested_origin,
            "registered_origin": registered_origin,
            "center_offset_xyz_m": [round(value, 6)
                                    for value in registration_offset],
            "vertical_center_offset_m": round(vertical_offset, 6),
            "horizontal_drift_during_calibration_m": round(horizontal_drift, 6),
            "tail_vertical_motion_m": round(tail_vertical_motion, 6),
            "settle_frames": args.object_settle_frames,
            "provenance": ("Measured by Bullet against the exported static stage; "
                           "not inferred from the language model."),
        }
        result["target"]["world_center"] = numeric_list(center)
        # Horizontal registration changes the navmesh queries as well as the
        # arm targets. Rebuild them from the measured physical object pose.
        queries = candidate_base_queries(center, reference_base, args.radii,
                                         args.angle_offsets_deg)
        write_json(args.output, result)
        gripper_links, grasp_link_names = spot_grasp_contact_links(robot)
        arm_links = set(int(link) for link in robot.params.arm_joints) | gripper_links
        link_names = robot_link_name_map(robot, arm_links)
        link_names.update(grasp_link_names)
        result["grasp_contact_links"] = {
            "link_ids": sorted(gripper_links),
            "link_names": {str(link): link_names.get(link, "unknown")
                           for link in sorted(gripper_links)},
            "policy": ("Open-gripper contact by these links is allowed during "
                       "the contact-ready approach; all environment and "
                       "non-gripper arm/object contacts remain blocking."),
        }
        robot_id, kettle_id = int(robot.sim_obj.object_id), int(kettle.object_id)
        seen, feasible = set(), []
        for query_info in queries:
            if len(result["trials"]) >= args.max_poses:
                break
            snapped = numeric_list(sim.pathfinder.snap_point(habitat_vector(query_info["query"])))
            if len(snapped) != 3 or not all(math.isfinite(value) for value in snapped):
                continue
            snap_offset = math.hypot(snapped[0] - query_info["query"][0],
                                    snapped[2] - query_info["query"][2])
            key = tuple(round(value, 3) for value in snapped)
            if snap_offset > args.max_snap_m or key in seen:
                continue
            seen.add(key)
            rotation = facing_rotation(snapped, center)
            robot.base_pos = habitat_vector(snapped)
            robot.base_rot = rotation
            robot.arm_joint_pos = numeric_list(initial)
            robot.gripper_joint_pos = open_gripper
            robot.update()
            initial_contacts = arm_contact_counts(sim, robot_id, kettle_id, arm_links)
            trial = {**query_info, "base_position": snapped, "base_rotation_rad": rotation,
                     "gripper_joint_positions": open_gripper,
                     "horizontal_snap_offset_m": round(snap_offset, 4),
                     "initial_arm_environment_contacts": len(initial_contacts["arm_environment"]),
                     "initial_arm_kettle_contacts": len(initial_contacts["arm_kettle"]),
                     "max_non_arm_contacts_ignored": len(initial_contacts["non_arm"])}
            if initial_contacts["arm_environment"] or initial_contacts["arm_kettle"]:
                trial["status"] = "initial_pose_collision"
                trial["initial_contact_details"] = {
                    "arm_environment": add_link_diagnostics(
                        initial_contacts["arm_environment"], link_names),
                    "arm_kettle": add_link_diagnostics(
                        initial_contacts["arm_kettle"], link_names),
                }
                result["trials"].append(trial)
                LOG.info("pose=%d rejected initial arm collision env=%d kettle=%d",
                         len(result["trials"]), len(initial_contacts["arm_environment"]),
                         len(initial_contacts["arm_kettle"]))
                continue

            base_transform = robot.base_transformation
            trial["approach_attempts"] = []
            passed = False
            for approach in clearance_approaches(
                    center, extent, snapped,
                    [args.clearance_m, *args.additional_clearances_m]):
                robot.arm_joint_pos = numeric_list(initial)
                robot.update()
                initial_ee_world = numeric_list(robot.ee_transform.translation)
                target_world = approach["world_target"]
                target_base = numeric_list(base_transform.inverted().transform_point(
                    mn.Vector3(target_world)))
                locked_arm_joint_indices = approach["locked_arm_joint_indices"]
                # Each approach starts from the same measured arm state. A
                # previous IK or segmented fallback must not bias this branch.
                ik.set_arm_state(initial)
                solution = numeric_list(ik.calc_ik(
                    np.asarray(target_base, dtype=np.float64),
                    locked_arm_joint_indices=locked_arm_joint_indices))
                limits_ok = within_limits(solution, lower, upper)
                fk_error = distance(target_base, numeric_list(ik.calc_fk(np.asarray(solution))))
                attempt = {"approach_name": approach["name"],
                           "clearance_m": approach["clearance_m"],
                           "approach_world_target": target_world,
                           "approach_base_target": target_base,
                           "orientation_strategy": approach["orientation_strategy"],
                           "locked_arm_joint_indices": locked_arm_joint_indices,
                           "joint_solution": solution,
                           "within_joint_limits": limits_ok,
                           "pybullet_fk_error_m": round(fk_error, 6)}
                if not limits_ok or fk_error > args.max_fk_error_m:
                    attempt["status"] = "ik_invalid"
                    trial["approach_attempts"].append(attempt)
                    LOG.info("pose=%d approach=%s clearance=%.2f status=ik_invalid pybullet_fk_error=%.4f",
                             len(result["trials"]) + 1, approach["name"],
                             approach["clearance_m"], fk_error)
                    continue

                path = interpolate_joint_path(initial, solution, args.max_joint_step_rad,
                                              args.minimum_steps)
                blocking = []
                max_non_arm = trial["max_non_arm_contacts_ignored"]
                for sample_index, joints in enumerate(path):
                    robot.arm_joint_pos = joints
                    robot.update()
                    contacts = arm_contact_counts(sim, robot_id, kettle_id, arm_links)
                    gripper_kettle = [contact for contact in contacts["arm_kettle"]
                                      if contact["robot_link_id"] in gripper_links]
                    non_gripper_kettle = [
                        contact for contact in contacts["arm_kettle"]
                        if contact["robot_link_id"] not in gripper_links]
                    gripper_penetration = max_contact_penetration(gripper_kettle)
                    sample_blocking = (len(contacts["arm_environment"]) +
                                       len(non_gripper_kettle) +
                                       int(gripper_penetration >
                                           args.max_gripper_penetration_m))
                    max_non_arm = max(max_non_arm, len(contacts["non_arm"]))
                    if sample_blocking:
                        blocking.append({
                            "sample_index": sample_index,
                            "arm_environment_contacts": len(contacts["arm_environment"]),
                            "non_gripper_arm_kettle_contacts": len(non_gripper_kettle),
                            "allowed_gripper_kettle_contacts": len(gripper_kettle),
                            "gripper_max_penetration_m": round(
                                gripper_penetration, 6),
                            "contact_details": {
                                "arm_environment": add_link_diagnostics(
                                    contacts["arm_environment"], link_names),
                                "arm_kettle": add_link_diagnostics(
                                    non_gripper_kettle, link_names),
                            }})
                        break

                # Measure FK at the intended solution, not at an early collision stop.
                robot.arm_joint_pos = solution
                robot.update()
                goal_ee = numeric_list(robot.ee_transform.translation)
                goal_world_error = distance(target_world, goal_ee)
                attempt.update({
                    "trajectory_sample_count": len(path),
                    "first_blocking_sample": blocking[0] if blocking else None,
                    "max_non_arm_contacts_ignored": max_non_arm,
                    "goal_ee_world": goal_ee,
                    "goal_habitat_fk_error_m": round(goal_world_error, 6)})
                passed = not blocking and goal_world_error <= args.max_fk_error_m
                attempt["trajectory_strategy"] = "direct_joint_interpolation"

                # A direct joint-space line can sweep the wrist or open gripper
                # through the table even when both endpoint configurations are
                # valid. Try a continuous seeded IK route that first lifts,
                # traverses above the target, and then descends.
                if (not passed and blocking and
                        goal_world_error <= args.max_fk_error_m):
                    segmented = {"strategy": "lift_traverse_descend", "segments": []}
                    seed_joints = numeric_list(initial)
                    segmented_ok = True
                    final_solution = None
                    final_goal_ee = None
                    for segment_index, waypoint in enumerate(clearance_waypoints(
                            initial_ee_world, target_world, args.lift_clearance_m)):
                        ik.set_arm_state(seed_joints)
                        waypoint_world = waypoint["world_target"]
                        waypoint_base = numeric_list(base_transform.inverted().transform_point(
                            mn.Vector3(waypoint_world)))
                        waypoint_solution = numeric_list(ik.calc_ik(
                            np.asarray(waypoint_base, dtype=np.float64),
                            locked_arm_joint_indices=locked_arm_joint_indices))
                        waypoint_limits_ok = within_limits(waypoint_solution, lower, upper)
                        waypoint_pb_error = distance(
                            waypoint_base,
                            numeric_list(ik.calc_fk(np.asarray(waypoint_solution))))
                        segment = {
                            "segment_index": segment_index,
                            "segment_name": waypoint["name"],
                            "world_target": waypoint_world,
                            "base_target": waypoint_base,
                            "joint_solution": waypoint_solution,
                            "within_joint_limits": waypoint_limits_ok,
                            "pybullet_fk_error_m": round(waypoint_pb_error, 6),
                        }
                        if (not waypoint_limits_ok or
                                waypoint_pb_error > args.max_fk_error_m):
                            segment["status"] = "ik_invalid"
                            segmented["segments"].append(segment)
                            segmented_ok = False
                            break

                        segment_path = interpolate_joint_path(
                            seed_joints, waypoint_solution,
                            args.max_joint_step_rad, args.minimum_steps)
                        segment_blocking = None
                        for segment_sample_index, joints in enumerate(segment_path):
                            robot.arm_joint_pos = joints
                            robot.update()
                            contacts = arm_contact_counts(
                                sim, robot_id, kettle_id, arm_links)
                            gripper_kettle = [
                                contact for contact in contacts["arm_kettle"]
                                if contact["robot_link_id"] in gripper_links]
                            non_gripper_kettle = [
                                contact for contact in contacts["arm_kettle"]
                                if contact["robot_link_id"] not in gripper_links]
                            gripper_penetration = max_contact_penetration(
                                gripper_kettle)
                            if (contacts["arm_environment"] or
                                    non_gripper_kettle or
                                    gripper_penetration >
                                    args.max_gripper_penetration_m):
                                segment_blocking = {
                                    "sample_index": segment_sample_index,
                                    "arm_environment_contacts": len(
                                        contacts["arm_environment"]),
                                    "non_gripper_arm_kettle_contacts": len(
                                        non_gripper_kettle),
                                    "allowed_gripper_kettle_contacts": len(
                                        gripper_kettle),
                                    "gripper_max_penetration_m": round(
                                        gripper_penetration, 6),
                                    "contact_details": {
                                        "arm_environment": add_link_diagnostics(
                                            contacts["arm_environment"], link_names),
                                        "arm_kettle": add_link_diagnostics(
                                            non_gripper_kettle, link_names),
                                    },
                                }
                                break
                        robot.arm_joint_pos = waypoint_solution
                        robot.update()
                        waypoint_goal_ee = numeric_list(robot.ee_transform.translation)
                        waypoint_habitat_error = distance(
                            waypoint_world, waypoint_goal_ee)
                        segment.update({
                            "trajectory_sample_count": len(segment_path),
                            "first_blocking_sample": segment_blocking,
                            "goal_ee_world": waypoint_goal_ee,
                            "goal_habitat_fk_error_m": round(
                                waypoint_habitat_error, 6),
                        })
                        segment["status"] = (
                            "feasible" if not segment_blocking and
                            waypoint_habitat_error <= args.max_fk_error_m
                            else "collision_or_goal_fk_error")
                        segmented["segments"].append(segment)
                        if segment["status"] != "feasible":
                            segmented_ok = False
                            break
                        seed_joints = waypoint_solution
                        final_solution = waypoint_solution
                        final_goal_ee = waypoint_goal_ee

                    segmented["status"] = "feasible" if segmented_ok else "failed"
                    attempt["segmented_approach"] = segmented
                    if segmented_ok and final_solution is not None:
                        passed = True
                        attempt["trajectory_strategy"] = "lift_traverse_descend"
                        attempt["joint_solution"] = final_solution
                        attempt["goal_ee_world"] = final_goal_ee
                        attempt["goal_habitat_fk_error_m"] = round(
                            distance(target_world, final_goal_ee), 6)

                attempt["status"] = ("feasible" if passed
                                     else "approach_collision_or_goal_fk_error")
                trial["approach_attempts"].append(attempt)
                LOG.info("pose=%d approach=%s clearance=%.2f status=%s goal_fk_error=%.4f",
                         len(result["trials"]) + 1, approach["name"],
                         approach["clearance_m"], attempt["status"], goal_world_error)
                if passed:
                    trial["selected_approach"] = attempt
                    break

            trial["status"] = "feasible" if passed else "no_feasible_approach"
            result["trials"].append(trial)
            LOG.info("pose=%d radius=%.2f angle=%+.0f status=%s",
                     len(result["trials"]), query_info["radius_m"],
                     query_info["angle_offset_deg"], trial["status"])
            if passed:
                feasible.append(trial)
                break
        ik.close()
        selected = feasible[0] if feasible else None
        result.update({"status": "spot_safe_base_pose_found" if selected else "no_safe_base_pose_found",
                       "selected_base_pose": selected,
                       "safe_base_pose_found": selected is not None,
                       "skill_precondition_updates": [
                           {"step_id": "st_01_01", "predicate": "navigable",
                            "status": "true" if selected else "unknown",
                            "reason": "navmesh_pose_and_arm_clearance_validated" if selected else "no_safe_sampled_pose"},
                           {"step_id": "st_01_02", "predicate": "collision_free_approach",
                            "status": "true" if selected else "false",
                            "reason": "no_environment_or_non_gripper_collision" if selected else "no_safe_sampled_pose"},
                           {"step_id": "st_01_03", "predicate": "graspable", "status": "unknown",
                            "reason": "grasp_orientation_and_closure_not_validated"}],
                       "resource_summary": {"process_peak_rss_mb": peak_rss_mb(),
                                            "system_memory": system_memory_mb(),
                                            "elapsed_s": round(time.monotonic() - started, 3)}})
    write_json(args.output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("alignment_json", "scene_graph_json", "ik_json", "stage_config", "navmesh",
                 "spot_urdf", "kettle_config", "output", "log_file"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--role-id", default="electric_kettle")
    parser.add_argument("--axis-mode", choices=["x_z_minus_y", "x_z_y", "identity"], default="x_z_minus_y")
    parser.add_argument("--radii", nargs="+", type=float, default=[0.65, 0.8, 0.95])
    parser.add_argument("--angle-offsets-deg", nargs="+", type=float, default=[0, -20, 20, -40, 40])
    parser.add_argument("--max-poses", type=int, default=15)
    parser.add_argument("--max-snap-m", type=float, default=0.25)
    parser.add_argument("--clearance-m", type=float, default=0.06)
    parser.add_argument("--additional-clearances-m", nargs="*", type=float,
                        default=[])
    parser.add_argument("--lift-clearance-m", type=float, default=0.06)
    parser.add_argument("--max-gripper-penetration-m", type=float, default=0.005)
    parser.add_argument("--kettle-origin-y", type=float, default=1.31381)
    parser.add_argument("--object-settle-frames", type=int, default=300)
    parser.add_argument("--max-registration-horizontal-drift-m", type=float,
                        default=0.15)
    parser.add_argument("--max-registration-vertical-offset-m", type=float,
                        default=0.30)
    parser.add_argument("--max-registration-tail-motion-m", type=float,
                        default=0.01)
    parser.add_argument("--max-joint-step-rad", type=float, default=0.08)
    parser.add_argument("--minimum-steps", type=int, default=20)
    parser.add_argument("--max-fk-error-m", type=float, default=0.04)
    args = parser.parse_args()
    args.log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(args.log_file, encoding="utf-8")])
    try:
        result = run(args)
        LOG.info("status=%s safe_base_pose=%s result=%s", result["status"], result["safe_base_pose_found"], args.output)
        return 0 if result["status"] == "spot_safe_base_pose_found" else 2
    except Exception as exc:
        LOG.exception("Spot base-pose search failed")
        write_json(args.output, {"schema_version": "1.0", "status": "error",
                                 "error": {"type": type(exc).__name__, "message": str(exc)},
                                 "resource_summary": {"process_peak_rss_mb": peak_rss_mb(),
                                                      "system_memory": system_memory_mb()}})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
