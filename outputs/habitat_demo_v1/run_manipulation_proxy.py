#!/usr/bin/env python3
"""ConceptGraphs-grounded Scene201 Habitat attach/carry/release proxy demo.

This does not simulate a gripper, arm trajectory, or quadruped locomotion.
"""
import argparse
import logging
import math
import sys
import subprocess
from pathlib import Path

from run_navigation_demo import (read_json, write_json, rss_mb, gpu_free_mb,
    flatten_steps, targets, cg_to_habitat, habitat_vector, select_coordinates,
    choose_start, navigate, horizontal_distance, peak_rss_mb, system_memory_mb)

LOG = logging.getLogger("habitat_manipulation_proxy")
EXPECTED_SKILLS = ["navigate_to_object", "position_arm_for_object", "grasp_object",
                   "lift_held_object", "navigate_to_object", "position_arm_for_object",
                   "place_object_on_surface"]


def contact_near_target(contact, object_id, board, extent, top, margin=0.15):
    """Check a kettle-to-stage contact against the CG sideboard top footprint.

    This is spatial evidence, not a semantic instance ID for the static mesh.
    """
    if contact.object_id_a == object_id:
        point = contact.position_on_a_in_ws
    elif contact.object_id_b == object_id:
        point = contact.position_on_b_in_ws
    else:
        return False
    return (abs(float(point[0]) - board[0]) <= float(extent[0]) / 2 + margin
            and abs(float(point[2]) - board[2]) <= float(extent[1]) / 2 + margin
            and abs(float(point[1]) - top) <= margin)


def tail_motion_m(positions):
    if not positions:
        return None
    origin = positions[0]
    return max(math.dist(origin, point) for point in positions)


def run(args):
    import habitat_sim
    from habitat_sim.agent import ActionSpec, ActuationSpec, AgentConfiguration
    from habitat_sim.physics import MotionType, PhysicsSimulationLibrary

    if not math.isfinite(args.proxy_reach_m) or not 0 < args.proxy_reach_m <= 1.5:
        raise ValueError("Proxy reach threshold must be finite and in (0, 1.5] metres")

    plan, alignment = read_json(args.skill_plan_json), read_json(args.alignment_json)
    nodes = read_json(args.scene_graph_json)
    steps = flatten_steps(plan)
    if [s["skill_id"] for s in steps] != EXPECTED_SKILLS:
        raise ValueError("This proxy only supports the exact 7-step kettle-to-sideboard skill sequence")
    if alignment.get("status") != "success" or plan.get("status") != "proposal_needs_environment_validation":
        raise ValueError("Plan or alignment not ready")
    nav_targets = targets(alignment, plan, nodes)
    if len(nav_targets) != 2:
        raise ValueError("Expected exactly two navigations")
    kettle_id, board_id = nav_targets[0]["element_id"], nav_targets[1]["element_id"]
    board = cg_to_habitat(nav_targets[1]["cg_center"], "x_z_minus_y")
    board_node = next(n for n in nodes if int(n["id"]) == board_id)
    top = float(board_node["bbox_center"][2]) + float(board_node["bbox_extent"][2]) / 2
    if steps[2]["arguments"]["target"]["element_id"] != kettle_id or \
       steps[6]["arguments"]["surface"]["element_id"] != board_id:
        raise ValueError("Skill plan references inconsistent object instances")
    if args.preflight_only:
        result = {"status": "preflight_passed", "kettle_id": kettle_id,
                  "sideboard_id": board_id, "skills": EXPECTED_SKILLS}
        write_json(args.output, result)
        return result
    for p in (args.stage_config, args.object_config, args.sideboard_config, args.navmesh):
        if not p.is_file():
            raise ValueError(f"Missing resource: {p}")
    # Habitat-Sim 0.2.3 does not export built_with_bullet from utils.settings.
    # Its binding may expose the build flag; the active simulator is checked
    # again after scene load, so an absent flag is not treated as a pass.
    bullet_build = getattr(getattr(habitat_sim, "bindings", None), "built_with_bullet", None)
    if bullet_build is False:
        raise RuntimeError("Installed Habitat-Sim has no Bullet physics; dynamic placement cannot be tested")
    pf = habitat_sim.PathFinder()
    if not pf.load_nav_mesh(str(args.navmesh)):
        raise ValueError("Navmesh could not be loaded")
    axis, snapped = select_coordinates(pf, nav_targets, "x_z_minus_y", 2.0)
    start = choose_start(pf, snapped[0]["nav_goal"], args.start_seed, args.start_selection)
    result = {"schema_version": "1.0", "status": "preflight_passed",
              "scope": "proxy attach/carry/release; not robot-arm grasp or quadruped execution",
              "input": {"plan": str(args.skill_plan_json), "alignment": str(args.alignment_json),
                        "scene_graph": str(args.scene_graph_json)},
              "coordinate_mode": axis,
              "scene_graph_ids": {"kettle": kettle_id, "sideboard": board_id},
              "object_ids": {"kettle": kettle_id, "sideboard": board_id},
              "steps": [], "resources": []}
    result["test_fault"] = args.test_fault
    result["start_sampling"] = {"seed": args.start_seed, "selection": args.start_selection,
                                "position": start,
                                "scope": "Navmesh-reachable start at least 1.5 m horizontally from first goal"}
    LOG.info("Selected navigation start: seed=%s policy=%s position=%s",
             args.start_seed, args.start_selection, start)
    if args.assume_grasp_after_approach:
        result["execution_policy"] = {
            "grasp_rule": "assume_grasp_after_validated_proximity",
            "proximity_threshold_m": args.proxy_reach_m,
            "physical_grasp_verified": False,
            "arm_ik_verified_in_this_run": False,
            "quadruped_locomotion_verified": False,
            "continuous_proxy_execution": True,
            "map_incremental_update_enabled": False,
        }
    write_json(args.output, result)
    cfg = habitat_sim.SimulatorConfiguration()
    cfg.scene_id = str(args.stage_config)
    cfg.create_renderer = False
    cfg.enable_physics = True
    agent_cfg = AgentConfiguration()
    agent_cfg.action_space = {
        "move_forward": ActionSpec("move_forward", ActuationSpec(amount=0.25)),
        "turn_left": ActionSpec("turn_left", ActuationSpec(amount=10.0)),
        "turn_right": ActionSpec("turn_right", ActuationSpec(amount=10.0)),
    }
    LOG.info("Loading split Habitat stage; rss_mb=%s gpu_free_mb=%s", rss_mb(), gpu_free_mb())
    with habitat_sim.Simulator(habitat_sim.Configuration(cfg, [agent_cfg])) as sim:
        physics_backend = sim.get_physics_simulation_library()
        LOG.info("physics_backend=%s", physics_backend)
        if physics_backend != PhysicsSimulationLibrary.Bullet:
            raise RuntimeError(f"Habitat scene has no Bullet physics backend: {physics_backend}")
        sim.pathfinder.load_nav_mesh(str(args.navmesh))
        state = sim.get_agent(0).get_state()
        state.position = habitat_vector(start)
        sim.get_agent(0).set_state(state)
        result["resources"].append({"phase": "stage_loaded", "rss_mb": rss_mb(),
                                    "peak_rss_mb": peak_rss_mb(),
                                    "system_memory": system_memory_mb(),
                                    "gpu_free_mb": gpu_free_mb()})
        mgr = sim.get_object_template_manager()
        sideboard_ids = mgr.load_configs(str(args.sideboard_config))
        if not sideboard_ids:
            raise RuntimeError("Habitat could not load sideboard object template")
        sideboard_template = mgr.get_template_by_id(sideboard_ids[0])
        sideboard_obj = sim.get_rigid_object_manager().add_object_by_template_handle(
            sideboard_template.handle)
        if sideboard_obj is None:
            raise RuntimeError("Habitat could not instantiate sideboard")
        sideboard_obj.motion_type = MotionType.KINEMATIC
        sideboard_obj.translation = habitat_vector([-5.77159, 0.00711, -1.49913])
        sim.step_physics(1.0 / 60.0)
        # In Habitat 0.2.3 a GLB rigid object's translation is COM-aligned.
        # Audit the imported top and compensate only a bounded vertical COM
        # offset, so the separately exported cabinet returns to its CG pose.
        calibration_ray = habitat_sim.geo.Ray()
        calibration_ray.origin = habitat_vector([board[0] - 0.05, top + 0.5,
                                                  board[2] - 0.05])
        calibration_ray.direction = habitat_vector([0.0, -1.0, 0.0])
        calibration_hits = sim.cast_ray(calibration_ray, max_distance=1.5).hits
        calibration_hit = calibration_hits[0] if calibration_hits else None
        if calibration_hit is None or calibration_hit.object_id != sideboard_obj.object_id:
            raise RuntimeError("Cannot identify sideboard top for COM-offset calibration")
        imported_top = float(calibration_hit.point[1])
        vertical_correction = top - imported_top
        if not 0.0 <= vertical_correction <= 0.6:
            raise RuntimeError(f"Sideboard registration offset out of safe range: {vertical_correction:.3f} m")
        sideboard_obj.translation = habitat_vector([
            -5.77159, 0.00711 + vertical_correction, -1.49913])
        sideboard_obj.motion_type = MotionType.STATIC
        sim.step_physics(1.0 / 60.0)
        result["sideboard_registration"] = {"method": "bounded_vertical_top_alignment",
            "imported_top_m": imported_top, "cg_top_m": top,
            "vertical_correction_m": vertical_correction,
            "note": "Compensates rigid-object COM offset; not an independently measured pose"}
        result["steps"].append({"skill_id": "instantiate_sideboard",
                                "status": "static_instance_created",
                                "cg_element_id": board_id,
                                "habitat_object_id": int(sideboard_obj.object_id),
                                "position": list(map(float, sideboard_obj.translation))})
        ids = mgr.load_configs(str(args.object_config))
        if not ids:
            raise RuntimeError("Habitat could not load kettle object template")
        template = mgr.get_template_by_id(ids[0])
        obj = sim.get_rigid_object_manager().add_object_by_template_handle(template.handle)
        if obj is None:
            raise RuntimeError("Habitat could not instantiate kettle")
        initial = cg_to_habitat(nav_targets[0]["cg_center"], axis)
        obj.motion_type = MotionType.KINEMATIC
        obj.translation = habitat_vector([initial[0], 1.31381, initial[2]])
        result["steps"].append({"skill_id": "instantiate_object", "status": "proxy_created",
                                "cg_element_id": kettle_id,
                                "habitat_object_id": int(obj.object_id),
                                "position": list(map(float, obj.translation))})
        write_json(args.output, result)

        first = navigate(sim, snapped[0]["nav_goal"], args.max_actions, 0.6, 25)
        result["steps"].append({"skill_id": "navigate_to_object", **first})
        if first["status"] != "arrived":
            result["status"] = "navigation_failed"
            write_json(args.output, result)
            return result
        agent_pos = sim.get_agent(0).get_state().position
        reach = horizontal_distance(agent_pos, obj.translation)
        # This is a proximity check only, not a grasp affordance or collision-free IK test.
        if reach > args.proxy_reach_m:
            result["status"] = "proxy_reach_failed"
            result["reach_m"] = reach
            write_json(args.output, result)
            return result
        result["steps"].extend([
            {"skill_id": "position_arm_for_object", "status": "proxy_only_not_executed"},
            {"skill_id": "grasp_object", "status": "proxy_attached", "reach_m": round(reach, 3)},
            {"skill_id": "lift_held_object", "status": "proxy_lifted"},
        ])
        if args.assume_grasp_after_approach:
            result["steps"][-2]["evidence_type"] = "assumed_after_navigation_and_proximity_check"
            result["steps"][-2]["physical_grasp_verified"] = False
        LOG.info("Approach passed: object_id=%s horizontal_distance_m=%.3f threshold_m=%.3f; grasp is scripted, not physically verified",
                 kettle_id, reach, args.proxy_reach_m)
        write_json(args.output, result)
        def carry():
            pos = sim.get_agent(0).get_state().position
            obj.translation = habitat_vector([float(pos[0]), 1.35, float(pos[2])])
        carry()
        second = navigate(sim, snapped[1]["nav_goal"], args.max_actions, 0.6, 25,
                          on_action=carry)
        result["steps"].append({"skill_id": "navigate_to_object", **second})
        if second["status"] != "arrived":
            result["status"] = "navigation_failed"
            write_json(args.output, result)
            return result
        proximity = horizontal_distance(sim.get_agent(0).get_state().position, board)
        if proximity > args.proxy_reach_m:
            result["status"] = "proxy_place_reach_failed"
            result["reach_m"] = proximity
            write_json(args.output, result)
            return result
        result["steps"].append({"skill_id": "position_arm_for_object",
                                "status": "proxy_only_not_executed"})
        # The graph's sideboard bbox top gives only a candidate height. It is
        # not a measured free/support surface, so final verification is mandatory.
        # A centre ray alone could miss clutter under the kettle footprint.
        # Require the centre and four near-corner rays to hit this exact
        # sideboard rigid-object ID at approximately the same top height.
        surface_trials = []
        candidates = [(dx, dz) for dx in (0.0, -0.05, 0.05)
                      for dz in (0.0, -0.05, 0.05, -0.10, 0.10)]
        chosen = None
        for dx, dz in candidates:
            samples = []
            for ox, oz in ((0.0, 0.0), (-0.07, -0.07), (-0.07, 0.07),
                           (0.07, -0.07), (0.07, 0.07)):
                ray = habitat_sim.geo.Ray()
                ray.origin = habitat_vector([board[0] + dx + ox, top + 0.5,
                                             board[2] + dz + oz])
                ray.direction = habitat_vector([0.0, -1.0, 0.0])
                hits = sim.cast_ray(ray, max_distance=1.0).hits
                first = hits[0] if hits else None
                samples.append({"object_id": int(first.object_id) if first else None,
                                "height": float(first.point[1]) if first else None})
            valid = (all(s["object_id"] == sideboard_obj.object_id for s in samples)
                     and all(abs(s["height"] - top) <= 0.15 for s in samples))
            surface_trials.append({"offset_xz": [dx, dz], "valid": valid,
                                   "samples": samples})
            if valid:
                chosen = (dx, dz, samples[0]["height"])
                break
        result["surface_selection"] = {"status": "selected" if chosen else "no_clear_surface",
                                       "candidate_trials": surface_trials}
        if chosen is None:
            result["status"] = "no_clear_sideboard_surface"
            write_json(args.output, result)
            return result
        dx, dz, measured_top = chosen
        release = [board[0] + dx, measured_top + 0.15, board[2] + dz]
        if args.test_fault == "before_place":
            result["steps"].append({"skill_id": "place_object_on_surface",
                                    "status": "injected_failure",
                                    "note": "Controlled regression fault; not observed physical failure"})
            result["status"] = "placement_test_failure"
            write_json(args.output, result)
            return result
        obj.translation = habitat_vector(release)
        obj.motion_type = MotionType.DYNAMIC
        tail_positions = []
        stage_contact_frames = 0
        candidate_support_frames = 0
        direct_sideboard_contact_frames = 0
        contact_sample = []
        direct_contact_sample = []
        board_extent_h = [float(board_node["bbox_extent"][0]),
                          float(board_node["bbox_extent"][1])]
        total_frames = max(120, round(args.settle_seconds * 60))
        tail_start = total_frames - 60
        for frame in range(total_frames):
            sim.step_physics(1.0 / 60.0)
            if frame >= tail_start:
                tail_positions.append(list(map(float, obj.translation)))
                contacts = [c for c in sim.get_physics_contact_points()
                            if c.object_id_a == obj.object_id or c.object_id_b == obj.object_id]
                if contacts:
                    stage_contact_frames += 1
                matched = [c for c in contacts if contact_near_target(
                    c, obj.object_id, board, board_extent_h, measured_top)]
                if matched:
                    candidate_support_frames += 1
                    if len(contact_sample) < 5:
                        c = matched[0]
                        p = (c.position_on_a_in_ws if c.object_id_a == obj.object_id
                             else c.position_on_b_in_ws)
                        contact_sample.append({"frame": frame, "object_ids":
                            [int(c.object_id_a), int(c.object_id_b)],
                            "point_on_kettle": list(map(float, p))})
                direct = [c for c in matched if
                          {int(c.object_id_a), int(c.object_id_b)} ==
                          {int(obj.object_id), int(sideboard_obj.object_id)}]
                if direct:
                    direct_sideboard_contact_frames += 1
                    if len(direct_contact_sample) < 5:
                        c = direct[0]
                        p = (c.position_on_a_in_ws if c.object_id_a == obj.object_id
                             else c.position_on_b_in_ws)
                        direct_contact_sample.append({"frame": frame,
                            "object_ids": [int(c.object_id_a), int(c.object_id_b)],
                            "point_on_kettle": list(map(float, p))})
        final = list(map(float, obj.translation))
        # A geometric proxy, not contact/semantic support proof. Falling through
        # or off furniture must fail. Report all measurements for manual review.
        horizontal_error = horizontal_distance(final, board)
        height_error = abs(final[1] - (measured_top + 0.083))
        placed = horizontal_error <= 0.35 and height_error <= 0.20
        motion = tail_motion_m(tail_positions)
        stable = motion is not None and motion <= 0.02
        supported_candidate = (placed and stable and
                               direct_sideboard_contact_frames >= 20)
        verdict = ("proxy_instance_support_passed" if supported_candidate else
                   "proxy_instance_support_failed")
        result["steps"].append({"skill_id": "place_object_on_surface",
                                "status": verdict,
                                "release_position": release, "settled_position": final,
                                "sideboard_top_from_graph_m": top,
                                "sideboard_top_measured_by_ray_m": measured_top,
                                "horizontal_error_m": round(horizontal_error, 3),
                                "height_error_m": round(height_error, 3),
                                "geometry_passed": placed, "stable_last_second": stable,
                                "simulated_settle_seconds": total_frames / 60,
                                "max_motion_last_second_m": round(motion, 4),
                                "contact_frames_last_second": stage_contact_frames,
                                "candidate_sideboard_top_contact_frames_last_second":
                                    candidate_support_frames,
                                "direct_sideboard_contact_frames_last_second":
                                    direct_sideboard_contact_frames,
                                "sideboard_habitat_object_id": int(sideboard_obj.object_id),
                                "contact_samples": contact_sample,
                                "direct_sideboard_contact_samples": direct_contact_sample,
                                "note": "Contact pair confirms exported sideboard rigid instance; no robot gripper or arm execution"})
        result["resources"].append({"phase": "after_execution", "rss_mb": rss_mb(),
                                    "peak_rss_mb": peak_rss_mb(),
                                    "system_memory": system_memory_mb(),
                                    "gpu_free_mb": gpu_free_mb()})
        result["resource_summary"] = {
            "process_peak_rss_mb": peak_rss_mb(),
            "system_memory_after_execution": system_memory_mb(),
            "measurement": "Linux /proc/self/status VmHWM",
        }
        result["status"] = verdict
    write_json(args.output, result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("alignment_json", "skill_plan_json", "scene_graph_json", "stage_config",
                "object_config", "sideboard_config", "navmesh", "output", "log_file"):
        p.add_argument("--" + key.replace("_", "-"), type=Path, required=True)
    p.add_argument("--max-actions", type=int, default=500)
    p.add_argument("--proxy-reach-m", type=float, default=1.5)
    p.add_argument("--settle-seconds", type=float, default=10.0)
    p.add_argument("--preflight-only", action="store_true")
    p.add_argument("--assume-grasp-after-approach", action="store_true",
                   help="Export a task-level episode with explicit assumed-grasp labels")
    p.add_argument("--test-fault", choices=("none", "before_place"), default="none",
                   help="Controlled regression fault injection; never use for positive training data")
    p.add_argument("--start-seed", type=int, default=201)
    p.add_argument("--start-selection", choices=("closest", "first"), default="closest")
    p.add_argument("--scene-memory-output-dir", type=Path,
                   help="Optional isolated scene-memory snapshot after verified placement; never changes source scene")
    a = p.parse_args()
    a.log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(a.log_file, encoding="utf-8")])
    try:
        result = run(a)
        if a.scene_memory_output_dir and result["status"] == "proxy_instance_support_passed":
            memory_script = Path(__file__).resolve().parent.parent / "scene_memory_v1/update_memory.py"
            subprocess.run([sys.executable, str(memory_script),
                "--scene-dir", str(a.scene_graph_json.parent), "--proxy-json", str(a.output),
                "--output-dir", str(a.scene_memory_output_dir)], check=True)
            LOG.info("Updated scene memory: %s", a.scene_memory_output_dir)
        if a.assume_grasp_after_approach and not a.preflight_only:
            from export_proxy_training_episode import build_episode
            episode_path = a.output.parent / "task_level_episode.json"
            write_json(episode_path, build_episode(result, read_json(a.skill_plan_json)))
            LOG.info("Task-level training episode: %s", episode_path)
        LOG.info("status=%s result=%s", result["status"], a.output)
        return 0 if result["status"] in ("preflight_passed", "proxy_instance_support_passed") else 2
    except Exception as exc:
        LOG.exception("Habitat manipulation proxy failed")
        write_json(a.output, {"status": "error", "error": {"type": type(exc).__name__,
                    "message": str(exc)}})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
