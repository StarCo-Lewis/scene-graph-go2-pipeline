#!/usr/bin/env python3
"""Instantiate Habitat-Lab 0.2.3 Spot and exercise one arm joint and gripper."""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path

from run_navigation_demo import gpu_free_mb, habitat_vector, peak_rss_mb, rss_mb, system_memory_mb, write_json

LOG = logging.getLogger("spot_arm_smoke")


def numeric_list(values):
    return [float(value) for value in values]


def choose_safe_joint_target(current, lower, upper, delta=0.12):
    """Return a bounded one-joint target and the changed joint index."""
    current = numeric_list(current)
    lower = numeric_list(lower)
    upper = numeric_list(upper)
    if not current or not (len(current) == len(lower) == len(upper)):
        raise ValueError("arm joint vectors must be non-empty and have equal length")
    clearance = 0.03
    for index, value in enumerate(current):
        positive = upper[index] - clearance - value
        negative = value - (lower[index] + clearance)
        if positive >= delta:
            target = current.copy()
            target[index] = value + delta
            return target, index
        if negative >= delta:
            target = current.copy()
            target[index] = value - delta
            return target, index
    raise ValueError("no arm joint has enough limit margin for the smoke-test motion")


def max_abs_delta(a, b):
    if len(a) != len(b):
        raise ValueError("joint vectors have different lengths")
    return max((abs(float(x) - float(y)) for x, y in zip(a, b)), default=0.0)


def vector3(value):
    return [float(value[0]), float(value[1]), float(value[2])]


def step_robot(sim, robot, frames, sample_every, samples, stage):
    for frame in range(frames):
        sim.step_physics(1.0 / 60.0)
        update = getattr(robot, "update", None)
        if callable(update):
            update()
        if frame % sample_every == 0 or frame == frames - 1:
            samples.append({"stage": stage, "frame": frame + 1, "rss_mb": rss_mb()})
            LOG.info("physics progress stage=%s frame=%d/%d rss_mb=%s",
                     stage, frame + 1, frames, rss_mb())


def resolve_start(pathfinder, requested):
    query = habitat_vector(requested)
    snapped = numeric_list(pathfinder.snap_point(query))
    if len(snapped) != 3 or not all(math.isfinite(x) for x in snapped):
        snapped = numeric_list(pathfinder.get_random_navigable_point())
    if len(snapped) != 3 or not all(math.isfinite(x) for x in snapped):
        raise ValueError("could not obtain a finite navigable Spot base position")
    return snapped


def run(args):
    if not args.stage_config.is_file() or not args.navmesh.is_file() or not args.spot_urdf.is_file():
        raise ValueError("stage config, navmesh, or Spot URDF is missing")

    import habitat_sim
    from habitat.robots.spot_robot import SpotRobot
    from habitat_sim.agent import AgentConfiguration

    cfg = habitat_sim.SimulatorConfiguration()
    cfg.scene_id = str(args.stage_config)
    cfg.create_renderer = False
    cfg.enable_physics = True
    agent_cfg = AgentConfiguration()
    samples = [{"stage": "pre_scene_load", "rss_mb": rss_mb(),
                "gpu_free_mb": gpu_free_mb()}]
    result = {
        "schema_version": "1.0",
        "status": "starting",
        "scope": "Headless fixed-base Spot arm and gripper interface smoke test; no IK, dynamic stability, grasp, carry, or placement claim.",
        "inputs": {"stage_config": str(args.stage_config), "navmesh": str(args.navmesh),
                   "spot_urdf": str(args.spot_urdf)},
        "resource_samples": samples,
    }
    write_json(args.output, result)
    LOG.info("loading Scene201 with Bullet physics; rss_mb=%s", rss_mb())
    started = time.monotonic()
    with habitat_sim.Simulator(habitat_sim.Configuration(cfg, [agent_cfg])) as sim:
        if not sim.pathfinder.load_nav_mesh(str(args.navmesh)):
            raise ValueError("Scene201 navmesh failed to load")
        start = resolve_start(sim.pathfinder, args.start)
        samples.append({"stage": "after_scene_load", "rss_mb": rss_mb(),
                        "gpu_free_mb": gpu_free_mb()})
        LOG.info("instantiating fixed-base Spot at %s", start)
        robot = SpotRobot(str(args.spot_urdf), sim, limit_robo_joints=True, fixed_base=True)
        robot.reconfigure()
        robot.reset()
        robot.base_pos = habitat_vector(start)
        robot.base_rot = float(args.base_rotation)
        update = getattr(robot, "update", None)
        if callable(update):
            update()
        samples.append({"stage": "after_spot_load", "rss_mb": rss_mb(),
                        "gpu_free_mb": gpu_free_mb()})

        initial_arm = numeric_list(robot.arm_joint_pos)
        lower, upper = robot.arm_joint_limits
        target_arm, changed_joint = choose_safe_joint_target(initial_arm, lower, upper, args.arm_delta)
        initial_ee = vector3(robot.ee_transform.translation)
        LOG.info("arm joint test index=%d initial=%.4f target=%.4f",
                 changed_joint, initial_arm[changed_joint], target_arm[changed_joint])
        if args.control_mode == "kinematic":
            robot.arm_joint_pos = target_arm
            if callable(update):
                update()
        else:
            robot.arm_motor_pos = target_arm
            step_robot(sim, robot, args.settle_frames, args.sample_every, samples, "arm_motion")
        final_arm = numeric_list(robot.arm_joint_pos)
        final_ee = vector3(robot.ee_transform.translation)
        arm_delta = max_abs_delta(initial_arm, final_arm)

        if args.control_mode == "kinematic":
            robot.gripper_joint_pos = numeric_list(robot.params.gripper_open_state)
            if callable(update):
                update()
        else:
            robot.open_gripper()
            step_robot(sim, robot, args.settle_frames, args.sample_every, samples, "gripper_open")
        open_pos = numeric_list(robot.gripper_joint_pos)
        if args.control_mode == "kinematic":
            robot.gripper_joint_pos = numeric_list(robot.params.gripper_closed_state)
            if callable(update):
                update()
        else:
            robot.close_gripper()
            step_robot(sim, robot, args.settle_frames, args.sample_every, samples, "gripper_close")
        closed_pos = numeric_list(robot.gripper_joint_pos)
        gripper_delta = max_abs_delta(open_pos, closed_pos)

        arm_passed = arm_delta >= args.min_arm_motion
        gripper_passed = gripper_delta >= args.min_gripper_motion
        result.update({
            "status": "spot_arm_smoke_passed" if arm_passed and gripper_passed
                      else "spot_arm_smoke_failed",
            "control_mode": args.control_mode,
            "spot": {"base_position": start, "fixed_base": True,
                     "articulated_object_id": int(robot.sim_obj.object_id)},
            "arm_test": {"changed_joint_index": changed_joint,
                         "command_delta_rad": args.arm_delta,
                         "initial_joint_positions": initial_arm,
                         "target_joint_positions": target_arm,
                         "final_joint_positions": final_arm,
                         "observed_max_joint_delta_rad": round(arm_delta, 6),
                         "minimum_required_delta_rad": args.min_arm_motion,
                         "initial_ee_translation": initial_ee,
                         "final_ee_translation": final_ee,
                         "passed": arm_passed},
            "gripper_test": {"open_joint_positions": open_pos,
                             "closed_joint_positions": closed_pos,
                             "observed_max_delta_rad": round(gripper_delta, 6),
                             "minimum_required_delta_rad": args.min_gripper_motion,
                             "passed": gripper_passed},
            "resource_summary": {"process_peak_rss_mb": peak_rss_mb(),
                                 "system_memory": system_memory_mb(),
                                 "elapsed_s": round(time.monotonic() - started, 3)},
        })
    write_json(args.output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-config", type=Path, required=True)
    parser.add_argument("--navmesh", type=Path, required=True)
    parser.add_argument("--spot-urdf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--log-file", type=Path, required=True)
    parser.add_argument("--start", type=float, nargs=3, default=[-3.03, 0.1, -7.65])
    parser.add_argument("--base-rotation", type=float, default=0.0)
    parser.add_argument("--arm-delta", type=float, default=0.12)
    parser.add_argument("--min-arm-motion", type=float, default=0.02)
    parser.add_argument("--min-gripper-motion", type=float, default=0.1)
    parser.add_argument("--settle-frames", type=int, default=120)
    parser.add_argument("--sample-every", type=int, default=30)
    parser.add_argument("--control-mode", choices=["kinematic", "physics"], default="kinematic",
                        help="Kinematic is the safe Scene201 default; physics is an explicit diagnostic.")
    args = parser.parse_args()
    args.log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(args.log_file, encoding="utf-8")])
    try:
        result = run(args)
        LOG.info("status=%s result=%s", result["status"], args.output)
        return 0 if result["status"] == "spot_arm_smoke_passed" else 2
    except Exception as exc:
        LOG.exception("Spot arm smoke test failed")
        write_json(args.output, {"schema_version": "1.0", "status": "error",
                                 "scope": "No manipulation success claim.",
                                 "error": {"type": type(exc).__name__, "message": str(exc)},
                                 "resource_summary": {"process_peak_rss_mb": peak_rss_mb(),
                                                      "system_memory": system_memory_mb()}})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
