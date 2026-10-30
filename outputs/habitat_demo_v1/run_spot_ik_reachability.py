#!/usr/bin/env python3
"""Check whether Spot IK can reach a grounded ConceptGraph object target."""
from __future__ import annotations

import argparse
import json
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

LOG = logging.getLogger("spot_ik_reachability")


def numeric_list(values):
    return [float(value) for value in values]


def grounded_node(alignment, nodes, role_id):
    binding = alignment.get("context_bindings", {}).get(role_id, {})
    ids = binding.get("element_ids", [])
    if alignment.get("status") != "success" or len(ids) != 1:
        raise ValueError(f"role {role_id} must have exactly one successful grounded element")
    wanted = int(ids[0])
    matches = [node for node in nodes if int(node.get("id", -1)) == wanted]
    if len(matches) != 1 or matches[0].get("bbox_center") is None:
        raise ValueError(f"grounded node {wanted} is missing or has no bbox_center")
    return matches[0]


def approach_candidates(center, extent, clearance=0.06):
    """Create deterministic center and axis-aligned surface approach points."""
    center = numeric_list(center)
    extent = numeric_list(extent)
    if len(center) != 3 or len(extent) != 3 or not all(x > 0 for x in extent):
        raise ValueError("object center and positive 3D extent are required")
    half = [value / 2.0 for value in extent]
    offsets = [
        ("center", [0.0, 0.0, 0.0]),
        ("above", [0.0, half[1] + clearance, 0.0]),
        ("positive_x", [half[0] + clearance, 0.0, 0.0]),
        ("negative_x", [-(half[0] + clearance), 0.0, 0.0]),
        ("positive_z", [0.0, 0.0, half[2] + clearance]),
        ("negative_z", [0.0, 0.0, -(half[2] + clearance)]),
    ]
    return [{"name": name, "world_target": [center[i] + offset[i] for i in range(3)]}
            for name, offset in offsets]


def within_limits(joints, lower, upper, tolerance=1e-4):
    if not (len(joints) == len(lower) == len(upper)):
        return False
    return all(math.isfinite(float(value)) and
               float(lo) - tolerance <= float(value) <= float(hi) + tolerance
               for value, lo, hi in zip(joints, lower, upper))


def distance(a, b):
    if len(a) != len(b):
        raise ValueError("vectors have different lengths")
    return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))


def select_spot_arm_joints(joints):
    """Select seven arm joints and the gripper/EE link from a full Spot URDF."""
    arm_chain = [joint for joint in joints
                 if joint["movable"] and joint["link_name"].startswith("arm0.link_")]
    arm_chain.sort(key=lambda item: item["index"])
    if len(arm_chain) != 8:
        raise ValueError(f"expected 8 movable arm/gripper links in full Spot URDF, found {len(arm_chain)}")
    return arm_chain[:7], arm_chain[7]


class FullSpotIkHelper:
    """Position IK adapter for the full Spot URDF, with all non-arm DOFs locked."""

    def __init__(self, urdf_path, arm_start, ee_local_offset, gripper_position=0.0):
        import pybullet as p
        import numpy as np

        self.p = p
        self.np = np
        self.client_id = p.connect(p.DIRECT)
        self.robot_id = p.loadURDF(str(urdf_path), basePosition=[0, 0, 0], useFixedBase=True,
                                   flags=p.URDF_USE_INERTIA_FROM_FILE,
                                   physicsClientId=self.client_id)
        joints = []
        for index in range(p.getNumJoints(self.robot_id, physicsClientId=self.client_id)):
            info = p.getJointInfo(self.robot_id, index, physicsClientId=self.client_id)
            joints.append({"index": index, "joint_name": info[1].decode("utf-8"),
                           "joint_type": int(info[2]), "q_index": int(info[3]),
                           "lower": float(info[8]), "upper": float(info[9]),
                           "link_name": info[12].decode("utf-8"),
                           "movable": int(info[2]) != int(p.JOINT_FIXED)})
        self.joints = joints
        self.arm_joints, self.ee_joint = select_spot_arm_joints(joints)
        self.ee_local_offset = numeric_list(ee_local_offset)
        if len(self.ee_local_offset) != 3:
            raise ValueError("end-effector local offset must contain three values")
        self.movable = [joint for joint in joints if joint["movable"]]
        self.movable.sort(key=lambda item: item["q_index"])
        if len(arm_start) != 7:
            raise ValueError("Spot arm start state must contain seven joints")
        self.state = {joint["index"]: 0.0 for joint in self.movable}
        for joint, value in zip(self.arm_joints, arm_start):
            self.state[joint["index"]] = float(value)
        self.state[self.ee_joint["index"]] = float(gripper_position)
        self.gripper_position = float(gripper_position)
        self.set_arm_state(arm_start)

    def describe(self):
        return {"arm_joints": self.arm_joints, "ee_link": self.ee_joint,
                "ee_local_offset": self.ee_local_offset,
                "locked_gripper_position": self.gripper_position,
                "movable_joint_count": len(self.movable)}

    def set_arm_state(self, arm_state):
        for joint, value in zip(self.arm_joints, arm_state):
            self.state[joint["index"]] = float(value)
        for joint in self.movable:
            self.p.resetJointState(self.robot_id, joint["index"], self.state[joint["index"]],
                                   physicsClientId=self.client_id)

    def _link_state(self, arm_state):
        self.set_arm_state(arm_state)
        return self.p.getLinkState(self.robot_id, self.ee_joint["index"],
                                   computeForwardKinematics=1, physicsClientId=self.client_id)

    def calc_fk(self, arm_state):
        state = self._link_state(arm_state)
        ee_position, _ = self.p.multiplyTransforms(
            state[4], state[5], self.ee_local_offset, [0.0, 0.0, 0.0, 1.0])
        return self.np.asarray(ee_position, dtype=self.np.float64)

    def calc_link_orientation(self, arm_state):
        """Return the PyBullet end-effector link-frame quaternion (xyzw)."""
        state = self._link_state(arm_state)
        return self.np.asarray(state[5], dtype=self.np.float64)

    def _calc_position_ik_with_locked_arm_joints(
            self, target, locked_arm_joint_indices, max_iterations=200):
        """Solve position IK while keeping selected arm joints exactly fixed.

        PyBullet's null-space IK does not reliably lock a joint when its lower
        and upper limits are made almost equal.  On the full Spot model that
        approach can return multi-turn, out-of-limit arm states.  This local
        damped-least-squares solver optimizes only the unlocked arm joints, so
        the wrist sweep used by the grasp search is a real constraint.
        """
        target = self.np.asarray(target, dtype=self.np.float64)
        locked = {int(index) for index in locked_arm_joint_indices}
        if any(index < 0 or index >= len(self.arm_joints) for index in locked):
            raise ValueError("locked arm joint index is outside the seven-joint arm")
        active = [index for index in range(len(self.arm_joints))
                  if index not in locked]
        if not active:
            return self.np.asarray(
                [self.state[joint["index"]] for joint in self.arm_joints],
                dtype=self.np.float64)

        arm_state = self.np.asarray(
            [self.state[joint["index"]] for joint in self.arm_joints],
            dtype=self.np.float64)
        locked_values = arm_state.copy()
        lower = self.np.empty(len(self.arm_joints), dtype=self.np.float64)
        upper = self.np.empty(len(self.arm_joints), dtype=self.np.float64)
        for index, joint in enumerate(self.arm_joints):
            lo, hi = float(joint["lower"]), float(joint["upper"])
            if hi < lo:
                lo, hi = -2.0 * math.pi, 2.0 * math.pi
            lower[index], upper[index] = lo, hi

        finite_difference = 1e-4
        damping = 2e-3
        max_step_norm = 0.12
        tolerance = 5e-5
        best_state = arm_state.copy()
        best_error = float(self.np.linalg.norm(target - self.calc_fk(arm_state)))

        for _ in range(max_iterations):
            current = self.calc_fk(arm_state)
            error = target - current
            error_norm = float(self.np.linalg.norm(error))
            if error_norm < best_error:
                best_error = error_norm
                best_state = arm_state.copy()
            if error_norm <= tolerance:
                break

            jacobian = self.np.zeros((3, len(active)), dtype=self.np.float64)
            for column, joint_index in enumerate(active):
                plus = arm_state.copy()
                minus = arm_state.copy()
                plus[joint_index] = min(upper[joint_index],
                                        plus[joint_index] + finite_difference)
                minus[joint_index] = max(lower[joint_index],
                                         minus[joint_index] - finite_difference)
                span = plus[joint_index] - minus[joint_index]
                if span > 0.0:
                    jacobian[:, column] = (
                        self.calc_fk(plus) - self.calc_fk(minus)) / span

            system = (jacobian @ jacobian.T
                      + (damping ** 2) * self.np.eye(3, dtype=self.np.float64))
            try:
                delta = jacobian.T @ self.np.linalg.solve(system, error)
            except self.np.linalg.LinAlgError:
                delta = self.np.linalg.pinv(jacobian) @ error
            delta_norm = float(self.np.linalg.norm(delta))
            if delta_norm > max_step_norm:
                delta *= max_step_norm / delta_norm

            improved = False
            for scale in (1.0, 0.5, 0.25, 0.125, 0.0625):
                candidate = arm_state.copy()
                candidate[active] += scale * delta
                candidate = self.np.minimum(self.np.maximum(candidate, lower), upper)
                for joint_index in locked:
                    candidate[joint_index] = locked_values[joint_index]
                candidate_error = float(
                    self.np.linalg.norm(target - self.calc_fk(candidate)))
                if candidate_error + 1e-9 < error_norm:
                    arm_state = candidate
                    improved = True
                    break
            if not improved:
                break

        final_error = float(self.np.linalg.norm(target - self.calc_fk(arm_state)))
        if final_error < best_error:
            best_state = arm_state.copy()
        self.set_arm_state(best_state)
        return best_state

    def calc_ik(self, target, target_orientation=None,
                locked_arm_joint_indices=None):
        lock_epsilon = 1e-6
        locked_arm_joint_indices = set(locked_arm_joint_indices or [])
        if target_orientation is None and locked_arm_joint_indices:
            return self._calc_position_ik_with_locked_arm_joints(
                target, locked_arm_joint_indices)
        lower, upper, ranges, rest = [], [], [], []
        arm_ids = {joint["index"] for joint in self.arm_joints}
        locked_arm_ids = {
            self.arm_joints[index]["index"]
            for index in locked_arm_joint_indices
            if 0 <= int(index) < len(self.arm_joints)
        }
        for joint in self.movable:
            value = self.state[joint["index"]]
            if joint["index"] in arm_ids:
                if joint["index"] in locked_arm_ids:
                    lo, hi = value - lock_epsilon, value + lock_epsilon
                else:
                    lo, hi = joint["lower"], joint["upper"]
                    if hi < lo:
                        lo, hi = -2.0 * math.pi, 2.0 * math.pi
            else:
                lo, hi = value - lock_epsilon, value + lock_epsilon
            lower.append(lo)
            upper.append(hi)
            ranges.append(max(hi - lo, 2.0 * lock_epsilon))
            rest.append(value)
        target = self.np.asarray(target, dtype=self.np.float64)
        if target_orientation is not None:
            target_orientation = self.np.asarray(
                target_orientation, dtype=self.np.float64)
            if target_orientation.shape != (4,):
                raise ValueError("target orientation must be an xyzw quaternion")
            # The requested point is Habitat's EE offset from the final link.
            # With orientation fixed, rotate that offset and solve for the
            # underlying PyBullet link-frame origin.
            rotated_offset, _ = self.p.multiplyTransforms(
                [0.0, 0.0, 0.0], numeric_list(target_orientation),
                self.ee_local_offset,
                [0.0, 0.0, 0.0, 1.0])
            link_target = target - self.np.asarray(
                rotated_offset, dtype=self.np.float64)
            solution = self.p.calculateInverseKinematics(
                self.robot_id, self.ee_joint["index"], numeric_list(link_target),
                targetOrientation=numeric_list(target_orientation),
                lowerLimits=lower, upperLimits=upper, jointRanges=ranges,
                restPoses=rest, maxNumIterations=300, residualThreshold=1e-6,
                physicsClientId=self.client_id)
            by_index = {joint["index"]: float(value)
                        for joint, value in zip(self.movable, solution)}
            arm_solution = self.np.asarray(
                [by_index[joint["index"]] for joint in self.arm_joints],
                dtype=self.np.float64)
            self.set_arm_state(arm_solution)
            return arm_solution
        # PyBullet targets the link origin, while Habitat defines the EE at a
        # local offset on that link. Iteratively compensate using link rotation.
        link_target = target.copy()
        arm_solution = None
        for _ in range(8):
            solution = self.p.calculateInverseKinematics(
                self.robot_id, self.ee_joint["index"], numeric_list(link_target),
                lowerLimits=lower, upperLimits=upper, jointRanges=ranges, restPoses=rest,
                maxNumIterations=200, residualThreshold=1e-6, physicsClientId=self.client_id)
            by_index = {joint["index"]: float(value)
                        for joint, value in zip(self.movable, solution)}
            arm_solution = self.np.asarray(
                [by_index[joint["index"]] for joint in self.arm_joints], dtype=self.np.float64)
            state = self._link_state(arm_solution)
            rotated_offset, _ = self.p.multiplyTransforms(
                [0.0, 0.0, 0.0], state[5], self.ee_local_offset, [0.0, 0.0, 0.0, 1.0])
            link_target = target - self.np.asarray(rotated_offset, dtype=self.np.float64)
        return arm_solution

    def close(self):
        if self.client_id >= 0:
            self.p.disconnect(physicsClientId=self.client_id)
            self.client_id = -1


def resolve_start(pathfinder, requested):
    snapped = numeric_list(pathfinder.snap_point(habitat_vector(requested)))
    if len(snapped) != 3 or not all(math.isfinite(x) for x in snapped):
        raise ValueError("requested Spot pose cannot be snapped to the Scene201 navmesh")
    return snapped


def run(args):
    for path in (args.alignment_json, args.scene_graph_json, args.stage_config,
                 args.navmesh, args.spot_urdf):
        if not path.is_file():
            raise ValueError(f"required input is missing: {path}")
    alignment = read_json(args.alignment_json)
    nodes = read_json(args.scene_graph_json)
    node = grounded_node(alignment, nodes, args.role_id)
    world_center = cg_to_habitat(node["bbox_center"], args.axis_mode)
    # Extents change axis order but never sign.
    raw_extent = numeric_list(node.get("bbox_extent", []))
    if len(raw_extent) != 3:
        raise ValueError(f"grounded node {node['id']} has no valid bbox_extent")
    world_extent = [raw_extent[0], raw_extent[2], raw_extent[1]] if args.axis_mode != "identity" else raw_extent

    import magnum as mn
    import numpy as np
    import habitat_sim
    from habitat.robots.spot_robot import SpotRobot
    from habitat_sim.agent import AgentConfiguration

    cfg = habitat_sim.SimulatorConfiguration()
    cfg.scene_id = str(args.stage_config)
    cfg.create_renderer = False
    cfg.enable_physics = True
    result = {
        "schema_version": "1.0",
        "status": "starting",
        "scope": "Kinematic position-only IK feasibility; no collision-free path, grasp orientation, contact, or dynamic execution claim.",
        "target": {"role_id": args.role_id, "element_id": int(node["id"]),
                   "object_tag": node.get("object_tag"), "cg_center": node["bbox_center"],
                   "habitat_world_center": world_center, "habitat_world_extent": world_extent},
        "inputs": {"alignment": str(args.alignment_json), "scene_graph": str(args.scene_graph_json),
                   "stage_config": str(args.stage_config), "navmesh": str(args.navmesh),
                   "spot_urdf": str(args.spot_urdf)},
        "resource_samples": [{"stage": "pre_scene_load", "rss_mb": rss_mb(),
                              "gpu_free_mb": gpu_free_mb()}],
    }
    write_json(args.output, result)
    started = time.monotonic()
    with habitat_sim.Simulator(habitat_sim.Configuration(cfg, [AgentConfiguration()])) as sim:
        if not sim.pathfinder.load_nav_mesh(str(args.navmesh)):
            raise ValueError("Scene201 navmesh failed to load")
        base_position = resolve_start(sim.pathfinder, args.start)
        robot = SpotRobot(str(args.spot_urdf), sim, limit_robo_joints=True, fixed_base=True)
        robot.reconfigure()
        robot.reset()
        robot.base_pos = habitat_vector(base_position)
        robot.base_rot = float(args.base_rotation)
        open_gripper = numeric_list(robot.params.gripper_open_state)
        if len(open_gripper) != 1:
            raise ValueError("Spot IK adapter expects exactly one gripper joint")
        robot.gripper_joint_pos = open_gripper
        robot.update()
        initial_joints = np.asarray(robot.arm_joint_pos, dtype=np.float64)
        lower, upper = robot.arm_joint_limits
        ik = FullSpotIkHelper(args.spot_urdf, initial_joints, robot.ee_local_offset,
                              gripper_position=open_gripper[0])
        base_transform = robot.base_transformation

        habitat_initial_world = numeric_list(robot.ee_transform.translation)
        habitat_initial_base = numeric_list(
            base_transform.inverted().transform_point(mn.Vector3(habitat_initial_world)))
        pybullet_initial_base = numeric_list(ik.calc_fk(initial_joints))
        frame_error = distance(habitat_initial_base, pybullet_initial_base)
        frame_ok = frame_error <= args.max_frame_error_m
        result["frame_calibration"] = {
            "habitat_initial_ee_in_base": habitat_initial_base,
            "pybullet_initial_fk_in_base": pybullet_initial_base,
            "error_m": round(frame_error, 6),
            "maximum_allowed_error_m": args.max_frame_error_m,
            "passed": frame_ok,
        }
        result["pybullet_joint_mapping"] = ik.describe()
        result["spot_pose"] = {"base_position": base_position,
                               "base_rotation_rad": args.base_rotation,
                               "initial_arm_joints": numeric_list(initial_joints),
                               "gripper_joint_positions": open_gripper}
        result["resource_samples"].append({"stage": "after_spot_and_ik_load",
                                           "rss_mb": rss_mb(), "gpu_free_mb": gpu_free_mb()})
        candidates = []
        if frame_ok:
            for candidate in approach_candidates(world_center, world_extent, args.clearance_m):
                world_target = candidate["world_target"]
                base_target = numeric_list(base_transform.inverted().transform_point(mn.Vector3(world_target)))
                solution = numeric_list(ik.calc_ik(np.asarray(base_target, dtype=np.float64)))
                limits_ok = within_limits(solution, lower, upper)
                fk_base = numeric_list(ik.calc_fk(np.asarray(solution, dtype=np.float64)))
                fk_error = distance(base_target, fk_base)
                robot.arm_joint_pos = solution
                robot.update()
                habitat_fk_world = numeric_list(robot.ee_transform.translation)
                habitat_error = distance(world_target, habitat_fk_world)
                passed = limits_ok and fk_error <= args.max_fk_error_m and habitat_error <= args.max_fk_error_m
                candidates.append({**candidate, "base_target": base_target,
                                   "joint_solution": solution, "within_joint_limits": limits_ok,
                                   "pybullet_fk_base": fk_base,
                                   "pybullet_fk_error_m": round(fk_error, 6),
                                   "habitat_fk_world": habitat_fk_world,
                                   "habitat_fk_error_m": round(habitat_error, 6),
                                   "passed": passed})
                LOG.info("candidate=%s limits=%s pb_error=%.4f habitat_error=%.4f passed=%s",
                         candidate["name"], limits_ok, fk_error, habitat_error, passed)
        robot.arm_joint_pos = numeric_list(initial_joints)
        passed_candidates = [candidate for candidate in candidates if candidate["passed"]]
        result["candidates"] = candidates
        result["selected_candidate"] = (min(passed_candidates,
                                               key=lambda item: item["habitat_fk_error_m"])
                                          if passed_candidates else None)
        result["reachable"] = bool(passed_candidates)
        result["status"] = ("spot_ik_reachability_passed" if passed_candidates else
                            "ik_frame_calibration_failed" if not frame_ok else
                            "ik_target_unreachable_at_test_pose")
        result["skill_precondition_update"] = {
            "step_id": "st_01_02",
            "predicate": "reachable",
            "status": "true" if passed_candidates else "false" if frame_ok else "unknown",
            "reason": "position_only_ik_validated" if passed_candidates else
                      "no_valid_ik_candidate_at_test_pose" if frame_ok else
                      "ik_coordinate_frames_not_calibrated",
        }
        result["resource_summary"] = {"process_peak_rss_mb": peak_rss_mb(),
                                      "system_memory": system_memory_mb(),
                                      "elapsed_s": round(time.monotonic() - started, 3)}
        ik.close()
    write_json(args.output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alignment-json", type=Path, required=True)
    parser.add_argument("--scene-graph-json", type=Path, required=True)
    parser.add_argument("--stage-config", type=Path, required=True)
    parser.add_argument("--navmesh", type=Path, required=True)
    parser.add_argument("--spot-urdf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--log-file", type=Path, required=True)
    parser.add_argument("--role-id", default="electric_kettle")
    parser.add_argument("--axis-mode", choices=["x_z_minus_y", "x_z_y", "identity"],
                        default="x_z_minus_y")
    parser.add_argument("--start", type=float, nargs=3, default=[-3.03, 0.1, -7.65])
    parser.add_argument("--base-rotation", type=float, default=0.0)
    parser.add_argument("--clearance-m", type=float, default=0.06)
    parser.add_argument("--max-frame-error-m", type=float, default=0.03)
    parser.add_argument("--max-fk-error-m", type=float, default=0.04)
    args = parser.parse_args()
    args.log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(args.log_file, encoding="utf-8")])
    try:
        result = run(args)
        LOG.info("status=%s reachable=%s result=%s", result["status"], result["reachable"], args.output)
        return 0 if result["status"] == "spot_ik_reachability_passed" else 2
    except Exception as exc:
        LOG.exception("Spot IK reachability test failed")
        write_json(args.output, {"schema_version": "1.0", "status": "error",
                                 "reachable": None,
                                 "error": {"type": type(exc).__name__, "message": str(exc)},
                                 "resource_summary": {"process_peak_rss_mb": peak_rss_mb(),
                                                      "system_memory": system_memory_mb()}})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
