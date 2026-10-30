#!/usr/bin/env python3
"""One live Scene201 simulator; arbitrary serial combinations of five proxy primitives."""
from __future__ import annotations

import argparse
import copy
import logging
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

from sequential_proxy_core import SerialExecutor, PARAMETERS, ExecutionRejected, validate_step, placement_checks
from runtime_constraints import check_runtime_initial_constraints, RuntimeConstraintError
from visual_audit import VisualAudit, add_arguments as add_audit_arguments, configure_camera
from proxy_safety import (POLICY as MOTION_POLICY, checked_motion, robot_segment_check, rotate_point,
                          collision_center_registration)
from stage_collision_config import prepare as prepare_stage_config, verify_loaded as verify_stage_config
from run_navigation_demo import (read_json, write_json, cg_to_habitat, habitat_vector,
    horizontal_distance, select_coordinates, choose_start, navigate, peak_rss_mb, system_memory_mb)
from run_manipulation_proxy import contact_near_target, tail_motion_m
from proxy_placement import (POLICY, candidate_offsets, footprint_samples, descent_probe_positions,
                             blocking_contacts, yaw_rotated_bounds, yaw_quaternion)

OUT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(OUT / "skill_matching_v1"))
sys.path.insert(0, str(OUT / "scene_memory_v1"))
sys.path.insert(0, str(OUT / "object_alignment_v1"))
from runtime_goal_protocol import (CONTRACT as STATE_GOAL_CONTRACT, STATE_RELATIONS,
                                   validate_state_goal_set, check_state_goal)
from action_sequence import validate_sequence, refresh_sequence
from match_skills import build_result, validate_proposal
from update_memory import initialize, apply_events, digest

SCENE = Path("/mnt/d/NebulaAI/output/batches/scene201_vith_full_20260915_181750/seq_0001__hf_house_201")
LOG = logging.getLogger("continuous_proxy")


def rigid_object_diagnostics(obj):
    result = {}
    for name in ("rotation_wxyz", "linear_velocity", "angular_velocity", "motion_type"):
        try:
            if name == "rotation_wxyz":
                q = obj.rotation
                result[name] = [float(q.scalar), *map(float, q.vector)]
            elif name == "motion_type":
                result[name] = str(obj.motion_type)
            else:
                result[name] = list(map(float, getattr(obj, name)))
        except Exception as exc:
            result[name] = None
            result.setdefault("unavailable_fields", {})[name] = str(exc)
    return result


def finalize_execution_evidence(summary):
    """Report observations independently of overall task success."""
    rounds = summary["rounds"]
    actions = [a for row in rounds for a in row["actions"]]
    summary["attempted_action_count"] = len(actions)
    summary["executed_action_count"] = sum(a["status"] != "failed" for a in actions)
    summary["cross_round_held_object_continuations"] = [r["round"] for r in rounds[1:]
        if r["begin_state"]["held_object_id"] is not None and r["continuity_with_previous_round"]]
    summary["uninterrupted_simulator_execution"] = bool(rounds) and summary["simulator_creation_count"] == 1 and all(
        r["continuity_with_previous_round"] for r in rounds)
    if rounds:
        last = rounds[-1]
        summary["final_state"] = last.get("end_state", last["actions"][-1]["after"] if last["actions"] else last["begin_state"])


def contract():
    return {"adapter_mode": "serial_scene201_proxy", "supported_skill_ids": list(PARAMETERS),
            "fixed_order_required": False, "max_held_objects": 1,
            "mapped_instance_ids": [67, 13], "physical_grasp_verified": False,
            "placement_policy": copy.deepcopy(POLICY),
            "motion_safety_policy": copy.deepcopy(MOTION_POLICY),
            "goal_protocol": copy.deepcopy(STATE_GOAL_CONTRACT),
            "parameter_policy": "Only grounded target/object/surface arguments are supported; reject others"}


def build_fixture_episode(library):
    refs = {67: {"role_id": "kettle", "element_id": 67}, 13: {"role_id": "sideboard", "element_id": 13}}
    def step(skill, target, surface=None):
        key = "object" if skill in ("lift_held_object", "place_object_on_surface") else "target"
        arguments = {key: copy.deepcopy(refs[target])}
        if surface is not None:
            arguments["surface"] = copy.deepcopy(refs[surface])
        return {"skill_id": skill, "arguments": arguments}
    route = [step("navigate_to_object", target) for target in [67, 13, 67]]
    pickup = [step("position_arm_for_object", 67), step("grasp_object", 67), step("lift_held_object", 67)]
    putdown = [step("navigate_to_object", 13), step("position_arm_for_object", 13), step("place_object_on_surface", 67, 13)]
    rows = [
        ("Visit the kettle, visit the sideboard, then return to the kettle.", route, None, False),
        ("Pick up and lift the kettle; keep holding it for the next instruction.", pickup, 67, True),
        ("Carry the held kettle to the sideboard and place it there.", putdown, None, False),
        ("Go to the moved kettle, pick it up and lift it again.", [step("navigate_to_object", 67)] + pickup, 67, True),
        ("Place the held kettle back on the sideboard.", putdown, None, False),
    ]
    rounds = []
    for index, (instruction, steps, held, lifted) in enumerate(rows, 1):
        goal = ({"subject_role": "kettle", "relation": "on", "object_role": "sideboard"}
                if index in (3, 5) else {"subject_role": "robot", "relation": "near", "object_role": "kettle"})
        alignment = {"status": "success", "context_bindings": {
            "kettle": {"element_ids": [67]}, "sideboard": {"element_ids": [13]}},
            "task_decomposition": {"instruction": instruction,
                "subtasks": [{"subtask_id": "st_01", "description": instruction, "dependencies": []}],
                "initial_scene_constraints": [], "goal_state": [dict(goal, constraint_id="goal_01", required=True)]}}
        proposal = {"subtask_plans": [{"subtask_id": "st_01", "intent": instruction, "steps": steps}]}
        rounds.append({"round": index, "alignment": alignment, "plan": build_result(alignment, library, proposal),
                       "expected_end": {"held_object_id": held, "lifted": lifted}})
    return {"schema_version": "1.0", "planning_origin": "hand_authored_executor_fixture",
            "fresh_llm_calls": False, "rounds": rounds, "training_eligible": False}


def validate_episode(episode, library, nodes):
    if not isinstance(episode.get("rounds"), list) or not episode["rounds"]:
        raise ValueError("Episode needs at least one round")
    active_ids = {n["id"] for n in nodes}
    round_ids = set()
    for row in episode["rounds"]:
        if type(row.get("round")) is not int or row["round"] in round_ids:
            raise ValueError("Round IDs must be unique integers")
        round_ids.add(row["round"])
        plan, alignment = row["plan"], row["alignment"]
        if alignment.get("status") != "success" or plan.get("status") != "proposal_needs_environment_validation":
            raise ValueError("Unresolved plan or object alignment")
        if plan.get("instruction") != alignment["task_decomposition"]["instruction"]:
            raise ValueError("Plan instruction does not match alignment")
        validate_proposal(plan, alignment, library)
        validate_sequence(plan)
        goals = alignment["task_decomposition"].get("goal_state", [])
        if not goals or not any(g.get("required", True) for g in goals):
            raise ValueError("Required completion goal missing")
        if any(g.get("relation") not in ({"near", "on"} | STATE_RELATIONS) for g in goals):
            raise ValueError("Unsupported runtime goal relation")
        validate_state_goal_set(goals, alignment['context_bindings'], active_ids)
        if any(g.get("relation") == "near" and g.get("subject_role") != "robot" for g in goals):
            raise ValueError("Runtime near goals require robot as subject")
        steps = [s for task in plan["subtask_plans"] for s in task["steps"]]
        if not steps or any(not task["steps"] for task in plan["subtask_plans"]):
            raise ValueError("Empty subtask cannot be silently skipped")
        step_ids = [s.get("step_id") for s in steps]
        if any(not isinstance(sid, str) or not sid for sid in step_ids) or len(set(step_ids)) != len(step_ids):
            raise ValueError("Unique nonempty step IDs required even for legacy plans")
        expected = row.get("expected_end", {})
        if (not isinstance(expected, dict) or set(expected) - {"held_object_id", "lifted"}
                or ("held_object_id" in expected and expected["held_object_id"] is not None
                    and (type(expected["held_object_id"]) is not int or expected["held_object_id"] != 67))
                or ("lifted" in expected and not isinstance(expected["lifted"], bool))):
            raise ValueError("Unsupported expected runtime-state assertion")
        for action in steps:
            validate_step(action, alignment["context_bindings"], active_ids, {67, 13})


def inject_fault(episode, fault):
    episode = copy.deepcopy(episode)
    if fault == "place_without_grasp":
        row = copy.deepcopy(episode["rounds"][2])
        row["round"] = 1
        # Even a well-formed placement skill cannot run without a held object.
        row["plan"]["subtask_plans"][0]["steps"] = [row["plan"]["subtask_plans"][0]["steps"][-1]]
        row["plan"] = refresh_sequence(row["plan"])
        episode["rounds"] = [row] + episode["rounds"][1:]
    elif fault == "duplicate_grasp":
        row = episode["rounds"][1]
        duplicate = copy.deepcopy(row["plan"]["subtask_plans"][0]["steps"][1])
        duplicate["step_id"] = "duplicate_grasp"
        row["plan"]["subtask_plans"][0]["steps"].append(duplicate)
        row["plan"] = refresh_sequence(row["plan"])
    elif fault in ("missing_object", "unsupported_skill"):
        action = episode["rounds"][0]["plan"]["subtask_plans"][0]["steps"][0]
        if fault == "missing_object":
            action["arguments"]["target"]["element_id"] = 999
        else:
            action["skill_id"] = "teleport_object"
    episode["test_fault"] = fault
    return episode


class HabitatBackend:
    """Scene-specific asset registration; dispatch itself is not a seven-step script."""
    def __init__(self, sim, nodes, memory, root, reach, max_actions, settle_seconds):
        import habitat_sim
        from habitat_sim.physics import MotionType
        self.sim, self.hs, self.motion = sim, habitat_sim, MotionType
        self.nodes = {n["id"]: n for n in nodes}
        self.active_ids = set(self.nodes)
        self.mapped_ids = {67, 13}
        self.memory, self.root, self.reach = memory, root, reach
        self.max_actions, self.settle_seconds = max_actions, settle_seconds
        self.object_offsets = {67: [0., 0., 0.], 13: [0., 0., 0.]}
        self.support = {}
        self.instances = {}
        self.events = 0
        self.motion_evidence = []
        self.registration_report = {"schema_version":"1.0", "training_eligible":False,
                                    "status":"registering", "objects":{}}
        self._register()
        # Validate the spawn before navigation or any costly live LLM request.
        try:
            check = self._move(67, list(map(float, self.instances[67].translation)),
                               "initial_pose_validation", commit=False)
            self.registration_report.update(status="initial_pose_sampled_clear", collision_check=check)
        except ExecutionRejected as exc:
            self.registration_report.update(status="initial_pose_rejected", error=str(exc),
                                            collision_check=exc.details)
            raise ExecutionRejected("INITIAL_OBJECT_POSE_REJECTED", exc.details) from exc
        finally:
            write_json(self.root / "initial_asset_registration.json", self.registration_report)

    def _register(self):
        assets = SCENE / "habitat_split_assets_v2"
        manager = self.sim.get_object_template_manager()
        for node_id, filename in ((13, "sideboard_static.object_config.json"), (67, "kettle_dynamic.object_config.json")):
            templates = manager.load_configs(str(assets / filename))
            if not templates:
                raise ValueError(f"Cannot load object template for {node_id}")
            template = manager.get_template_by_id(templates[0])
            obj = self.sim.get_rigid_object_manager().add_object_by_template_handle(template.handle)
            if obj is None:
                raise ValueError(f"Cannot instantiate node {node_id}")
            self.instances[node_id] = obj
            obj.motion_type = self.motion.KINEMATIC
            if node_id == 13:
                # Same bounded COM calibration as the already validated Scene201 proxy.
                board = cg_to_habitat(self.nodes[13]["bbox_center"], "x_z_minus_y")
                top = self.board_top()
                obj.translation = habitat_vector([-5.77159, 0.00711, -1.49913])
                self.sim.step_physics(1. / 60)
                ray = self.hs.geo.Ray()
                ray.origin = habitat_vector([board[0] - .05, top + .5, board[2] - .05])
                ray.direction = habitat_vector([0., -1., 0.])
                hits = self.sim.cast_ray(ray, max_distance=1.5).hits
                if not hits or hits[0].object_id != obj.object_id:
                    raise ValueError("Cannot calibrate sideboard top")
                correction = top - float(hits[0].point[1])
                if not 0 <= correction <= .6:
                    raise ValueError("Sideboard registration offset out of range")
                obj.translation = habitat_vector([-5.77159, .00711 + correction, -1.49913])
                obj.motion_type = self.motion.STATIC
                self.object_offsets[13] = [board[i] - float(obj.translation[i]) for i in range(3)]
            else:
                center = cg_to_habitat(self.nodes[67]["bbox_center"], "x_z_minus_y")
                registration = collision_center_registration(obj, center)
                self.object_offsets[67] = registration["graph_center_offset_from_rigid_origin"]
                obj.translation = habitat_vector(registration["translation"])
                registration["template_compute_COM_from_shape"] = getattr(template, "compute_COM_from_shape", None)
                registration["previous_fixed_offset_removed"] = [0., 0.08618998527526855, 0.]
                self.registration_report["objects"]["67"] = registration
                LOG.info("KETTLE_REGISTRATION graph_center=%s rigid_translation=%s derived_center_offset=%s",
                         center, registration["translation"], self.object_offsets[67])
        self.sim.step_physics(1. / 60)

    def board_top(self):
        node = self.nodes[13]
        return float(node["bbox_center"][2]) + float(node["bbox_extent"][2]) / 2

    def center(self, target):
        obj = self.instances[target]
        return [float(obj.translation[i]) + self.object_offsets[target][i] for i in range(3)]

    def snapshot(self):
        agent = self.sim.get_agent(0).get_state()
        q = agent.rotation
        return {"robot_position": list(map(float, agent.position)),
                "robot_rotation_wxyz": [float(q.real), *list(map(float, q.imag))],
                "objects": {str(key): {"com": list(map(float, value.translation)),
                    "center": self.center(key), "support_id": self.support.get(key),
                    "rigid_body_diagnostics": rigid_object_diagnostics(value)}
                    for key, value in self.instances.items()},
                "memory_version": self.memory["version"]}

    def require_near(self, target):
        distance = horizontal_distance(self.sim.get_agent(0).get_state().position, self.center(target))
        if distance > self.reach:
            raise ExecutionRejected(f"PROXIMITY_REJECTED: {target} distance={distance:.3f}")

    def require_graspable_instance(self, target):
        if target != 67:
            raise ExecutionRejected("INSTANCE_NOT_SUPPORTED_FOR_PROXY_GRASP")

    def require_placeable_pair(self, target, surface):
        if (target, surface) != (67, 13):
            raise ExecutionRejected("UNSUPPORTED_OBJECT_SURFACE_PAIR")

    def attach(self, target):
        obj = self.instances[target]
        start = list(map(float, obj.translation))
        agent = list(map(float, self.sim.get_agent(0).get_state().position))
        # A supported object must clear its old surface before horizontal proxy transfer.
        height = max(1.2, start[1])
        clearance = max(height, start[1]) + .15
        self._move(target, [agent[0], height, agent[2]], "assumed_grasp_transfer",
                   via=([start[0], clearance, start[2]], [agent[0], clearance, agent[2]]))
        obj.motion_type = self.motion.KINEMATIC
        self.support.pop(target, None)

    def lift(self, target):
        self.carry(target)

    def carry(self, target, height=1.35):
        agent = self.sim.get_agent(0).get_state().position
        return self._move(target, [float(agent[0]), height, float(agent[2])], "carry_or_lift")

    def _move(self, target, goal, phase, **kwargs):
        if not hasattr(self, "motion_evidence"):
            self.motion_evidence = []
        try:
            report = checked_motion(self, target, goal, phase, **kwargs)
        except ExecutionRejected as exc:
            self.motion_evidence.append(copy.deepcopy(exc.details))
            LOG.warning("PROXY_MOTION_REJECTED phase=%s target=%s error=%s", phase, target, exc)
            raise
        self.motion_evidence.append(report)
        return report

    def navigate(self, target, held):
        if target == held:
            raise ExecutionRejected("NAVIGATION_TO_HELD_OBJECT_UNDEFINED")
        center = self.center(target)
        entity = self.memory["entities"].get(str(target), {})
        if not entity.get("active"):
            raise ExecutionRejected("NO object can be used in this task")
        memory_center = entity["node"]["bbox_center"]
        target_from_memory = cg_to_habitat(memory_center, "x_z_minus_y")
        memory_error = math.dist(target_from_memory, center)
        if memory_error > 1e-4:
            raise ExecutionRejected(f"MEMORY_SIMULATOR_POSE_MISMATCH: {memory_error:.6f} m")
        node = {"element_id": target, "cg_center": memory_center}
        _, snapped = select_coordinates(self.sim.pathfinder, [node], "x_z_minus_y", 2.)
        last_state = self.sim.get_agent(0).get_state()
        def before_action(action):
            nonlocal last_state
            last_state = self.sim.get_agent(0).get_state()
            if action != "move_forward":
                return
            p = list(map(float, last_state.position))
            q = last_state.rotation
            delta = rotate_point([0., 0., -.25], [float(q.real), *map(float, q.imag)])
            raw_goal = habitat_vector([p[i]+delta[i] for i in range(3)])
            predicted = list(map(float, self.sim.pathfinder.try_step(habitat_vector(p), raw_goal)))
            robot_segment_check(self, p, predicted)
            self.motion_evidence.append({"phase": "robot_navigation_pre_step", "start": p,
                "goal": predicted, "status": "sampled_clear", "robot_body_collision_verified": False})
            if held is not None:
                self._move(held, [predicted[0], 1.35, predicted[2]], "pre_navigation_carry", commit=False)
        def callback():
            try:
                p = list(map(float, self.sim.get_agent(0).get_state().position))
                robot_segment_check(self, list(map(float, last_state.position)), p)
                if held is not None:
                    self.carry(held)
            except ExecutionRejected as exc:
                # Unexpected post-step mismatch: restore the last safe agent pose;
                # never hide that a step occurred or pretend simulation time rolled back.
                attempted = p
                self.sim.get_agent(0).set_state(last_state)
                if isinstance(exc.details, dict):
                    exc.details.update(attempted_robot_position=attempted,
                        robot_restored_to_previous_safe_pose=True, simulation_clock_not_rolled_back=True)
                raise
            audit = getattr(self, "audit", None)
            if audit is not None:
                audit.capture("navigation_post_step", force=False)
        result = navigate(self.sim, snapped[0]["nav_goal"], self.max_actions, .6, 25,
                          on_action=callback, before_action=before_action)
        if result["status"] == "arrived":
            self.require_near(target)
        return dict(result, target_id=target, target_center=target_from_memory,
                    target_source="current_memory_confirmed_by_live_instance_readback",
                    observed_instance_center=center, memory_target_error_m=memory_error)

    def _choose_placement(self, target, surface, upright_rotation, orientations=None):
        """Transactional volume probing: no physics integration or memory commit."""
        obj, board_obj = self.instances[target], self.instances[surface]
        board = cg_to_habitat(self.nodes[13]["bbox_center"], "x_z_minus_y")
        try:
            bounds = obj.collision_shape_aabb
            lower, upper = list(map(float, bounds.min)), list(map(float, bounds.max))
            plans = []
            for yaw, rotation in orientations if orientations is not None else [(0., upright_rotation)]:
                rotated_lower, rotated_upper = yaw_rotated_bounds(lower, upper, yaw)
                offsets = candidate_offsets(self.nodes[13]["bbox_extent"][:2], rotated_lower, rotated_upper)
                for dx, dz in offsets:
                    plans.append((dx, dz, yaw, rotation, rotated_lower, rotated_upper))
            plans.sort(key=lambda item: (round(math.hypot(item[0], item[1]), 8), item[2]))
        except Exception as exc:
            raise ExecutionRejected("PLACEMENT_COLLISION_BOUNDS_UNAVAILABLE",
                {"release_performed": False, "reason": str(exc), "placement_policy": POLICY}) from exc
        saved = {name: getattr(obj, name) for name in
                 ("translation", "rotation", "motion_type", "linear_velocity", "angular_velocity", "awake")}
        top, chosen, chosen_yaw, candidates = self.board_top(), None, None, []
        try:
            # Kinematic-to-static checks can be filtered out by Bullet. Probe as
            # DYNAMIC, but never step the world; restore the held state in finally.
            obj.motion_type = self.motion.DYNAMIC
            obj.linear_velocity = obj.angular_velocity = habitat_vector([0., 0., 0.])
            for dx, dz, yaw, rotation, rotated_lower, rotated_upper in plans:
                obj.rotation = rotation
                samples = footprint_samples(rotated_lower, rotated_upper)
                heights, ray_reports = [], []
                candidate = {"offset_xz": [dx, dz], "yaw_deg": yaw, "rotation_wxyz": yaw_quaternion(yaw),
                             "rotated_collision_bounds": {"min": rotated_lower, "max": rotated_upper},
                             "rays": ray_reports, "valid": False,
                             "volume_probes": [], "reason": "support_footprint_not_verified"}
                candidates.append(candidate)
                for ox, oz in samples:
                    ray = self.hs.geo.Ray()
                    ray.origin = habitat_vector([board[0] + dx + ox, top + .5, board[2] + dz + oz])
                    ray.direction = habitat_vector([0., -1., 0.])
                    # The temporarily moved kettle is not an obstacle to its own
                    # support query. All stage and other-object hits stay visible.
                    hits = [h for h in self.sim.cast_ray(ray, max_distance=1.).hits
                            if int(h.object_id) != int(obj.object_id)]
                    ray_reports.append({"origin": list(map(float, ray.origin)),
                        "first_hit_object_id": int(hits[0].object_id) if hits else None,
                        "first_hit_position": list(map(float, hits[0].point)) if hits else None})
                    if not hits or hits[0].object_id != board_obj.object_id or abs(float(hits[0].point[1]) - top) > .15:
                        break
                    heights.append(float(hits[0].point[1]))
                if len(heights) != len(samples):
                    continue
                poses = descent_probe_positions(board[0] + dx, board[2] + dz, max(heights), rotated_lower[1])
                candidate.update(reason="volume_probe_blocked", measured_support_top_m=max(heights))
                for pose in poses:
                    obj.translation = habitat_vector(pose)
                    obj.awake = True
                    self.sim.perform_discrete_collision_detection()
                    blockers = blocking_contacts(self.sim.get_physics_contact_points(),
                                                 int(obj.object_id), int(board_obj.object_id))
                    candidate["volume_probes"].append({"position": pose, "blocking_contacts": blockers})
                    if blockers:
                        LOG.info("PLACEMENT_CANDIDATE_REJECTED offset=%s yaw=%s reason=volume_probe_blocked obstacles=%s",
                                 [dx, dz], yaw, sorted({b["other_object_id"] for b in blockers}))
                        break
                else:
                    chosen = poses[0]
                    chosen_yaw = yaw
                    candidate.update(valid=True, reason="support_and_volume_preflight_passed")
                    LOG.info("PLACEMENT_CANDIDATE offset=%s yaw=%s release=%s probes=%s",
                             candidate["offset_xz"], yaw, chosen, len(poses))
                    break
        finally:
            obj.motion_type = saved["motion_type"]
            obj.rotation, obj.translation = saved["rotation"], saved["translation"]
            obj.linear_velocity, obj.angular_velocity = saved["linear_velocity"], saved["angular_velocity"]
            obj.awake = saved["awake"]
            # Do not leave probe manifolds as if they were execution observations.
            self.sim.perform_discrete_collision_detection()
        preparation = {"placement_policy": copy.deepcopy(POLICY), "surface_candidates": candidates,
                       "local_collision_bounds": {"min": lower, "max": upper},
                       "candidate_count": len(candidates), "held_pose_restored_after_search": True,
                       "candidate_budget": len(plans), "release_yaw_deg": chosen_yaw,
                       "release_rotation_wxyz": yaw_quaternion(chosen_yaw) if chosen_yaw is not None else None,
                       "release_position": chosen, "release_performed": False}
        if chosen is None:
            LOG.warning("PLACEMENT_PREFLIGHT_FAILED candidates=%s volume_blocked=%s support_unverified=%s",
                        len(candidates), sum(c["reason"] == "volume_probe_blocked" for c in candidates),
                        sum(c["reason"] == "support_footprint_not_verified" for c in candidates))
            raise ExecutionRejected("NO_CLEAR_SUPPORT_SURFACE", preparation)
        return preparation

    def place(self, target, surface):
        self.require_placeable_pair(target, surface)
        import magnum as mn
        obj, board_obj = self.instances[target], self.instances[surface]
        board = cg_to_habitat(self.nodes[13]["bbox_center"], "x_z_minus_y")
        top = self.board_top()
        before_release = rigid_object_diagnostics(obj)
        orientations = []
        for yaw in POLICY["allowed_release_yaws_deg"]:
            q = yaw_quaternion(yaw)
            orientations.append((yaw, mn.Quaternion(mn.Vector3(*q[1:]), q[0])))
        preparation = self._choose_placement(target, surface, orientations[0][1], orientations)
        chosen = preparation["release_position"]
        q = preparation["release_rotation_wxyz"]
        upright = mn.Quaternion(mn.Vector3(*q[1:]), q[0])
        start = list(map(float, obj.translation))
        clearance = max(start[1], chosen[1]) + .15
        self._move(target, chosen, "pre_release_transfer", rotation=upright,
                   via=([start[0], clearance, start[2]], [chosen[0], clearance, chosen[2]]))
        obj.motion_type = self.motion.DYNAMIC
        obj.rotation, obj.translation = upright, habitat_vector(chosen)
        obj.linear_velocity = obj.angular_velocity = habitat_vector([0., 0., 0.])
        obj.awake = True
        released_state = rigid_object_diagnostics(obj)
        if getattr(self, "audit", None) is not None:
            self.audit.capture("placement_release", held_state_pending_action_commit=True)
        positions, contact_frames, raw_pair_frames = [], 0, 0
        contact_pairs = {}
        frames = max(120, round(self.settle_seconds * 60))
        for frame in range(frames):
            self.sim.step_physics(1. / 60)
            if frame % 15 == 0 and getattr(self, "audit", None) is not None:
                self.audit.capture("placement_settling", physics_frame=frame + 1,
                                   held_state_pending_action_commit=True)
            if frame >= frames - 60:
                positions.append(list(map(float, obj.translation)))
                contacts = self.sim.get_physics_contact_points()
                for c in contacts:
                    pair = sorted([int(c.object_id_a), int(c.object_id_b)])
                    if int(obj.object_id) in pair:
                        key = ":".join(map(str, pair))
                        contact_pairs[key] = contact_pairs.get(key, 0) + 1
                raw_pair = [c for c in contacts
                    if {int(c.object_id_a), int(c.object_id_b)} == {int(obj.object_id), int(board_obj.object_id)}]
                raw_pair_frames += bool(raw_pair)
                direct = [c for c in raw_pair
                    if contact_near_target(c, obj.object_id, board, self.nodes[13]["bbox_extent"][:2], top)]
                contact_frames += bool(direct)
        final, motion = list(map(float, obj.translation)), tail_motion_m(positions)
        validation = placement_checks(final, board, top, motion, contact_frames)
        passed = validation["passed"]
        settled_state = rigid_object_diagnostics(obj)
        LOG.info("PLACEMENT target=%s surface=%s passed=%s failed_checks=%s metrics=%s raw_pair_frames=%s",
                 target, surface, passed, validation["failed_checks"], validation["metrics"], raw_pair_frames)
        if passed:
            self.support[target] = surface
            # Freeze after verified settling: later nav must not silently advance a new physical experiment.
            obj.motion_type = self.motion.KINEMATIC
        return {"status": "proxy_instance_support_passed" if passed else "proxy_instance_support_failed",
                "validation": validation, "release_performed": True,
                "before_release": before_release, "at_release": released_state, "after_settling": settled_state,
                "release_position": chosen, "surface_candidates": preparation["surface_candidates"], "board_top_m": top,
                "placement_preparation": preparation,
                "simulator_object_ids": {"object": int(obj.object_id), "surface": int(board_obj.object_id)},
                "raw_object_surface_contact_frames": raw_pair_frames,
                "tail_object_contact_point_counts_by_pair": contact_pairs,
                "settled_position": final, "direct_support_contact_frames": contact_frames,
                "tail_motion_m": motion, "simulated_settle_seconds": frames / 60,
                "placed_object_frozen_after_verification": passed}

    def sync_memory(self, action_id, reference):
        center = self.center(67)
        cg_center = [center[0], -center[2], center[1]]
        previous = self.memory["entities"]["67"]["node"]["bbox_center"]
        desired_support = self.support.get(67)
        old_support = next((e["target"] for e in self.memory["edges"] if e["source"] == 67 and e["relation"] == "a on b"), None)
        if math.dist(cg_center, previous) <= 1e-6 and desired_support == old_support:
            return
        self.events += 1
        event = {"event_id": f"continuous_{self.events:04d}", "object_id": 67, "operation": "move",
                 "coordinate_frame": "conceptgraphs_world", "position": cg_center,
                 "support_object_id": desired_support, "action_id": action_id, "action_status": "completed",
                 "observed_at": datetime.now(timezone.utc).isoformat(),
                 "evidence": {"verified": True, "source": "simulator_observation", "reference": str(reference),
                              "scope": "Observed proxy translation, NOT physical grasp verification"}}
        self.memory = apply_events(self.memory, [event])


def evaluate_goals(alignment, backend, executor=None):
    result = []
    bindings = alignment["context_bindings"]
    goals = alignment["task_decomposition"]["goal_state"]
    validate_state_goal_set(goals, bindings, backend.active_ids)
    snapshot = executor.snapshot() if executor is not None else None
    for goal in goals:
        if goal['relation'] in STATE_RELATIONS:
            result.append(check_state_goal(goal, bindings, snapshot, backend.active_ids, 'proxy_executor_readback'))
            continue
        subject, obj = goal["subject_role"], goal["object_role"]
        ids = bindings.get(obj, {}).get("element_ids", [])
        if goal["relation"] == "near" and subject == "robot":
            passed = len(ids) == 1 and horizontal_distance(backend.sim.get_agent(0).get_state().position, backend.center(ids[0])) <= backend.reach
        elif goal["relation"] == "on":
            subjects = bindings.get(subject, {}).get("element_ids", [])
            passed = bool(subjects and ids) and all(backend.support.get(s) in ids for s in subjects)
        else:
            passed = False
        result.append({"constraint_id": goal["constraint_id"], "required": goal.get("required", True),
                       "status": "observed_in_proxy" if passed else "not_satisfied"})
    return result


def save_memory(backend, directory):
    write_json(directory / "memory_state.json", backend.memory)
    write_json(directory / "scene/scene_graph.json", [e["node"] for e in backend.memory["entities"].values() if e["active"]])
    write_json(directory / "scene/scene_graph_edges.json", backend.memory["edges"])


def run(args, summary, round_provider=None):
    nodes = read_json(args.scene_dir / "scene_graph.json")
    edges = read_json(args.scene_dir / "scene_graph_edges.json")
    base_digest = digest({"nodes": nodes, "edges": edges})
    summary["source_graph_digest_before_execution"] = base_digest
    if {n["id"] for n in nodes} != {67, 13}:
        raise ValueError("Scene201 adapter requires the verified kettle/sideboard graph; not arbitrary asset mapping")
    library = read_json(OUT / "quadruped_atomic_skill_library.v0.1.json")
    library["skills"] = [s for s in library["skills"] if s["skill_id"] in PARAMETERS]
    library["execution_contract"] = contract()
    episode = (round_provider.episode if round_provider is not None else
               read_json(args.episode_json) if args.episode_json else build_fixture_episode(library))
    if args.test_fault != "none" and args.episode_json:
        raise ValueError("Fault injection is only supported for the built-in fixture")
    episode = inject_fault(episode, args.test_fault)
    write_json(args.output_dir / "input_episode.json", episode)
    if round_provider is None:
        validate_episode(episode, library, nodes)
    summary.update(planning_origin=episode.get("planning_origin", "external_episode"), fresh_llm_calls=False)
    write_json(args.output_dir / "execution_contract.json", contract())
    if args.preflight_only:
        summary.update(status="preflight_passed", simulation_executed=False)
        return
    assets = SCENE / "habitat_split_assets_v2"
    navmesh = SCENE / "scene_package/hf_house_201/hf_house_201.navmesh"
    for path in (assets / "scene201_without_kettle.stage_config.json", assets / "kettle_dynamic.object_config.json", assets / "sideboard_static.object_config.json", navmesh):
        if not path.is_file():
            raise ValueError(f"Missing asset: {path}")
    import habitat_sim
    from habitat_sim.agent import AgentConfiguration, ActionSpec, ActuationSpec
    from habitat_sim.physics import PhysicsSimulationLibrary
    cfg = habitat_sim.SimulatorConfiguration()
    stage_config, stage_report = prepare_stage_config(assets / "scene201_without_kettle.stage_config.json", args.output_dir)
    cfg.scene_id = str(stage_config)
    summary["stage_collision_configuration"] = stage_report
    cfg.create_renderer, cfg.enable_physics = bool(getattr(args, "visual_audit", False)), True
    agent_cfg = AgentConfiguration()
    if cfg.create_renderer:
        configure_camera(habitat_sim, agent_cfg)
    agent_cfg.action_space = {name: ActionSpec(name, ActuationSpec(amount=amount))
        for name, amount in (("move_forward", .25), ("turn_left", 10.), ("turn_right", 10.))}
    LOG.info("Opening one continuous Habitat simulator; no round reset or respawn")
    with habitat_sim.Simulator(habitat_sim.Configuration(cfg, [agent_cfg])) as sim, VisualAudit(args, summary) as audit:
        summary["simulator_creation_count"] = 1
        summary["simulation_executed"] = True
        if sim.get_physics_simulation_library() != PhysicsSimulationLibrary.Bullet:
            raise ValueError("Bullet physics required for placement verification")
        verify_stage_config(sim, stage_report, args.output_dir)
        LOG.info("STAGE_COLLISION_CONFIG loaded_margin_m=%.6f stage_collidable=True",
                 stage_report['loaded_stage_margin_m'])
        if not sim.pathfinder.load_nav_mesh(str(navmesh)):
            raise ValueError("Navmesh load failed")
        _, snapped = select_coordinates(sim.pathfinder, [{"element_id": 67, "cg_center": next(n["bbox_center"] for n in nodes if n["id"] == 67)}], "x_z_minus_y", 2.)
        start = choose_start(sim.pathfinder, snapped[0]["nav_goal"], args.seed)
        state = sim.get_agent(0).get_state()
        state.position = habitat_vector(start)
        sim.get_agent(0).set_state(state)
        summary["initial_start_sampling_count"] = 1
        memory = initialize(nodes, edges, args.scene_dir)
        memory["execution_domain"] = "habitat_task_level_proxy"
        memory["physical_object_movement_verified"] = False
        summary["initial_asset_registration_json"] = str(args.output_dir / "initial_asset_registration.json")
        backend = HabitatBackend(sim, nodes, memory, args.output_dir,
                                 args.proxy_reach_m, args.max_actions, args.settle_seconds)
        executor, previous_end = SerialExecutor(backend), None
        backend.audit = audit
        audit.bind(backend, executor)
        audit.capture("initial_state")
        for row in episode["rounds"]:
            round_dir = args.output_dir / f"round_{row['round']}"
            begin = executor.snapshot()
            audit.context = {"round": row["round"], "instruction": row.get("instruction", row.get("plan", {}).get("instruction"))}
            audit.capture("round_begin")
            if round_provider is not None:
                summary["fresh_llm_calls"] = True
                try:
                    row = round_provider.plan(row, executor, backend, library, round_dir)
                    validate_episode({"rounds": [row]}, library,
                                     [e["node"] for e in backend.memory["entities"].values() if e["active"]])
                    if executor.snapshot() != begin:
                        raise ValueError("Planning mutated the live simulator/runtime state")
                except Exception as exc:
                    summary["rounds"].append({"round": row["round"], "status": "failed",
                        "instruction": row.get("instruction"), "failed_stage": "live_llm_planning",
                        "error": str(exc), "begin_state": begin, "end_state": executor.snapshot(),
                        "continuity_with_previous_round": previous_end is None or begin == previous_end,
                        "actions": [], "training_eligible": False})
                    raise
            continuity = previous_end is None or begin == previous_end
            record = {"round": row["round"], "instruction": row["plan"]["instruction"], "status": "running",
                      "begin_state": begin, "continuity_with_previous_round": continuity,
                      "training_eligible": False, "actions": []}
            if round_provider is not None:
                record["planning_evidence"] = row["planning_evidence"]
            summary["rounds"].append(record)
            if not continuity:
                record.update(status="failed", error="Cross-round robot/object/held-state continuity broken", end_state=begin)
                raise ValueError(record["error"])
            try:
                record['initial_runtime_constraint_checks'] = check_runtime_initial_constraints(
                    row['alignment'], begin, args.proxy_reach_m)
            except RuntimeConstraintError as exc:
                record.update(status='failed', failed_stage='initial_runtime_constraint_review',
                    error=str(exc), runtime_constraint_error=exc.details, end_state=executor.snapshot())
                write_json(round_dir / 'execution.json', record)
                raise
            write_json(round_dir / "skill_plan.json", row["plan"])
            LOG.info("START round=%s held=%s memory_version=%s", row["round"], executor.held_object_id, backend.memory["version"])
            for subtask in row["plan"]["subtask_plans"]:
                for step in subtask["steps"]:
                    before = executor.snapshot()
                    action_id = step.get('action_id',f"round_{row['round']}_{step['step_id']}")
                    audit.context = {"round": row["round"], "instruction": record["instruction"],
                                     "action_id": action_id, "skill_id": step["skill_id"]}
                    audit.capture("before_action", action_status="running")
                    try:
                        motion_start = len(backend.motion_evidence)
                        result = executor.execute(step, row["alignment"])
                        result["motion_safety_checks"] = copy.deepcopy(backend.motion_evidence[motion_start:])
                        # JSON stores readback, not a predicted action effect.
                        write_json(round_dir / f"{step['step_id']}.readback.json", executor.snapshot())
                        backend.sync_memory(action_id, round_dir / f"{step['step_id']}.readback.json")
                        record["actions"].append(dict(result, before=before, after=executor.snapshot()))
                        save_memory(backend, round_dir / "scene_memory")
                        audit.capture("after_action", action_status=result["status"])
                    except Exception as exc:
                        after = executor.snapshot()
                        audit.capture("failed_action", action_status="failed", error=str(exc))
                        write_json(round_dir / f"{step['step_id']}.readback.json", after)
                        record.update(status="failed", failed_action=action_id, error=str(exc))
                        record["scene_reconciliation_required"] = True
                        record["actions"].append({"step_id": step["step_id"], "skill_id": step["skill_id"],
                            "status": "failed", "error": str(exc), "before": before, "after": after,
                            "backend_result": getattr(exc, "details", None),
                            "motion_safety_checks": copy.deepcopy(backend.motion_evidence[motion_start:])})
                        record["end_state"] = after
                        record["memory_commit_skipped_for_failed_action"] = True
                        write_json(round_dir / "execution.json", record)
                        raise
                    write_json(round_dir / "execution.json", record)
                    write_json(args.output_dir / "pipeline_summary.json", summary)
                    LOG.info("END action=%s skill=%s status=%s held=%s", action_id, step["skill_id"], result["status"], executor.held_object_id)
            goals = evaluate_goals(row["alignment"], backend, executor)
            expected = row.get("expected_end", {})
            expected_passed = all(getattr(executor, key, object()) == value for key, value in expected.items())
            if "expected_navigation_targets" in row:
                expected_passed = expected_passed and [a.get("target_id") for a in record["actions"]
                    if a["skill_id"] == "navigate_to_object"] == row["expected_navigation_targets"]
            passed = expected_passed and all(g["status"] == "observed_in_proxy" for g in goals if g["required"])
            record.update(status="passed" if passed else "failed", goal_checks=goals,
                          expected_runtime_state_passed=expected_passed, end_state=executor.snapshot())
            previous_end = record["end_state"]
            write_json(round_dir / "execution.json", record)
            LOG.info("END round=%s passed=%s held=%s", row["round"], passed, executor.held_object_id)
            if not passed:
                raise ValueError(f"Round {row['round']} completion conditions failed; subsequent rounds skipped")
        summary.update(status="passed", simulation_executed=True,
                       process_peak_rss_gib=round(peak_rss_mb() / 1024, 3) if peak_rss_mb() else None,
                       system_memory_after_execution=system_memory_mb(), final_state=executor.snapshot())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scene-dir", type=Path, default=SCENE / "vlm_scenegraph_regression/scene201_verified_20260921_113316/scene")
    p.add_argument("--episode-json", type=Path, help="External round alignment/plan objects; never silently rewritten")
    p.add_argument("--seed", type=int, default=201)
    p.add_argument("--max-actions", type=int, default=500)
    p.add_argument("--proxy-reach-m", type=float, default=1.5)
    p.add_argument("--settle-seconds", type=float, default=10.)
    p.add_argument("--preflight-only", action="store_true")
    add_audit_arguments(p)
    p.add_argument("--test-fault", choices=("none", "place_without_grasp", "duplicate_grasp", "missing_object", "unsupported_skill"), default="none")
    p.add_argument("--output-dir", type=Path, default=SCENE / "continuous_sequence_proxy" / datetime.now().strftime("scene201_%Y%m%d_%H%M%S_%f"))
    args = p.parse_args()
    if args.audit_stride <= 0 or args.audit_max_frames <= 0:
        p.error("Audit stride and frame budget must be positive")
    if not math.isfinite(args.proxy_reach_m) or not 0 < args.proxy_reach_m <= 1.5 or not math.isfinite(args.settle_seconds) or args.settle_seconds < 2 or args.max_actions <= 0:
        p.error("Invalid proximity, settle duration or action budget")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", handlers=[
        logging.StreamHandler(sys.stdout), logging.FileHandler(args.output_dir / "console.log", encoding="utf-8")])
    summary = {"schema_version": "1.0", "diagnostics_version": "explicit_stage_margin_collider_center_guard_v6",
               "status": "running", "rounds": [], "training_eligible": False,
               "test_fault": args.test_fault,
               "simulation_executed": False, "uninterrupted_simulator_execution": False,
               "simulator_creation_count": 0, "initial_start_sampling_count": 0,
               "scope": "Single-agent, single-held-object task-level proxy serial dispatch, Scene201 assets only",
               "limitations": ["No real arm/gripper/quadruped control", "No new perception or navmesh reconstruction",
                   "Fixture plans are not new LLM training examples", "Placed object is frozen after verified dynamic settling",
                   "Upright placement pose and short drop are proxy operations, not executed arm trajectories",
                   "Sampled body contacts and swept rays reject blocked proxy motions; not exact continuous collision detection",
                   "Only five primitive skill types and two mapped Scene201 instances are supported"]}
    try:
        run(args, summary)
    except Exception as exc:
        LOG.exception("Continuous sequence failed; remaining actions/rounds skipped")
        summary.update(status="failed", error=str(exc))
    finalize_execution_evidence(summary)
    summary["base_graph_unchanged"] = None
    if "source_graph_digest_before_execution" in summary:
        try:
            current_digest = digest({"nodes": read_json(args.scene_dir / "scene_graph.json"),
                                     "edges": read_json(args.scene_dir / "scene_graph_edges.json")})
            summary["base_graph_unchanged"] = current_digest == summary["source_graph_digest_before_execution"]
            if not summary["base_graph_unchanged"]:
                summary.update(status="failed", source_graph_error="Source graph unexpectedly changed")
        except Exception as exc:
            summary["source_graph_check_error"] = str(exc)
            summary.update(status="failed")
    summary["process_peak_rss_gib"] = round(peak_rss_mb() / 1024, 3) if peak_rss_mb() else None
    write_json(args.output_dir / "pipeline_summary.json", summary)
    LOG.info("status=%s summary=%s", summary["status"], args.output_dir / "pipeline_summary.json")
    return 0 if summary["status"] in ("passed", "preflight_passed") else 2


if __name__ == "__main__":
    raise SystemExit(main())
