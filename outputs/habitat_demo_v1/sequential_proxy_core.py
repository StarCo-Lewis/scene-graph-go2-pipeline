"""Closed-world serial dispatch independent of Habitat for offline testing."""
from __future__ import annotations

import copy
import math

PARAMETERS = {
    "navigate_to_object": {"target"}, "position_arm_for_object": {"target"},
    "grasp_object": {"target"}, "lift_held_object": {"object"},
    "place_object_on_surface": {"object", "surface"},
}


def serial_state_contract():
    """Review/planning declaration of this executor, NOT physical observations.

    Keep changes to execute() and these effects together; offline transition
    tests check the declaration against the executor rather than an LLM verdict.
    Each call returns a fresh value so request preparation cannot alter it.
    """
    return {
        "version": "serial_proxy_state_semantics_v1",
        "source": "sequential_proxy_core.SerialExecutor.execute",
        "scope": "Symbolic/proxy state only; no physical grasp or arm verification",
        "fields": {
            "held_object_id": "ID of the currently attached/held object; null means none.",
            "lifted": "Whether the held object has been lifted under proxy effects; not arm preparation.",
            "prepared_target_id": "Temporary approach/arm-preparation target for a later grasp or place. NOT a holding flag or attachment ID. Null does NOT mean release.",
        },
        "successful_action_effects": {
            "navigate_to_object": {"held_object_id": "unchanged", "lifted": "unchanged",
                                   "prepared_target_id": "null"},
            "position_arm_for_object": {"held_object_id": "unchanged", "lifted": "unchanged",
                                        "prepared_target_id": "arguments.target.element_id"},
            "grasp_object": {"held_object_id": "arguments.target.element_id", "lifted": "false",
                             "prepared_target_id": "unchanged"},
            "lift_held_object": {"held_object_id": "unchanged", "lifted": "true",
                                 "prepared_target_id": "unchanged"},
            "place_object_on_surface": {"held_object_id": "null", "lifted": "false",
                                        "prepared_target_id": "null"},
        },
        "preconditions": [
            "Navigation with a held object requires lifted=true; navigation never releases it.",
            "Grasp requires an empty gripper and prepared_target_id equal to the grasp target.",
            "Place requires the requested object held and lifted, and prepared_target_id equal to the destination surface.",
            "Preparation also requires proximity; graspability and placement-pair gates remain in force.",
        ],
        "interpretation_rules": [
            "held_object_id=H, lifted=true, prepared_target_id=null is VALID after carrying H. Do not infer release or a broken hold from null preparation.",
            "Do not add preparation just to keep holding; preparation is required for a subsequent grasp/place, not for maintaining attachment.",
            "Assess holding/release from held_object_id and the action trace, not prepared_target_id. Still reject genuine release, wrong objects, missing actions or violated constraints.",
            "If backend.place returns a failed support result, release has already cleared held/lifted/prepared and execution is rejected; failure is never success.",
        ],
    }


class ExecutionRejected(ValueError):
    def __init__(self, message, details=None):
        super().__init__(message)
        self.details = copy.deepcopy(details)


def placement_checks(final, board, top, motion, contact_frames):
    """Expose the existing four gates, without changing their thresholds."""
    metrics = {"horizontal_error_m": math.hypot(final[0] - board[0], final[2] - board[2]),
               "com_height_error_m": abs(final[1] - (top + .083)),
               "tail_motion_m": motion, "direct_support_contact_frames": contact_frames}
    limits = {"horizontal_error_m": .35, "com_height_error_m": .20,
              "tail_motion_m": .02, "direct_support_contact_frames": 20}
    checks = {name: math.isfinite(value) and
              (value >= limits[name] if name == "direct_support_contact_frames" else value <= limits[name])
              for name, value in metrics.items()}
    return {"metrics": metrics, "thresholds": limits, "checks": checks,
            "failed_checks": [name for name, passed in checks.items() if not passed],
            "passed": all(checks.values())}


def validate_step(step, bindings, active_ids, mapped_ids):
    skill = step.get("skill_id")
    if skill not in PARAMETERS:
        raise ExecutionRejected(f"UNSUPPORTED_SKILL: {skill}")
    arguments = step.get("arguments")
    if not isinstance(arguments, dict) or set(arguments) != PARAMETERS[skill]:
        raise ExecutionRejected(f"UNSUPPORTED_OR_MISSING_SKILL_PARAMETER: {skill}")
    for argument in arguments.values():
        if (not isinstance(argument, dict) or set(argument) != {"role_id", "element_id"}
                or type(argument["element_id"]) is not int
                or not isinstance(argument["role_id"], str)):
            raise ExecutionRejected("INVALID_OBJECT_BINDING")
        element = argument["element_id"]
        if element not in active_ids:
            raise ExecutionRejected("NO object can be used in this task")
        if element not in bindings.get(argument["role_id"], {}).get("element_ids", []):
            raise ExecutionRejected("UNGROUNDED_OBJECT")
        if element not in mapped_ids:
            raise ExecutionRejected("OBJECT_HAS_NO_SIMULATOR_INSTANCE")


class SerialExecutor:
    """A finite-state proxy, not a physical affordance/value function."""
    def __init__(self, backend):
        self.backend = backend
        self.held_object_id = None
        self.lifted = False
        self.prepared_target_id = None

    def snapshot(self):
        return {"held_object_id": self.held_object_id, "lifted": self.lifted,
                "prepared_target_id": self.prepared_target_id,
                "backend": copy.deepcopy(self.backend.snapshot())}

    def execute(self, step, alignment):
        validate_step(step, alignment.get("context_bindings", {}),
                      self.backend.active_ids, self.backend.mapped_ids)
        skill, args = step["skill_id"], step["arguments"]
        target = args.get("target", args.get("object"))["element_id"]
        if skill == "navigate_to_object":
            if self.held_object_id is not None and not self.lifted:
                raise ExecutionRejected("HELD_OBJECT_MUST_BE_LIFTED_BEFORE_CARRY_NAVIGATION")
            result = self.backend.navigate(target, self.held_object_id)
            if result.get("status") != "arrived":
                raise ExecutionRejected(f"NAVIGATION_FAILED: {result.get('status')}", result)
            self.prepared_target_id = None
        elif skill == "position_arm_for_object":
            self.backend.require_near(target)
            self.prepared_target_id = target
            result = {"status": "proxy_preparation_only", "arm_motion_executed": False}
        elif skill == "grasp_object":
            if self.held_object_id is not None:
                raise ExecutionRejected("GRIPPER_NOT_EMPTY")
            if self.prepared_target_id != target:
                raise ExecutionRejected("PROXY_PREPARATION_REQUIRED")
            self.backend.require_graspable_instance(target)
            self.backend.require_near(target)
            self.backend.attach(target)
            self.held_object_id, self.lifted = target, False
            result = {"status": "proxy_attached", "physical_grasp_verified": False}
        elif skill == "lift_held_object":
            if self.held_object_id != target:
                raise ExecutionRejected("OBJECT_NOT_HELD")
            self.backend.lift(target)
            self.lifted = True
            result = {"status": "proxy_lifted", "physical_arm_motion_verified": False}
        else:
            surface = args["surface"]["element_id"]
            if self.held_object_id != target or not self.lifted:
                raise ExecutionRejected("LIFTED_HELD_OBJECT_REQUIRED")
            if self.prepared_target_id != surface:
                raise ExecutionRejected("SURFACE_PROXY_PREPARATION_REQUIRED")
            self.backend.require_placeable_pair(target, surface)
            self.backend.require_near(surface)
            result = self.backend.place(target, surface)
            # Release happened even if support validation fails. Do not retain a fictitious held state.
            self.held_object_id, self.lifted, self.prepared_target_id = None, False, None
            if result.get("status") != "proxy_instance_support_passed":
                raise ExecutionRejected("PLACEMENT_SUPPORT_NOT_VERIFIED", result)
        result=dict(result, step_id=step["step_id"], skill_id=skill,
                    held_object_id=self.held_object_id)
        if 'action_id' in step:
            result.update(action_id=step['action_id'],action_ordinal=step['action_ordinal'])
        return result
