"""Derived serial action indexes; subtask_plans remains the source of truth."""
from __future__ import annotations

import copy
from action_identity import assign_action_identity, validate_action_identity


POLICY = "preserve_order_except_adjacent_duplicate_navigation_within_subtask"


def validate_subtask_dependencies(subtasks):
    seen = set()
    for subtask in subtasks:
        sid = subtask.get("subtask_id")
        if not isinstance(sid, str) or not sid or sid in seen:
            raise ValueError("subtask IDs must be unique nonempty strings")
        dependencies = subtask.get("dependencies", [])
        if (not isinstance(dependencies, list) or
                any(not isinstance(dep, str) for dep in dependencies) or
                len(dependencies) != len(set(dependencies)) or
                any(dep not in seen for dep in dependencies)):
            raise ValueError(f"Invalid dependencies for {sid}: require earlier subtask IDs; no cycles or forward references")
        seen.add(sid)


def sequence_metadata(plans):
    validate_subtask_dependencies(plans)
    sequence, groups, seen, previous = [], [], set(), None
    for plan in plans:
        step_ids = []
        for step in plan["steps"]:
            step_id = step.get("step_id")
            if not isinstance(step_id, str) or not step_id or step_id in seen:
                raise ValueError("action step IDs must be unique nonempty strings")
            seen.add(step_id)
            step_ids.append(step_id)
            sequence.append({"ordinal": len(sequence) + 1, "step_id": step_id,
                             "subtask_id": plan["subtask_id"],
                             "depends_on": [previous] if previous else []})
            if 'action_id' in step:
                sequence[-1]['action_id']=step['action_id']
            previous = step_id
        groups.append({"subtask_id": plan["subtask_id"], "action_count": len(step_ids),
                       "step_ids": step_ids})
    return {"action_sequence": sequence, "action_count": len(sequence),
            "subtask_action_mapping": groups,
            "sequence_semantics": "serial; no parallel execution or physical success is asserted",
            "sequence_policy": POLICY}


def refresh_sequence(plan):
    result = copy.deepcopy(plan)
    if 'action_identity' in result:
        result=assign_action_identity(result)
    result.update(sequence_metadata(result["subtask_plans"]))
    return result


def validate_sequence(plan):
    validate_action_identity(plan)
    if "action_sequence" not in plan:
        return  # Legacy plans remain readable.
    expected = sequence_metadata(plan["subtask_plans"])
    for field in ("action_sequence", "action_count", "subtask_action_mapping"):
        if plan.get(field) != expected[field]:
            raise ValueError(f"Stale or inconsistent derived {field}; regenerate from subtask_plans")


def removable_navigation_ids(plan):
    """Only local identical calls qualify; A -> B -> A is never a duplicate."""
    removable = set()
    for subtask in plan["subtask_plans"]:
        for previous, current in zip(subtask["steps"], subtask["steps"][1:]):
            if (previous["skill_id"] == current["skill_id"] == "navigate_to_object"
                    and previous["arguments"] == current["arguments"]):
                removable.add(current["step_id"])
    return removable


def adapter_shape_check(plan, contract):
    skills = [step["skill_id"] for task in plan["subtask_plans"] for step in task["steps"]]
    if not contract:
        return {"status": "no_executor_contract", "execution_status": "not_executed"}
    allowed = contract.get("supported_skill_ids")
    expected = contract.get("supported_ordered_sequence")
    unsupported = sorted(set(skills) - set(allowed)) if allowed is not None else []
    matches = not unsupported and (expected is None or skills == expected)
    return {"status": "supported_shape" if matches else "unsupported_shape",
            "adapter_mode": contract.get("adapter_mode"),
            "unsupported_skill_ids": unsupported,
            "ordered_sequence_matches": skills == expected if expected is not None else None,
            "claim": "Shape check only; object parameters, reachability and outcomes require adapter validation.",
            "execution_status": "not_executed"}
