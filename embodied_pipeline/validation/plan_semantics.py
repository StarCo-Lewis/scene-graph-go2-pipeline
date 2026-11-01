"""Deterministic semantic gate on the model's own decomposition and plan.

These invariants need no reference answer and no environment observation:
a robot cannot satisfy two mutually exclusive terminal location goals, the
declared terminal goal of a navigation plan must be its last destination, and
an explicit preservation requirement must survive as a structured process
constraint instead of free text. Reachability, physical feasibility and answer
correctness stay out of scope.
"""

import re

# v2: unplanned subtasks checked independently, None roles never match, null-tolerant inputs.
VERSION = "plan_semantics_audit_v2"

# Predicates that place the robot at one location; two of them cannot hold at
# the same terminal state for objects that are not the same object.
LOCATION_RELATIONS = frozenset({"near", "at", "next_to", "beside", "in_front_of"})
NAVIGATION_SKILL = "navigate_to_object"

PRESERVATION_PATTERN = re.compile(
    r"\b(?:do\s+not|don'?t|never|avoid|without|leave|keep)\b[^.;]{0,60}?"
    r"\b(?:mov\w*|touch\w*|disturb\w*|relocat\w*|displac\w*|pick(?:ing)?\s+up|"
    r"carry\w*|lift\w*|in\s+place|untouched|where\s+(?:it|they)\s+(?:is|are))\b",
    re.IGNORECASE)
SCENE_WIDE_PATTERN = re.compile(
    r"\b(?:any|all|every)\b\s+(?:\w+\s+){0,2}?objects?\b|"
    r"\b(?:anything|everything|nothing)\b", re.IGNORECASE)


def _issue(code, severity, detail, message_zh):
    """severity: 'defect' marks a semantic error, 'warning' marks a contract limit.

    Severity never stops collection by itself; the caller decides what to do.
    """
    return {"code": code, "severity": severity, "detail": detail, "message_zh": message_zh}


def _as_dict(value):
    """Model output may carry null or a wrong type anywhere; treat it as absent."""
    return value if isinstance(value, dict) else {}


def _as_list(value):
    return value if isinstance(value, list) else []


def terminal_location_goals(decomposition):
    """Required end-state goals that place the robot at one object."""
    return [g for g in _as_list(_as_dict(decomposition).get("goal_state"))
            if isinstance(g, dict) and g.get("required", True) and g.get("subject_role") == "robot"
            and isinstance(g.get("relation"), str) and g["relation"] in LOCATION_RELATIONS]


def _resolved_targets(role, bindings):
    """Bound element IDs for a role, or None when the role is missing or unresolved."""
    if role is None or not isinstance(bindings, dict):
        return None
    try:
        entry = _as_dict(bindings.get(role))
        ids = entry.get("element_ids")
        return frozenset(ids) if isinstance(ids, list) and ids else None
    except TypeError:  # unhashable role or element IDs
        return None


def _navigation_targets(plan):
    """Ordered (role_id, element_id) pairs of a navigation-only plan, or None."""
    steps = [s for t in _as_list(_as_dict(plan).get("subtask_plans"))
             for s in _as_list(_as_dict(t).get("steps"))]
    if not steps or any(_as_dict(s).get("skill_id") != NAVIGATION_SKILL for s in steps):
        return None
    targets = []
    for step in steps:
        target = _as_dict(_as_dict(step.get("arguments")).get("target"))
        targets.append((target.get("role_id"), target.get("element_id")))
    return targets


def _check_exclusive_terminal_goals(goals, bindings, issues):
    """Group terminal goals by the object they resolve to; disjoint groups conflict."""
    groups, unbound, missing = [], False, []
    for goal in goals:
        role = goal.get("object_role")
        if role is None:
            # A goal without an object cannot be grouped with, or matched to, anything.
            missing.append(goal.get("constraint_id"))
            continue
        ids = _resolved_targets(role, bindings)
        if ids is None:
            unbound = True
            ids = frozenset([("unbound_role", repr(role))])
        for existing in groups:
            if existing["targets"] & ids:
                existing["targets"] |= ids
                existing["constraint_ids"].append(goal.get("constraint_id"))
                break
        else:
            groups.append({"targets": set(ids), "object_role": role,
                           "constraint_ids": [goal.get("constraint_id")]})
    if missing:
        issues.append(_issue(
            "terminal_goal_missing_object_role", "defect", {"constraint_ids": missing},
            "\u7ec8\u6001\u4f4d\u7f6e\u76ee\u6807\u7f3a\u5c11 object_role\uff0c\u65e0\u6cd5\u786e\u5b9a\u673a\u5668\u4eba\u5e94\u9760\u8fd1\u54ea\u4e2a\u7269\u4f53\u3002"))
    if len(groups) > 1:
        issues.append(_issue(
            "mutually_exclusive_terminal_robot_location", "defect",
            {"goal_count": len(goals), "distinct_target_groups": len(groups),
             "object_roles": [g["object_role"] for g in groups],
             "constraint_ids": [i for g in groups for i in g["constraint_ids"]],
             "binding_evidence": "unbound_roles_assumed_distinct" if unbound
                                 else "bound_element_ids_disjoint"},
            "\u7ec8\u6001\u8981\u6c42\u673a\u5668\u4eba\u540c\u65f6\u9760\u8fd1\u591a\u4e2a\u4e0d\u540c\u7269\u4f53\uff1b"
            "\u4e2d\u95f4\u5230\u8bbf\u5e94\u4fdd\u7559\u4e3a\u6709\u5e8f\u5b50\u4efb\u52a1\u4e0e\u4f9d\u8d56\uff0c"
            "\u53ea\u6709\u6700\u540e\u4e00\u4e2a\u76ee\u7684\u5730\u662f\u7ec8\u6001\u76ee\u6807\u3002"))
    return groups


def _check_final_destination(goals, plan, bindings, issues):
    targets = _navigation_targets(plan)
    if not targets or not goals:
        return
    last_role, last_element = targets[-1]
    if last_role is None and last_element is None:
        # Unknown destination: report the gap instead of matching None to None.
        issues.append(_issue(
            "final_navigation_target_unresolved", "warning",
            {"visited_target_count": len(targets)},
            "\u8ba1\u5212\u6700\u540e\u4e00\u4e2a\u5bfc\u822a\u6b65\u9aa4\u6ca1\u6709\u53ef\u8bc6\u522b\u7684\u76ee\u6807\uff0c\u65e0\u6cd5\u6838\u5bf9\u7ec8\u6001\u76ee\u6807\u3002"))
        return

    def covers_last(goal):
        role = goal.get("object_role")
        if role is not None and role == last_role:
            return True
        ids = _resolved_targets(role, bindings)
        if ids is None or last_element is None:
            return False
        try:
            return last_element in ids
        except TypeError:  # unhashable element_id from malformed output
            return False

    matched = any(covers_last(goal) for goal in goals)
    if not matched:
        issues.append(_issue(
            "terminal_goal_not_final_navigation_target", "defect",
            {"final_navigation_target": {"role_id": last_role, "element_id": last_element},
             "terminal_goal_roles": [g.get("object_role") for g in goals],
             "visited_target_count": len(targets)},
            "\u58f0\u660e\u7684\u7ec8\u6001\u76ee\u6807\u4e0d\u662f\u8ba1\u5212\u91cc\u6700\u540e\u4e00\u4e2a\u5bfc\u822a\u76ee\u7684\u5730\u3002"))


def _check_unplanned(plan, issues):
    """Independent of plan shape and terminal goals: any remaining gap is a defect."""
    remaining = _as_dict(plan).get("unplanned_subtasks")
    if remaining:
        issues.append(_issue("unplanned_subtasks_remain", "defect",
                             {"unplanned_subtasks": remaining},
                             "\u8ba1\u5212\u4e2d\u4ecd\u6709\u672a\u89c4\u5212\u7684\u5b50\u4efb\u52a1\u3002"))


def _check_preservation(decomposition, instruction, issues):
    texts = [instruction or decomposition.get("instruction") or ""]
    texts += [_as_dict(s).get("description") for s in _as_list(decomposition.get("subtasks"))]
    hits = [t for t in texts if isinstance(t, str) and PRESERVATION_PATTERN.search(t)]
    if not hits:
        return
    scene_wide = any(SCENE_WIDE_PATTERN.search(t) for t in hits)
    structured = decomposition.get("process_constraints")
    if not isinstance(structured, list):
        structured = None
    detail = {"matched_text": hits[0][:200], "scene_wide_scope": scene_wide,
              "structured_constraint_count": len(structured or []),
              "process_constraint_contract": decomposition.get("process_constraint_contract")}
    if not structured:
        detail["reason"] = ("contract_absent" if "process_constraint_contract" not in decomposition
                            else "contract_returned_empty")
        issues.append(_issue(
            "preservation_requirement_not_structured", "defect", detail,
            "\u6307\u4ee4\u8981\u6c42\u4e0d\u79fb\u52a8\u7269\u4f53\uff0c\u4f46 process_constraints \u4e3a\u7a7a\uff1b"
            "\u4fdd\u62a4\u53ea\u505c\u7559\u5728\u6587\u5b57\uff0c\u5de5\u7a0b\u4fa7\u6ca1\u6709\u7ed3\u6784\u5316\u7ea6\u675f\u3002"))
    elif scene_wide:
        issues.append(_issue(
            "preservation_scope_wider_than_contract", "warning", detail,
            "\u4fdd\u62a4\u8303\u56f4\u662f\u5168\u573a\u666f\u91cf\u8bcd\uff0c\u5f53\u524d\u5951\u7ea6\u53ea\u80fd\u9010\u89d2\u8272\u8868\u8fbe\uff0c"
            "\u4e0d\u80fd\u58f0\u79f0\u5df2\u8986\u76d6\u5168\u90e8\u7269\u4f53\u3002"))


def _suggested_terminal_goal(decomposition, plan):
    """The destination the terminal goal should name, for later repair work."""
    targets = _navigation_targets(plan)
    if targets:
        role, element = targets[-1]
        if role is not None or element is not None:
            return {"source": "final_navigation_target", "role_id": role, "element_id": element}
    subtasks = _as_list(decomposition.get("subtasks"))
    for task in reversed(subtasks):
        roles = _as_list(_as_dict(task).get("required_roles"))
        if roles and _as_dict(roles[0]).get("role_id") is not None:
            return {"source": "last_subtask_required_role", "role_id": roles[0]["role_id"],
                    "subtask_id": task.get("subtask_id")}
    return None


def audit(decomposition, bindings=None, plan=None, instruction=None):
    """Return a structured verdict. This function never stops a run by itself;
    collection always keeps the model's own output and the caller decides what
    the severity means for downstream use."""
    if not isinstance(decomposition, dict):
        raise ValueError("Semantic audit requires the decomposition object")
    goals, issues = terminal_location_goals(decomposition), []
    _check_exclusive_terminal_goals(goals, bindings, issues)
    _check_final_destination(goals, plan, bindings, issues)
    _check_unplanned(plan, issues)
    _check_preservation(decomposition, instruction, issues)
    defects = [i for i in issues if i["severity"] == "defect"]
    result = {
        "version": VERSION,
        "inputs_available": {"decomposition": True, "object_bindings": bool(bindings),
                             "navigation_plan": _navigation_targets(plan) is not None},
        "subtask_count": len(_as_list(decomposition.get("subtasks"))),
        "terminal_location_goal_count": len(goals),
        "terminal_goal_roles": [g.get("object_role") for g in goals],
        "process_constraint_count": len(_as_list(decomposition.get("process_constraints"))),
        "issues": issues,
        "issue_codes": [i["code"] for i in issues],
        "defect_codes": [i["code"] for i in defects],
        "passed": not issues,
        "has_defect": bool(defects),
        "scope": ("Deterministic invariants over the model's own output. Not reachability, "
                  "physical feasibility, execution success or reference-answer grading."),
    }
    if any(i["code"] in ("mutually_exclusive_terminal_robot_location",
                         "terminal_goal_not_final_navigation_target") for i in defects):
        # Evidence for a later repair pass; the model's own goal_state is never rewritten here.
        result["suggested_terminal_goal"] = _suggested_terminal_goal(decomposition, plan)
    return result
