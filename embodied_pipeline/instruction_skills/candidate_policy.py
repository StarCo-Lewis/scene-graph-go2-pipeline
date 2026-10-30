"""Opt-in candidate availability repair; never inserts executable actions."""

import copy

POLICIES = ("legacy", "prerequisite_complete")
PLACEMENT_SUPPORT = {"place_object_on_surface": ("lift_held_object",)}


def complete_candidates(retrieval, library, policy="legacy"):
    """Extend candidate closure without changing cosine scores or source metadata.

    The serial placement adapter requires a lifted held object. Offer its state
    producers as candidates even when retrieval only selects navigation/placement.
    These are possible prerequisites, not unconditional actions: existing held
    and lifted state may already satisfy the requirement.
    """
    if policy not in POLICIES:
        raise ValueError(f"Unknown candidate policy: {policy}")
    if policy == "legacy":
        return retrieval, None
    by_id = {s["skill_id"]: s for s in library["skills"]}
    result = copy.deepcopy(retrieval)
    audit = dict(policy=policy, version=1,
                 scope="Candidate availability only; no actions inserted or executed",
                 state_sensitive_action_selection="Left to the planner using the unchanged runtime state",
                 prerequisite_overlay={k: list(v) for k, v in PLACEMENT_SUPPORT.items()},
                 subtasks=[])
    for row in result["subtasks"]:
        before = list(row["candidate_skill_ids"])
        selected = set(before)
        todo = list(before)
        edges = []
        while todo:
            skill = todo.pop()
            if skill not in by_id:
                raise ValueError(f"Unknown candidate skill: {skill}")
            dependencies = list(by_id[skill]["semantic_profile"]["prerequisite_skills"])
            dependencies += list(PLACEMENT_SUPPORT.get(skill, ()))
            for dep in dependencies:
                if dep not in by_id:
                    raise ValueError(f"Missing prerequisite skill: {skill} -> {dep}")
                if dep not in selected:
                    selected.add(dep)
                    todo.append(dep)
                    edges.append(dict(skill_id=skill, added_prerequisite=dep))
        row["candidate_skill_ids"] = sorted(selected)
        audit["subtasks"].append(dict(subtask_id=row["subtask_id"],
            baseline_candidate_skill_ids=before,
            candidate_skill_ids=row["candidate_skill_ids"],
            added_skill_ids=sorted(selected-set(before)), expansion_edges=edges))
    result["candidate_completion_policy"] = dict(name=policy, version=1)
    return result, audit
