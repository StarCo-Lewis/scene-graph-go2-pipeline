"""Observe whole proxy trajectories; never replace effects or authorize success."""

from ..planning.process_contract import grounded_restrictions

# All currently available atomic skills have explicit argument semantics.
# Unknown skills fail closed for this audit until their effects are described.
KNOWN = {"navigate_to_object", "position_arm_for_object", "grasp_object",
         "lift_held_object", "place_object_on_surface"}
PAYLOAD_SLOT = {"grasp_object": "target", "lift_held_object": "object",
                "place_object_on_surface": "object"}


def audit(alignment, initial, feedback):
    contract = grounded_restrictions(alignment)
    if contract is None:
        return None
    protected = {i for r in contract["restrictions"] for i in r["element_ids"]}
    violations, unknown = [], []
    fields = ("center", "support_id", "room_hypothesis")
    for obj in protected:
        baseline = initial["backend"]["objects"].get(str(obj))
        if baseline is None:
            unknown.append(dict(object_id=obj, reason="missing_initial_object"))
            continue
        states = [(t["action_index"], t, t["end_state"]) for t in feedback["trace"]]
        end = feedback["resulting_state"]
        states.append(("final", None, dict(objects=end["backend"]["objects"],
                                           held_object_id=end["held_object_id"], lifted=end["lifted"])))
        for index, step, state in states:
            reasons = []
            if step:
                skill = step["skill_id"]
                if skill not in KNOWN:
                    unknown.append(dict(action_index=index, reason="unsupported_skill_effect", skill_id=skill))
                if skill in PAYLOAD_SLOT:
                    payload = step["arguments"].get(PAYLOAD_SLOT[skill])
                    if not isinstance(payload, dict) or type(payload.get("element_id")) is not int:
                        unknown.append(dict(action_index=index, reason="missing_manipulation_payload"))
                    elif payload["element_id"] == obj:
                        reasons.append("protected_object_manipulation_attempt")
            current = state.get("objects", {}).get(str(obj))
            if current is None or any(f not in current or f not in baseline for f in fields):
                unknown.append(dict(action_index=index, object_id=obj, reason="missing_object_state"))
            elif any(current[f] != baseline[f] for f in fields):
                reasons.append("protected_object_state_changed")
            if "held_object_id" not in state or "lifted" not in state:
                unknown.append(dict(action_index=index, object_id=obj, reason="missing_gripper_state"))
            elif ((state["held_object_id"] == obj) != (initial["held_object_id"] == obj)
                  or (state["held_object_id"] == obj and state["lifted"] != initial["lifted"])):
                reasons.append("protected_object_held_state_changed")
            if reasons:
                violations.append(dict(action_index=index, object_id=obj, reasons=reasons))
    complete = (feedback["status"] == "proxy_actions_completed"
                and len(feedback["trace"]) == feedback["proposed_action_count"]
                and feedback["attempted_action_count"] == len(feedback["trace"]))
    return dict(version=contract["version"], protected_object_ids=sorted(protected),
                passed=complete and not violations and not unknown,
                trace_complete=complete, violations=violations, unknown=unknown,
                scope="Observed symbolic proxy trajectory; not physical verification",
                execution_or_memory_blocked=False)
