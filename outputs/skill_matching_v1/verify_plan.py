#!/usr/bin/env python3
"""Evidence-aware symbolic review of an unexecuted atomic-skill proposal."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from observation_interface import validate_observations


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=True, indent=2) + "\n", encoding="ascii")


def fact_key(subject: Any, predicate: str) -> tuple[str, str]:
    return str(subject), predicate


def resolve_symbol(symbol: Any, arguments: dict[str, Any]) -> Any:
    if isinstance(symbol, str) and symbol.startswith("$"):
        value = arguments.get(symbol[1:])
        if isinstance(value, dict):
            return value["element_id"]
        return value
    return symbol


def scene_relations(edges: list[dict[str, Any]]) -> list[tuple[int, str, int]]:
    relations = []
    for edge in edges:
        try:
            a, b = int(edge["source"]), int(edge["target"])
        except (KeyError, TypeError, ValueError):
            continue
        relation = str(edge.get("relation", "")).lower()
        if relation == "a on b":
            relations.append((a, "on", b))
        elif relation == "b on a":
            relations.append((b, "on", a))
        elif relation == "a in b":
            relations.append((a, "in", b))
        elif relation == "b in a":
            relations.append((b, "in", a))
    return relations


def observed_state(alignment: dict[str, Any], nodes: list[dict[str, Any]],
                   edges: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    state: dict[tuple[str, str], dict[str, Any]] = {}
    known_ids = {int(node["id"]) for node in nodes}
    for node in nodes:
        node_id = int(node["id"])
        state[fact_key(node_id, "exists")] = {"value": True, "source": "scene_graph_node"}
        if node.get("bbox_center") is not None:
            state[fact_key(node_id, "localized")] = {"value": True, "source": "scene_graph_bbox"}
    for a, relation, b in scene_relations(edges):
        if a in known_ids and b in known_ids:
            key = fact_key(a, relation)
            # More than one observed support is ambiguous; do not choose one silently.
            old = state.get(key)
            if old is not None and old["value"] != b:
                state.pop(key, None)
            elif old is None:
                state[key] = {"value": b, "source": "scene_graph_edge"}
    for role_id, binding in alignment.get("context_bindings", {}).items():
        for node_id in binding.get("element_ids", []):
            if node_id not in known_ids:
                raise ValueError(f"binding {role_id} refers to absent scene node {node_id}")
    return state


def role_ids(alignment: dict[str, Any], role_id: str) -> list[int | str]:
    if role_id == "robot":
        return ["robot"]
    binding = alignment.get("context_bindings", {}).get(role_id)
    return binding.get("element_ids", []) if isinstance(binding, dict) else []


def check_initial_claims(alignment: dict[str, Any], observed: dict[tuple[str, str], dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for claim in alignment["task_decomposition"].get("initial_scene_constraints", []):
        subjects = role_ids(alignment, claim["subject_role"])
        objects = role_ids(alignment, claim["object_role"])
        checks = []
        for subject in subjects:
            fact = observed.get(fact_key(subject, claim["relation"]))
            if fact is None:
                checks.append("unknown")
            elif fact["value"] in objects:
                checks.append("observed_true")
            elif claim["relation"] in {"on", "in"}:
                checks.append("observed_conflict")
            else:
                checks.append("unknown")
        status = "observed_true" if checks and all(x == "observed_true" for x in checks) else (
            "observed_conflict" if "observed_conflict" in checks else "unknown")
        output.append({"constraint_id": claim["constraint_id"], "status": status,
                       "required": claim.get("required", True), "trusted_as_initial_fact": status == "observed_true"})
    return output


def lookup(state: dict[tuple[str, str], dict[str, Any]], subject: Any,
           predicate: str, expected: Any) -> dict[str, Any]:
    fact = state.get(fact_key(subject, predicate))
    if fact is None:
        return {"status": "unknown", "reason": "no_observed_or_predicted_fact"}
    if fact["value"] == expected:
        return {"status": "true", "reason": fact["source"]}
    return {"status": "false", "reason": fact["source"]}


def check_condition(condition: dict[str, Any], arguments: dict[str, Any],
                    state: dict[tuple[str, str], dict[str, Any]]) -> dict[str, Any]:
    if "any_of" in condition:
        options = [check_condition(item, arguments, state) for item in condition["any_of"]]
        status = "true" if any(x["status"] == "true" for x in options) else (
            "false" if all(x["status"] == "false" for x in options) else "unknown")
        return {"predicate": "any_of", "status": status, "options": options}
    subject = resolve_symbol(condition.get("subject"), arguments)
    predicate = condition.get("predicate", "")
    expected = resolve_symbol(condition.get("object", condition.get("value")), arguments)
    check = lookup(state, subject, predicate, expected)
    return {"subject": subject, "predicate": predicate, "expected": expected, **check}


def apply_effects(skill: dict[str, Any], arguments: dict[str, Any],
                  state: dict[tuple[str, str], dict[str, Any]], step_id: str,
                  conditional: bool) -> list[dict[str, Any]]:
    applied = []
    for effect in skill.get("effects", []):
        if "update" in effect:
            continue  # A pose update cannot be predicted without geometry.
        subject = resolve_symbol(effect.get("subject"), arguments)
        predicate = effect.get("predicate")
        value = resolve_symbol(effect.get("object", effect.get("value")), arguments)
        if "value_from" in effect:
            value = resolve_symbol(effect["value_from"], arguments)
        if predicate is None:
            continue
        # These predicates have a single current value. The new hypothetical
        # effect replaces the earlier observed/predicted value in this branch.
        state[fact_key(subject, predicate)] = {
            "value": value, "source": "predicted_effect", "after_step": step_id,
            "conditional": conditional,
        }
        applied.append({"subject": subject, "predicate": predicate, "value": value,
                        "provenance": "predicted_effect", "conditional": conditional})
    return applied


def flatten_steps(skill_plan: dict[str, Any]) -> list[dict[str, Any]]:
    return [step for subtask in skill_plan.get("subtask_plans", []) for step in subtask.get("steps", [])]


def goal_skill_match(alignment: dict[str, Any], library: dict[str, Any],
                     steps: list[dict[str, Any]], observed: dict[tuple[str, str], dict[str, Any]]) -> list[dict[str, Any]]:
    """Unify grounded binary goals with declared skill effects, never with skill names."""
    matches = []
    for goal in alignment["task_decomposition"].get("goal_state", []):
        instances = []
        subjects = role_ids(alignment, goal["subject_role"])
        objects = role_ids(alignment, goal["object_role"])
        for subject in subjects:
            # Quantity roles may contain several possible targets. A particular
            # subject needs one matching object, not every object in the role.
            candidates = []
            selected = []
            for object_id in objects:
                for skill in library["skills"]:
                    for effect in skill.get("effects", []):
                        if effect.get("update") or effect.get("predicate") != goal["relation"]:
                            continue
                        lhs, rhs = effect.get("subject"), effect.get("object")
                        if not (isinstance(lhs, str) and isinstance(rhs, str)):
                            continue
                        bindings = {}
                        compatible = True
                        for symbol, role, instance in ((lhs, goal["subject_role"], subject),
                                                       (rhs, goal["object_role"], object_id)):
                            if symbol.startswith("$"):
                                prior = bindings.get(symbol[1:])
                                if prior is not None and prior != (role, instance):
                                    compatible = False
                                bindings[symbol[1:]] = (role, instance)
                            elif symbol != instance:
                                compatible = False
                        if not compatible:
                            continue
                        required = {p["name"] for p in skill["parameters"] if p.get("required")}
                        if not required.issubset(bindings):
                            continue
                        arguments = {name: {"role_id": role, "element_id": node_id}
                                     for name, (role, node_id) in bindings.items()}
                        candidates.append({"skill_id": skill["skill_id"], "arguments": arguments,
                                           "effect": {"subject": subject, "predicate": goal["relation"],
                                                      "object": object_id}})
                        for step in steps:
                            if step["skill_id"] == skill["skill_id"] and all(
                                step["arguments"].get(name) == arg for name, arg in arguments.items()
                            ):
                                selected.append(step["step_id"])
            already_observed = any(
                observed.get(fact_key(subject, goal["relation"]), {}).get("value") == object_id
                for object_id in objects
            )
            instances.append({"subject_id": subject, "target_ids": objects,
                              "library_candidates": candidates,
                              "matched_step_ids": sorted(set(selected)),
                              "status": "observed_initially" if already_observed else (
                                  "matched_in_plan" if selected else (
                                      "not_expressible_by_library" if not candidates else "not_selected_in_plan"))})
        matches.append({"constraint_id": goal["constraint_id"], "required": goal.get("required", True),
                        "instances": instances,
                        "status": "covered" if instances and all(i["status"] in {"observed_initially", "matched_in_plan"}
                                                                  for i in instances) else "unresolved"})
    return matches


def conservative_optimize(steps: list[dict[str, Any]], removable_ids=None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Only adjacent identical, side-effect-idempotent calls may be collapsed."""
    idempotent = {"navigate_to_object", "navigate_to_location", "position_arm_for_object"}
    output = []
    removals = []
    for step in steps:
        previous = output[-1] if output else None
        if (previous is not None and step["skill_id"] in idempotent
                and (removable_ids is None or step["step_id"] in removable_ids)
                and previous["skill_id"] == step["skill_id"]
                and previous["arguments"] == step["arguments"]):
            removals.append({"step_id": step["step_id"], "duplicate_of": previous["step_id"],
                             "assumption": "The immediately preceding identical skill succeeds."})
        else:
            output.append(step)
    return output, removals


def review_redundancy(steps: list[dict[str, Any]], library: dict[str, Any]) -> dict[str, Any]:
    """Flag non-local savings without silently changing a safety-relevant plan."""
    skills = {skill["skill_id"]: skill for skill in library["skills"]}
    navigation = []
    for step in steps:
        if step["skill_id"] == "navigate_to_object":
            target = step.get("arguments", {}).get("target", {})
            if isinstance(target, dict):
                navigation.append({"step_id": step["step_id"], "target_id": target.get("element_id")})
    revisits = []
    for i, entry in enumerate(navigation):
        if entry["target_id"] is not None and any(
            prior["target_id"] == entry["target_id"] for prior in navigation[:i]
        ):
            revisits.append({"step_id": entry["step_id"], "target_id": entry["target_id"],
                             "classification": "review_route_order_not_safe_to_delete_automatically"})
    used_predicates = {condition.get("predicate")
                       for skill in skills.values() for condition in skill.get("preconditions", [])
                       if "predicate" in condition}
    inspection_steps = [step["step_id"] for step in steps
                        if step["skill_id"] == "inspect_object"]
    unused_inspection = inspection_steps if "inspected" not in used_predicates else []
    route = [entry["target_id"] for entry in navigation]
    transitions = sum(a != b for a, b in zip(route, route[1:]))
    return {
        "navigation_route": route,
        "inter_target_transitions": transitions,
        "navigation_revisits": revisits,
        "inspection_steps_without_symbolic_consumers": unused_inspection,
        "recommendation": (
            "Consider grouping inspection and manipulation at each target to reduce revisits, "
            "but retain early inspections if they establish identity, safety, or affordances. "
            "No non-adjacent step was automatically removed."
        ),
        "automatic_non_adjacent_removals": 0,
    }


def review(alignment: dict[str, Any], skill_plan: dict[str, Any], library: dict[str, Any],
           nodes: list[dict[str, Any]], edges: list[dict[str, Any]],
           external_observations: dict[str, Any] | None = None) -> dict[str, Any]:
    if alignment.get("status") != "success":
        raise ValueError("object alignment must be successful")
    if skill_plan.get("status") != "proposal_needs_environment_validation":
        raise ValueError("skill plan must be a structurally valid proposal")
    from action_sequence import validate_sequence, removable_navigation_ids
    validate_sequence(skill_plan)
    skills = {item["skill_id"]: item for item in library["skills"]}
    observed = observed_state(alignment, nodes, edges)
    if external_observations is not None:
        known_ids = {int(node["id"]) for node in nodes}
        predicates = set(library.get("predicate_vocabulary", [])) | {"support_surface"}
        for fact in validate_observations(external_observations, known_ids, predicates):
            observed[fact_key(fact["subject"], fact["predicate"])] = {
                "value": fact["value"], "source": fact["source"],
                "observed_at": fact["observed_at"],
            }
    initial_checks = check_initial_claims(alignment, observed)
    state = dict(observed)
    original_steps = flatten_steps(skill_plan)
    removable = removable_navigation_ids(skill_plan) if skill_plan.get("sequence_policy") else None
    optimized, removals = conservative_optimize(original_steps, removable)
    skill_alignment = goal_skill_match(alignment, library, optimized, observed)
    trace = []
    failed = False
    conditional_chain = False
    for step in optimized:
        skill = skills.get(step["skill_id"])
        if skill is None:
            raise ValueError(f"unknown skill {step['skill_id']}")
        checks = [check_condition(c, step["arguments"], state)
                  for c in skill.get("preconditions", [])]
        for arg_name, required in skill.get("required_affordances", {}).items():
            for affordance in required:
                subject = resolve_symbol("$" + arg_name, step["arguments"])
                checks.append({"subject": subject, "predicate": affordance,
                               "expected": True,
                               **lookup(state, subject, affordance, True)})
        blocked = any(c["status"] == "false" for c in checks)
        unknown = any(c["status"] == "unknown" for c in checks)
        if blocked:
            failed = True
        conditional_chain = conditional_chain or unknown
        effects = [] if blocked else apply_effects(skill, step["arguments"], state,
                                                   step["step_id"], conditional_chain)
        trace.append({"step_id": step["step_id"], "skill_id": step["skill_id"],
                      "preconditions": checks,
                      "symbolic_status": "blocked" if blocked else (
                          "conditional" if unknown or conditional_chain else "symbolically_admissible"),
                      "predicted_effects": effects,
                      "execution_status": "not_executed"})
    goals = []
    for goal in alignment["task_decomposition"].get("goal_state", []):
        subjects = role_ids(alignment, goal["subject_role"])
        objects = role_ids(alignment, goal["object_role"])
        evaluations = []
        for subject in subjects:
            fact = state.get(fact_key(subject, goal["relation"]))
            if fact is None or fact["value"] not in objects:
                evaluations.append("not_supported")
            elif fact["source"] == "predicted_effect":
                evaluations.append("predicted_only")
            else:
                evaluations.append("observed_initially")
        status = "not_supported" if not evaluations or "not_supported" in evaluations else (
            "predicted_only" if "predicted_only" in evaluations else "observed_initially")
        goals.append({"constraint_id": goal["constraint_id"], "status": status,
                      "required": goal.get("required", True)})
    required_missing = not goals or any(g["required"] and g["status"] == "not_supported" for g in goals)
    required_unmatched = any(g["required"] and g["status"] == "unresolved" for g in skill_alignment)
    status = "invalid_symbolic_plan" if failed or required_missing or required_unmatched else "conditionally_valid_symbolic_plan"
    bound_subjects = {str(i) for role in alignment["context_bindings"].values()
                      for i in role["element_ids"]} | {"robot"}
    observed_facts = [{"subject": s, "predicate": p, **v}
                      for (s, p), v in observed.items()
                      if s in bound_subjects]
    predicted_facts = [{"subject": s, "predicate": p, **v}
                       for (s, p), v in state.items() if v["source"] == "predicted_effect"]
    return {"schema_version": "1.0", "status": status,
            "claim": "Symbolic predictions assume prior skills succeed; no action was executed.",
            "observed_state": {"facts": observed_facts, "initial_claim_checks": initial_checks},
            "plan_optimization": {"original_steps": len(original_steps),
                                  "retained_steps": len(optimized), "removed_adjacent_duplicates": removals,
                                  "policy": "Only adjacent identical idempotent calls are removed; non-adjacent travel and inspections are retained.",
                                  "redundancy_review": review_redundancy(optimized, library)},
            "symbolic_trace": trace,
            "goal_skill_alignment": skill_alignment,
            "predicted_state": {"facts": predicted_facts, "not_observed": True},
            "goal_check": goals,
            "physical_feasibility": "unknown"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Conservative symbolic plan review")
    parser.add_argument("--alignment-json", type=Path, required=True)
    parser.add_argument("--skill-plan-json", type=Path, required=True)
    parser.add_argument("--skill-library", type=Path, required=True)
    parser.add_argument("--scene-dir", type=Path, required=True)
    parser.add_argument("--observations-json", type=Path,
                        help="Optional timestamped robot/simulator observation snapshot")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--redundancy-output", type=Path,
                        help="Optional separate report for route and inspection redundancy")
    args = parser.parse_args()
    result = review(read_json(args.alignment_json), read_json(args.skill_plan_json),
                    read_json(args.skill_library),
                    read_json(args.scene_dir / "scene_graph.json"),
                    read_json(args.scene_dir / "scene_graph_edges.json"),
                    read_json(args.observations_json) if args.observations_json else None)
    write_json(args.output, result)
    if args.redundancy_output:
        write_json(args.redundancy_output, {
            "schema_version": "1.0",
            "status": "review_only",
            "source_review": str(args.output),
            **result["plan_optimization"],
        })
    print(json.dumps(result, ensure_ascii=True, indent=2))
    return 0 if result["status"] == "conditionally_valid_symbolic_plan" else 2


if __name__ == "__main__":
    raise SystemExit(main())
