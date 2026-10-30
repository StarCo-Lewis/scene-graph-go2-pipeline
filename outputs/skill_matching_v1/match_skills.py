#!/usr/bin/env python3
"""Offline skill-plan proposal and conservative validation.

ProgPrompt inspires the constrained skill/object inventory and examples.
SayCan inspires separating usefulness from feasibility; no learned value
function or execution success probability is claimed here.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
import urllib.error
import urllib.request
import io
from pathlib import Path
from typing import Any
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'llm_audit_v1'))
from response_audit import ResponseAudit, response_text, validation as record_validation
from action_sequence import validate_subtask_dependencies, refresh_sequence, adapter_shape_check
from action_identity import assign_action_identity
from skill_output_normalization import normalize_and_record, NormalizationError


LOG = logging.getLogger("skill_matching")
DEFAULT_MODEL = "qwen3.8-flash"
DEFAULT_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


class PlanError(ValueError):
    pass


def request_json_with_retry(req, timeout=240, attempts=3, audit=None):
    if timeout <= 0 or not 1 <= attempts <= 3:
        raise PlanError("Request timeout must be positive; attempts must be in [1, 3]")
    for attempt in range(1, attempts + 1):
        LOG.info("LLM HTTP request attempt %d/%d timeout=%ss", attempt, attempts, timeout)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                if audit is None:
                    return json.load(response)
                raw = response_text(response)
                audit.received(attempt, raw)
                try:
                    return json.loads(raw)
                except (ValueError, TypeError) as exc:
                    audit.parsed(raw, 'rejected', exc)
                    raise
        except urllib.error.HTTPError as exc:
            body = exc.read() if audit is not None else None
            if audit is not None:
                audit.received(attempt, body, exc.code)
            if exc.code not in (408, 429, 500, 502, 503, 504):
                raise PlanError(f"LLM HTTP {exc.code}: {(body if body is not None else exc.read()).decode('utf-8', 'replace')[:500]}") from exc
            error = f"HTTP {exc.code}"
            exc.close()
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            if audit is not None:
                audit.transport_error(attempt, exc)
            error = str(exc)
        if attempt == attempts:
            raise PlanError(f"LLM request failed after {attempts} attempts: {error}")
        delay = 5 * attempt
        LOG.warning("Transient LLM request failure: %s; retrying in %ss. A repeated API request may incur additional cost.", error, delay)
        time.sleep(delay)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=True, indent=2) + "\n", encoding="ascii")


def compact_skills(library: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [{
        "skill_id": skill["skill_id"],
        "name": skill.get("name", skill["skill_id"]),
        "description": skill.get("description_en", skill.get("description_zh", "")),
        "required_affordances": skill.get("required_affordances", {}),
        "parameters": [{"name": p["name"], "type": p["type"], "required": p.get("required", False)}
                       for p in skill["parameters"]],
        "preconditions": skill.get("preconditions", []),
        "effects": skill.get("effects", []),
    } for skill in library["skills"]]
    for row, skill in zip(rows, library['skills']):
        if 'semantic_profile' in skill:
            row['semantic_profile'] = {k: skill['semantic_profile'][k]
                for k in ('aliases', 'not_equivalent')}
    return rows


def bindings_from_alignment(alignment: dict[str, Any]) -> dict[str, list[int]]:
    if alignment.get("status") != "success":
        return {}
    output = {}
    for role_id, item in alignment.get("context_bindings", {}).items():
        ids = item.get("element_ids", [])
        if not isinstance(ids, list) or not ids or any(not isinstance(i, int) for i in ids):
            raise PlanError(f"invalid binding for role {role_id}")
        output[role_id] = ids
    return output


def request_plan(alignment: dict[str, Any], library: dict[str, Any], model: str,
                 base_url: str, api_key_env: str, timeout=240, attempts=3,
                 validation_error=None, rejected_proposal=None, diagnostics_dir=None,
                 audit_context=None, retrieval=None, widen_inventory=False) -> dict[str, Any]:
    key = os.environ.get(api_key_env, "").strip()
    if not key:
        raise PlanError(f"{api_key_env} is not configured")
    decomposition = alignment["task_decomposition"]
    bindings = bindings_from_alignment(alignment)
    prompt = {
        "instruction": decomposition["instruction"],
        "subtasks": [{"subtask_id": s["subtask_id"], "description": s["description"],
                      "dependencies": s.get("dependencies", [])} for s in decomposition["subtasks"]],
        "goal_state": decomposition.get("goal_state", []),
        "available_objects": bindings,
        "available_skills": compact_skills(library),
        "execution_contract": library.get("execution_contract"),
        "example": {
            "subtask_id": "example_get_object",
            "intent": "Pick up a grounded object",
            "steps": [
                {"skill_id": "navigate_to_object", "arguments": {"target": {"role_id": "example_object", "element_id": 1}}},
                {"skill_id": "position_arm_for_object", "arguments": {"target": {"role_id": "example_object", "element_id": 1}}},
                {"skill_id": "grasp_object", "arguments": {"target": {"role_id": "example_object", "element_id": 1}}},
            ],
        },
    }
    allowed = {s['skill_id'] for s in library['skills']}
    if retrieval is not None:
        per_task = {s['subtask_id']: s['candidate_skill_ids'] for s in retrieval['subtasks']}
        if set(per_task) != {s['subtask_id'] for s in decomposition['subtasks']}:
            raise PlanError('Retrieval subtask IDs mismatch')
        candidates = set().union(*(set(ids) for ids in per_task.values()))
        if not candidates or not candidates <= allowed:
            raise PlanError('Invalid retrieved candidate IDs')
        prompt['skill_retrieval'] = {'candidate_skill_ids_per_subtask': per_task,
            'widened_inventory': widen_inventory,
            'rule': 'Use the per-subtask candidates unless widened_inventory is true. '
                    'Scores are text similarity, not probabilities. Preconditions and current state '
                    'still decide which prerequisites are necessary. Do not execute all candidates.'}
        if not widen_inventory:
            prompt['available_skills'] = [s for s in prompt['available_skills'] if s['skill_id'] in candidates]
            allowed = candidates
    if validation_error:
        prompt['previous_validation_error'] = validation_error
        prompt['correction_requirement'] = (
            'Generate the entire proposal again from the unchanged instruction, subtasks and current state. '
            'Use only exact available skill IDs and grounded arguments. Each subtask_plans entry is a '
            'subtask object; each steps entry is an atomic {skill_id,arguments} object, never a subtask '
            'or an empty object. The rejected response is deliberately omitted; do not copy its structure.')
        # Retain rejected_proposal in the caller/audit, not the correction prompt.
        # Recursive wrappers and empty placeholders must not become a retry example.
    prompt['example']['steps'] = [step for step in prompt['example']['steps'] if step['skill_id'] in allowed]
    navigation_only = allowed == {'navigate_to_object'} and bool(bindings)
    if navigation_only:
        # Avoid showing grasp skills or a nested solution wrapper for a
        # single-skill navigation adapter. This is schema guidance, not an
        # assertion that navigation will physically succeed.
        first_subtask = decomposition['subtasks'][0]['subtask_id']
        first_role = next(iter(bindings))
        first_id = bindings[first_role][0]
        prompt['example'] = {
            'schema_version': '1.0',
            'subtask_plans': [{'subtask_id': first_subtask, 'intent': 'Navigate to grounded target',
                              'steps': [{'skill_id': 'navigate_to_object',
                                         'arguments': {'target': {'role_id': first_role, 'element_id': first_id}}}]}],
        }
    else:
        prompt['example'] = {'schema_version': '1.0', 'subtask_plans': [prompt['example']]}
    prompt['required_subtask_plan_order'] = [s['subtask_id'] for s in decomposition['subtasks']]
    system = (
        "You are a robot skill-plan PROPOSAL module. Return JSON only, in English. "
        "Return {schema_version:'1.0',subtask_plans:[{subtask_id,intent,steps:[{skill_id,arguments}]}]}. "
        "Use exactly the given subtask IDs in required_subtask_plan_order, once each, without adding example IDs. "
        "Only the root contains subtask_plans. Only subtask objects contain subtask_id, intent and steps. "
        "Only atomic action objects occur inside steps; never nest subtask objects or another steps list there. "
        "Use only listed skill IDs verbatim; never rename or translate skill IDs. "
        "A subtask is an intent, NOT necessarily one action: emit as many ordered atomic steps as needed. "
        "Preserve intermediate requested actions, explicit visit order and necessary repeated skills. "
        "Never combine several skill IDs into one step or add a macro skill absent from the library. "
        "Do not infer that reaching the final goal alone satisfies intermediate instructions. "
        "Each step must directly contain skill_id and arguments; do not nest a solution, use type=navigate_to, "
        "add preconditions/effects, or return an empty object. "
        "Each scene-element argument must be {role_id,element_id}, where element_id is listed "
        "for that role in available_objects. Never invent an object, skill, state, affordance or success probability. "
        + ("The example demonstrates the exact schema using an available grounded object; "
           "choose the actual navigation target from the task goal. " if navigation_only else
           "The example demonstrates the complete root/subtask/action shape only, NOT a solution or initial state; "
           "its example subtask IDs and object are not available for this task. ")
        +
        "For repeated objects choose distinct available element IDs. Do not claim execution occurred. "
        "If no plan can be proposed, return an empty steps list for that subtask."
    )
    payload = {"model": model, "temperature": 0, "response_format": {"type": "json_object"}, "enable_thinking": False,
               "messages": [{"role": "system", "content": system},
                            {"role": "user", "content": json.dumps(prompt, ensure_ascii=True)}]}
    req = urllib.request.Request(base_url.rstrip("/") + "/chat/completions",
                                 data=json.dumps(payload).encode("utf-8"),
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                                 method="POST")
    LOG.info("Requesting skill proposal from %s", model)
    audit = ResponseAudit(diagnostics_dir, 'skill_matching', audit_context, req)
    content = None
    try:
        result = (request_json_with_retry(req, timeout, attempts, audit=audit) if diagnostics_dir is not None
                  else request_json_with_retry(req, timeout, attempts))
        content = result["choices"][0]["message"]["content"]
        raw_content = content
        content = content.strip()
        if content.startswith("```json"):
            content = content[7:].strip().removesuffix("```").strip()
        proposal = json.loads(content)
        audit.parsed(raw_content)
        return proposal
    except urllib.error.HTTPError as exc:
        raise PlanError(f"LLM HTTP {exc.code}: {exc.read().decode('utf-8', 'replace')[:500]}") from exc
    except (urllib.error.URLError, KeyError, IndexError, TypeError, AttributeError, json.JSONDecodeError) as exc:
        audit.parsed(content, 'rejected', exc)
        raise PlanError(f"LLM proposal failed: {exc}") from exc


def validate_proposal(proposal: dict[str, Any], alignment: dict[str, Any],
                      library: dict[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(proposal, dict) or not isinstance(proposal.get("subtask_plans"), list):
        raise PlanError("proposal requires subtask_plans array")
    expected = [s["subtask_id"] for s in alignment["task_decomposition"]["subtasks"]]
    try:
        validate_subtask_dependencies(alignment["task_decomposition"]["subtasks"])
    except ValueError as exc:
        raise PlanError(str(exc)) from exc
    plans = proposal["subtask_plans"]
    actual = [p.get("subtask_id") if isinstance(p, dict) else None for p in plans]
    if actual != expected:
        keys = [sorted(p) if isinstance(p, dict) else type(p).__name__ for p in plans]
        raise PlanError(f"subtask plans must match decomposition IDs and order exactly; "
                        f"expected={expected!r}; received={actual!r}; entry_keys={keys!r}. "
                        "subtask_plans entries require subtask_id/intent/steps, not direct atomic actions")
    skills = {s["skill_id"]: s for s in library["skills"]}
    bindings = bindings_from_alignment(alignment)
    for plan_index, plan in enumerate(plans):
        if not isinstance(plan.get("intent"), str) or not isinstance(plan.get("steps"), list):
            raise PlanError(f"invalid plan structure for {plan['subtask_id']}")
        for index, step in enumerate(plan["steps"], 1):
            path = f"subtask_plans[{plan_index}].steps[{index-1}]"
            if not isinstance(step, dict):
                raise PlanError(f"{path} must be an atomic action object; received {type(step).__name__}")
            wrappers = set(step) & {'subtask_id', 'steps', 'subtask_plans', 'solution'}
            if wrappers:
                raise PlanError(f"{path}: nested subtask/wrapper is not an atomic action; "
                                f"wrapper_keys={sorted(wrappers)!r}; use direct skill_id/arguments")
            skill_id = step.get('skill_id')
            if not isinstance(skill_id, str) or not skill_id.strip():
                raise PlanError(f"{path}.skill_id must be a nonempty string; received {skill_id!r}; "
                                f"available keys={sorted(step)!r}; empty actions are invalid")
            if skill_id not in skills:
                raise PlanError(f"unknown skill at {plan['subtask_id']} step {index}: {skill_id!r}; "
                                f"path={path}.skill_id; allowed={sorted(skills)!r}")
            spec = skills[step["skill_id"]]
            args = step.get("arguments")
            if not isinstance(args, dict):
                raise PlanError(f"arguments must be an object at {plan['subtask_id']} step {index}")
            params = {p["name"]: p for p in spec["parameters"]}
            if set(args) - set(params):
                raise PlanError(f"unknown parameters for {step['skill_id']}: {sorted(set(args)-set(params))}")
            for name, parameter in params.items():
                if parameter.get("required") and name not in args:
                    raise PlanError(f"missing required parameter {name} for {step['skill_id']}")
                if name not in args:
                    continue
                value = args[name]
                if parameter["type"] in {"scene_element", "location", "surface", "container", "person", "control"}:
                    if not isinstance(value, dict) or set(value) != {"role_id", "element_id"}:
                        raise PlanError(f"{step['skill_id']}.{name} must be a role_id/element_id binding")
                    if value["element_id"] not in bindings.get(value["role_id"], []):
                        raise PlanError(f"ungrounded object in {step['skill_id']}.{name}")
                elif parameter["type"] in {"number", "height_m"}:
                    if isinstance(value, bool) or not isinstance(value, (int, float)):
                        raise PlanError(f"{step['skill_id']}.{name} must be numeric")
                elif not isinstance(value, str) or not value.strip():
                    raise PlanError(f"{step['skill_id']}.{name} must be a nonempty string")
    return plans


def deterministic_navigation_proposal(alignment: dict[str, Any], library: dict[str, Any]) -> dict[str, Any] | None:
    """Compile the uniquely grounded one-skill navigation case, else defer to LLM."""
    contract = library.get('execution_contract') or {}
    if (contract.get('adapter_mode') != 'navigation' or
            contract.get('supported_ordered_sequence') != ['navigate_to_object'] or
            [s.get('skill_id') for s in library.get('skills', [])] != ['navigate_to_object']):
        return None
    decomposition = alignment['task_decomposition']
    instruction = decomposition.get('instruction', '')
    # Do not collapse requests for inspection/manipulation into navigation.
    if not isinstance(instruction, str) or re.search(
            r'\b(inspect|check|verify|measure|scan|photograph|grasp|pick up|lift|place|deliver|carry)\b',
            instruction, flags=re.IGNORECASE):
        return None
    subtasks = decomposition.get('subtasks', [])
    goals = [g for g in decomposition.get('goal_state', []) if g.get('required', True)]
    if len(subtasks) != 1 or len(goals) != 1:
        return None
    goal = goals[0]
    if goal.get('subject_role') != 'robot' or goal.get('relation') != 'near':
        return None
    role_id = goal.get('object_role')
    ids = bindings_from_alignment(alignment).get(role_id, [])
    if len(ids) != 1:
        return None
    for constraint in decomposition.get('initial_scene_constraints', []):
        if constraint.get('required', True) and constraint.get('subject_role') != 'robot':
            checks = alignment.get('alignment', {}).get('relation_checks', [])
            if not any(c.get('constraint_id') == constraint.get('constraint_id') and
                       c.get('status') == 'observed_true' for c in checks):
                return None
    proposal = {'schema_version': '1.0', 'subtask_plans': [{
        'subtask_id': subtasks[0]['subtask_id'],
        'intent': subtasks[0].get('description', 'Navigate to grounded target'),
        'steps': [{'skill_id': 'navigate_to_object',
                   'arguments': {'target': {'role_id': role_id, 'element_id': ids[0]}}}],
    }]}
    validate_proposal(proposal, alignment, library)
    return proposal


def check_preconditions(skill: dict[str, Any], arguments: dict[str, Any]) -> list[dict[str, str]]:
    """Only grounded existence is known; all physical/dynamic facts are unknown."""
    checks = []
    for condition in skill.get("preconditions", []):
        if "any_of" in condition:
            checks.append({"predicate": "any_of", "status": "unknown", "reason": "dynamic_state_unavailable"})
            continue
        subject = condition.get("subject", "")
        predicate = condition.get("predicate", "")
        if subject.startswith("$"):
            value = arguments.get(subject[1:])
            if predicate == "exists" and isinstance(value, dict) and value.get("element_id") is not None:
                checks.append({"predicate": predicate, "status": "true", "reason": "grounded_scene_node"})
                continue
        checks.append({"predicate": predicate, "status": "unknown", "reason": "environment_state_unavailable"})
    for arg_name, affordances in skill.get("required_affordances", {}).items():
        for affordance in affordances:
            checks.append({"predicate": affordance, "status": "unknown", "reason": "affordance_not_observed"})
    return checks


def build_result(alignment: dict[str, Any], library: dict[str, Any],
                 proposal: dict[str, Any] | None) -> dict[str, Any]:
    if alignment.get("status") != "success":
        return {"schema_version": "1.0", "status": "unresolved", "stage": "object_alignment",
                "reason": "Required objects were not fully grounded; skill proposal was skipped.",
                "source_error": alignment.get("error")}
    if proposal is None:
        raise PlanError("proposal required after successful object alignment")
    plans = validate_proposal(proposal, alignment, library)
    skills = {s["skill_id"]: s for s in library["skills"]}
    reviewed = []
    empty_subtasks = []
    for plan in plans:
        if not plan["steps"]:
            empty_subtasks.append(plan["subtask_id"])
        steps = []
        for index, step in enumerate(plan["steps"], 1):
            spec = skills[step["skill_id"]]
            checks = check_preconditions(spec, step["arguments"])
            steps.append({"step_id": f"{plan['subtask_id']}_{index:02d}",
                          "skill_id": step["skill_id"], "arguments": step["arguments"],
                          "precondition_checks": checks,
                          "feasibility": "unknown" if any(c["status"] == "unknown" for c in checks) else "symbolically_admissible",
                          "execution_status": "not_executed"})
        subtask = alignment["task_decomposition"]["subtasks"][len(reviewed)]
        reviewed.append({"subtask_id": plan["subtask_id"], "intent": plan["intent"],
                         "dependencies": subtask.get("dependencies", []), "steps": steps})
    status = "unresolved" if empty_subtasks else "proposal_needs_environment_validation"
    result = {"schema_version": "1.0", "status": status,
            "method": "ProgPrompt-inspired constrained proposal with conservative SayCan-inspired feasibility checks",
            "claim": "No learned value function, success probability, robot execution, or state transition is asserted.",
            "instruction": alignment["task_decomposition"].get("instruction"),
            "unplanned_subtasks": empty_subtasks,
            "subtask_plans": reviewed,
            "deferred": {"initial_scene_constraints": alignment["task_decomposition"].get("initial_scene_constraints", []),
                         "goal_state": alignment["task_decomposition"].get("goal_state", [])}}
    result = assign_action_identity(result,(library.get('execution_contract') or {}).get('action_identity_scope'))
    result = refresh_sequence(result)
    result["adapter_support"] = adapter_shape_check(result, library.get("execution_contract"))
    LOG.info("Matched %d subtasks to %d ordered atomic actions; adapter=%s",
             len(reviewed), result["action_count"], result["adapter_support"]["status"])
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Propose and validate grounded atomic-skill plans")
    parser.add_argument("--alignment-json", type=Path, required=True)
    parser.add_argument("--skill-library", type=Path, required=True)
    parser.add_argument("--proposal-json", type=Path, help="Offline proposal; skips LLM")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--log-file", type=Path)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=DEFAULT_URL)
    parser.add_argument("--api-key-env", default="DASHSCOPE_API_KEY")
    parser.add_argument("--request-timeout", type=int, default=240)
    parser.add_argument("--request-attempts", type=int, default=3)
    args = parser.parse_args()
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if args.log_file:
        args.log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(args.log_file, encoding="utf-8"))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", handlers=handlers)
    try:
        alignment = read_json(args.alignment_json)
        library = read_json(args.skill_library)
        if alignment.get("status") != "success":
            LOG.info("Object alignment unresolved; skipping LLM and skill matching")
            result = build_result(alignment, library, None)
        else:
            deterministic_used = False
            normalization = None
            if args.proposal_json:
                proposal = read_json(args.proposal_json)
            else:
                proposal = deterministic_navigation_proposal(alignment, library)
                if proposal is not None:
                    deterministic_used = True
                    LOG.info('Compiled unique grounded navigation skill without LLM skill-proposal request')
                    write_json(args.output.parent / 'deterministic_navigation_proposal.json', proposal)
                else:
                    validation_error = None
                    rejected_proposal = None
                    for schema_attempt in range(1, 4):
                        proposal = request_plan(alignment, library, args.model, args.base_url, args.api_key_env,
                                                args.request_timeout, args.request_attempts,
                                                diagnostics_dir=args.output.parent, audit_context={'validation_attempt': schema_attempt},
                                                validation_error=validation_error, rejected_proposal=rejected_proposal)
                        write_json(args.output.parent / f'llm_skill_proposal_attempt_{schema_attempt}.json', proposal)
                        try:
                            proposal, normalization = normalize_and_record(proposal, 'nested_json', library,
                                args.output.parent, f'llm_skill_attempt_{schema_attempt}',
                                f'llm_skill_proposal_attempt_{schema_attempt}.json')
                            validate_proposal(proposal, alignment, library)
                            record_validation(args.output.parent, 'skill_matching', schema_attempt, 'accepted')
                            break
                        except (PlanError, NormalizationError) as exc:
                            validation_error = str(exc)
                            record_validation(args.output.parent, 'skill_matching', schema_attempt, 'rejected', exc,
                                              evidence=f'llm_skill_proposal_attempt_{schema_attempt}.json')
                            rejected_proposal = proposal
                            LOG.warning('Rejected skill proposal attempt %d/3: %s', schema_attempt, exc)
                            if schema_attempt == 3:
                                raise
            result = build_result(alignment, library, proposal)
            if normalization is not None:
                result['format_normalization'] = normalization
            if deterministic_used:
                result['proposal_origin'] = 'deterministic_grounded_navigation'
                result['method'] = 'Deterministic grounded single-skill navigation with symbolic feasibility checks'
        write_json(args.output, result)
        print(json.dumps(result, ensure_ascii=True, indent=2))
        return 2 if result["status"] == "unresolved" else 0
    except (PlanError, OSError, ValueError, KeyError) as exc:
        LOG.error("Skill matching failed: %s", exc)
        result = {"schema_version": "1.0", "status": "error", "error": {"code": "SKILL_PLAN_ERROR", "message": str(exc)}}
        write_json(args.output, result)
        print(json.dumps(result, ensure_ascii=True, indent=2), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
