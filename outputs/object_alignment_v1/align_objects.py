#!/usr/bin/env python3
"""High-level task decomposition and open-vocabulary ConceptGraphs alignment.

This stage intentionally stops after binding task object requirements to scene
graph nodes. It never reads SceneSmith ground truth and never invents objects.
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import math
import os
import pickle
import re
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from llm_transport import request_json
from response_audit import ResponseAudit, validation as record_validation
from matrix_type_contract import MODES, check_mode, response_format, enforce_and_record
from decomposition_wire import FLAT_PROMPT, from_flat_roles
from structural_matrix import (DECOMPOSITION_PROMPT, from_decomposition_matrix,
                               matrix_contract_context, make_binding_codebook, from_binding_matrix)
from runtime_goal_protocol import STATE_RELATIONS, PROMPT as STATE_GOAL_PROMPT, normalize_state_goal, validate_state_goal_set


ERROR_MESSAGE = "NO object can be used in this task"
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen3.8-flash"
LOGGER = logging.getLogger("task_object_alignment")


class AlignmentError(RuntimeError):
    pass


def preserve_original_instruction(decomposition, instruction):
    """User instruction is authoritative; retain model rewrite for diagnostics."""
    if decomposition.get('instruction') != instruction:
        decomposition['model_instruction'] = decomposition.get('instruction')
    decomposition['instruction'] = instruction
    return decomposition


def validate_planning_capability_claims(decomposition, execution_contract=None):
    """Reject confusion between graph checks and physical execution skills."""
    capabilities = (execution_contract or {}).get('planning_capabilities', {})
    if capabilities.get('validate_supplied_on_in_edges') is not True:
        return decomposition
    relevant = any(c.get('relation') in ('on', 'in') and c.get('required', True)
                   for c in decomposition.get('initial_scene_constraints', []))
    if not relevant:
        return decomposition
    for subtask in decomposition.get('subtasks', []):
        description = str(subtask.get('description', '')).lower()
        mentions_graph_check = re.search(r'\b(edge|edges|scene.graph relation|graph relation|observed relation|relation from graph)\b', description)
        denies_execution = re.search(r'\b(not supported|unsupported|cannot|can.t|unable|unresolved)\b.{0,100}\b(execution|navigation|skill|robot)\b', description)
        if mentions_graph_check and denies_execution:
            raise AlignmentError(
                f"{subtask.get('subtask_id')}: supplied-edge relation validation is a planning capability, "
                "not an unsupported physical execution skill; correct the description without dropping the relation"
            )
    return decomposition


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Decompose a high-level instruction and align required objects."
    )
    parser.add_argument("--scene-dir", type=Path, required=True)
    parser.add_argument("--matching-mode", choices=("clip", "llm_graph"), default="clip")
    parser.add_argument("--binding-proposal-json", type=Path,
                        help="Offline fixture only; not a real LLM binding result")
    parser.add_argument("--instruction")
    parser.add_argument("--decomposition-json", type=Path)
    parser.add_argument('--identifier-allocation-json',type=Path,
                        help='Saved program allocation required for offline typed matrix replay only')
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--decomposition-output", type=Path)
    parser.add_argument("--log-file", type=Path)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--api-key-env", default="DASHSCOPE_API_KEY")
    parser.add_argument("--request-timeout", type=float, default=180)
    parser.add_argument("--request-attempts", type=int, default=1)
    parser.add_argument("--validation-attempts",type=int,choices=(1,2,3),default=3,
                        help="LLM output validation attempts; 1 records errors without corrected proposals")
    parser.add_argument("--decomposition-format", choices=("nested", "flat_roles", "matrix"), default="nested")
    parser.add_argument("--binding-format", choices=("nested_json", "matrix"), default="nested_json")
    parser.add_argument('--matrix-contract', choices=MODES, default='strict_schema',
                        help='Matrix generation/type policy; no silent provider fallback')
    parser.add_argument("--execution-contract", type=Path,
                        help="Optional executor capability context; does not assert scene state")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--min-score", type=float, default=0.24)
    parser.add_argument("--min-metadata-score", type=float, default=0.20)
    parser.add_argument("--visual-weight", type=float, default=0.65)
    parser.add_argument("--metadata-weight", type=float, default=0.35)
    args = parser.parse_args()
    if not math.isfinite(args.request_timeout) or args.request_timeout <= 0 or not 1 <= args.request_attempts <= 3:
        parser.error("Positive request timeout and 1..3 request attempts required")
    if bool(args.instruction) == bool(args.decomposition_json):
        parser.error("provide exactly one of --instruction or --decomposition-json")
    if args.top_k < 1:
        parser.error("--top-k must be positive")
    if not math.isclose(args.visual_weight + args.metadata_weight, 1.0, abs_tol=1e-6):
        parser.error("visual and metadata weights must sum to 1")
    if not 0.0 <= args.min_score <= 1.0:
        parser.error("--min-score must be in [0, 1]")
    if not 0.0 <= args.min_metadata_score <= 1.0:
        parser.error("--min-metadata-score must be in [0, 1]")
    return args


def configure_logging(log_file: Path | None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if log_file is not None:
        log_file = log_file.resolve()
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=handlers,
        force=True,
    )


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=True, indent=2) + "\n", encoding="ascii")


def strip_json_fence(text: str) -> str:
    text = text.strip()
    match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else text


def call_llm(
    instruction: str,
    model: str,
    base_url: str,
    api_key_env: str,
    max_attempts: int = 3,
    execution_contract=None,
    diagnostics_dir=None,
    request_timeout=180,
    request_attempts=1,
    decomposition_format="nested",
    matrix_contract="legacy_json",
) -> dict[str, Any]:
    check_mode(matrix_contract)
    typed_version='structural_typed5_v1' if matrix_contract=='typed5_schema' else 'structural_typed9_v1'
    if matrix_contract in ('typed9_schema','typed5_schema') and decomposition_format!='matrix':
        raise AlignmentError('Typed identifiers require matrix decomposition')
    api_key = os.environ.get(api_key_env, "").strip()
    if not api_key:
        raise AlignmentError(f"{api_key_env} is not configured")

    system_prompt = """You are a hierarchical task-analysis module for a mobile manipulator.
Convert one high-level instruction into concise, ordered subtasks and object requirements.
Return JSON only. Never output scene object IDs and never claim that an object exists.
Use stable task-level role_id values when the same physical object is reused.
Each required_roles entry MUST include the exact key "role_id" with a nonempty
string such as "electric_kettle". Do not substitute "id", "role", or "object_id",
and do not use numbers, arrays, null, or blank strings as role_id.
All JSON keys and values must be in English. Translate the input when needed.
The root instruction field must contain the normalized English translation, not
the original non-English text. Each object requirement must contain role_id,
name, semantic_query_en, semantic_labels, quantity, and required.
semantic_labels is a short list of English object-category names that would be
valid for the role, such as ["cup", "drinking cup", "tumbler"]. Do not include
locations, adjectives, or unrelated fallback categories. relation_hints is a list
of {relation, reference_role, required}; use only on, in, near, or none.
Each subtask must contain subtask_id, description, dependencies, and required_roles.
The root must contain schema_version, instruction, task_name, subtasks,
initial_scene_constraints, and goal_state. Each constraint contains constraint_id,
subject_role, relation, object_role, and required. Initial constraints describe the
world before execution. Goal state describes the desired world after execution.
Never put a goal relation into initial_scene_constraints.
Do not infer an initial location merely from the instruction's requested destination.
If the initial relation is not stated by the instruction, return [] for
initial_scene_constraints. Unknown initial state must stay unknown.
Do not generate robot skills in this stage.
Checking an initial on/in relation against supplied scene-graph edges is a
planning/grounding operation, not a robot action. Never describe this check as
unavailable merely because the execution contract supports only navigation or
proxy manipulation. If edges are absent, keep the relation unknown; do not
claim it is observed. Physical inspection remains unsupported unless the
execution contract explicitly provides an inspection skill.
goal_state must contain at least one explicit required completion condition.
For navigation tasks, use subject_role "robot", relation "near", and the
destination object's role_id as object_role. "robot" is a reserved executor
symbol, not a scene object: never include it in required_roles or request
scene grounding for it. Example: {"constraint_id":"goal_01",
"subject_role":"robot","relation":"near","object_role":"sideboard","required":true}.

Follow this shape exactly (required_roles contains objects, never strings):
{
  "schema_version": "1.0",
  "instruction": "Normalized English instruction",
  "task_name": "Short English task name",
  "subtasks": [{
    "subtask_id": "st_01",
    "description": "English subtask description",
    "dependencies": [],
    "required_roles": [{
      "role_id": "stable_snake_case_id",
      "name": "English object name",
      "semantic_query_en": "English visual-semantic query",
      "semantic_labels": ["object category", "valid synonym"],
      "quantity": 1,
      "required": true
    }]
  }],
  "initial_scene_constraints": [{
    "constraint_id": "init_01", "subject_role": "object_role_id",
    "relation": "on", "object_role": "support_role_id", "required": true
  }],
  "goal_state": [{
    "constraint_id": "goal_01", "subject_role": "object_role_id",
    "relation": "on", "object_role": "destination_role_id", "required": true
  }]
}"""
    system_prompt += '\n' + STATE_GOAL_PROMPT
    if decomposition_format == "flat_roles":
        system_prompt = FLAT_PROMPT
    elif decomposition_format == "matrix":
        system_prompt = DECOMPOSITION_PROMPT
    elif decomposition_format != "nested":
        raise AlignmentError("Unsupported decomposition wire format")
    identifier_book=None
    if matrix_contract in ('typed9_schema','typed5_schema'):
        from typed_matrix_ids import allocation, decomp_prompt
        digits=5 if matrix_contract=='typed5_schema' else 9
        identifier_book=allocation(digits=digits)
        system_prompt=decomp_prompt(system_prompt,digits)
        if diagnostics_dir is not None:
            write_json(Path(diagnostics_dir)/'identifier_allocation.json',identifier_book)
    user_data={"instruction": instruction,
                              "execution_contract": (matrix_contract_context(execution_contract,
                                  typed_version if identifier_book is not None else 'structural_matrix_v1')
                                  if decomposition_format == 'matrix' else execution_contract)}
    if identifier_book is not None:
        user_data['identifier_allocation']=identifier_book
    user_prompt = json.dumps(user_data, ensure_ascii=False)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    last_error = "unknown schema error"
    for attempt in range(1, max_attempts + 1):
        payload = {
            "model": model,
            "temperature": 0,
            "response_format": (response_format('decomposition', matrix_contract,identifier_book)
                                if decomposition_format == 'matrix' else {'type':'json_object'}),
            "enable_thinking": False,
            "messages": messages,
        }
        request = urllib.request.Request(
            base_url.rstrip("/") + "/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        LOGGER.info("Calling LLM decomposition model: %s (attempt %d/%d)", model, attempt, max_attempts)
        content = "{}"
        response_audit = ResponseAudit(diagnostics_dir, 'decomposition', {'validation_attempt': attempt}, request)
        try:
            result = (request_json(request, timeout=request_timeout, attempts=request_attempts, audit=response_audit)
                      if diagnostics_dir is not None else request_json(request, timeout=request_timeout, attempts=request_attempts))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")[:1000]
            raise AlignmentError(f"LLM HTTP {exc.code}: {body}") from exc
        except urllib.error.URLError as exc:
            raise AlignmentError(f"LLM request failed: {exc.reason}") from exc
        try:
            content = result["choices"][0]["message"]["content"]
            if diagnostics_dir is not None:
                write_json(Path(diagnostics_dir) / f"llm_decomposition_attempt_{attempt}.json",
                           response_audit.clean({"attempt": attempt, "model": model, "response_content": content,
                            "decomposition_wire_format": decomposition_format,
                            "matrix_contract": matrix_contract if decomposition_format == 'matrix' else None,
                            "finish_reason": result["choices"][0].get("finish_reason"),
                            "usage": result.get("usage"), "response_model": result.get("model"),
                            "scope": "Raw model content only; credential redaction; no API key or request headers"}))
            if result["choices"][0].get("finish_reason") == "length":
                raise AlignmentError("LLM response truncated (finish_reason=length); incomplete output rejected")
            candidate = json.loads(content if decomposition_format=='matrix' and matrix_contract in ('strict_schema','typed9_schema','typed5_schema')
                                   else strip_json_fence(content))
            if decomposition_format == "flat_roles":
                candidate = from_flat_roles(candidate)
            elif decomposition_format == "matrix":
                candidate, type_record = enforce_and_record(candidate, 'decomposition', matrix_contract,
                    diagnostics_dir, f'llm_decomposition_attempt_{attempt}',identifier_book)
                if identifier_book is not None:
                    from typed_matrix_ids import decode_decomposition
                    candidate=decode_decomposition(candidate,identifier_book)
                else:
                    candidate = from_decomposition_matrix(candidate)
                candidate['matrix_type_contract'] = dict(mode=matrix_contract,
                    version=type_record['version'], normalized=type_record['changed'])
                if diagnostics_dir is not None:
                    write_json(Path(diagnostics_dir) / f'llm_decomposition_decoded_attempt_{attempt}.json', candidate)
                    write_json(Path(diagnostics_dir) / f'decomposition_codebook_attempt_{attempt}.json', identifier_book if identifier_book is not None else {
                        'schema_version': 'structural_matrix_v1',
                        'roles': [[i, f'role_{i:03d}'] for i, _ in enumerate(json.loads(strip_json_fence(content))['roles'], 1)],
                        'tasks': [[i, f'st_{i:03d}'] for i, _ in enumerate(candidate['subtasks'], 1)],
                        'relations': [[1,'on'],[2,'in'],[3,'near'],[4,'held'],[5,'lifted']],
                        'scope': [[1,'initial_scene_constraints'],[2,'goal_state']],
                        'robot_subject_code': 0, 'source': 'explicit_model_definitions_not_scene_IDs'})
            validated = validate_decomposition(candidate)
            validate_planning_capability_claims(validated, execution_contract)
            if not validated["goal_state"] or not any(g["required"] for g in validated["goal_state"]):
                raise AlignmentError("goal_state must contain at least one required completion condition")
            response_audit.parsed(content)
            record_validation(diagnostics_dir, 'decomposition', attempt, 'accepted',
                              context={'request_id_local': response_audit.id})
            return validated
        except (KeyError, IndexError, TypeError, ValueError, AlignmentError) as exc:
            last_error = str(exc)
            response_audit.parsed(content, 'rejected', exc)
            record_validation(diagnostics_dir, 'decomposition', attempt, 'rejected', exc,
                              context={'request_id_local': response_audit.id})
            LOGGER.warning("Rejected LLM decomposition attempt %d: %s", attempt, last_error)
            if attempt < max_attempts:
                if decomposition_format == 'matrix':
                    messages = [{'role': 'system', 'content': system_prompt},
                        {'role': 'user', 'content': user_prompt + '\nPrevious output failed validation: '
                         + last_error + (f'\nReturn complete {typed_version} tables with the SAME allocation hash. '
                             if identifier_book is not None else '\nReturn complete structural_matrix_v1 tables from unchanged inputs. ')
                         + 'No flattening, invented references, deleted requirements or substitute plan.'}]
                    continue
                if decomposition_format == "flat_roles":
                    # Do not train the next completion on malformed recursive JSON.
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt + "\nPrevious output failed validation: "
                         + last_error + "\nGenerate a NEW complete flat object_roles table and subtasks reference table. "
                         "Do not output empty roles, required_roles, nested subtasks or an example answer. "
                         "Preserve the complete original instruction; do not infer missing objects."}]
                    continue
                messages.extend([
                    {"role": "assistant", "content": content if 'content' in locals() else "{}"},
                    {
                        "role": "user",
                        "content": (
                            "Your JSON failed validation: " + last_error
                            + ". Return a corrected complete JSON object only. "
                            "Every required_roles entry must be an object with a nonempty string role_id, "
                            "name, semantic_query_en, semantic_labels, quantity, and required. "
                            "Use the same role_id in required_roles and constraints. "
                            "Do not add an example object or silently drop user requirements. "
                            "Scene-graph relation validation is a planning operation; do not describe it "
                            "as an unsupported physical robot skill."
                        ),
                    },
                ])
    raise AlignmentError(f"LLM decomposition remained invalid after {max_attempts} attempts: {last_error}")


def validate_decomposition(data: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(data, dict) or not isinstance(data.get("subtasks"), list):
        raise AlignmentError("decomposition must contain a subtasks array")
    if not data["subtasks"]:
        raise AlignmentError("decomposition contains no subtasks")

    subtask_ids: set[str] = set()
    role_specs: dict[str, dict[str, Any]] = {}
    role_groups = []
    for index, subtask in enumerate(data["subtasks"], 1):
        sid = subtask.get("subtask_id")
        if not isinstance(sid, str) or not sid:
            raise AlignmentError(f"subtask {index} has invalid subtask_id")
        if sid in subtask_ids:
            raise AlignmentError(f"duplicate subtask_id: {sid}")
        subtask_ids.add(sid)
        if not isinstance(subtask.get("dependencies", []), list):
            raise AlignmentError(f"{sid}.dependencies must be an array")
        roles = subtask.get("required_roles")
        if not isinstance(roles, list):
            raise AlignmentError(f"{sid}.required_roles must be an array")
        role_groups.append((sid, roles))
    constraint_roles = data.get('constraint_roles', [])
    if not isinstance(constraint_roles, list):
        raise AlignmentError('constraint_roles must be an array')
    role_groups.append(('constraint_roles', constraint_roles))
    for sid, roles in role_groups:
        for role_index, role in enumerate(roles, 1):
            if not isinstance(role, dict):
                raise AlignmentError(
                    f"{sid}.required_roles[{role_index}] must be an object, not {type(role).__name__}"
                )
            role_id = role.get("role_id")
            query = role.get("semantic_query_en")
            if not isinstance(role_id, str) or not role_id:
                raise AlignmentError(f"{sid}.required_roles[{role_index}].role_id must be a nonempty string; "
                                     f"received {role_id!r}; available keys={sorted(role.keys())}")
            if role_id == "robot":
                raise AlignmentError("robot is a reserved executor, not a scene object requirement")
            if not isinstance(query, str) or not query.strip():
                raise AlignmentError(f"{sid}.{role_id} requires semantic_query_en")
            quantity = role.get("quantity", 1)
            if not isinstance(quantity, int) or quantity < 1:
                raise AlignmentError(f"{sid}.{role_id} has invalid quantity")
            role.setdefault("name", role_id)
            role.setdefault("required", True)
            labels = role.get("semantic_labels")
            if not isinstance(labels, list) or not labels or not all(
                isinstance(label, str) and label.strip() for label in labels
            ):
                raise AlignmentError(f"{sid}.{role_id} requires semantic_labels")
            existing = role_specs.get(role_id)
            normalized = {
                "role_id": role_id,
                "name": role["name"],
                "semantic_query_en": query.strip(),
                "semantic_queries_en": [query.strip()],
                "semantic_labels": [str(label).strip().lower() for label in labels],
                "quantity": quantity,
                "required": bool(role["required"]),
            }
            if existing:
                if normalized["semantic_query_en"] not in existing["semantic_queries_en"]:
                    existing["semantic_queries_en"].append(normalized["semantic_query_en"])
                for label in normalized["semantic_labels"]:
                    if label not in existing["semantic_labels"]:
                        existing["semantic_labels"].append(label)
                existing["quantity"] = max(existing["quantity"], quantity)
                existing["required"] = existing["required"] or normalized["required"]
            else:
                role_specs[role_id] = normalized

    for subtask in data["subtasks"]:
        unknown = set(subtask.get("dependencies", [])) - subtask_ids
        if unknown:
            raise AlignmentError(f"{subtask['subtask_id']} has unknown dependencies: {sorted(unknown)}")
    assert_acyclic(data["subtasks"])
    data["task_roles"] = list(role_specs.values())
    data["object_requirements"] = list(role_specs.values())
    data["initial_scene_constraints"] = validate_constraints(
        data.get("initial_scene_constraints"), "initial_scene_constraints", set(role_specs)
    )
    data["goal_state"] = validate_constraints(
        data.get("goal_state"), "goal_state", set(role_specs)
    )
    action_role_ids = {role['role_id'] for subtask in data['subtasks']
                       for role in subtask['required_roles']}
    constraint_role_ids = [role['role_id'] for role in constraint_roles]
    if len(set(constraint_role_ids)) != len(constraint_role_ids) or action_role_ids.intersection(constraint_role_ids):
        raise AlignmentError('constraint_roles must define distinct constraint-only roles, not duplicate action roles')
    referenced = {constraint[key] for constraint in data['initial_scene_constraints'] + data['goal_state']
                  for key in ('subject_role', 'object_role') if constraint[key] != 'robot'}
    unused = sorted(set(constraint_role_ids) - referenced)
    if unused:
        raise AlignmentError(f'constraint_roles must be referenced by a validated constraint/goal; '
                             f'unreferenced_role_ids={unused}')
    if any(g['relation'] in STATE_RELATIONS and role_specs[g['object_role']]['quantity'] != 1
           for g in data['goal_state'] + data['initial_scene_constraints']):
        raise AlignmentError('State goals require a single-object role, quantity=1')
    return data


def validate_constraints(
    constraints: Any, field_name: str, role_ids: set[str]
) -> list[dict[str, Any]]:
    if not isinstance(constraints, list):
        raise AlignmentError(f"{field_name} must be an array")
    allowed_relations = {"on", "in", "near"}
    if field_name in ('goal_state','initial_scene_constraints'):
        allowed_relations |= STATE_RELATIONS
    normalized = []
    seen_ids = set()
    for index, item in enumerate(constraints, 1):
        if not isinstance(item, dict):
            raise AlignmentError(f"{field_name}[{index}] must be an object")
        constraint_id = item.get("constraint_id")
        subject = item.get("subject_role")
        relation = str(item.get("relation", "")).lower()
        object_role = item.get("object_role")
        if not isinstance(constraint_id, str) or not constraint_id or constraint_id in seen_ids:
            raise AlignmentError(f"{field_name}[{index}] has invalid or duplicate constraint_id")
        if (subject not in role_ids and subject != "robot") or object_role not in role_ids:
            raise AlignmentError(f"{constraint_id} references an unknown role")
        if subject == "robot" and relation not in ({"near"} | STATE_RELATIONS):
            raise AlignmentError(f"{constraint_id}: unsupported reserved robot predicate in {field_name}")
        if relation not in allowed_relations:
            raise AlignmentError(f"{constraint_id} has unsupported relation: {relation}")
        if type(item.get('required', True)) is not bool:
            raise AlignmentError(f'{constraint_id}: required must be a JSON boolean')
        seen_ids.add(constraint_id)
        value = {
            "constraint_id": constraint_id,
            "subject_role": subject,
            "relation": relation,
            "object_role": object_role,
            "required": item.get("required", True),
        }
        try:
            if relation in STATE_RELATIONS:
                # Same predicate schema, different time scope. Initial assertions
                # are checked against the input snapshot, never used as evidence.
                normalize_state_goal(dict(item, relation=relation))
                value = normalize_state_goal(dict(value, expected_value=item.get('expected_value', True)))
            elif 'expected_value' in item:
                raise ValueError('expected_value is only supported for held/lifted state goals')
        except ValueError as exc:
            raise AlignmentError(f'{constraint_id}: {exc}') from exc
        normalized.append(value)
    try:
        validate_state_goal_set(normalized)
    except ValueError as exc:
        raise AlignmentError(str(exc)) from exc
    return normalized


def assert_acyclic(subtasks: list[dict[str, Any]]) -> None:
    deps = {item["subtask_id"]: set(item.get("dependencies", [])) for item in subtasks}
    remaining = set(deps)
    while remaining:
        ready = {sid for sid in remaining if not (deps[sid] & remaining)}
        if not ready:
            raise AlignmentError("subtask dependency graph contains a cycle")
        remaining -= ready


@dataclass
class SceneData:
    nodes: list[dict[str, Any]]
    edges: list[dict[str, Any]]
    visual_features: Any


def load_scene(scene_dir: Path) -> SceneData:
    scene_dir = scene_dir.resolve()
    graph_path = scene_dir / "scene_graph.json"
    edges_path = scene_dir / "scene_graph_edges.json"
    map_path = scene_dir / "map" / "scene_map_cfslam_pruned.pkl.gz"
    for path in (graph_path, edges_path, map_path):
        if not path.is_file():
            raise AlignmentError(f"required scene artifact missing: {path}")

    LOGGER.info("Loading ConceptGraphs artifacts from %s", scene_dir)
    nodes = load_json(graph_path)
    edges = load_json(edges_path)
    if not isinstance(nodes, list):
        raise AlignmentError("scene_graph.json is invalid")
    if not nodes:
        LOGGER.info("Scene graph has no active nodes")
        return SceneData(nodes=[], edges=edges, visual_features=[])
    ids = [node.get("id") for node in nodes]
    if len(ids) != len(set(ids)):
        raise AlignmentError("scene graph contains duplicate node IDs")

    with gzip.open(map_path, "rb") as stream:
        serialized = pickle.load(stream)
    if isinstance(serialized, dict):
        serialized = serialized.get("objects")
    if not isinstance(serialized, list):
        raise AlignmentError("unsupported ConceptGraphs map format")
    feature_by_id: dict[int, Any] = {}
    for index, item in enumerate(serialized):
        caption = item.get("caption_dict", {})
        node_id = caption.get("id", index)
        feature_by_id[int(node_id)] = item.get("clip_ft")
    try:
        ordered_features = [feature_by_id[int(node["id"])] for node in nodes]
    except KeyError as exc:
        raise AlignmentError(f"scene node has no CLIP feature: {exc}") from exc
    LOGGER.info("Loaded %d nodes and %d relations", len(nodes), len(edges))
    return SceneData(nodes=nodes, edges=edges, visual_features=ordered_features)


def node_text(node: dict[str, Any]) -> str:
    tags = node.get("possible_tags") or []
    return ". ".join(
        part for part in [str(node.get("object_tag", "")), ", ".join(map(str, tags)), str(node.get("caption", ""))]
        if part
    )


def normalize_label(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.lower()))


def node_category_labels(node: dict[str, Any]) -> set[str]:
    """Return only explicit scene-graph labels, never free-form caption mentions."""
    values = [node.get("object_tag", ""), *(node.get("possible_tags") or [])]
    return {normalize_label(str(value)) for value in values if normalize_label(str(value))}


def category_matches(node: dict[str, Any], allowed_labels: list[str]) -> bool:
    observed = node_category_labels(node)
    allowed = {normalize_label(label) for label in allowed_labels}
    return bool(observed & allowed)


def tensorize_features(features: list[Any], torch: Any, device: str) -> Any:
    values = []
    for value in features:
        if value is None:
            raise AlignmentError("ConceptGraphs map contains a missing clip_ft")
        tensor = value.detach() if hasattr(value, "detach") else torch.as_tensor(value)
        values.append(tensor.float().reshape(-1))
    matrix = torch.stack(values).to(device)
    return matrix / matrix.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def encode_texts(texts: list[str], model: Any, tokenizer: Any, torch: Any, device: str) -> Any:
    with torch.no_grad():
        tokens = tokenizer(texts).to(device)
        features = model.encode_text(tokens).float()
        return features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def score_roles(
    roles: list[dict[str, Any]],
    scene: SceneData,
    device: str,
    top_k: int,
    min_score: float,
    min_metadata_score: float,
    visual_weight: float,
    metadata_weight: float,
) -> dict[str, list[dict[str, Any]]]:
    if not scene.nodes:
        return {role["role_id"]: [] for role in roles}
    # This is the same hard category gate used below, not a new matching rule.
    # Avoid loading the encoder when no role has any eligible active node.
    if not any(category_matches(node, role["semantic_labels"])
               for role in roles for node in scene.nodes):
        LOGGER.info("No active node passes the semantic category gate")
        return {role["role_id"]: [] for role in roles}
    try:
        import open_clip
        import torch
    except ImportError as exc:
        raise AlignmentError("open_clip and torch are required in the ActiveSplat environment") from exc

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise AlignmentError("CUDA was requested but is unavailable; use --device cpu if necessary")
    LOGGER.info("Loading OpenCLIP ViT-H-14 on %s", device)
    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-H-14", pretrained="laion2b_s32b_b79k", device=device
    )
    model.eval()
    tokenizer = open_clip.get_tokenizer("ViT-H-14")
    visual = tensorize_features(scene.visual_features, torch, device)
    metadata = encode_texts([node_text(node) for node in scene.nodes], model, tokenizer, torch, device)

    scored: dict[str, list[dict[str, Any]]] = {}
    for role in roles:
        queries = role.get("semantic_queries_en", [role["semantic_query_en"]])
        query_features = encode_texts(queries, model, tokenizer, torch, device)
        visual_scores = query_features @ visual.T
        metadata_scores = query_features @ metadata.T
        visual_scores = visual_scores.max(dim=0).values
        metadata_scores = metadata_scores.max(dim=0).values
        total_scores = visual_weight * visual_scores + metadata_weight * metadata_scores

        ranked = torch.argsort(total_scores, descending=True).tolist()
        candidates = []
        for index in ranked:
            total = float(total_scores[index].item())
            meta = float(metadata_scores[index].item())
            if total < min_score or meta < min_metadata_score:
                continue
            node = scene.nodes[index]
            if not category_matches(node, role["semantic_labels"]):
                continue
            candidates.append({
                "element_id": int(node["id"]),
                "object_tag": node.get("object_tag"),
                "score": round(total, 6),
                "evidence": {
                    "visual_cosine": round(float(visual_scores[index].item()), 6),
                    "metadata_cosine": round(meta, 6),
                    "possible_tags": node.get("possible_tags", []),
                    "matched_semantic_labels": sorted(
                        node_category_labels(node)
                        & {normalize_label(label) for label in role["semantic_labels"]}
                    ),
                    "caption": node.get("caption", ""),
                },
            })
            if len(candidates) >= top_k:
                break
        scored[role["role_id"]] = candidates
        LOGGER.info(
            "Role %-24s candidates=%d best_score=%s",
            role["role_id"],
            len(candidates),
            candidates[0]["score"] if candidates else "none",
        )
    return scored


def normalized_edges(edges: list[dict[str, Any]]) -> set[tuple[int, str, int]]:
    output: set[tuple[int, str, int]] = set()
    for edge in edges:
        try:
            a, b = int(edge["source"]), int(edge["target"])
        except (KeyError, TypeError, ValueError):
            continue
        relation = str(edge.get("relation", "")).strip().lower()
        if relation == "a on b":
            output.add((a, "on", b))
        elif relation == "b on a":
            output.add((b, "on", a))
        elif relation == "a in b":
            output.add((a, "in", b))
        elif relation == "b in a":
            output.add((b, "in", a))
    return output


def top_down_prune(
    roles: list[dict[str, Any]],
    candidates: dict[str, list[dict[str, Any]]],
    edges: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Apply task-level quantity, identity, and relation constraints."""
    bindings: dict[str, Any] = {}
    errors: list[dict[str, Any]] = []
    used_ids: set[int] = set()
    ordered = sorted(roles, key=lambda role: (len(candidates[role["role_id"]]), role["role_id"]))
    for role in ordered:
        role_id = role["role_id"]
        pool = list(candidates[role_id])

        selected = [item for item in pool if item["element_id"] not in used_ids][: role["quantity"]]
        if len(selected) < role["quantity"]:
            if role["required"]:
                errors.append({
                    "role_id": role_id,
                    "name": role["name"],
                    "required_quantity": role["quantity"],
                    "matched_quantity": len(selected),
                    "reason": "insufficient_scene_graph_candidates",
                    "top_candidates": pool,
                })
            continue
        element_ids = [item["element_id"] for item in selected]
        used_ids.update(element_ids)
        bindings[role_id] = {
            "element_ids": element_ids,
            "selected": selected,
            "alternatives": [item for item in pool if item["element_id"] not in element_ids],
        }

    return bindings, errors


def build_result(
    scene_dir: Path,
    decomposition: dict[str, Any],
    candidates: dict[str, list[dict[str, Any]]],
    bindings: dict[str, Any],
    errors: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    if errors:
        return {
            "schema_version": "1.0",
            "status": "error",
            "error": {
                "code": "NO_OBJECT_CAN_BE_USED_IN_THIS_TASK",
                "message": ERROR_MESSAGE,
                "details": errors,
                "interpretation": "The object may exist in the physical scene but is absent or insufficiently represented in the explored open-vocabulary scene graph. No object was fabricated.",
            },
            "instruction": decomposition.get("instruction"),
            "scene_reference": str(scene_dir),
            "task_decomposition": decomposition,
            "candidate_evidence": candidates,
            "deferred_validation": {
                "initial_scene_constraints": decomposition["initial_scene_constraints"],
                "goal_state": decomposition["goal_state"],
                "reason": "Deferred to Environment-Aware Memory and execution-state tracking.",
            },
        }
    return {
        "schema_version": "1.0",
        "status": "success",
        "instruction": decomposition.get("instruction"),
        "scene_reference": str(scene_dir),
        "task_decomposition": decomposition,
        "context_bindings": bindings,
        "deferred_validation": {
            "initial_scene_constraints": decomposition["initial_scene_constraints"],
            "goal_state": decomposition["goal_state"],
            "reason": "Deferred to Environment-Aware Memory and execution-state tracking.",
        },
        "alignment": {
            "phase": "object_existence_alignment",
            "method": "ViT-H-14 cosine Top-K, semantic category gate, quantity and distinct-identity validation",
            "top_k": args.top_k,
            "min_score": args.min_score,
            "min_metadata_score": args.min_metadata_score,
            "visual_weight": args.visual_weight,
            "metadata_weight": args.metadata_weight,
            "invented_objects": False,
        },
    }


def main() -> int:
    args = parse_args()
    configure_logging(args.log_file)
    try:
        LOGGER.info("Starting independent task-to-object alignment")
        if args.instruction and args.identifier_allocation_json:
            raise AlignmentError('Live planning allocates a fresh scope; saved allocations are offline-only')
        decomposition = (
            call_llm(args.instruction, args.model, args.base_url, args.api_key_env,
                     max_attempts=args.validation_attempts,
                     execution_contract=load_json(args.execution_contract) if args.execution_contract else None,
                     request_timeout=args.request_timeout, request_attempts=args.request_attempts,
                     decomposition_format=args.decomposition_format,
                     matrix_contract=args.matrix_contract,
                     diagnostics_dir=args.decomposition_output.parent if args.decomposition_output else args.output.parent)
            if args.instruction
            else load_json(args.decomposition_json)
        )
        if not args.instruction and args.decomposition_format == 'matrix':
            identifier_book=None
            if args.matrix_contract in ('typed9_schema','typed5_schema'):
                if not args.identifier_allocation_json:
                    raise AlignmentError('Offline typed decomposition requires --identifier-allocation-json; no allocation inferred')
                identifier_book=load_json(args.identifier_allocation_json)
                write_json(args.output.parent/'identifier_allocation.json',identifier_book)
            write_json(args.output.parent / 'offline_decomposition_matrix.json', decomposition)
            decomposition, _ = enforce_and_record(decomposition, 'decomposition', args.matrix_contract,
                args.output.parent, 'offline_decomposition_matrix',identifier_book)
            if identifier_book is not None:
                from typed_matrix_ids import decode_decomposition
                decomposition=decode_decomposition(decomposition,identifier_book)
            else:
                decomposition = from_decomposition_matrix(decomposition)
        if args.instruction:
            decomposition = preserve_original_instruction(decomposition, args.instruction)
        decomposition = validate_decomposition(decomposition)
        validate_planning_capability_claims(
            decomposition, load_json(args.execution_contract) if args.execution_contract else None)
        if args.decomposition_output is not None:
            write_json(args.decomposition_output.resolve(), decomposition)
            LOGGER.info("Saved validated LLM decomposition: %s", args.decomposition_output.resolve())
        LOGGER.info(
            "Validated decomposition: %d subtasks, %d task roles",
            len(decomposition["subtasks"]),
            len(decomposition["task_roles"]),
        )
        if args.matching_mode == "llm_graph":
            from scene_input_interface import normalize_scene_input
            from llm_graph_binding import (request_validated_bindings, validate_bindings, validate_edges,
                                           check_binding_relations, partition_initial_constraints)
            graph_constraints, runtime_constraints = partition_initial_constraints(decomposition['initial_scene_constraints'])
            nodes = normalize_scene_input(load_json(args.scene_dir / "scene_graph.json"), decomposition['instruction'])['nodes']
            nodes = [n for n in nodes if n.get('active', True)]
            edge_path = args.scene_dir / 'scene_graph_edges.json'
            edges = validate_edges(load_json(edge_path) if edge_path.exists() else [], nodes)
            binding_attempt = 1
            if args.binding_proposal_json:
                proposal = load_json(args.binding_proposal_json)
                if args.binding_format == 'matrix':
                    book = make_binding_codebook(decomposition['task_roles'], nodes, decomposition.get('identifier_context'))
                    write_json(args.output.parent / 'object_binding_codebook.json', book)
                    write_json(args.output.parent / 'offline_binding_matrix.json', proposal)
                    proposal, _ = enforce_and_record(proposal, 'object_binding', args.matrix_contract,
                        args.output.parent, 'offline_binding_matrix', book)
                    proposal = from_binding_matrix(proposal, decomposition['task_roles'], nodes, book)
                write_json(args.output.parent / 'llm_binding_proposal.json', proposal)
            else:
                binding_context = {}
                if args.matrix_contract in ('typed9_schema','typed5_schema'):
                    binding_context['identifier_context']=decomposition['identifier_context']
                if args.execution_contract:
                    context = load_json(args.execution_contract).get('conversation_context')
                    if context is not None:
                        binding_context['conversation_context'] = context
                    inference = load_json(args.execution_contract).get('inference_policy')
                    if inference is not None:
                        binding_context['inference_policy'] = inference
                    capabilities = load_json(args.execution_contract).get('symbolic_scene_capabilities')
                    if capabilities is not None:
                        binding_context['scene_capabilities'] = capabilities
                    boundary = load_json(args.execution_contract).get('factual_boundary')
                    if boundary is not None:
                        binding_context['factual_boundary'] = boundary
                proposal, bindings, errors, binding_attempt = request_validated_bindings(
                    decomposition['task_roles'], nodes, args.model, args.base_url, args.api_key_env,
                    instruction=decomposition['instruction'], edges=edges,
                    request_timeout=args.request_timeout, request_attempts=args.request_attempts,
                    max_validation_attempts=args.validation_attempts,
                    diagnostics_dir=args.output.parent, initial_constraints=graph_constraints,
                    output_format=args.binding_format, matrix_contract=args.matrix_contract, **binding_context)
            try:
                if args.binding_proposal_json:
                    bindings, errors = validate_bindings(proposal, decomposition['task_roles'], nodes)
                relation_checks, relation_errors = check_binding_relations(bindings, graph_constraints, edges)
            except (ValueError, KeyError, TypeError) as exc:
                record_validation(args.output.parent, 'object_binding', binding_attempt, 'rejected', exc,
                                  evidence='llm_binding_proposal.json')
                raise
            errors.extend(relation_errors)
            record_validation(args.output.parent, 'object_binding', binding_attempt, 'rejected' if errors else 'accepted',
                              error=json.dumps(errors, ensure_ascii=True) if errors else None,
                              context={'proposal_origin': 'offline_fixture' if args.binding_proposal_json else 'llm'},
                              evidence={'proposal_file': 'llm_binding_proposal.json',
                                        'relation_checks': relation_checks})
            candidates = {}
        else:
            scene = load_scene(args.scene_dir)
            candidates = score_roles(
            decomposition["task_roles"], scene, args.device, args.top_k,
            args.min_score, args.min_metadata_score,
            args.visual_weight, args.metadata_weight,
            )
            bindings, errors = top_down_prune(decomposition["task_roles"], candidates, scene.edges)
        if not errors:
            validate_state_goal_set(decomposition['goal_state'], bindings)
            validate_state_goal_set(decomposition['initial_scene_constraints'], bindings)
        result = build_result(args.scene_dir.resolve(), decomposition, candidates, bindings, errors, args)
        if args.matching_mode == 'llm_graph':
            result['alignment'] = {'phase': 'object_existence_alignment', 'method': 'LLM graph metadata binding',
                'binding_schema_attempts': binding_attempt,
                'binding_output_format': args.binding_format,
                'decomposition_output_format': args.decomposition_format,
                'matrix_contract': args.matrix_contract,
                'binding_correction_policy': f'At most {args.validation_attempts} validated LLM proposals; no automatic ID conversion; unresolved decisions are terminal',
                'relation_checks': relation_checks, 'edges_supplied': len(edges),
                'invented_objects': False, 'visual_verified': False,
                'offline_fixture': bool(args.binding_proposal_json), 'semantic_correctness_independently_verified': False,
                'constraint_routing': {
                    'object_relation_constraint_ids': [c['constraint_id'] for c in graph_constraints],
                    'runtime_deferred_constraints': runtime_constraints,
                    'runtime_validation_status': 'pending_executor_validation' if runtime_constraints else 'not_required',
                    'robot_state_inferred_from_scene_edges': False}}
            if any(e['reason'] == 'ambiguous' for e in errors):
                result['error']['code'] = 'OBJECT_BINDING_AMBIGUOUS'
                result['error']['message'] = 'Object binding requires clarification'
            elif relation_errors:
                result['error']['code'] = 'OBJECT_RELATION_UNVERIFIED'
                result['error']['message'] = 'Required initial object relation is unknown or conflicting'
        write_json(args.output.resolve(), result)
        LOGGER.info("Finished with status=%s; result=%s", result["status"], args.output.resolve())
        print(json.dumps(result, ensure_ascii=True, indent=2))
        return 0 if result["status"] == "success" else 2
    except Exception as exc:
        LOGGER.exception("Pipeline failed")
        cause = exc
        infrastructure_error = False
        while cause is not None:
            # AlignmentError also inherits RuntimeError but represents rejected model data.
            if isinstance(cause, OSError) or type(cause) is RuntimeError:
                infrastructure_error = True
            cause = cause.__cause__ or cause.__context__
        result = {
            "schema_version": "1.0",
            "status": "error",
            "error": {"code": "PIPELINE_ERROR", "message": str(exc),
                      "category": "infrastructure_error" if infrastructure_error else "model_or_input_validation_error"},
        }
        write_json(args.output.resolve(), result)
        print(json.dumps(result, ensure_ascii=True, indent=2), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
