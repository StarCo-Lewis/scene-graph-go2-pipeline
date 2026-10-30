"""Reproducible format/reference checks, not answer or feasibility grading."""

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re

VERSION = "collection_format_reference_quality_v1"


def _json_value(value):
    if isinstance(value, (str, Path)):
        return json.loads(Path(value).read_text(encoding="utf-8"))
    return value


def _canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"), allow_nan=False)


def content_hash(value):
    """Content digest only; it is not a count of independent valid samples."""
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _unordered_rows(value):
    if isinstance(value, list):
        return sorted(value, key=_canonical)
    return value


def _canonical_spec(value):
    if isinstance(value, dict):
        result = {key: _canonical_spec(item) for key, item in value.items()}
        for key in ("final_predicates", "support_relations", "preserve_objects", "unknown_support_object_ids"):
            if key in result:
                result[key] = _unordered_rows(result[key])
        return result
    if isinstance(value, list):
        return [_canonical_spec(item) for item in value]
    return value


def near_duplicate_key(task, reference):
    """Group rewordings with the same scene, goals, input state and reference.

    Action order is preserved. Goal/state/constraint conjunction order is ignored.
    The resulting groups need later review; this is not semantic deduplication.
    """
    reference = reference or {}
    if any(key not in task for key in ("goal", "initial_state", "constraints")) or "steps" not in reference:
        raise ValueError("Near-duplicate signature requires goal, initial_state, constraints and reference steps")
    signature = {"signature_version": VERSION, "scene_id": task.get("scene_id"),
                 "goal": _unordered_rows(_canonical_spec(task.get("goal"))),
                 "initial_state": _unordered_rows(_canonical_spec(task.get("initial_state"))),
                 "constraints": _unordered_rows(_canonical_spec(task.get("constraints"))),
                 "reference_steps": _canonical_spec(reference.get("steps")),
                 "reference_dependencies": _unordered_rows(reference.get("dependencies"))}
    return content_hash(signature)


def duplicate_groups(tasks, references):
    """Return audit groups only; do not discard records or assign valid status."""
    groups = defaultdict(list)
    for task in tasks:
        rid = task.get("record_id")
        reference = references.get(rid, {})
        groups[near_duplicate_key(task, reference)].append(rid)
    return [{"signature_sha256": digest, "record_ids": ids, "requires_review": True}
            for digest, ids in sorted(groups.items()) if len(ids) > 1]


def _check_dependencies(steps, dependencies, issues, *, id_key, scope):
    if not isinstance(steps, list):
        issues.append({"code": "steps_not_array", "scope": scope})
        return
    ids = [step.get(id_key) if isinstance(step, dict) else None for step in steps]
    if any(not isinstance(i, str) or not i for i in ids) or len(ids) != len(set(i for i in ids if isinstance(i, str))):
        issues.append({"code": "invalid_or_duplicate_step_ids", "scope": scope})
        return
    order = {step_id: index for index, step_id in enumerate(ids)}
    graph = {step_id: [] for step_id in ids}
    for before, after, path in dependencies:
        if before not in order or after not in order:
            issues.append({"code": "unknown_dependency_reference", "scope": scope, "path": path,
                           "before": before, "after": after})
            continue
        graph[before].append(after)
        if order[before] >= order[after]:
            issues.append({"code": "dependency_not_previous_step", "scope": scope, "path": path,
                           "before": before, "after": after})
    visiting, done = set(), set()
    def visit(node):
        if node in visiting:
            return False
        if node in done:
            return True
        visiting.add(node)
        if any(not visit(child) for child in graph[node]):
            return False
        visiting.remove(node)
        done.add(node)
        return True
    if not all(visit(node) for node in ids):
        issues.append({"code": "dependency_cycle", "scope": scope})


def _embedded_dependencies(steps, id_key, issues, scope, dependency_key="dependencies"):
    edges = []
    if not isinstance(steps, list):
        return edges
    for index, step in enumerate(steps):
        if not isinstance(step, dict):
            continue
        deps = step.get(dependency_key, [])
        if not isinstance(deps, list) or any(not isinstance(item, str) for item in deps):
            issues.append({"code": "dependencies_not_string_array", "scope": scope, "index": index})
            continue
        for before in deps:
            edges.append((before, step.get(id_key), f"{scope}[{index}].dependencies"))
    return edges


def _element_refs(value, path="root"):
    if isinstance(value, dict):
        for key, item in value.items():
            if key in ("element_id", "object_id", "support_id", "destination_id", "held_object_id"):
                if item is not None or key not in ("destination_id", "held_object_id", "support_id"):
                    yield path + "." + key, item
            elif key in ("preserve_objects", "grounded_object_ids", "unknown_support_object_ids") and isinstance(item, list):
                for index, element in enumerate(item):
                    yield path + "." + key + f"[{index}]", element
            else:
                yield from _element_refs(item, path + "." + str(key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _element_refs(item, f"{path}[{index}]")


def validate_record(task, reference, graph_nodes, *, decomposition=None, skill_plan=None):
    """Check only syntax/IDs/DAG/references; decoded output checks are optional.

    ``decomposition`` and ``skill_plan`` may be dictionaries or JSON file paths.
    This does not compare a proposed plan to ``reference`` or inspect goal truth.
    """
    format_issues, reference_issues = [], []
    scene = task.get("scene_id")
    rid = task.get("record_id")
    if not isinstance(scene, str) or not scene or not isinstance(rid, str) or not re.fullmatch(re.escape(scene) + r"_[0-9]{8}", rid):
        format_issues.append({"code": "record_id_not_scene_plus_eight_digits"})
    if not isinstance(task.get("instruction"), str) or not task["instruction"].strip():
        format_issues.append({"code": "missing_instruction"})
    for key in ("initial_state", "goal", "constraints"):
        if not isinstance(task.get(key), (dict, list)):
            format_issues.append({"code": "missing_or_invalid_task_field", "field": key})
    if not isinstance(graph_nodes, list):
        reference_issues.append({"code": "graph_nodes_not_array"})
        graph_nodes = []
    raw_ids = [node.get("id") if isinstance(node, dict) else None for node in graph_nodes]
    graph_ids = {value for value in raw_ids if type(value) is int}
    if len(graph_ids) != len(raw_ids):
        reference_issues.append({"code": "graph_ids_not_unique_integers"})
    referenced = task.get("grounded_object_ids", task.get("referenced_objects", []))
    if not isinstance(referenced, list):
        reference_issues.append({"code": "referenced_objects_not_array"})
    else:
        for index, item in enumerate(referenced):
            element = item.get("element_id") if isinstance(item, dict) else item
            if type(element) is not int or element not in graph_ids:
                reference_issues.append({"code": "unknown_task_object_reference", "index": index,
                                         "element_id": element})
    for field in ("initial_state", "goal", "constraints"):
        for path, element in _element_refs(task.get(field), "task." + field):
            if type(element) is not int or element not in graph_ids:
                reference_issues.append({"code": "unknown_task_spec_object_reference", "path": path, "element_id": element})
    if reference is not None:
        if not isinstance(reference, dict):
            format_issues.append({"code": "reference_not_object"})
        else:
            steps = reference.get("steps", [])
            deps = _embedded_dependencies(steps, "step_id", reference_issues, "reference.steps", "depends_on")
            external = reference.get("dependencies", [])
            if not isinstance(external, list):
                reference_issues.append({"code": "reference_dependencies_not_array"})
            else:
                for index, row in enumerate(external):
                    if not isinstance(row, list) or len(row) != 2 or not all(isinstance(v, str) for v in row):
                        reference_issues.append({"code": "invalid_reference_dependency_row", "index": index})
                    else:
                        deps.append((row[0], row[1], f"reference.dependencies[{index}]"))
            _check_dependencies(steps, deps, reference_issues, id_key="step_id", scope="reference.steps")
            for path, element in _element_refs(reference, "reference"):
                if type(element) is not int or element not in graph_ids:
                    reference_issues.append({"code": "unknown_reference_object", "path": path, "element_id": element})
    for kind, value in (("decomposition", decomposition), ("skill_plan", skill_plan)):
        if value is None:
            continue
        try:
            parsed = _json_value(value)
            if not isinstance(parsed, dict):
                raise ValueError("Decoded output is not an object")
            if kind == "decomposition":
                steps = parsed.get("subtasks")
                deps = _embedded_dependencies(steps, "subtask_id", reference_issues, "decomposition.subtasks")
                _check_dependencies(steps, deps, reference_issues, id_key="subtask_id", scope="decomposition.subtasks")
            else:
                if not isinstance(parsed.get("subtask_plans"), list):
                    format_issues.append({"code": "skill_subtask_plans_not_array"})
                for path, element in _element_refs(parsed, "skill_plan"):
                    if type(element) is not int or element not in graph_ids:
                        reference_issues.append({"code": "unknown_plan_object_reference", "path": path, "element_id": element})
        except (OSError, ValueError, TypeError) as exc:
            format_issues.append({"code": "output_unreadable", "scope": kind, "error_type": type(exc).__name__})
    return {"quality_version": VERSION, "record_id": rid,
            "format_valid": not format_issues, "reference_valid": not reference_issues,
            "format_issues": format_issues, "reference_issues": reference_issues,
            "semantic_evaluation": "not_performed", "reference_answer_comparison": "not_performed",
            "physical_feasibility_evaluation": "not_performed", "valid_candidate": None,
            "modality_validation": "not_performed_by_this_helper",
            "status": "format_reference_checks_only_not_semantically_graded"}
