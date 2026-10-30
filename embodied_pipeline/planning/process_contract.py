"""Explicit process restrictions, separate from action coverage and final goals.

The extension is installed only inside the isolated alignment worker. Legacy
engine files and historical wire contracts remain unchanged.
"""

import copy
from contextlib import contextmanager

from .._legacy import load

VERSION = "process_constraints_v1"
PROMPT = """
PROCESS CONSTRAINT CONTRACT
In addition to the existing root fields, return process_constraint_contract
equal to "process_constraints_v1" and process_constraints (an array).
Each process_constraints row is [object_role_code, restriction_code].
restriction_code 1 = do_not_manipulate: do not grasp, lift, place or otherwise
manipulate the object; its pose/support and held state must remain unchanged
throughout this round. Use this for an explicit "leave it here", "keep it
untouched" or "do not move it" instruction. The object must be a defined,
required role. Reuse that role in other tables; never invent an object.
Restrictions apply to the whole round, not just the final state. Use [] when
there is no such supported restriction. Do not invent protections.
tasks contains ONLY intentions requiring actions. Do not create a task for
"leave X untouched", "ensure X stays here", or another passive restriction.
Put supported protection in process_constraints and retain its meaning in the
affected action task description. Constraint-only roles do not require tasks.
Keep initial factual assertions and final goals in constraints with their
existing scope codes. A final on relation is not a process restriction.
Other prohibitions/order/uncertainty must remain in the affected task text;
do not misencode them as restriction 1. This version only machine-checks
do_not_manipulate. Never omit a requested action to satisfy a restriction.
At least one actionable task is required by this transport-planning profile;
passive-only instructions are unsupported, not a reason to invent an action.
"""


def compose_process_prompt(base):
    # Amend the exhaustive field list as well as the detailed rules.
    return base.replace(
        "dependencies, constraints (all arrays of fixed-column rows).",
        "dependencies, constraints (all arrays of fixed-column rows), "
        "process_constraint_contract and process_constraints.") + "\n" + PROMPT


@contextmanager
def installed():
    """Extend typed5 generation, strict decoding and canonical validation."""
    typed, engine, types = load("typed_matrix_ids"), load("align_objects"), load("matrix_type_contract")
    original_schema = typed.schema
    original_decode = typed.decode_decomposition
    original_validate = engine.validate_decomposition

    def schema(stage, book=None, digits=None):
        result = original_schema(stage, book, digits)
        if stage == "decomposition":
            if typed.width(book) != 5:
                raise ValueError("Process constraints require typed5_schema")
            result["properties"].update(
                process_constraint_contract=types.text(VERSION),
                process_constraints=types.table([
                    types.integer(typed.pool(book, 2)), types.integer((1,))]))
            result["required"] += ["process_constraint_contract", "process_constraints"]
        return result

    def decode(raw, book):
        types.validate_owned_schema(raw, schema("decomposition", book))
        rows = raw["process_constraints"]
        defined = {r[0]: r for r in raw["roles"]}
        if len({tuple(r) for r in rows}) != len(rows):
            raise ValueError("Duplicate process constraint")
        for role, _ in rows:
            if role not in defined or defined[role][4] != 1:
                raise ValueError("Process restriction requires a defined required object role")
        legacy = copy.deepcopy(raw)
        legacy.pop("process_constraint_contract")
        legacy.pop("process_constraints")
        # A temporary inventory bridge lets the unchanged decoder validate all
        # role definitions. It is removed before binding/action generation.
        referenced = {r[1] for r in legacy["task_roles"]}
        referenced.update(r[i] for r in legacy["constraints"] for i in (2, 4))
        process_only = {r[0] for r in rows} - referenced
        for role in sorted(process_only):
            legacy["task_roles"].append([legacy["tasks"][0][0], role])
        typed.schema = original_schema
        try:
            result = original_decode(legacy, book)
        finally:
            typed.schema = schema
        role_ids = {code: rid for rid, code in result["identifier_context"]["role_codes"].items()}
        only_ids = {role_ids[c] for c in process_only}
        first = result["subtasks"][0]
        added = [r for r in first["required_roles"] if r["role_id"] in only_ids]
        first["required_roles"] = [r for r in first["required_roles"] if r["role_id"] not in only_ids]
        result["constraint_roles"].extend(added)
        result["process_constraint_contract"] = VERSION
        result["process_constraints"] = [
            dict(role_id=role_ids[role], restriction="do_not_manipulate") for role, _ in rows]
        return result

    def validate(data):
        if data.get("process_constraint_contract") != VERSION:
            raise ValueError("Missing process constraint contract")
        working = copy.deepcopy(data)
        rows = working["process_constraints"]
        if (not isinstance(rows, list) or any(
                not isinstance(r, dict) or set(r) != {"role_id", "restriction"}
                or r["restriction"] != "do_not_manipulate" or not isinstance(r["role_id"], str)
                for r in rows)):
            raise ValueError("Invalid canonical process constraint")
        ids = [r["role_id"] for r in rows]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate process constraint")
        roles = {r["role_id"]: r for t in working["subtasks"] for r in t["required_roles"]}
        roles.update({r["role_id"]: r for r in working.get("constraint_roles", [])})
        if any(r not in roles or roles[r].get("required") is not True for r in ids):
            raise ValueError("Process restriction requires a defined required object role")
        linked = {c[k] for f in ("initial_scene_constraints", "goal_state")
                  for c in working[f] for k in ("subject_role", "object_role")}
        extra = [r for r in working.get("constraint_roles", [])
                 if r["role_id"] in ids and r["role_id"] not in linked]
        extra_ids = {r["role_id"] for r in extra}
        working["constraint_roles"] = [r for r in working.get("constraint_roles", [])
                                       if r["role_id"] not in extra_ids]
        working["subtasks"][0]["required_roles"].extend(extra)
        result = original_validate(working)
        result["subtasks"][0]["required_roles"] = [
            r for r in result["subtasks"][0]["required_roles"] if r["role_id"] not in extra_ids]
        result["constraint_roles"].extend(extra)
        return result

    typed.schema, typed.decode_decomposition = schema, decode
    engine.validate_decomposition = validate
    try:
        yield
    finally:
        typed.schema, typed.decode_decomposition = original_schema, original_decode
        engine.validate_decomposition = original_validate


def grounded_restrictions(alignment):
    decomposition = alignment.get("task_decomposition", {})
    if decomposition.get("process_constraint_contract") != VERSION:
        return None
    restrictions = []
    for row in decomposition["process_constraints"]:
        if row["restriction"] != "do_not_manipulate":
            raise ValueError("Unsupported process restriction")
        ids = alignment["context_bindings"].get(row["role_id"], {}).get("element_ids")
        if not ids or any(type(i) is not int for i in ids):
            raise ValueError("Process constraint role was not bound")
        restrictions.append(dict(row, element_ids=list(ids)))
    return dict(version=VERSION, restrictions=restrictions,
                scope="Entire round; action coverage applies only to subtasks",
                rule="Do not manipulate protected objects; preserve their pose/support and held state")
