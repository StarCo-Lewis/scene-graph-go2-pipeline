"""Real-graph planning-only smoke test; never dispatch actions or commit memory."""

import argparse
import copy
from datetime import datetime
import json
import logging
import os
from pathlib import Path
import shutil
import time
from types import SimpleNamespace

from .._legacy import ROOT, load
from .. import planning, instruction_skills as skills, validation
from ..validation import plan_semantics
from ..perception.real_rgbd import sha, write
from ..perception.conceptgraphs_scene import load_real_scene

TASK = ROOT / "embodied_pipeline/perception/d3_619_navigation_smoke.json"
LIBRARY = ROOT / "outputs/quadruped_atomic_skill_library.v0.1.json"
SKILLS = ("navigate_to_object", "position_arm_for_object", "grasp_object",
          "lift_held_object", "place_object_on_surface")
INSTRUCTION_SCHEMA = "real_scene_navigation_instruction_v2"
ORDERED_GOAL_POLICY = (
    " Temporal goal contract ordered_navigation_terminal_v1: goal_state means conditions that must "
    "hold at the END of the entire task, not a list of every visited destination. For 'visit A then B', "
    "create separate ordered visit subtasks A -> B; put only robot near B in final goal_state. "
    "A is an intermediate milestone verified when its navigation action completes; do not require "
    "robot near A to remain true after leaving A. For A -> B -> A, retain all three visits and finish "
    "near A. Encode each visit as its own subtask with an explicit dependency on the preceding visit. "
    "Do not delete visits to satisfy final goals, do not weaken unrelated preservation constraints, "
    "and do not invent a visited predicate or unsupported wire columns. If explicitly requested final "
    "conditions conflict, report clarification instead of silently discarding them."
)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def prepare(task_path, root):
    task = read(task_path)
    if task.get("schema_version") == "real_scene_navigation_instruction_v1":
        raise ValueError("Unbound instruction-v1 task; re-prepare with real_scene_navigation_instruction_v2")
    if task.get("schema_version") not in ("real_scene_navigation_smoke_v1", INSTRUCTION_SCHEMA):
        raise ValueError("Unsupported smoke task schema")
    instruction_led = task["schema_version"] == INSTRUCTION_SCHEMA
    if instruction_led:
        expected_metadata = task.get("scene_metadata_sha256")
        if (not isinstance(expected_metadata, str) or len(expected_metadata) != 64 or
                any(c not in "0123456789abcdef" for c in expected_metadata)):
            raise ValueError("Instruction-v2 requires preparation-time scene_metadata_sha256; re-prepare task")
    if task.get("navigation_mode", "single_goal") not in ("single_goal", "ordered_goals"):
        raise ValueError("Unsupported navigation mode")
    if instruction_led and "target_element_id" in task:
        raise ValueError("Instruction-led tasks must not contain a preselected reference target")
    if not isinstance(task.get("instruction"), str) or not task["instruction"].strip():
        raise ValueError("A nonempty user instruction is required")
    source = (ROOT / task["scene_dir"]).resolve()
    hashes = {name: sha(source / name) for name in
              ("scene_graph.json", "scene_graph_edges.json", "scene_metadata.json")}
    if (hashes["scene_graph.json"] != task["scene_graph_sha256"] or
            hashes["scene_graph_edges.json"] != task["scene_edges_sha256"]):
        raise ValueError("Task belongs to a different graph revision; review task before running")
    if instruction_led and hashes["scene_metadata.json"] != expected_metadata:
        raise ValueError("Task metadata hash mismatch; re-prepare task before running")
    bundle = load_real_scene(source)
    target = None
    if not instruction_led:
        target = next((n for n in bundle["nodes"] if n["id"] == task["target_element_id"]), None)
        if target is None or target.get("active", True) is False:
            raise ValueError("Task target is absent or inactive")
    # A new output directory prevents accidental overwrite of a previous live run.
    root.mkdir(parents=True, exist_ok=False)
    scene = root / "input_memory/scene"
    scene.mkdir(parents=True)
    for name in hashes:
        shutil.copyfile(source / name, scene / name)
    contract = {
        "schema_version": "real_scene_planning_only_v1",
        "mode": "proposal_only_no_executor",
        "execution_ready": False,
        "planning_capabilities": {"validate_supplied_on_in_edges": True},
        "factual_boundary": (
            "Use supplied graph IDs and descriptions. Geometry and edges are map/model evidence, "
            "not physical verification. Missing relations and actual robot state are unknown. "
            "Do not invent initial robot proximity, holding, lift, preparation or navigation success. "
            "Generate a proposed task and actions, not an execution report. Unknown physical "
            "preconditions remain requirements for a future real-robot backend."),
        "unobserved_runtime_fields": ["robot_pose", "held_object_id", "lifted", "prepared_target_id",
                                      "reachable", "navigable", "ros_map_transform"],
        "coordinate_frame": bundle["metadata"]["coordinate_frame"],
        "action_identity_scope": {"case_id": task["task_id"], "round": 1, "run_id": root.name},
    }
    library = read(LIBRARY)
    if task.get("navigation_mode") == "ordered_goals":
        contract["factual_boundary"] += (
            " This is an ordered multi-goal navigation task. Preserve every explicitly requested visit "
            "and its order, including a return visit if requested. Create meaningful visit subtasks "
            "and ordered navigate_to_object actions with the corresponding bound target roles. "
            "Do not collapse A then B into only B, do not add manipulation actions, and do not "
            "invent targets for unresolved references. All targets remain in the same scene frame." + ORDERED_GOAL_POLICY)
        contract["ordered_navigation_goal_policy_version"] = "ordered_navigation_terminal_v1"
    library["skills"] = [s for s in library["skills"] if s["skill_id"] in SKILLS]
    if {s["skill_id"] for s in library["skills"]} != set(SKILLS):
        raise ValueError("Required existing skill definitions are missing")
    library["execution_contract"] = contract
    library["integration_note"] = (
        "Five existing symbolic skill specifications; original parameters/defaults retained. "
        "No proxy parameter projection. Robot profile is a design specification, not observed capability.")
    library = skills.enrich(library)
    write(root / "task_input.json", task)
    write(root / "round_input.json", {"instruction": task["instruction"]})
    write(root / "execution_contract.json", contract)
    write(root / "execution_skill_library.json", library)
    write(root / "input_audit.json", {"source_scene": str(source), "source_hashes": hashes,
        "node_count": len(bundle["nodes"]), "edge_count": len(bundle["edges"]),
        "target_from_graph": target,
        "target_authority": "user_instruction_via_llm_object_binding" if instruction_led else "fixed_reference_test",
        "skill_library_sha256": sha(LIBRARY),
        "scene_filtered": False, "runtime_state_fabricated": False, "memory_committed": False})
    return task, bundle, library, source, hashes


def check_navigation(plan, task, bundle=None, alignment=None):
    steps = [s for t in plan["subtask_plans"] for s in t["steps"]]
    if task.get("schema_version") == INSTRUCTION_SCHEMA:
        nodes = {n["id"]: n for n in (bundle or {}).get("nodes", [])}
        bindings = (alignment or {}).get("context_bindings", {})
        issues, targets = [], []
        multi = task.get("navigation_mode") == "ordered_goals"
        if not steps or (not multi and len(steps) != 1):
            issues.append({"code": "single_navigation_action_required"})
        if plan.get("unplanned_subtasks"):
            issues.append({"code": "unplanned_subtasks_remain"})
        for s in steps:
            ref = s.get("arguments", {}).get("target", {})
            target = ref.get("element_id")
            targets.append(target)
            reason = None
            if s.get("skill_id") != "navigate_to_object":
                reason = "non_navigation_action"
            elif type(target) is not int or target not in nodes or nodes[target].get("active", True) is False:
                reason = "target_absent_or_inactive"
            elif target not in bindings.get(ref.get("role_id"), {}).get("element_ids", []):
                reason = "target_disagrees_with_object_binding"
            if reason:
                issues.append({"code": reason, "step_id": s.get("step_id")})
        if multi and targets:
            for goal in plan.get("deferred", {}).get("goal_state", []):
                if goal.get("subject_role") == "robot" and goal.get("relation") == "near" and goal.get("required", True):
                    goal_targets = bindings.get(goal.get("object_role"), {}).get("element_ids", [])
                    if goal_targets and targets[-1] not in goal_targets:
                        issues.append({"code": "intermediate_visit_declared_as_terminal_goal",
                            "constraint_id": goal.get("constraint_id"), "goal_targets": goal_targets,
                            "final_navigation_target": targets[-1],
                            "message_zh": "中间访问目标被写成必须在任务结束时仍满足的目标；应保留访问步骤与依赖，区分阶段目标和最终状态。"})
        return {"passed": not issues, "action_count": len(steps), "issues": issues,
                "unexpected_step_ids": [i["step_id"] for i in issues if "step_id" in i],
                "bound_target_element_ids": targets, "repeated_navigation": len(steps) > 1,
                "navigation_mode": task.get("navigation_mode", "single_goal"),
                "reference_target_comparison_performed": False,
                "semantic_correctness_independently_verified": False,
                "claim": "Instruction-led structural and grounding checks only. Symbolic goal review is separate; operator must review intent and physical feasibility."}
    wrong = [s["step_id"] for s in steps if s["skill_id"] != "navigate_to_object" or
             s["arguments"].get("target", {}).get("element_id") != task["target_element_id"]]
    return {"passed": bool(steps) and not wrong, "action_count": len(steps),
            "unexpected_step_ids": wrong, "repeated_navigation": len(steps) > 1,
            "claim": "Task-specific output check only; neither reachability nor execution success."}


def generate(args, root, task, bundle, library, summary, encoder):
    summary["stage"] = "decomposition_and_object_binding"
    print("[REAL SCENE PLAN] START decomposition and object binding", flush=True)
    options = SimpleNamespace(**dict(vars(args), output_dir=root,
        alignment_wire_format="matrix", matrix_contract="typed5_schema", alignment_validation_attempts=1))
    alignment = planning.decompose_and_bind(options, {"instruction": task["instruction"]}, summary)
    audit = plan_semantics.audit(alignment["task_decomposition"],
                                 bindings=alignment.get("context_bindings"),
                                 instruction=task["instruction"])
    write(root / "decomposition_audit.json", audit)
    summary["decomposition_audit"] = {k: audit[k] for k in
                                      ("version", "passed", "has_defect", "issue_codes")}
    summary["stage"] = "full_skill_inventory"
    summary["candidate_policy"] = "full_inventory_no_cosine_v1"
    print("[REAL SCENE PLAN] START full skill inventory and skill matrix (cosine disabled)", flush=True)
    retrieval = skills.candidates(alignment, library)
    write(root / "skill_retrieval.json", retrieval)
    book = skills.make_codebook(alignment, library)
    write(root / "action_codebook.json", book)
    summary["stage"] = "llm_skill_matrix"
    # Do not use the deterministic navigation compiler or an Oracle action list.
    raw = skills.request_matrix(alignment, library, book, args.model, args.base_url,
        "DASHSCOPE_API_KEY", retrieval=retrieval, widen_inventory=False, timeout=180, attempts=2,
        diagnostics_dir=root, audit_context={"scope": "real_scene_planning_only"},
        matrix_contract="typed5_schema")
    write(root / "llm_matrix_proposal.json", raw)
    normalized, note = validation.normalize_and_record(raw, "matrix", library, root,
        "llm_skill", "llm_matrix_proposal.json", matrix_contract="typed5_schema")
    summary["format_normalization"] = note
    decoded = validation.decode_matrix(normalized, book, alignment, library)
    write(root / "decoded_matrix.json", decoded)
    write(root / "candidate_membership_observation.json",
          skills.check_membership(decoded, retrieval, widened=True))
    plan = skills.build_result(alignment, library, decoded)
    write(root / "skill_plan.json", plan)
    summary["skill_plan_json_generated"] = True
    summary["stage"] = "static_review_not_execution"
    check = check_navigation(plan, task, bundle, alignment)
    write(root / "task_output_check.json", check)
    audit = plan_semantics.audit(alignment["task_decomposition"],
                                 bindings=alignment.get("context_bindings"), plan=plan,
                                 instruction=task["instruction"])
    write(root / "decomposition_audit.json", audit)
    summary["decomposition_audit"] = {k: audit[k] for k in
                                      ("version", "passed", "has_defect", "issue_codes")}
    review = load("verify_plan").review(alignment, plan, library, bundle["nodes"], bundle["edges"])
    write(root / "static_review.json", review)
    # Review may examine an optimized copy. Never export that copy as the model's plan.
    engineering = {"schema_version": "real_robot_proposal_envelope_v1",
        "status": "proposal_not_executable", "execution_ready": False,
        "task_id": task["task_id"], "instruction": task["instruction"],
        "target_authority": "user_instruction_via_llm_object_binding" if task.get("schema_version") == INSTRUCTION_SCHEMA else "fixed_reference_test",
        "coordinate_frame": bundle["metadata"]["coordinate_frame"], "unit": "metre",
        "source_graph_sha256": task["scene_graph_sha256"],
        "source_edges_sha256": task["scene_edges_sha256"],
        "plan": copy.deepcopy(plan), "task_output_check": check,
        "decomposition_audit": audit,
        "quality_flags": audit["defect_codes"],
        "static_review_status": review["status"],
        "geometry_contract": "Resolve bound element IDs in the source graph. Object bbox centre is NOT a collision-free navigation goal pose.",
        "required_before_dispatch": ["Register actual robot pose and map TF", "Choose and validate a collision-free approach pose",
            "Validate navigation/collision/stop tolerances", "Bind skills to real interfaces and obtain execution feedback"],
        "robot_goal_pose": None, "simulation_executed": False, "robot_executed": False,
        "memory_committed": False}
    write(root / "engineering_plan.json", engineering)
    summary["engineering_json_generated"] = True
    initial_checks = review.get("observed_state", {}).get("initial_claim_checks", [])
    initial_problem = any(c.get("required", True) and c.get("status") not in
                          ("observed_true", "true") for c in initial_checks)
    passed = (check["passed"] and review["status"] == "conditionally_valid_symbolic_plan"
              and not initial_problem and not audit["has_defect"])
    summary.update(status="planning_json_generated_not_executed" if passed else "planning_output_requires_review",
        action_count=plan["action_count"], task_check_passed=check["passed"],
        static_review_status=review["status"], initial_claims_require_review=initial_problem,
        engineering_plan=str(root / "engineering_plan.json"))
    summary["review_issues"] = check.get("issues", [])


def run(args):
    root = (args.output_dir or ROOT / "outputs/real_scene_runs" /
            ("d3_619_plan_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))).resolve()
    task, bundle, library, source, hashes = prepare(args.task_json, root)
    summary = {"status": "preflight_only_not_model_test", "stages": [], "stage": "preflight",
        "engineering_json_generated": False, "fresh_llm_calls": False,
        "simulation_executed": False, "robot_executed": False, "memory_committed": False,
        "execution_ready": False, "node_count": len(bundle["nodes"]), "edge_count": len(bundle["edges"])}
    summary.update(model=args.model, matrix_contract="typed5_schema",
                   semantic_retries=0, deterministic_plan_substitution=False)
    print(f"[REAL SCENE PLAN] Output: {root}", flush=True)
    print(f"[REAL SCENE PLAN] Alignment log: {root / 'alignment.console.log'}", flush=True)
    write(root / "planning_summary.json", summary)
    encoder = None
    started = time.monotonic()
    try:
        if not args.preflight_only:
            if not os.environ.get("DASHSCOPE_API_KEY"):
                raise RuntimeError("DASHSCOPE_API_KEY is not set; no request was made")
            summary.update(candidate_policy="full_inventory_no_cosine_v1", embedding_used=False)
            summary["fresh_llm_calls"] = True
            generate(args, root, task, bundle, library, summary, encoder)
    except Exception as exc:
        summary.update(status="planning_failed_not_executed", error_type=type(exc).__name__, error=str(exc))
        logging.exception("Planning-only smoke failed; no executor or memory commit was invoked")
    finally:
        if encoder is not None:
            encoder.close()
        summary["input_files_unchanged"] = all(sha(source / name) == value for name, value in hashes.items())
        if not summary["input_files_unchanged"]:
            summary["status"] = "input_integrity_failed"
        summary["duration_seconds"] = time.monotonic() - started
        write(root / "planning_summary.json", summary)
        print(f"status={summary['status']} summary={root / 'planning_summary.json'}", flush=True)
    return 0 if summary["status"] in ("preflight_only_not_model_test", "planning_json_generated_not_executed") else 2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-json", type=Path, default=TASK)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--model", default="qwen3.8-flash")
    parser.add_argument("--base-url", default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    parser.add_argument("--embedding-python", type=Path, default=Path("/home/lewis/miniforge3/envs/skill_embed/bin/python"))
    parser.add_argument("--embedding-model", type=Path, default=Path("/mnt/d/NebulaAI/models/skill_embedding_minilm_v1"))
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
