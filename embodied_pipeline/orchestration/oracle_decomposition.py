"""Paired archived-state Oracle-D trials; no decomposition model call or recovery."""
import argparse
import copy
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime

from .._legacy import ROOT, load
from .. import execution, errors, memory
from ..introspection import verify_legacy, write_run_manifest
from .proposal import propose_once

DEFAULT_BUNDLE = ROOT / "outputs/oracle_decomposition_review_20261020/oracle_decomposition_review.json"
DEFAULT_APPROVAL = DEFAULT_BUNDLE.with_name("human_approval.json")
INPUT_NAMES = (
    "round_input.json", "runtime_state.json", "execution_contract.json",
    "execution_skill_library.json", "identifier_allocation.json",
    "input_memory/scene/scene_graph.json", "input_memory/scene/scene_graph_edges.json")


class BranchTransportError(RuntimeError):
    """A recorded API transport failure, safe to isolate between snapshot branches."""

    def __init__(self, proposal):
        super().__init__(proposal["error"])
        self.proposal = proposal


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def content_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()


def check_english_input(value):
    """Reject Chinese text rather than silently translating archived evidence."""
    if re.search(r"[\u3400-\u9fff]", json.dumps(value, ensure_ascii=False)):
        raise ValueError("Chinese text found in model input; English input is required")


def source_path(reference):
    path = (ROOT / reference["path"]).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError("Source path is outside the workspace")
    if sha(path) != reference["sha256"]:
        raise ValueError("Archived evidence changed: " + reference["path"])
    return path


def selected_cases(args):
    bundle, approval = read(args.bundle), read(args.approval)
    if approval.get("approved") is not True or sha(args.bundle) != approval["bundle_sha256"]:
        raise ValueError("Missing approval or bundle changed after approval")
    available = {c["case_id"]: c for c in bundle["cases"] if c["oracle_payload"] is not None}
    ids = args.case_ids or ["C01"] + [k for k in available if k != "C01"]
    if len(ids) != len(set(ids)) or set(ids) - available.keys():
        raise ValueError("Select distinct cases with an approved injectable payload")
    result = [available[k] for k in ids]
    for case in result:
        expected = approval["payload_sha256_by_case"].get(case["case_id"])
        if expected != content_hash(case["oracle_payload"]):
            raise ValueError("Payload changed after approval: " + case["case_id"])
        if read(ROOT / case["payload_file"]) != case["oracle_payload"]:
            raise ValueError("Standalone payload differs from reviewed bundle")
        for reference in case["source_files"].values():
            source_path(reference)
    return result


def check_decomposition(payload, allocation, contract):
    """Same owned schema and domain checks; no correction or binding substitute."""
    alignment_engine = load("align_objects")
    decoded = load("typed_matrix_ids").decode_decomposition(copy.deepcopy(payload), allocation)
    decoded = alignment_engine.validate_decomposition(decoded)
    alignment_engine.validate_planning_capability_claims(decoded, contract)
    if decoded["instruction"] != payload["instruction"]:
        raise ValueError("Instruction changed")
    return decoded


def prepare_branch(case, arm, destination):
    planning = destination / "planning"
    for name in INPUT_NAMES:
        target = planning / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_path(case["source_files"][name]), target)
    payload = case["original_model_output"] if arm == "original_replay" else case["oracle_payload"]
    for name in INPUT_NAMES:
        check_english_input(read(planning / name))
    check_english_input(payload)
    if payload["instruction"] != read(planning / "round_input.json")["instruction"]:
        raise ValueError("Archived instruction differs from replay payload")
    memory.write_json(planning / "injected_decomposition.json", payload)
    state = read(planning / "runtime_state.json")
    contract = read(planning / "execution_contract.json")
    if state != contract["current_runtime_state"]:
        raise ValueError("Runtime/contract state mismatch")
    memory.write_json(destination / "memory_checkpoint.json", dict(
        state=state, history=contract["conversation_context"]["proxy_execution_history"]))
    record = dict(case_id=case["case_id"], arm=arm, source_batch=case["batch"],
                  source_episode=case["episode"], source_round=case["round"],
                  injected_stage="decomposition_only", decomposition_model_called=False,
                  payload_sha256=content_hash(payload),
                  allocation_policy="archived_allocation_replayed_in_isolated_branch",
                  full_scene_and_history_preserved=True, private_evaluation_forwarded=False,
                  english_input_check="passed_no_chinese_text",
                  original_artifacts_overwritten=False)
    try:
        check_decomposition(payload, read(planning / "identifier_allocation.json"), contract)
        record["decomposition_precheck"] = "accepted"
    except (ValueError, RuntimeError, KeyError, TypeError) as exc:
        record.update(decomposition_precheck="rejected", error=str(exc))
    memory.write_json(destination / "intervention.json", record)
    return planning, record


def alignment_from_saved(args, descriptor, summary):
    """Use the existing offline-decomposition/live-binding CLI, with no injected binding."""
    planning = args.output_dir
    command = [
        sys.executable, str(ROOT / "outputs/object_alignment_v1/align_objects.py"),
        "--decomposition-json", str(planning / "injected_decomposition.json"),
        "--identifier-allocation-json", str(planning / "identifier_allocation.json"),
        "--scene-dir", str(planning / "input_memory/scene"),
        "--matching-mode", "llm_graph", "--execution-contract", str(planning / "execution_contract.json"),
        "--model", args.model, "--base-url", args.base_url,
        "--request-timeout", "180", "--request-attempts", "2",
        "--validation-attempts", "1", "--decomposition-format", "matrix",
        "--binding-format", "matrix", "--matrix-contract", "typed5_schema",
        "--output", str(planning / "object_alignment.json"),
        "--decomposition-output", str(planning / "task_decomposition.json")]
    with (planning / "alignment.console.log").open("w", encoding="utf-8") as log:
        completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    summary["stages"].append(dict(stage="saved_decomposition_and_live_object_binding",
                                  exit_code=completed.returncode))
    alignment = read(planning / "object_alignment.json")
    if completed.returncode or alignment.get("status") != "success":
        summary["upstream_error"] = alignment.get("error")
        raise ValueError("Saved decomposition/live binding failed: " +
                         alignment.get("error", {}).get("message", "unknown"))
    if alignment["task_decomposition"]["instruction"] != descriptor["instruction"]:
        raise ValueError("Instruction changed during replay")
    # Match live decomposition metadata; this field is not used to generate bindings.
    alignment["task_decomposition"]["matrix_type_contract"] = dict(
        mode="typed5_schema", version=load("matrix_type_contract").VERSION, normalized=False)
    memory.write_json(planning / "object_alignment.json", alignment)
    memory.write_json(planning / "task_decomposition.json", alignment["task_decomposition"])
    return alignment


def evaluate_private(case, initial, feedback):
    """Post-execution symbolic reference checks, not a model judge or execution gate."""
    expected = case["private_evaluation"]
    state = feedback["resulting_state"]
    trace = feedback["trace"]
    placements = [[step["arguments"]["object"]["element_id"],
                   step["arguments"]["surface"]["element_id"]]
                  for step in trace if step["skill_id"] == "place_object_on_surface"
                  and step["status"] == "proxy_action_succeeded"]
    objects = state["backend"]["objects"]
    support_ok = all(objects[str(obj)]["support_id"] == surface
                     for obj, surface in expected["placement_events_in_order"])
    protected = expected["protected_object_ids"]
    violations = []
    for step in trace:
        for obj in protected:
            before = initial["backend"]["objects"][str(obj)]
            after = step["end_state"]["objects"][str(obj)]
            targeted = step["skill_id"] in ("grasp_object", "lift_held_object", "place_object_on_surface") and any(
                isinstance(v, dict) and v.get("element_id") == obj
                for v in step["arguments"].values())
            if targeted or before != after or step["end_state"]["held_object_id"] == obj:
                violations.append(dict(action_index=step["action_index"], object_id=obj))
    hints = {v["role_code"]: v["expected_scene_id"] for v in expected["role_grounding_reference"]}
    goal_checks = []
    for row in case["oracle_payload"]["constraints"]:
        cid, scope, subject, relation, obj, wanted, required = row
        if scope != 2 or not required:
            continue
        target = hints[obj]
        if relation == 1:
            satisfied = objects[str(hints[subject])]["support_id"] == target
        elif relation in (4, 5):
            actual = state["held_object_id"] == target and (relation == 4 or state["lifted"])
            satisfied = actual == bool(wanted)
        elif relation == 3 and subject == 10001:
            a, b = state["backend"]["robot_position"], objects[str(target)]["center"]
            satisfied = math.hypot(a[0]-b[0], a[2]-b[2]) <= 1.5
        else:
            satisfied = None
        goal_checks.append(dict(constraint_code=cid, satisfied=satisfied))
    checks = dict(
        proxy_actions_completed=feedback["status"] == "proxy_actions_completed",
        placement_order_matches=placements == expected["placement_events_in_order"],
        placement_supports_match=support_ok,
        final_held_matches=state["held_object_id"] == expected["final_held_object_id"],
        all_required_oracle_goals_match=all(g["satisfied"] is True for g in goal_checks),
        protected_objects_untouched=not violations)
    return dict(checks=checks, private_reference_passed=all(checks.values()),
                actual_placements=placements, protected_violations=violations, goal_checks=goal_checks,
                scope="Symbolic authored reference checks only; not independent physical truth",
                evaluated_after_commit=True, full_semantics_human_audit_required=True)


def execute_branch(args, case, arm, folder, planning, encoder, *,
                   alignment_provider=None, decomposition_origin=None, live_decomposition=False):
    initial = read(planning / "runtime_state.json")
    contract = read(planning / "execution_contract.json")
    nodes = read(planning / "input_memory/scene/scene_graph.json")
    descriptor = read(planning / "round_input.json")
    proposal, plan, alignment = propose_once(args, planning, encoder,
        alignment_provider=alignment_provider or alignment_from_saved)
    proposal.update(decomposition_origin=decomposition_origin or
                    ("archived_model_replay" if arm == "original_replay" else "human_reviewed_oracle"),
                    decomposition_model_called=live_decomposition and
                    any(planning.rglob("decomposition_*_request.json")),
                    fresh_llm_calls=any(planning.rglob("*_request.json")))
    memory.write_json(planning / "planning_summary.json", proposal)
    if read(planning / "runtime_state.json") != initial:
        raise RuntimeError("Planning mutated the input snapshot")
    if proposal["status"] == "infrastructure_error":
        if "LLM transport failed after " in proposal["error"]:
            raise BranchTransportError(proposal)
        raise RuntimeError(proposal["error"])
    tracked = tuple(int(n) for n in initial["backend"]["objects"])
    compact = lambda value: load("dialogue_chain_cases").compact_state(value, tracked)
    feedback = (execution.execute_proxy(plan, alignment, read(planning / "execution_skill_library.json"),
                                       nodes, copy.deepcopy(initial), compact)
                if plan is not None else errors.unavailable_result(initial, proposal.get("error")))
    memory.write_json(folder / "execution_feedback.json", feedback)
    history = contract["conversation_context"]["proxy_execution_history"]
    history_ids = tuple(int(n) for n in history[-1]["end_state"]["objects"]) if history else tracked
    history_compact = lambda value: load("dialogue_chain_cases").compact_state(value, history_ids)
    candidate, entry = memory.commit_feedback(
        folder, planning, descriptor, initial,
        history, feedback,
        {"conversation_context": contract["conversation_context"],
         "rooms": contract["author_assigned_rooms"],
         "control_node_rooms": contract["control_node_rooms"]}, nodes, history_compact)
    memory.write_json(folder / "memory_after_feedback.json", candidate)
    evaluation = evaluate_private(case, initial, feedback)
    process_check = feedback.get("process_constraint_evaluation")
    if process_check is not None:
        evaluation["process_constraint_evaluation"] = process_check
        evaluation["checks"]["declared_process_constraints_passed"] = process_check["passed"]
        evaluation["private_reference_passed"] = all(evaluation["checks"].values())
    memory.write_json(folder / "evaluation.json", evaluation)
    return dict(case_id=case["case_id"], arm=arm, mode=case["mode"],
                status=feedback["status"], planning_status=proposal["status"],
                error=feedback.get("error"), action_count=feedback["proposed_action_count"],
                successful_actions=feedback["successful_action_count"],
                private_reference_passed=evaluation["private_reference_passed"],
                process_constraints_passed=process_check["passed"] if process_check is not None else None,
                memory_version=candidate["backend"]["memory_version"],
                historical_input_memory_version=initial["backend"]["memory_version"],
                token_usage=proposal["token_usage"])


def run(args, *, prepare=None, execute=None, experiment=None, input_names=INPUT_NAMES,
        branch_keys=None):
    if args.repeat < 1:
        raise ValueError("repeat must be positive")
    arms = getattr(args, "arms", ["original_replay", "oracle_decomposition"])
    allowed = set(experiment["arms"]) if experiment else {"original_replay", "oracle_decomposition"}
    if not arms or len(set(arms)) != len(arms) or not set(arms) <= allowed:
        raise ValueError("Select distinct supported experiment arms")
    cases = selected_cases(args)
    if branch_keys is not None:
        available = {(c["case_id"], r, a) for c in cases
                     for r in range(1, args.repeat + 1) for a in arms}
        if not branch_keys or not set(branch_keys) <= available:
            raise ValueError("Invalid or empty branch selection")
    integrity = verify_legacy()
    if integrity["status"] != "unchanged":
        raise ValueError("Legacy source hashes differ from the reviewed baseline")
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=False)
    memory.write_json(root / "approved_bundle.json", read(args.bundle))
    memory.write_json(root / "human_approval.json", read(args.approval))
    summary = dict(status="preparing", profile="oracle_decomposition_paired_snapshot_v1",
                   case_ids=[c["case_id"] for c in cases], repeats=args.repeat,
                   planned_branch_count=(len(branch_keys) if branch_keys is not None else
                                         len(cases)*len(arms)*args.repeat), processed_branch_count=0,
                   arms=arms, branches=[], preflight=[],
                   model=args.model, fresh_llm_calls=False, decomposition_model_calls=0,
                   branches_are_independent=True, historical_inputs_preserved=True,
                   simulation_executed=False, physical_grasp_verified=False,
                   review_only_cases_excluded=True, legacy_integrity=integrity,
                   control_interpretation="Identity cases resample downstream only; not Oracle repair",
                   all_inputs_preserved_except_decomposition=True)
    policy = "full_inventory_no_cosine_v1"
    summary.update(candidate_policy=policy,
                   additional_interventions=[] if policy == "legacy" else
                   ["Full skill inventory supplied without cosine; execution rules unchanged"],
                   all_inputs_preserved_except_decomposition=(policy == "legacy"))
    if policy != "legacy" or len(arms) != 2:
        summary["profile"] = "oracle_decomposition_selected_arms_snapshot_v2"
        summary["comparison_scope"] = (
            "Selected-arm resampling against historical results; not a contemporaneous paired ablation.")
    if experiment:
        summary.update(experiment)
    summary_name = "comparison_summary.json" if experiment else "oracle_summary.json"
    run_label = "intent-comparison" if experiment else "oracle"
    dest = root / summary_name
    memory.write_json(dest, summary)
    encoder = None
    model_hash = None
    started = time.monotonic()
    input_hashes = {}
    engine_paths = list((ROOT / "embodied_pipeline").rglob("*.py"))
    engine_paths += [ROOT / f["path"] for f in read(ROOT / "embodied_pipeline/legacy_manifest.json")["files"]]
    engine_hashes = {str(p): sha(p) for p in engine_paths}
    memory.write_json(root / "source_hashes.json", engine_hashes)
    handler = logging.FileHandler(root / "run.console.log", encoding="utf-8")
    logging.getLogger().addHandler(handler)
    try:
        for repeat in range(1, args.repeat+1):
            for case in cases:
                # Alternate order across cases/repeats to reduce systematic arm-order effects.
                arms = summary["arms"] if (cases.index(case)+repeat) % 2 else summary["arms"][::-1]
                for arm in arms:
                    if branch_keys is not None and (case["case_id"], repeat, arm) not in branch_keys:
                        continue
                    folder = root / "cases" / case["case_id"] / f"repeat_{repeat:02}" / arm
                    planning, record = (prepare or prepare_branch)(case, arm, folder)
                    record.update(repeat=repeat, path=folder.relative_to(root).as_posix())
                    summary["preflight"].append(record)
                    if arm == "oracle_decomposition" and record["decomposition_precheck"] != "accepted":
                        raise ValueError("Approved Oracle payload failed precheck")
                    for name in input_names:
                        input_hashes[str(planning / name)] = sha(planning / name)
        memory.write_json(dest, summary)
        if args.preflight_only:
            summary["status"] = "preflight_completed_no_model_calls"
        elif not os.environ.get("DASHSCOPE_API_KEY", "").strip():
            summary.update(status="blocked_environment", error="DASHSCOPE_API_KEY is not configured in this process")
        else:
            summary.update(candidate_policy="full_inventory_no_cosine_v1", embedding_used=False)
            for record in summary["preflight"]:
                case = next(c for c in cases if c["case_id"] == record["case_id"])
                folder = root / record["path"]
                summary.update(status="running", active_branch=record["path"])
                memory.write_json(dest, summary)
                logging.info("START %s case=%s arm=%s repeat=%s", run_label, case["case_id"], record["arm"], record["repeat"])
                try:
                    row = (execute or execute_branch)(args, case, record["arm"], folder, folder / "planning", encoder)
                except BranchTransportError as exc:
                    if not getattr(args, "continue_on_transport_error", False):
                        raise
                    # A transport failure never permits continuation after input mutation.
                    if not all(sha(p) == h for p, h in input_hashes.items()):
                        raise RuntimeError("Input changed during failed transport; batch integrity failure") from exc
                    state = read(folder / "planning/runtime_state.json")
                    row = dict(case_id=case["case_id"], arm=record["arm"], mode=case["mode"],
                        status="transport_error", planning_status="infrastructure_error",
                        error=str(exc), error_category="transport_error", action_count=0,
                        successful_actions=0, private_reference_passed=None,
                        evaluation_performed=False, memory_committed=False,
                        historical_input_memory_version=state["backend"]["memory_version"],
                        token_usage=exc.proposal["token_usage"],
                        failed_stage=exc.proposal.get("failed_stage"),
                        continuation="Recorded transport failure; continue with independent branch")
                    memory.write_json(folder / "branch_error.json", row)
                    logging.warning("Transport failure recorded; continuing next independent branch: %s", record["path"])
                row.update(repeat=record["repeat"], path=record["path"])
                summary["branches"].append(row)
                summary["processed_branch_count"] += 1
                summary["transport_error_branch_count"] = sum(
                    b["status"] == "transport_error" for b in summary["branches"])
                memory.write_json(dest, summary)
                logging.info("END %s case=%s arm=%s status=%s", run_label, case["case_id"], record["arm"], row["status"])
            summary["status"] = ("completed_with_transport_errors" if summary.get("transport_error_branch_count")
                                 else "completed")
    except KeyboardInterrupt:
        summary["status"] = "interrupted"
    except Exception as exc:
        summary.update(status="harness_error", error=str(exc))
        logging.exception("%s experiment stopped", run_label)
    finally:
        if encoder is not None:
            memory.write_json(root / "embedding_metrics.json", encoder.stats())
            encoder.close()
        summary.update(duration_seconds=time.monotonic()-started,
                       remaining_branch_count=summary["planned_branch_count"]-summary["processed_branch_count"],
                       engine_unchanged=all(sha(p) == h for p, h in engine_hashes.items()),
                       archived_sources_unchanged=all(
                           (ROOT / ref["path"]).is_file() and sha(ROOT / ref["path"]) == ref["sha256"]
                           for c in cases for ref in c["source_files"].values()),
                       copied_inputs_unchanged=all(sha(p) == h for p, h in input_hashes.items()),
                       embedding_model_sha256=model_hash,
                       embedding_model_unchanged=None if model_hash is None else
                       load("skill_retrieval").model_fingerprint(args.embedding_model_dir) == model_hash)
        usage = load("run_crossroom_task").usage(root)
        summary["token_usage"] = usage
        summary["known_fee_cny_no_discount"] = (usage["prompt_tokens"]*.8 + usage["completion_tokens"]*2.7)/1e6
        summary["fee_scope"] = "Recorded provider usage only; missing/timeout usage is unknown, not zero."
        summary["fresh_llm_calls"] = any(root.rglob("*_request.json"))
        summary["decomposition_model_calls"] = sum(1 for _ in root.rglob("decomposition_*_request.json"))
        if not all(summary[k] for k in ("engine_unchanged", "archived_sources_unchanged", "copied_inputs_unchanged")):
            summary["status"] = "invalid_integrity"
        if summary["embedding_model_unchanged"] is False:
            summary["status"] = "invalid_integrity"
        memory.write_json(dest, summary)
        write_run_manifest(root)
        logging.getLogger().removeHandler(handler)
        handler.close()
    label = "comparison_summary" if experiment else "oracle_summary"
    print(f"status={summary['status']} {label}={dest}")
    return 0 if summary["status"] in ("completed", "completed_with_transport_errors",
                                       "preflight_completed_no_model_calls") else 2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--approval", type=Path, default=DEFAULT_APPROVAL)
    parser.add_argument("--case-ids", nargs="+")
    parser.add_argument("--arms", nargs="+", choices=("original_replay", "oracle_decomposition"),
                        default=["original_replay", "oracle_decomposition"])
    parser.add_argument("--candidate-policy", choices=("full_inventory_no_cosine_v1", "legacy", "prerequisite_complete"), default="full_inventory_no_cosine_v1",
                        help="Deprecated aliases accepted; current requests always use the full skill inventory without cosine")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--continue-on-transport-error", action="store_true",
                        help="Record API transport failure and continue independent branches")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--model", default="qwen3.8-flash")
    parser.add_argument("--base-url", default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    parser.add_argument("--embedding-python", type=Path, default=Path("/home/lewis/miniforge3/envs/skill_embed/bin/python"))
    parser.add_argument("--embedding-model-dir", type=Path, default=Path("/mnt/d/NebulaAI/models/skill_embedding_minilm_v1"))
    args = parser.parse_args(argv)
    args.matrix_contract = "typed5_schema"
    args.alignment_wire_format = "matrix"
    args.top_k, args.cosine_threshold, args.cosine_margin = 2, .35, .05
    if args.output_dir is None:
        args.output_dir = ROOT / "outputs/vlm_full_scene_v1/runs" / datetime.now().strftime("oracle_d_%Y%m%d_%H%M%S_%f")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    return run(args)
