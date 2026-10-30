"""Paired fresh-LLM decomposition prompts on the reviewed historical snapshots."""

import argparse
import logging
from pathlib import Path
import shutil
import subprocess
import sys
from functools import partial
from datetime import datetime
from types import SimpleNamespace

from .._legacy import ROOT, load
from .. import memory
from ..planning.intent_prompt import VERSION, ARMS, compose
from ..planning.process_contract import compose_process_prompt, VERSION as PROCESS_VERSION
from . import oracle_decomposition as oracle

# Live decomposition issues a fresh allocation. All factual inputs stay byte-identical.
INPUT_NAMES = tuple(n for n in oracle.INPUT_NAMES if n != "identifier_allocation.json")
PROCESS_PROFILE = "complete_intent_relevant_constraints_process_v2"


def system_prompt(base, arm, process_constraints=False):
    result = compose(base, arm)
    if process_constraints:
        result = compose_process_prompt(result)
    return load("typed_matrix_ids").decomp_prompt(result, 5)


def prepare_branch(case, arm, destination, *, process_constraints=False):
    planning = destination / "planning"
    for name in INPUT_NAMES:
        target = planning / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(oracle.source_path(case["source_files"][name]), target)
        oracle.check_english_input(oracle.read(target))
    state = oracle.read(planning / "runtime_state.json")
    contract = oracle.read(planning / "execution_contract.json")
    if state != contract["current_runtime_state"]:
        raise ValueError("Runtime/contract state mismatch")
    if oracle.read(planning / "round_input.json")["instruction"] != case["instruction_en"]:
        raise ValueError("Instruction differs from the reviewed snapshot")
    memory.write_json(destination / "memory_checkpoint.json", dict(
        state=state, history=contract["conversation_context"]["proxy_execution_history"]))
    base = load("align_objects").DECOMPOSITION_PROMPT
    # This is the exact system prompt after the existing typed5 namespace rewrite.
    prompt = system_prompt(base, arm, process_constraints)
    oracle.check_english_input(prompt)
    memory.write_json(destination / "decomposition_prompt.json",
                      dict(version=PROCESS_PROFILE if process_constraints else VERSION, arm=arm, system_prompt=prompt))
    record = dict(case_id=case["case_id"], arm=arm, source_batch=case["batch"],
        source_episode=case["episode"], source_round=case["round"],
        intervention="decomposition_system_prompt_only",
        prompt_sha256=oracle.content_hash(prompt), input_precheck="accepted",
        decomposition_precheck="not_applicable_fresh_generation",
        decomposition_model_called=False, oracle_payload_injected=False,
        full_scene_and_history_preserved=True, private_evaluation_forwarded=False,
        english_input_check="passed_no_chinese_text", original_artifacts_overwritten=False,
        allocation_policy="fresh_typed5_scope_per_live_branch")
    if process_constraints:
        record["process_constraint_contract"] = PROCESS_VERSION
    memory.write_json(destination / "intervention.json", record)
    return planning, record


def alignment_from_live(args, descriptor, summary):
    planning = args.output_dir
    command = [
        sys.executable, "-m", "embodied_pipeline.planning.intent_alignment_worker",
        "--prompt-arm", args.prompt_arm, "--instruction", descriptor["instruction"],
        "--scene-dir", str(planning / "input_memory/scene"),
        "--matching-mode", "llm_graph", "--execution-contract", str(planning / "execution_contract.json"),
        "--model", args.model, "--base-url", args.base_url,
        "--request-timeout", "180", "--request-attempts", "2",
        "--validation-attempts", "1", "--decomposition-format", "matrix",
        "--binding-format", "matrix", "--matrix-contract", "typed5_schema",
        "--output", str(planning / "object_alignment.json"),
        "--decomposition-output", str(planning / "task_decomposition.json")]
    if getattr(args, "process_constraints", False):
        command.append("--process-constraints")
    with (planning / "alignment.console.log").open("w", encoding="utf-8") as log:
        completed = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    summary["stages"].append(dict(stage="fresh_decomposition_and_live_object_binding",
                                  exit_code=completed.returncode))
    target = planning / "object_alignment.json"
    if not target.is_file():
        raise RuntimeError("Alignment worker produced no result; inspect alignment.console.log")
    alignment = oracle.read(target)
    if completed.returncode or alignment.get("status") != "success":
        summary["upstream_error"] = alignment.get("error")
        raise ValueError("Live decomposition/object binding failed: " +
                         alignment.get("error", {}).get("message", "unknown"))
    if alignment["task_decomposition"]["instruction"] != descriptor["instruction"]:
        raise ValueError("Instruction changed during live decomposition")
    return alignment


def execute_branch(args, case, arm, folder, planning, encoder):
    options = SimpleNamespace(**dict(vars(args), prompt_arm=arm))
    result = oracle.execute_branch(options, case, arm, folder, planning, encoder,
        alignment_provider=alignment_from_live,
        decomposition_origin="fresh_llm_" + arm, live_decomposition=True)
    result["mode"] = "fresh_llm_prompt_comparison"
    result["source_case_mode"] = case["mode"]
    decomp = planning / "task_decomposition.json"
    result["decomposition_validated"] = decomp.is_file()
    result["subtask_count"] = len(oracle.read(decomp)["subtasks"]) if decomp.is_file() else None
    alignment = oracle.read(planning / "object_alignment.json")
    result["object_binding_accepted"] = alignment.get("status") == "success"
    result["skill_matrix_accepted"] = (planning / "skill_plan_proposed.json").is_file()
    error = result.get("error") or ""
    result["binding_reason_duplicate"] = any(
        text in error for text in ("Repeated binding reason", "reasons contains duplicate rows"))
    return result


def resume_selection(args):
    """Select unfinished branches into a new run; completed failures stay completed."""
    source = args.resume_from.resolve()
    saved = oracle.read(source)
    if saved.get("profile") not in (VERSION, PROCESS_PROFILE) or saved.get("arms") != list(ARMS):
        raise ValueError("Resume source is not this prompt comparison profile")
    use_process = saved["profile"] == PROCESS_PROFILE
    if getattr(args, "process_constraints", None) not in (None, use_process):
        raise ValueError("Resume must preserve its process constraint contract")
    args.process_constraints = use_process
    if saved.get("status") not in ("harness_error", "interrupted", "blocked_environment",
                                   "preflight_completed_no_model_calls", "completed",
                                   "completed_with_transport_errors"):
        raise ValueError("Resume source must be stopped and have valid integrity")
    if not all(saved.get(k) is True for k in
               ("engine_unchanged", "archived_sources_unchanged", "copied_inputs_unchanged")):
        raise ValueError("Resume source failed integrity checks")
    if saved.get("candidate_policy") != "full_inventory_no_cosine_v1" or saved["model"] != args.model:
        raise ValueError("Resume model/candidate policy differs")
    if args.repeat not in (None, saved["repeats"]):
        raise ValueError("Resume inherits its repeat count")
    key = lambda row: (row["case_id"], row["repeat"], row["arm"])
    planned = [key(row) for row in saved["preflight"]]
    complete = [key(row) for row in saved["branches"]]
    if (len(set(planned)) != len(planned) or len(set(complete)) != len(complete)
            or not set(complete) <= set(planned)
            or len(planned) != saved["planned_branch_count"]
            or len(complete) != saved["processed_branch_count"]):
        raise ValueError("Resume source has inconsistent branch accounting")
    pending = set(planned) - set(complete)
    deferred = set()
    if args.case_ids is not None:
        requested = set(args.case_ids)
        if len(requested) != len(args.case_ids) or not requested <= {k[0] for k in pending}:
            raise ValueError("Select distinct cases with unfinished branches")
        deferred = {k for k in pending if k[0] not in requested}
        pending -= deferred
    if not pending:
        raise ValueError("No unfinished branches remain")
    base = load("align_objects").DECOMPOSITION_PROMPT
    for row in saved["preflight"]:
        if key(row) in pending:
            prompt = system_prompt(base, row["arm"], use_process)
            if oracle.content_hash(prompt) != row["prompt_sha256"]:
                raise ValueError("Decomposition prompt changed since the source run")
    args.repeat = saved["repeats"]
    args.case_ids = [c for c in saved["case_ids"] if any(k[0] == c for k in pending)]
    return pending, dict(
        resume_source=str(source), resume_source_sha256=oracle.sha(source),
        source_planned_branch_count=len(planned), source_completed_branch_count=len(complete),
        resumed_branch_keys=[dict(case_id=c, repeat=r, arm=a) for c, r, a in planned
                             if (c, r, a) in pending],
        deferred_branch_keys=[dict(case_id=c, repeat=r, arm=a) for c, r, a in planned
                             if (c, r, a) in deferred],
        resume_policy="Rerun unfinished branches from archived input snapshots in a new directory; preserve every completed result",
        retry_unit="Whole branch, including fresh decomposition; not continuation of a saved binding request")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-ids", nargs="+")
    parser.add_argument("--repeat", type=int)
    parser.add_argument("--resume-from", type=Path,
                        help="Prior comparison summary; rerun only unfinished branches in a new directory")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--process-constraints", dest="process_constraints", action="store_true",
                      help="Separate process restrictions from action tasks (default for new runs)")
    mode.add_argument("--legacy-process-contract", dest="process_constraints", action="store_false",
                      help="Use the historical task/constraint representation")
    parser.set_defaults(process_constraints=None)
    parser.add_argument("--stop-on-transport-error", action="store_true",
                        help="Stop the batch after a recorded API transport failure instead of continuing")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--model", default="qwen3.8-flash")
    parser.add_argument("--base-url", default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    parser.add_argument("--embedding-python", type=Path, default=Path("/home/lewis/miniforge3/envs/skill_embed/bin/python"))
    parser.add_argument("--embedding-model-dir", type=Path, default=Path("/mnt/d/NebulaAI/models/skill_embedding_minilm_v1"))
    args = parser.parse_args(argv)
    args.bundle, args.approval = oracle.DEFAULT_BUNDLE, oracle.DEFAULT_APPROVAL
    args.arms = list(ARMS)
    args.matrix_contract, args.alignment_wire_format = "typed5_schema", "matrix"
    args.candidate_policy = "full_inventory_no_cosine_v1"
    args.top_k, args.cosine_threshold, args.cosine_margin = 2, .35, .05
    args.continue_on_transport_error = not args.stop_on_transport_error
    branch_keys, resume_info = (resume_selection(args) if args.resume_from else (None, {}))
    if args.process_constraints is None:
        args.process_constraints = True
    if args.repeat is None:
        args.repeat = 1
    if args.output_dir is None:
        args.output_dir = ROOT / "outputs/vlm_full_scene_v1/runs" / datetime.now().strftime(
            "intent_constraints_%Y%m%d_%H%M%S_%f")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    experiment = dict(profile=PROCESS_PROFILE if args.process_constraints else VERSION, arms=list(ARMS),
        decomposition_source="fresh_llm_in_both_arms", oracle_payload_injected=False,
        control_interpretation="Same archived tasks and snapshots; both arms regenerate decomposition and downstream stages",
        comparison_scope="Paired prompt comparison with complete candidates in both arms; each repeat is an independent snapshot",
        all_inputs_preserved_except_decomposition=False,
        additional_interventions=["Treatment adds complete-intent/relevant-constraint instructions to decomposition system prompt"],
        shared_changes_from_historical_runs=["Full skill inventory without cosine filtering", "Fresh typed5 allocation per branch"],
        private_reference_scope="Reviewed Oracle reference is used only for post-execution evaluation",
        no_recovery=True,
        transport_failure_policy=("continue_independent_branches" if args.continue_on_transport_error else "stop_batch"),
        transport_failures_are_semantic_failures=False)
    if args.process_constraints:
        experiment["process_constraint_contract"] = PROCESS_VERSION
        experiment["shared_changes_from_historical_runs"].append(
            "Explicit process restriction table, grounding, action-request context and trajectory audit")
    experiment.update(resume_info)
    return oracle.run(args, prepare=partial(prepare_branch, process_constraints=args.process_constraints), execute=execute_branch,
                      experiment=experiment, input_names=INPUT_NAMES, branch_keys=branch_keys)


if __name__ == "__main__":
    raise SystemExit(main())
