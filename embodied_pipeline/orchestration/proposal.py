"""Existing one-shot proposal transaction, routed through module boundaries."""

import logging
import time
from types import SimpleNamespace

from .. import planning, instruction_skills as skills, validation, memory
from .._legacy import load
from ..planning.process_contract import grounded_restrictions


def propose_once(args, folder, encoder, *, alignment_provider=None):
    """Same stages, arguments, files and rejection behavior as the legacy path."""
    started = time.monotonic()
    profile = load("dialogue_execution_feedback").wire_profile(args)
    summary = dict(status="proposal_running", stages=[], fresh_llm_calls=True, **profile,
        simulation_executed=False, physical_grasp_verified=False,
        semantic_redecomposition_count=0, runtime_redecomposition_count=0,
        whole_instruction_review_requested=False, redundancy_review_requested=False,
        proposal_semantically_validated=False)
    plan = None
    alignment = None
    stage = "decomposition_and_binding"
    try:
        validation.apply_contract(folder)
        library = skills.enrich(memory.read_json(folder / "execution_skill_library.json"))
        memory.write_json(folder / "execution_skill_library.json", library)
        state = memory.read_json(folder / "runtime_state.json")
        nodes = memory.read_json(folder / "input_memory/scene/scene_graph.json")
        descriptor = memory.read_json(folder / "round_input.json")
        options = SimpleNamespace(**dict(vars(args), output_dir=folder, alignment_validation_attempts=1))
        provider = alignment_provider or planning.decompose_and_bind
        alignment = provider(options, descriptor, summary)
        memory.write_json(folder / "object_alignment_raw_semantic_1.json", alignment)
        alignment, closure = planning.close_runtime_objects(alignment, library, state, nodes)
        memory.write_json(folder / "runtime_object_closure_semantic_1.json", closure)
        memory.write_json(folder / "object_alignment.json", alignment)
        process = grounded_restrictions(alignment)
        if process is not None:
            library.setdefault("execution_contract", {})["process_constraints"] = process
            memory.write_json(folder / "process_constraints_grounded.json", process)
            summary["process_constraint_contract"] = process["version"]
        stage = "full_inventory_and_codebook"
        retrieval = skills.candidates(alignment, library)
        summary.update(candidate_policy=retrieval["policy"], embedding_used=False)
        memory.write_json(folder / "skill_retrieval_semantic_1.json", retrieval)
        book = skills.make_codebook(alignment, library)
        memory.write_json(folder / "action_codebook_semantic_1.json", book)
        stage = "matrix_protocol"
        raw = skills.request_matrix(alignment, library, book, args.model, args.base_url, "DASHSCOPE_API_KEY",
            retrieval=retrieval, widen_inventory=False, timeout=180, attempts=2,
            diagnostics_dir=folder, audit_context={"semantic_attempt": 1, "validation_attempt": 1,
                "skill_output_format": "matrix", "memory_mode": "proxy_execution_feedback"},
            matrix_contract=getattr(args, "matrix_contract", "legacy_json"))
        evidence = "llm_matrix_proposal_semantic_1_attempt_1.json"
        memory.write_json(folder / evidence, raw)
        normalized, note = validation.normalize_and_record(raw, "matrix", library, folder,
            "llm_skill_semantic_1_attempt_1", evidence,
            matrix_contract=getattr(args, "matrix_contract", "legacy_json"))
        summary["format_normalization"] = note
        decoded = validation.decode_matrix(normalized, book, alignment, library)
        memory.write_json(folder / "decoded_matrix_semantic_1_attempt_1.json", decoded)
        plan = skills.build_result(alignment, library, decoded)
        plan["proposal_origin"] = "fresh_llm_matrix_decoded_not_prevalidated"
        memory.write_json(folder / "skill_plan_proposed.json", plan)
        membership = skills.check_membership(decoded, retrieval, widened=True)
        membership.update(widened_inventory=False, check_only=True, execution_blocked=False)
        memory.write_json(folder / "skill_retrieval_membership_observation.json", membership)
        validation.record_validation(folder, "matrix_wire_protocol", 1, "accepted", evidence=evidence)
        summary.update(status="proposal_ready_not_semantically_validated", action_count=plan["action_count"],
            inventory_conflicts=membership["conflicts"])
    except (OSError, RuntimeError) as exc:
        summary.update(status="infrastructure_error", failed_stage=stage, error=str(exc))
        plan = None
        logging.error("Transport/runtime infrastructure error, not a simulated action failure: %s", exc)
    except (ValueError, KeyError, TypeError) as exc:
        upstream = summary.get("upstream_error") or {}
        infrastructure = upstream.get("category") == "infrastructure_error"
        summary.update(status="infrastructure_error" if infrastructure else "proposal_unavailable",
                       failed_stage=stage, error=str(exc))
        if stage == "matrix_protocol":
            validation.record_validation(folder, "matrix_wire_protocol", 1, "rejected", exc)
        plan = None
        logging.warning("%s without corrected proposal: %s", summary["status"], exc)
    finally:
        summary["duration_seconds"] = time.monotonic() - started
        summary["token_usage"] = load("run_crossroom_task").usage(folder)
        memory.write_json(folder / "planning_summary.json", summary)
    return summary, plan, alignment

