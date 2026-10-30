"""State/history commit and log persistence; no new state schema or policy."""

import copy
from .._legacy import load, resolve

_EXPORTS = {
    "read_json": ("run_navigation_demo", "read_json"),
    "write_json": ("run_navigation_demo", "write_json"),
    "initial_state": ("run_crossroom_task", "initial_state"),
    "compact_state": ("dialogue_chain_cases", "compact_state"),
    "ResponseAudit": ("response_audit", "ResponseAudit"),
}
__all__ = list(_EXPORTS) + ["commit_feedback"]


def __getattr__(name):
    return resolve(_EXPORTS, name)


def commit_feedback(folder, planning, descriptor, state, history, execution, cfg, nodes, compact):
    """Extracted unchanged feedback commit transaction, including failed turns.

    State/history share one pending-file replacement. Auxiliary files retain the
    legacy write order. This is symbolic bookkeeping, NOT observed scene memory;
    do not route it through evidence-gated update_memory.apply_events.
    """
    io = load("run_navigation_demo")
    n = descriptor["round"]
    candidate = copy.deepcopy(execution["resulting_state"])
    candidate["backend"]["memory_version"] = n
    results = [{k: copy.deepcopy(t[k]) for k in ("action_id", "skill_id", "arguments", "status", "error")}
               for t in execution["trace"]]
    entry = dict(round=n, instruction=descriptor["instruction"],
        source="proxy_execution_feedback_not_physical", begin_state=compact(state),
        end_state=compact(candidate),
        actions=[dict(skill_id=t["skill_id"], arguments=copy.deepcopy(t["arguments"]))
                 for t in execution["trace"] if t["status"] == "proxy_action_succeeded"],
        binding_notes=[], execution_status=execution["status"], error=execution["error"],
        action_results=results)
    binding = planning / "llm_binding_proposal.json"
    if binding.is_file():
        entry["binding_notes"] = io.read_json(binding).get("bindings", [])
    check_cfg = copy.deepcopy(cfg)
    check_cfg["conversation_context"]["proxy_execution_history"] = history + [entry]
    load("planning_start_contract").conversation_context(check_cfg, nodes)
    checkpoint = dict(state=candidate, history=history + [entry])
    pending = folder / "memory_checkpoint.pending.json"
    io.write_json(pending, checkpoint)
    pending.replace(folder / "memory_checkpoint.json")
    # The caller adopts candidate/history immediately, then saves the auxiliary
    # memory_after_feedback file, preserving legacy failure-side write ordering.
    return candidate, entry

