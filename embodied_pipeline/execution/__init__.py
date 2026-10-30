"""Symbolic proxy adapter and an explicitly unimplemented robot boundary."""

from typing import Protocol
import copy
from .._legacy import resolve, load

_EXPORTS = {
    "SerialExecutor": ("sequential_proxy_core", "SerialExecutor"),
    "SymbolicBackend": ("json_symbolic_review", "SymbolicBackend"),
}
__all__ = list(_EXPORTS) + ["execute_proxy", "ExecutionBackend", "RobotBackend"]


def execute_proxy(plan, alignment, library, nodes, state, compact):
    from ..validation.process_constraints import audit
    from ..planning.process_contract import grounded_restrictions
    contract = grounded_restrictions(alignment)
    # Capture every protected object even if the caller's display projection
    # tracks only the payload. This affects trace evidence, not proxy effects.
    protected = {i for r in (contract or {}).get("restrictions", []) for i in r["element_ids"]}
    def capture(snapshot):
        result = compact(snapshot)
        for obj in protected:
            if str(obj) in snapshot["backend"]["objects"]:
                result.setdefault("objects", {})[str(obj)] = copy.deepcopy(snapshot["backend"]["objects"][str(obj)])
        return result
    initial = copy.deepcopy(state)
    result = load("dialogue_execution_feedback").execute_proxy(plan, alignment, library, nodes, state, capture)
    check = audit(alignment, initial, result)
    if check is not None:
        result["process_constraint_evaluation"] = check
    return result


class ExecutionBackend(Protocol):
    def execute(self, plan, alignment, library, nodes, state, compact):
        """Return execution feedback; never substitute desired goals for results."""
        ...


class RobotBackend:
    """Interface reservation only. Never claims that robot execution happened."""

    def execute(self, plan, alignment, library, nodes, state, compact):
        raise NotImplementedError("Robot backend is not connected; use the symbolic proxy")


def __getattr__(name):
    return resolve(_EXPORTS, name)

