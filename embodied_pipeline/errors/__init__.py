"""Preserved error boundaries. No new retry, clarification or recovery policy."""

import copy
from .._legacy import resolve

_EXPORTS = {
    "ExecutionRejected": ("sequential_proxy_core", "ExecutionRejected"),
    "MatrixTypeError": ("matrix_type_contract", "MatrixTypeError"),
    "AlignmentError": ("align_objects", "AlignmentError"),
    "BindingError": ("llm_graph_binding", "BindingError"),
    "category": ("response_audit", "category"),
}
__all__ = list(_EXPORTS) + ["unavailable_result"]


def unavailable_result(state, error):
    """Same no-proposal result as the legacy feedback runner; no fake actions."""
    return dict(status="proposal_unavailable", trace=[], error=error,
                resulting_state=copy.deepcopy(state), proposed_action_count=0,
                attempted_action_count=0, successful_action_count=0,
                skipped_action_count=0, physical_execution=False)


def __getattr__(name):
    return resolve(_EXPORTS, name)

