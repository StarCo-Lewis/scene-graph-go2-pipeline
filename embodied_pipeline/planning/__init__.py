"""Task decomposition and OBJECT binding; not atomic skill selection."""

from .._legacy import resolve

_EXPORTS = {
    "decompose_and_bind": ("plan_live_round", "decompose_and_bind"),
    "call_llm": ("align_objects", "call_llm"),
    "request_bindings": ("llm_graph_binding", "request_bindings"),
    "close_runtime_objects": ("state_aware_planning", "close_runtime_objects"),
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    return resolve(_EXPORTS, name)

