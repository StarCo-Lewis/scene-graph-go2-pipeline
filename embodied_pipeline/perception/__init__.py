"""Scene-input boundary. No new perception, inference or room-topology logic."""

from .._legacy import resolve
from .conceptgraphs_scene import load_real_scene

_EXPORTS = {
    "scene_fixture": ("run_contextual_ambiguity", "scene_fixture"),
    "normalize_scene_input": ("scene_input_interface", "normalize_scene_input"),
    "validate_edges": ("llm_graph_binding", "validate_edges"),
    "prepare_feedback": ("dialogue_execution_feedback", "prepare_feedback"),
}
__all__ = list(_EXPORTS) + ["load_real_scene"]


def __getattr__(name):
    return resolve(_EXPORTS, name)

