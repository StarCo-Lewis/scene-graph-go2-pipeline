"""Existing validation APIs. Exporting a check does not enable a new gate."""

from .._legacy import resolve

_EXPORTS = {
    "apply_contract": ("strict_prompt_v2", "apply_contract"),
    "normalize_and_record": ("skill_output_normalization", "normalize_and_record"),
    "decode_matrix": ("action_matrix", "decode_matrix"),
    "enforce": ("matrix_type_contract", "enforce"),
    "decode_decomposition": ("typed_matrix_ids", "decode_decomposition"),
    "from_binding_matrix": ("structural_matrix", "from_binding_matrix"),
    "validate_decomposition": ("align_objects", "validate_decomposition"),
    "validate_bindings": ("llm_graph_binding", "validate_bindings"),
    "check_binding_relations": ("llm_graph_binding", "check_binding_relations"),
    "validate_state_goal_set": ("runtime_goal_protocol", "validate_state_goal_set"),
    "conversation_context": ("planning_start_contract", "conversation_context"),
    "review_symbolic": ("json_symbolic_review", "review_symbolic"),
    "record_validation": ("response_audit", "validation"),
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    return resolve(_EXPORTS, name)

