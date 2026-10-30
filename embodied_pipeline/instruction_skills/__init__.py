"""Full-inventory candidates, action proposal and parameter codebooks."""

from .._legacy import resolve
from .full_inventory import candidates

_EXPORTS = {
    "enrich": ("skill_retrieval", "enrich"),
    "check_membership": ("skill_retrieval", "check_membership"),
    "make_codebook": ("action_matrix", "make_codebook"),
    "matrix_payload": ("action_matrix", "matrix_payload"),
    "request_matrix": ("action_matrix", "request_matrix"),
    "build_result": ("match_skills", "build_result"),
}
__all__ = list(_EXPORTS) + ["candidates"]


def __getattr__(name):
    return resolve(_EXPORTS, name)

