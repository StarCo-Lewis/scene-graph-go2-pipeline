"""Unfiltered skill inventory for current pipeline requests; no embedding calls."""
POLICY = "full_inventory_no_cosine_v1"


def candidates(alignment, library, encoder=None, **legacy_options):
    """Keep the retrieval wire shape while offering every library skill per subtask.

    Legacy encoder/top-k arguments are accepted for caller compatibility only.
    Catalogue membership is not a claim of physical executability.
    """
    ids = [s["skill_id"] for s in library["skills"]]
    if not ids or any(not isinstance(s, str) or not s for s in ids) or len(set(ids)) != len(ids):
        raise ValueError("Skill inventory must contain unique nonempty skill IDs")
    tasks = alignment["task_decomposition"]["subtasks"]
    task_ids = [t["subtask_id"] for t in tasks]
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("Duplicate subtask IDs")
    return {"policy": POLICY, "cosine_filtering": False, "embedding_used": False,
            "inventory_size": len(ids),
            "scope": "All supplied library skills; selection, parameters, dependencies and feasibility remain subject to validation",
            "subtasks": [{"subtask_id": t["subtask_id"], "query": t["description"],
                          "candidate_skill_ids": list(ids), "policy": POLICY} for t in tasks]}
