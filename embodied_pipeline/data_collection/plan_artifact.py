"""Completeness of a delivered engineering_plan.json; dependency-free for consumers.

The collection worker writes engineering_plan.json for every claimed task, including
stopped ones. File existence therefore means "delivered", not "planned". Consumers
that count, score or copy plans must use these helpers instead of Path.exists().
"""

import json
from pathlib import Path

COMPLETE = "complete"
INCOMPLETE = "incomplete"


def is_complete(envelope):
    """True only for an envelope that carries a complete skill plan.

    Envelopes written before artifact_completeness existed were only written after
    a plan was built, so a dict plan without the field is treated as complete.
    """
    if not isinstance(envelope, dict) or not isinstance(envelope.get("plan"), dict):
        return False
    if "artifact_completeness" in envelope:
        return envelope.get("artifact_completeness") == COMPLETE and envelope.get("planning_completed") is True
    return True


def read_complete(path):
    """Return the envelope at path if it is a complete plan, else None (missing/unreadable/incomplete)."""
    path = Path(path)
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return value if is_complete(value) else None


def has_complete(path):
    return read_complete(path) is not None


def summary_planning_completed(summary):
    """Complete-plan flag from a planning_summary row; legacy rows only had engineering_json_generated."""
    if not isinstance(summary, dict):
        return False
    if "planning_completed" in summary:
        return summary.get("planning_completed") is True
    return summary.get("engineering_json_generated") is True
