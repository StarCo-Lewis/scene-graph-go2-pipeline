"""Validate optional observed facts; never accept simulated predictions as sensors."""
from __future__ import annotations

from datetime import datetime
from typing import Any


SOURCES = {
    "robot_telemetry", "camera_observation", "gripper_sensor", "navigation_map",
    "affordance_estimator", "human_verified", "simulator_observation",
}


def validate_observations(snapshot: dict[str, Any], known_ids: set[int],
                          predicates: set[str]) -> list[dict[str, Any]]:
    if not isinstance(snapshot, dict) or snapshot.get("schema_version") != "1.0":
        raise ValueError("observation snapshot requires schema_version 1.0")
    facts = snapshot.get("facts")
    if not isinstance(facts, list):
        raise ValueError("observation snapshot requires facts array")
    validated = []
    seen = set()
    for index, fact in enumerate(facts, 1):
        if not isinstance(fact, dict):
            raise ValueError(f"fact {index} must be an object")
        subject, predicate = fact.get("subject"), fact.get("predicate")
        source, observed_at = fact.get("source"), fact.get("observed_at")
        if subject != "robot" and (type(subject) is not int or subject not in known_ids):
            raise ValueError(f"fact {index} uses an ungrounded subject")
        if predicate not in predicates:
            raise ValueError(f"fact {index} uses an unsupported predicate")
        if source not in SOURCES:
            raise ValueError(f"fact {index} uses an untrusted source")
        if not isinstance(observed_at, str):
            raise ValueError(f"fact {index} requires observed_at")
        try:
            timestamp = datetime.fromisoformat(observed_at.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(f"fact {index} has invalid observed_at") from exc
        if timestamp.tzinfo is None:
            raise ValueError(f"fact {index} requires a timezone-aware timestamp")
        if "value" not in fact:
            raise ValueError(f"fact {index} requires value")
        value = fact["value"]
        if type(value) is int and value not in known_ids:
            raise ValueError(f"fact {index} refers to an ungrounded object value")
        if not (value is None or type(value) in {bool, int, str}):
            raise ValueError(f"fact {index} has unsupported value type")
        key = (str(subject), predicate)
        if key in seen:
            raise ValueError(f"duplicate observation for {key}")
        seen.add(key)
        validated.append({"subject": subject, "predicate": predicate, "value": value,
                          "source": source, "observed_at": observed_at})
    return validated
