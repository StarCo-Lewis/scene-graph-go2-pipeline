"""Shared wire contract. No robot SDK, controller tuning or coordinate guessing."""
import json
import math
import uuid

PROTOCOL = "nebula.navigation.tcp.v1"
MAX_LINE = 65536
POSE_KEYS = ("x", "y", "z", "roll", "pitch", "yaw")
GOAL_TYPES = {**{p+k: "double" for p in ("s_", "g_") for k in POSE_KEYS},
              **{k: "float" for k in ("pos_tol", "yaw_tol", "timeout", "max_speed")}}
# Operator-requested values updated 2026-10-23; not proof of a successful field test.
# Not a hardware speed guarantee. Other controller parameters remain unchanged.
FIELD_LIMITS = {"pos_tol": 0.05, "yaw_tol": 0.17, "timeout": 120.0, "max_speed": 0.5}


def clean(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    return value


def encode(value):
    data = json.dumps(clean(value), allow_nan=False, separators=(",", ":")).encode("utf-8") + b"\n"
    if len(data) > MAX_LINE:
        raise ValueError("Packet too large")
    return data


def decode(raw):
    if len(raw) > MAX_LINE:
        raise ValueError("Packet too large")
    value = json.loads(raw)
    if not isinstance(value, dict) or value.get("protocol") != PROTOCOL:
        raise ValueError("Incompatible bridge protocol; no fallback")
    return value


def validate_goal(packet):
    if packet.get("operator_confirmed") is not True:
        raise ValueError("Explicit operator authorization required")
    request_id = packet.get("request_id")
    if not isinstance(request_id, str) or str(uuid.UUID(request_id)) != request_id:
        raise ValueError("Canonical UUID request_id required")
    if not isinstance(packet.get("coordinate_frame"), str) or not packet["coordinate_frame"]:
        raise ValueError("Registered scene frame required")
    if not isinstance(packet.get("source_action_id"), str) or not packet["source_action_id"]:
        raise ValueError("Source action identity required")
    goal = packet.get("goal")
    if not isinstance(goal, dict) or set(goal) != set(GOAL_TYPES):
        raise ValueError("Exactly the 16 reviewed NavigateTo fields are required")
    for key, value in goal.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError("Nonfinite or invalid numeric field: " + key)
    for prefix in ("s_", "g_"):
        if any(goal[prefix+k] != 0 for k in ("z", "roll", "pitch")):
            raise ValueError("This field integration preserves z/roll/pitch = 0")
        if abs(goal[prefix+"yaw"]) > math.pi:
            raise ValueError("Yaw must use radians in [-pi, pi]")
    if any(goal[key] != expected for key, expected in FIELD_LIMITS.items()):
        raise ValueError("Motion request parameters differ from the reviewed field test; no silent tuning")
    return goal  # Unmodified; no speed scaling, path processing or parameter clamping.
