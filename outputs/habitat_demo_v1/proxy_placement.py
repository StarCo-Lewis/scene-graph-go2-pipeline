"""Pure policy for bounded, upright Scene201 proxy placement (not arm planning)."""
from __future__ import annotations

import math

POLICY = {
    "name": "upright_yaw_volume_checked_drop_v2",
    "allowed_release_yaws_deg": [0., 90., 180., 270.],
    "candidate_grid_step_m": .025,
    "release_bottom_gap_m": .04,
    "probe_bottom_gap_m": .005,
    "probe_step_m": .01,
    "max_center_offset_m": .30,
    "surface_edge_margin_m": .015,
    "blocking_contact_distance_m": .002,
    "support_penetration_limit_m": .001,
    "physics_advanced_during_search": False,
    "real_arm_motion_verified": False,
}


def validate_bounds(lower, upper):
    if len(lower) != 3 or len(upper) != 3 or any(
            not math.isfinite(a) or not math.isfinite(b) or a >= b for a, b in zip(lower, upper)):
        raise ValueError("INVALID_LOCAL_COLLISION_BOUNDS")


def yaw_quaternion(yaw_deg):
    if yaw_deg not in POLICY["allowed_release_yaws_deg"]:
        raise ValueError("UNSUPPORTED_PROXY_PLACEMENT_YAW")
    half = math.radians(yaw_deg) / 2
    return [math.cos(half), 0., math.sin(half), 0.]


def is_upright_rotation(q):
    try:
        return (len(q) == 4 and all(math.isfinite(v) for v in q)
                and abs(q[1]) <= 1e-6 and abs(q[3]) <= 1e-6
                and abs(sum(v * v for v in q) - 1.) <= 1e-6)
    except (TypeError, ValueError):
        return False


def same_upright_rotation(observed, expected):
    if not is_upright_rotation(observed) or not is_upright_rotation(expected):
        return False
    # Habitat uses float32 quaternions; q and -q encode the same rotation.
    return min(math.dist(observed, expected), math.dist(observed, [-v for v in expected])) <= 1e-6


def yaw_rotated_bounds(lower, upper, yaw_deg):
    validate_bounds(lower, upper)
    yaw_quaternion(yaw_deg)
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    corners = [(c * x + s * z, -s * x + c * z)
               for x in (lower[0], upper[0]) for z in (lower[2], upper[2])]
    return ([min(p[0] for p in corners), lower[1], min(p[1] for p in corners)],
            [max(p[0] for p in corners), upper[1], max(p[1] for p in corners)])


def candidate_offsets(board_extent_xz, lower, upper):
    """Fit the full upright local collider inside the graph footprint, near its center."""
    validate_bounds(lower, upper)
    if len(board_extent_xz) != 2 or any(not math.isfinite(v) or v <= 0 for v in board_extent_xz):
        raise ValueError("INVALID_SUPPORT_FOOTPRINT")
    offsets = []
    step = POLICY["candidate_grid_step_m"]
    values = [0.]
    for index in range(1, round(POLICY["max_center_offset_m"] / step) + 1):
        values.extend([-index * step, index * step])
    margin = POLICY["surface_edge_margin_m"]
    for dx in values:
        for dz in values:
            if math.hypot(dx, dz) > POLICY["max_center_offset_m"] + 1e-9:
                continue
            if all(offset + lower[axis] >= -extent / 2 + margin and
                   offset + upper[axis] <= extent / 2 - margin
                   for offset, axis, extent in ((dx, 0, board_extent_xz[0]), (dz, 2, board_extent_xz[1]))):
                offsets.append((dx, dz))
    # Stable order: closest to the supplied target first, no random candidate choice.
    return sorted(offsets, key=lambda offset: round(math.hypot(*offset), 8))


def footprint_samples(lower, upper):
    validate_bounds(lower, upper)
    return [(x, z) for x in (lower[0], (lower[0] + upper[0]) / 2, upper[0])
            for z in (lower[2], (lower[2] + upper[2]) / 2, upper[2])]


def descent_probe_positions(x, z, top, lower_y):
    """Discrete volume checks over only the short release-to-support descent."""
    release_y = top - lower_y + POLICY["release_bottom_gap_m"]
    last_y = top - lower_y + POLICY["probe_bottom_gap_m"]
    steps = math.ceil((release_y - last_y) / POLICY["probe_step_m"])
    return [[x, release_y + (last_y - release_y) * i / steps, z] for i in range(steps + 1)]


def blocking_contacts(contacts, object_id, surface_id):
    """Never relabel stage or unrelated object contact as intended support."""
    blockers = []
    for contact in contacts:
        a, b = int(contact.object_id_a), int(contact.object_id_b)
        if object_id not in (a, b):
            continue
        other = b if a == object_id else a
        distance = float(contact.contact_distance)
        blocked = (not math.isfinite(distance) or
                   (distance < -POLICY["support_penetration_limit_m"] if other == surface_id
                    else distance <= POLICY["blocking_contact_distance_m"]))
        if blocked:
            report = {"object_id": object_id, "other_object_id": other,
                      "contact_distance_m": distance if math.isfinite(distance) else None,
                      "reason": "support_penetration" if other == surface_id else "non_target_obstacle"}
            for name in ("position_on_a_in_ws", "position_on_b_in_ws"):
                point = getattr(contact, name, None)
                report[name] = list(map(float, point)) if point is not None else None
            blockers.append(report)
    return blockers
