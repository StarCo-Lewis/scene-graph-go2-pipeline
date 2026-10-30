"""Offline XY AABB approach mapping, not a collision-free path planner."""
import math

from .navigation_contract import number, pose

BBOX_CLEARANCE_M = 0.4
GROUND_REGION_TAGS = frozenset(("floor", "floorboard", "flooring", "ground", "ground plane"))


def is_ground_region(node):
    # A semantic label selects a candidate-generation rule, not a walkability claim.
    return str(node.get("object_tag", "")).strip().lower() in GROUND_REGION_TAGS


def bounds(node):
    vectors = []
    for key in ("bbox_center", "bbox_extent"):
        vector = node.get(key)
        if not isinstance(vector, list) or len(vector) != 3:
            raise ValueError("Navigation mapping requires source-map AABB center and full extent")
        vectors.append([number(v, key) for v in vector])
    center, extent = vectors
    if any(v < 0 for v in extent):
        raise ValueError("AABB extent must not be negative")
    return ([center[i] - extent[i]/2 for i in range(2)],
            [center[i] + extent[i]/2 for i in range(2)])


def bbox_distance(node, point):
    """Distance of robot navigation reference point to the XY rectangle, metres."""
    p = pose(point, "navigation_point")
    low, high = bounds(node)
    return math.hypot(*(max(low[i]-p[k], 0, p[k]-high[i])
                        for i, k in enumerate(("x", "y"))))


def clearance_value(value):
    return number(value, "bbox_clearance_m", positive=True)


def validate_clearance(node, point, clearance_m=BBOX_CLEARANCE_M):
    clearance_m = clearance_value(clearance_m)
    distance = bbox_distance(node, point)
    if distance < clearance_m - 1e-8:
        raise ValueError(f"Navigation point must be at least {clearance_m:g} m outside the target XY AABB; not its center")
    return distance


def bbox_approach(node, start, clearance_m=BBOX_CLEARANCE_M):
    """Nearest boundary projection, then the selected clearance outward toward the current start.

    Robot footprint, walls, support furniture and reachability are NOT checked.
    Inside/on ground-region targets, choose the nearest face and extend outwards.
    Solid-object interiors remain ambiguous and are rejected. No free-space claim.
    """
    clearance_m = clearance_value(clearance_m)
    p = pose(start, "current_start")
    low, high = bounds(node)
    xy = [p["x"], p["y"]]
    boundary = [min(max(xy[i], low[i]), high[i]) for i in range(2)]
    delta = [xy[i]-boundary[i] for i in range(2)]
    distance = math.hypot(*delta)
    if distance <= 1e-9:
        if not is_ground_region(node):
            raise ValueError(
                f"当前起点 ({p['x']:.3f}, {p['y']:.3f}) 位于目标 ID {node.get('id', '?')} "
                f"的 XY 包围盒内部/边界：X=[{low[0]:.3f}, {high[0]:.3f}]，"
                f"Y=[{low[1]:.3f}, {high[1]:.3f}]。请核对场景起点，或手动填写已核对的外侧停靠点。")
        # Deterministic tie order +X, -X, +Y, -Y; no arbitrary diagonal or centre goal.
        faces = [(high[0]-xy[0], [high[0], xy[1]], [1, 0]),
                 (xy[0]-low[0], [low[0], xy[1]], [-1, 0]),
                 (high[1]-xy[1], [xy[0], high[1]], [0, 1]),
                 (xy[1]-low[1], [xy[0], low[1]], [0, -1])]
        _, boundary, outward = min(faces, key=lambda item: item[0])
        goal_xy = [boundary[i]+clearance_m*outward[i] for i in range(2)]
    else:
        goal_xy = [boundary[i]+clearance_m*delta[i]/distance for i in range(2)]
    center = node["bbox_center"]
    goal = {"x": goal_xy[0], "y": goal_xy[1],
            "yaw": math.atan2(center[1]-goal_xy[1], center[0]-goal_xy[0])}
    validate_clearance(node, goal, clearance_m)
    return goal
