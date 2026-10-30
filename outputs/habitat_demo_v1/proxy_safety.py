"""Bounded geometry checks for proxy translations, NOT real robot/arm control."""
from __future__ import annotations

import copy
import itertools
import math

from sequential_proxy_core import ExecutionRejected
from proxy_placement import validate_bounds
from run_navigation_demo import habitat_vector

POLICY = {"name": "sampled_proxy_motion_guard_v1", "translation_step_m": .02,
          "rotation_step_deg": 5., "contact_penetration_tol_m": .001,
          "ray_start_tolerance_m": .002, "max_segment_m": 5.,
          "physics_advanced_during_probe": False, "continuous_collision_proof": False,
          "robot_body_collision_verified": False, "physical_arm_motion_verified": False}


class AuditCameraRejected(ValueError):
    pass


def quaternion_values(q):
    return [float(q.scalar), *map(float, q.vector)]


def rotate_point(p, q):
    w, x, y, z = q
    v = [x, y, z]
    cross = lambda a, b: [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]
    t = [2*a for a in cross(v, p)]
    second = cross(v, t)
    return [p[i] + w*t[i] + second[i] for i in range(3)]


def collision_center_registration(obj, graph_center):
    """Align the instantiated collider center, not the Blender free-joint origin.

    Restricted to this bounding-box-collider proxy. Does not infer a physical COM
    from the graph, or calibrate an arbitrary render asset.
    """
    lo = list(map(float, obj.collision_shape_aabb.min))
    hi = list(map(float, obj.collision_shape_aabb.max))
    validate_bounds(lo, hi)
    if len(graph_center) != 3 or not all(math.isfinite(v) for v in graph_center):
        raise ValueError("INVALID_GRAPH_REGISTRATION_CENTER")
    offset = rotate_point([(a+b)/2 for a,b in zip(lo, hi)], quaternion_values(obj.rotation))
    translation = [graph_center[i]-offset[i] for i in range(3)]
    return {"policy": "graph_bbox_center_to_instantiated_collision_bbox_center_v1",
            "graph_center": list(graph_center), "local_collision_bounds": {"min":lo, "max":hi},
            "graph_center_offset_from_rigid_origin": offset, "translation":translation,
            "physical_com_measured": False, "general_asset_calibration": False}


def motion_samples(start, goal, q0, q1):
    if any(len(v) != n for v, n in ((start, 3), (goal, 3), (q0, 4), (q1, 4))):
        raise ValueError("INVALID_PROXY_MOTION")
    if not all(math.isfinite(v) for seq in (start, goal, q0, q1) for v in seq):
        raise ValueError("NONFINITE_PROXY_MOTION")
    if any(abs(sum(v*v for v in q)-1.) > 1e-5 for q in (q0, q1)):
        raise ValueError("INVALID_PROXY_ROTATION")
    distance = math.dist(start, goal)
    if distance > POLICY["max_segment_m"]:
        raise ValueError("PROXY_SEGMENT_TOO_LONG")
    dot = sum(a*b for a, b in zip(q0, q1))
    if dot < 0:
        q1, dot = [-v for v in q1], -dot
    angle = 2*math.acos(min(1., max(0., dot)))
    count = max(1, math.ceil(distance/POLICY["translation_step_m"]),
                math.ceil(angle/math.radians(POLICY["rotation_step_deg"])))
    result = []
    for i in range(count+1):
        t = i/count
        theta = angle/2
        if theta <= 1e-8:
            q = [(1-t)*a+t*b for a, b in zip(q0, q1)]
        else:
            q = [(math.sin((1-t)*theta)*a+math.sin(t*theta)*b)/math.sin(theta) for a,b in zip(q0,q1)]
        norm = math.sqrt(sum(v*v for v in q))
        result.append(([a+(b-a)*t for a, b in zip(start, goal)], [v/norm for v in q]))
    return result


def segment_hits(sim, hs, start, end, ignore_ids=()):
    length = math.dist(start, end)
    if length <= 1e-8:
        return []
    ray = hs.geo.Ray()
    ray.origin = habitat_vector(start)
    ray.direction = habitat_vector([(end[i]-start[i])/length for i in range(3)])
    result = []
    for hit in sim.cast_ray(ray, max_distance=length).hits:
        if int(hit.object_id) in ignore_ids:
            continue
        point = list(map(float, hit.point))
        distance = math.dist(start, point)
        if not math.isfinite(distance):
            raise ValueError("NONFINITE_COLLISION_RAY_HIT")
        if POLICY["ray_start_tolerance_m"] < distance <= length + 1e-6:
            result.append({"other_object_id": int(hit.object_id), "position": point, "distance_m": distance})
    return result


def checked_motion(backend, target, goal, phase, rotation=None, commit=True, via=()):
    """Preflight the complete segment, restore probes, then apply sampled proxy poses."""
    obj = backend.instances[target]
    saved = {name: getattr(obj, name) for name in
             ("translation", "rotation", "motion_type", "linear_velocity", "angular_velocity", "awake")}
    q0 = quaternion_values(saved["rotation"])
    q1 = quaternion_values(rotation) if rotation is not None else q0
    report = {"phase": phase, "object_id": target, "start": list(map(float, saved["translation"])),
              "goal": list(map(float, goal)), "status": "checking", "probe_count": 0,
              "rotation_start_wxyz": q0, "rotation_goal_wxyz": q1,
              "committed": False, "policy": copy.deepcopy(POLICY)}
    try:
        bounds = obj.collision_shape_aabb
        lo, hi = list(map(float, bounds.min)), list(map(float, bounds.max))
        validate_bounds(lo, hi)
        report['local_collision_bounds'] = {'min':lo, 'max':hi}
        samples = [(report["start"], q0)]
        waypoints = [*via, report["goal"]]
        for index, waypoint in enumerate(waypoints):
            leg = motion_samples(samples[-1][0], waypoint, samples[-1][1],
                                 q1 if index == len(waypoints)-1 else q0)
            samples.extend(leg[1:])
        report["waypoints"] = copy.deepcopy(waypoints)
        offsets = list(itertools.product(*[(a, (a+b)/2, b) for a, b in zip(lo, hi)]))
        obj.motion_type = backend.motion.DYNAMIC
        obj.linear_velocity = obj.angular_velocity = habitat_vector([0., 0., 0.])
        import magnum as mn
        previous = None
        for index, (position, q) in enumerate(samples):
            obj.rotation = mn.Quaternion(mn.Vector3(*q[1:]), q[0])
            obj.translation = habitat_vector(position)
            obj.awake = True
            backend.sim.perform_discrete_collision_detection()
            report["probe_count"] += 1
            blockers = []
            for c in backend.sim.get_physics_contact_points():
                a, b = int(c.object_id_a), int(c.object_id_b)
                if int(obj.object_id) not in (a, b):
                    continue
                distance = float(c.contact_distance)
                if not math.isfinite(distance) or distance < -POLICY["contact_penetration_tol_m"]:
                    evidence = {"other_object_id": b if a == obj.object_id else a,
                                "contact_distance_m": distance if math.isfinite(distance) else None}
                    for name in ("position_on_a_in_ws", "position_on_b_in_ws", "contact_normal_on_b_in_ws"):
                        value = getattr(c, name, None)
                        if value is not None:
                            evidence[name] = [float(v) if math.isfinite(float(v)) else None for v in value]
                    blockers.append(evidence)
            if previous is not None and not blockers:
                old_position, old_q = previous
                for offset in offsets:
                    old = rotate_point(offset, old_q)
                    new = rotate_point(offset, q)
                    hits = segment_hits(backend.sim, backend.hs,
                        [old_position[i]+old[i] for i in range(3)],
                        [position[i]+new[i] for i in range(3)], {int(obj.object_id)})
                    if hits:
                        blockers = hits
                        break
            if blockers:
                report.update(status="blocked", blocked_probe_index=index, blocked_position=position,
                              blocked_rotation_wxyz=q,
                              blockers=blockers)
                raise ExecutionRejected("PROXY_OBJECT_PATH_BLOCKED", report)
            previous = (position, q)
    except ExecutionRejected:
        raise
    except Exception as exc:
        report.update(status="check_unavailable", error=str(exc))
        raise ExecutionRejected("PROXY_MOTION_CHECK_UNAVAILABLE", report) from exc
    finally:
        obj.motion_type = saved["motion_type"]
        obj.rotation, obj.translation = saved["rotation"], saved["translation"]
        obj.linear_velocity, obj.angular_velocity = saved["linear_velocity"], saved["angular_velocity"]
        obj.awake = saved["awake"]
        backend.sim.perform_discrete_collision_detection()
    report["status"] = "sampled_clear"
    if commit:
        import magnum as mn
        for index, (position, q) in enumerate(samples[1:], 1):
            obj.rotation = mn.Quaternion(mn.Vector3(*q[1:]), q[0])
            obj.translation = habitat_vector(position)
            audit = getattr(backend, "audit", None)
            if audit is not None and (index % 4 == 0 or index == len(samples)-1):
                audit.capture("proxy_motion", motion_phase=phase, motion_sample=index,
                              held_state_pending_action_commit=True)
        report["committed"] = True
    return report


def robot_segment_check(backend, start, end):
    """Navmesh point + three torso-height rays; no real quadruped volume claim."""
    if not backend.sim.pathfinder.is_navigable(habitat_vector(end)):
        raise ExecutionRejected("ROBOT_NAVMESH_POSITION_REJECTED", {"position": end})
    for height in (.3, .8, 1.2):
        a, b = [start[0], start[1]+height, start[2]], [end[0], end[1]+height, end[2]]
        ignore = {int(backend.instances[67].object_id)}
        hits = segment_hits(backend.sim, backend.hs, a, b, ignore)
        if hits:
            raise ExecutionRejected("ROBOT_PROXY_PATH_BLOCKED", {"start": start, "goal": end,
                                    "height_offset_m": height, "blockers": hits})


def camera_pose(sim, hs, robot, rotation, ignore_ids=()):
    """Short local camera boom with stage/other-object line-of-sight checks."""
    forward = rotate_point([0., 0., -1.], rotation)
    target = [robot[0], robot[1]+1.0, robot[2]]
    anchor = [robot[0], robot[1]+1.65, robot[2]]
    # Validate vertical mast and several nearby columns (near-plane margin).
    for ox, oz in ((0., 0.), (.04, 0.), (-.04, 0.), (0., .04), (0., -.04)):
        base = [robot[0]+ox, robot[1]+.05, robot[2]+oz]
        top = [anchor[0]+ox, anchor[1], anchor[2]+oz]
        if segment_hits(sim, hs, base, top, ignore_ids):
            raise AuditCameraRejected("AUDIT_CAMERA_MAST_BLOCKED")
    for boom in (.65, .35, .0):
        eye = [anchor[0]-forward[0]*boom, anchor[1], anchor[2]-forward[2]*boom]
        if any(segment_hits(sim, hs, [anchor[0]+ox, anchor[1]+oy, anchor[2]+oz],
                           [eye[0]+ox, eye[1]+oy, eye[2]+oz], ignore_ids)
               for ox, oy, oz in ((0., 0., 0.), (.04, 0., 0.), (-.04, 0., 0.),
                                 (0., .04, 0.), (0., -.04, 0.), (0., 0., .04), (0., 0., -.04))):
            continue
        # Never look through a wall to the robot marker.
        if boom and segment_hits(sim, hs, eye, target, ignore_ids):
            continue
        if boom == 0:
            target = [eye[i]+forward[i] for i in range(3)]
            target[1] -= .35
        return {"eye": eye, "target": target, "mode": "checked_short_chase" if boom else "checked_robot_mounted",
                "boom_m": boom, "geometry_checked": True,
                "scope": "Finite ray margin checks, not a full camera volume proof"}
    raise AuditCameraRejected("AUDIT_CAMERA_NO_CLEAR_VIEW")
