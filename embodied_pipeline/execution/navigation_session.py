"""Carry measured terminal poses between goals; never estimate pose from a target."""
import math


def arrived_pose(data):
    """Accept only a successful ROS terminal PoseStamped in the server's scene frame."""
    result = data.get("result", {})
    if (data.get("ros_goal_status") != 4 or result.get("success") is not True
            or result.get("reason") != "arrived"):
        return None
    return terminal_pose(data)


def terminal_pose(data):
    """A failed goal may still report a valid actual scene pose; never infer arrival."""
    if data.get("ros_goal_status") not in (4, 5, 6):
        return None
    result = data.get("result", {})
    if result.get("reason") in ("no_pose", "stale_pose", "pose_stale", "pose_timeout", "localization_lost"):
        return None
    stamped = result.get("final_pose") or {}
    if stamped.get("header", {}).get("frame_id") != "scene":
        return None
    pose = stamped.get("pose", {})
    p, q = pose.get("position", {}), pose.get("orientation", {})
    values = [p.get("x"), p.get("y")] + [q.get(k) for k in ("x", "y", "z", "w")]
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in values):
        return None
    x, y, qx, qy, qz, qw = values
    if abs(sum(v*v for v in (qx, qy, qz, qw)) - 1) > .02:
        return None
    return {"x": x, "y": y, "yaw": math.atan2(2*(qw*qz+qx*qy), 1-2*(qy*qy+qz*qz))}


class NavigationSession:
    """Local provenance, not a claim that the remote server has retained its transform."""
    def __init__(self):
        self.initial_pose = None
        self.scene_key = None
        self.completed = 0
        self.current_pose = None
        self.source_request_id = None

    def sent(self, start, scene_key):
        if self.scene_key is not None and self.scene_key != scene_key:
            raise ValueError("场景已变化，需要重新核对坐标系。")
        self.scene_key = scene_key
        if self.initial_pose is None:
            self.initial_pose = dict(start)
        self.current_pose = None
        self.source_request_id = None

    def finish(self, data, request_id, scene_key):
        self.current_pose = None
        self.source_request_id = None
        if self.scene_key != scene_key:
            return None
        value = terminal_pose(data)
        if value is not None:
            if arrived_pose(data) is not None:
                self.completed += 1
            self.current_pose = dict(value)
            self.source_request_id = request_id
        return value

    def invalidate_pose(self):
        # Disconnect/reset of UI does NOT reset the robot or its scene transform.
        self.current_pose = None
        self.source_request_id = None

    def provenance(self, start):
        carried = self.current_pose == start and self.source_request_id is not None
        return {"policy": "align_first_keep_scene_frame",
                "initial_scene_pose": self.initial_pose,
                "completed_segments_in_client_session": self.completed,
                "start_pose_source": "previous_server_final_pose" if carried else "operator_measured_scene_pose",
                "previous_result_request_id": self.source_request_id if carried else None,
                "remote_alignment_policy_verified_by_client": False}
