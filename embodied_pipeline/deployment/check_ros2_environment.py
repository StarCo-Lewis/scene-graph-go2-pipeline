"""Read-only ROS2/Python/message preflight. No rclpy.init, discovery or goals."""
import importlib
import json
import os
from pathlib import Path
import sys


def definitions(path):
    fields = [[]]
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        line = line.split("#", 1)[0].strip()
        if line == "---":
            fields.append([])
        elif line:
            dtype, name = line.split()
            fields[-1].append((dtype, name))
    if len(fields) != 3:
        raise ValueError("Expected Goal, Result and Feedback sections")
    return fields


def main():
    report = {"python": sys.executable, "python_version": sys.version.split()[0],
              "ros_distro": os.environ.get("ROS_DISTRO"), "domain": os.environ.get("ROS_DOMAIN_ID"),
              "rmw": os.environ.get("RMW_IMPLEMENTATION"), "checks": {}, "errors": [],
              "network_accessed": False, "robot_execution_attempted": False}
    for name in ("tkinter", "rclpy", "rosidl_runtime_py.convert", "nav_interfaces.action"):
        try:
            module = importlib.import_module(name)
            report["checks"][name] = {"imported": True, "path": getattr(module, "__file__", None)}
        except Exception as exc:
            report["checks"][name] = {"imported": False}
            report["errors"].append(name+": "+str(exc))
    if report["checks"]["nav_interfaces.action"]["imported"]:
        try:
            from nav_interfaces.action import NavigateTo
            from rclpy.type_support import check_for_type_support
            check_for_type_support(NavigateTo)  # Load native support, not just Python stubs.
            path = Path(__file__).parent / "ros2_client_ws/src/nav_interfaces/action/NavigateTo.action"
            normalize = {"float64": "double", "float32": "float", "bool": "boolean"}
            for name, fields in zip(("Goal", "Result", "Feedback"), definitions(path)):
                expected = {key: normalize.get(dtype, dtype) for dtype, key in fields}
                actual = {k: v.replace("/msg/", "/") for k, v in getattr(NavigateTo, name).get_fields_and_field_types().items()}
                if actual != expected:
                    raise ValueError(name+" schema mismatch: "+repr(actual))
            report["message_schema"] = "matches_local_navigation_repository_not_live_robot_verified"
        except Exception as exc:
            report["errors"].append("NavigateTo native support/schema: "+str(exc))
    if report["ros_distro"] != "foxy":
        report["errors"].append("Expected Foxy to match the supplied dog log; do not assume cross-distribution compatibility")
    if report["domain"] != "0":
        report["errors"].append("Set ROS_DOMAIN_ID=0 in the launch environment")
    report["status"] = "local_environment_ready_not_connected" if not report["errors"] else "environment_incomplete_not_connected"
    print(json.dumps(report, indent=2, ensure_ascii=True))
    return 0 if not report["errors"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
