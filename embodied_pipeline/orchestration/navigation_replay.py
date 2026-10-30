"""Safe default: write a request/preflight report without importing ROS."""
import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import uuid

from .._legacy import ROOT
from ..execution.navigation_contract import prepare, read, resolve, sha


def clean(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None  # Missing/non-finite measurements are never represented as zero.
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    return value


def write(path, data):
    path.write_text(json.dumps(clean(data), indent=2, ensure_ascii=True, allow_nan=False) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="embodied_pipeline/execution/d3_619_navigation_replay.json")
    parser.add_argument("--execute", action="store_true", help="Explicit live ROS2 navigation; robot may move")
    parser.add_argument("--confirm-motion", action="store_true", help="Operator authorizes this one goal")
    args = parser.parse_args(argv)
    run = ROOT / "outputs/real_scene_runs" / ("navigation_replay_" + datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8])
    run.mkdir(parents=True, exist_ok=False)
    summary = {"status": "preflight_failed_not_executed", "robot_execution_attempted": False,
               "memory_committed": False, "live_requested": args.execute}
    def emit(event, data):
        row = {"utc": datetime.now(timezone.utc).isoformat(), "event": event, "data": clean(data)}
        with (run / "events.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=True, allow_nan=False) + "\n")
            f.flush()
        print(event, flush=True)
    try:
        config_path = resolve(args.config)
        config = read(config_path)
        write(run / "config_snapshot.json", config)
        summary["config_sha256"] = sha(config_path)
        request = prepare(config)
        write(run / "navigation_request.json", request)
        summary.update(request)
        if args.execute:
            if not args.confirm_motion:
                raise ValueError("Live launch requires --execute --confirm-motion")
            if request["blockers"]:
                raise ValueError("Complete and verify configuration before live execution")
            for key, expected in config["ros_environment"].items():
                if os.environ.get(key) != str(expected):
                    raise ValueError(f"Environment mismatch: {key}; expected {expected!r}")
            from ..execution.ros2_navigation import execute
            summary.update(execute(request, emit))
        exit_code = 0 if summary["status"] in ("request_prepared_not_executed", "server_reported_arrival") else 2
    except (Exception, KeyboardInterrupt) as exc:
        summary["error"] = type(exc).__name__ + ": " + str(exc)
        summary["status"] = "launch_failed_or_interrupted"
        # Durable send evidence takes precedence over an exception during ROS cleanup.
        events = run / "events.jsonl"
        if events.exists() and '"event": "goal_send_started"' in events.read_text(encoding="utf-8"):
            summary["robot_execution_attempted"] = True
            summary["status"] = "execution_state_unknown_operator_check_required"
        exit_code = 2
    write(run / "run_summary.json", summary)
    print(f"status={summary['status']} summary={run / 'run_summary.json'}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
