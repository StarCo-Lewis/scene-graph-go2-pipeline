#!/usr/bin/env python3
"""Five far starts against the same restored memory target; no LLM requests."""
import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from run_memory_followup_navigation import SCENE
from run_navigation_demo import read_json, write_json
from run_task_level_regression import distinct_starts


def check_case(code, result):
    return (code == 0 and result.get("status") == "followup_proxy_passed"
            and result.get("training_eligible") is True
            and result.get("start_policy") == "sampled_far"
            and result.get("navigation", {}).get("actions", 0) > 0
            and result.get("initial_distance_to_new_object_m", 0) >= 3.0
            and result.get("distance_to_restored_object_m", float("inf")) <= 1.5
            and result.get("final_distance_to_old_position_m", 0) > 1.5
            and result.get("restored_bbox_position_error_m", float("inf")) <= 1e-4
            and result.get("memory_modified") is False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan-run", type=Path, default=SCENE / "memory_followup_planning/scene201_20260926_140047_110543")
    args = p.parse_args()
    root = SCENE / "memory_far_start_regression" / datetime.now().strftime("scene201_%Y%m%d_%H%M%S_%f")
    root.mkdir(parents=True, exist_ok=False)
    rows = []
    summary_path = root / "regression_summary.json"
    for seed in (201, 202, 203, 204, 205):
        output = root / f"seed_{seed}"
        command = [sys.executable, str(Path(__file__).with_name("run_memory_followup_navigation.py")),
            "--plan-run", str(args.plan_run), "--output-dir", str(output),
            "--start-policy", "sampled_far", "--start-seed", str(seed), "--min-start-distance-m", "3.0"]
        print(f"START memory_far seed={seed} output={output}", flush=True)
        with (root / f"seed_{seed}.console.log").open("w", encoding="utf-8") as log:
            proc = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        result = read_json(output / "followup_execution.json") if (output / "followup_execution.json").exists() else {}
        row = {"seed": seed, "passed": check_case(proc.returncode, result), "exit_code": proc.returncode,
               "output": str(output), "start_position": result.get("navigation", {}).get("start"),
               "actions": result.get("navigation", {}).get("actions"),
               "initial_distance_to_new_object_m": result.get("initial_distance_to_new_object_m"),
               "final_distance_to_new_object_m": result.get("distance_to_restored_object_m"),
               "final_distance_to_old_position_m": result.get("final_distance_to_old_position_m"),
               "peak_rss_gib": result.get("process_peak_rss_gib"), "error": result.get("error")}
        rows.append(row)
        peaks = [r["peak_rss_gib"] for r in rows if isinstance(r["peak_rss_gib"], (int, float))]
        summary = {"status": "running" if len(rows) < 5 else "passed" if all(r["passed"] for r in rows) else "failed",
            "scope": "Same restored memory and existing plan; varied far-start navigation only",
            "completed_cases": len(rows), "success_rate": sum(r["passed"] for r in rows) / len(rows),
            "separated_start_count": distinct_starts(rows), "start_separation_threshold_m": 0.25,
            "max_peak_rss_gib": max(peaks) if peaks else None, "cases": rows}
        if len(rows) == 5 and summary["status"] == "passed" and summary["separated_start_count"] < 3:
            summary["status"] = "insufficient_start_diversity"
        write_json(summary_path, summary)
        print(f"END memory_far seed={seed} passed={row['passed']} exit_code={proc.returncode}", flush=True)
    print(f"summary={summary_path}", flush=True)
    return 0 if summary["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
