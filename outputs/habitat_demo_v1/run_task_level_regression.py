#!/usr/bin/env python3
"""Sequential Habitat repeatability and controlled negative tests, not a benchmark."""
import argparse
import json
import os
import math
import subprocess
from datetime import datetime
from pathlib import Path

SCENE = "/mnt/d/NebulaAI/output/batches/scene201_vith_full_20260915_181750/seq_0001__hf_house_201"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def check_case(name, exit_code, episode):
    positive = name.startswith(("repeat_", "start_"))
    if positive:
        return (exit_code == 0 and episode.get("training_eligible") is True
                and episode.get("goal_observed_in_proxy") is True
                and episode.get("grasp_assumed") is True)
    expected = {"navigation_blocked": "navigation_failed",
                "proximity_rejected": "proxy_reach_failed",
                "placement_injected_failure": "placement_test_failure"}[name]
    return (exit_code == 2 and episode.get("source_execution_status") == expected
            and episode.get("training_eligible") is False
            and episode.get("goal_observed_in_proxy") is False
            and (name == "placement_injected_failure"
                 or episode.get("grasp_assumed") is False))


def distinct_starts(rows, minimum_separation=0.25):
    """Count a conservative greedy set of horizontally separated start points."""
    selected = []
    for row in rows:
        point = row.get("start_position")
        if not isinstance(point, list) or len(point) != 3:
            continue
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in point):
            continue
        if all(math.hypot(point[0] - other[0], point[2] - other[2]) >= minimum_separation
               for other in selected):
            selected.append(point)
    return len(selected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-root", type=Path, default=Path(SCENE))
    parser.add_argument("--vary-starts", action="store_true",
                        help="Five fixed RNG seeds with first-reachable sampling instead of nearest sampling")
    args = parser.parse_args()
    root = args.scene_root / "task_level_regression" / datetime.now().strftime("scene201_%Y%m%d_%H%M%S_%f")
    root.mkdir(parents=True, exist_ok=False)
    launcher = Path(__file__).with_name("run_scene201_task_level_demo.sh")
    cases = [(f"repeat_{i}", []) for i in range(1, 4)] + [
        ("navigation_blocked", ["--max-actions", "0"]),
        ("proximity_rejected", ["--proxy-reach-m", "0.01"]),
        ("placement_injected_failure", ["--test-fault", "before_place"]),
    ]
    if args.vary_starts:
        cases = [(f"start_{seed}", ["--start-seed", str(seed), "--start-selection", "first"])
                 for seed in (201, 202, 203, 204, 205)]
    rows = []
    summary_path = root / "regression_summary.json"
    for name, flags in cases:
        case_dir = root / name
        case_dir.mkdir()
        env = dict(os.environ, DEMO_RUN=str(case_dir))
        print(f"START {name} output={case_dir}", flush=True)
        with (case_dir / "launcher.log").open("w", encoding="utf-8") as output:
            proc = subprocess.run(["bash", str(launcher), *flags], env=env,
                                  stdout=output, stderr=subprocess.STDOUT)
        row = {"case": name, "exit_code": proc.returncode,
               "output_directory": str(case_dir), "passed": False}
        try:
            episode = read(case_dir / "task_level_episode.json")
            resource = read(case_dir / "resource_summary.json")
            row.update(passed=check_case(name, proc.returncode, episode),
                       episode_status=episode.get("source_execution_status"),
                       training_eligible=episode.get("training_eligible"),
                       peak_rss_gib=resource.get("peak_rss_gib"),
                       start_position=episode.get("start_sampling", {}).get("position"),
                       start_seed=episode.get("start_sampling", {}).get("seed"),
                       navigation=[{k: s.get(k) for k in
                                    ("status", "actions", "planned_geodesic_m", "final_horizontal_distance_m")}
                                   for s in episode.get("steps", [])
                                   if s.get("skill_id") == "navigate_to_object"])
        except (OSError, ValueError, KeyError) as exc:
            row["error"] = str(exc)
        rows.append(row)
        repeats = [r for r in rows if r["case"].startswith(("repeat_", "start_"))]
        peaks = [r["peak_rss_gib"] for r in rows
                 if isinstance(r.get("peak_rss_gib"), (int, float))]
        summary = {
            "schema_version": "1.0", "status": "running" if len(rows) < len(cases)
                else ("passed" if all(r["passed"] for r in rows) else "failed"),
            "scope": ("Same scene and task, five start seeds; navigation-start robustness only"
                      if args.vary_starts else "Same scene, task and start seed; repeatability only, not generalization"),
            "completed_cases": len(rows), "planned_cases": len(cases),
            "positive_repeat_pass_rate": sum(r["passed"] for r in repeats) / len(repeats),
            "max_peak_rss_gib": max(peaks) if peaks else None,
            "negative_test_scope": ("Not run in varying-start mode" if args.vary_starts else
                                    "Zero navigation budget, restrictive proximity, controlled placement fault"),
            "separated_start_count": distinct_starts(rows),
            "start_separation_threshold_m": 0.25,
            "start_diversity_passed": distinct_starts(rows) >= 3 if args.vary_starts else None,
            "cases": rows,
        }
        if args.vary_starts and len(rows) == len(cases) and not summary["start_diversity_passed"]:
            summary["status"] = "insufficient_start_diversity"
        summary_path.write_text(json.dumps(summary, ensure_ascii=True, indent=2) + "\n", encoding="ascii")
        print(f"END {name} passed={row['passed']} exit_code={proc.returncode}", flush=True)
    print(f"summary={summary_path}", flush=True)
    return 0 if summary["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
