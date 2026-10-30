#!/usr/bin/env python3
"""Three continuous positive episodes plus four fail-closed cases, no API."""
import argparse
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from run_continuous_sequence import SCENE, OUT
from run_navigation_demo import read_json, write_json
from proxy_placement import POLICY, same_upright_rotation


def evaluate_case(result, returncode, fault):
    if fault == "none":
        rounds = result.get("rounds", [])
        checks = {"completed": returncode == 0 and result.get("status") == "passed",
                  "one_simulator": result.get("simulator_creation_count") == 1,
                  "one_initial_start": result.get("initial_start_sampling_count") == 1,
                  "continuous": result.get("uninterrupted_simulator_execution") is True,
                  "five_rounds_passed": len(rounds) == 5 and all(r.get("status") == "passed" for r in rounds),
                  "sixteen_actions": result.get("executed_action_count") == 16,
                  "source_unchanged": result.get("base_graph_unchanged") is True,
                  "cross_round_holding": result.get("cross_round_held_object_continuations") == [3, 5],
                  "training_scope_not_overclaimed": result.get("training_eligible") is False}
        route = [s.get("target_id") for s in rounds[0].get("actions", [])] if rounds else []
        checks["necessary_revisit_executed"] = route == [67, 13, 67]
        try:
            original = rounds[0]["actions"][0]["target_center"]
            moved = rounds[3]["actions"][0]["target_center"]
            checks["moved_object_target_used"] = original != moved
        except (KeyError, IndexError):
            checks["moved_object_target_used"] = False
        placements = [a for row in rounds for a in row.get("actions", [])
                      if a.get("skill_id") == "place_object_on_surface"]
        checks["two_support_verified_placements"] = len(placements) == 2 and all(
            a.get("status") == "proxy_instance_support_passed" and
            a.get("validation", {}).get("checks", {}).get("direct_support_contact_frames") is True
            for a in placements)
        checks["volume_checked_placements"] = len(placements) == 2 and all(
            a.get("placement_preparation", {}).get("placement_policy", {}).get("name") == POLICY["name"] and
            any(c.get("valid") is True and c.get("volume_probes") and
                all(not probe.get("blocking_contacts") for probe in c["volume_probes"])
                for c in a.get("surface_candidates", [])) for a in placements)
        checks["upright_zero_velocity_releases"] = len(placements) == 2 and all(
            same_upright_rotation(a.get("at_release", {}).get("rotation_wxyz"),
                                  a.get("placement_preparation", {}).get("release_rotation_wxyz")) and
            a.get("at_release", {}).get("linear_velocity") == [0., 0., 0.] and
            a.get("at_release", {}).get("angular_velocity") == [0., 0., 0.] for a in placements)
        return checks
    errors = {"place_without_grasp": "LIFTED_HELD_OBJECT_REQUIRED",
              "duplicate_grasp": "GRIPPER_NOT_EMPTY", "missing_object": "ungrounded object",
              "unsupported_skill": "unknown skill"}
    checks = {"expected_rejection": returncode == 2 and result.get("status") == "failed" and errors[fault] in result.get("error", ""),
              "not_training_eligible": result.get("training_eligible") is False}
    if fault in ("missing_object", "unsupported_skill"):
        checks["no_simulator_started"] = result.get("simulator_creation_count") == 0
    else:
        expected_last = 1 if fault == "place_without_grasp" else 2
        rounds = result.get("rounds", [])
        checks["later_rounds_skipped"] = bool(rounds) and rounds[-1]["round"] == expected_last and rounds[-1]["status"] == "failed"
    return checks


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=SCENE / "continuous_sequence_regression" / datetime.now().strftime("scene201_%Y%m%d_%H%M%S_%f"))
    p.add_argument("--scene-dir", type=Path, default=SCENE / "vlm_scenegraph_regression/scene201_verified_20260921_113316/scene")
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    summary = {"status": "running", "cases": [], "training_eligible": False,
               "scope": "Every positive case is one uninterrupted simulator; separate cases intentionally create separate simulators"}
    cases = [(f"seed_{seed}", seed, "none") for seed in (201, 202, 203)]
    cases += [(fault, 201, fault) for fault in ("place_without_grasp", "duplicate_grasp", "missing_object", "unsupported_skill")]
    for name, seed, fault in cases:
        output = args.output_dir / name
        print(f"START {name} output={output}", flush=True)
        command = [sys.executable, str(OUT / "habitat_demo_v1/run_continuous_sequence.py"), "--seed", str(seed),
                   "--test-fault", fault, "--scene-dir", str(args.scene_dir), "--output-dir", str(output)]
        row = {"case": name, "passed": False}
        try:
            with (args.output_dir / f"{name}.console.log").open("w", encoding="utf-8") as log:
                child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                         text=True, encoding="utf-8", errors="replace")
                for line in child.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()
                code = child.wait()
            result = read_json(output / "pipeline_summary.json")
            checks = evaluate_case(result, code, fault)
            row.update(passed=all(checks.values()), checks=checks, exit_code=code,
                       process_peak_rss_gib=result.get("process_peak_rss_gib"), summary=str(output / "pipeline_summary.json"))
        except (OSError, ValueError, KeyError) as exc:
            row["error"] = str(exc)
        summary["cases"].append(row)
        write_json(args.output_dir / "regression_summary.json", summary)
        print(f"END {name} passed={row['passed']} exit_code={row.get('exit_code')}", flush=True)
    summary["status"] = "passed" if all(r["passed"] for r in summary["cases"]) else "failed"
    write_json(args.output_dir / "regression_summary.json", summary)
    print(f"status={summary['status']} summary={args.output_dir / 'regression_summary.json'}", flush=True)
    return 0 if summary["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
