#!/usr/bin/env python3
"""Second task: a grounded navigation-only fixture, not fresh LLM planning."""
import argparse
import copy
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from run_navigation_demo import read_json, write_json, flatten_steps, cg_to_habitat, horizontal_distance

SCENE = Path("/mnt/d/NebulaAI/output/batches/scene201_vith_full_20260915_181750/seq_0001__hf_house_201")


def fixture(alignment, plan, nodes):
    if alignment.get("status") != "success":
        raise ValueError("Successful source alignment required")
    binding = alignment.get("context_bindings", {}).get("sideboard", {})
    ids = binding.get("element_ids", [])
    if len(ids) != 1:
        raise ValueError("NO object can be used in this task")
    node = next((n for n in nodes if n.get("id") == ids[0]), None)
    if node is None or node.get("bbox_center") is None:
        raise ValueError("NO object can be used in this task")
    step = next((s for s in flatten_steps(plan)
                 if s.get("skill_id") == "navigate_to_object"
                 and s.get("arguments", {}).get("target", {}).get("role_id") == "sideboard"
                 and s["arguments"]["target"].get("element_id") == ids[0]), None)
    if step is None:
        raise ValueError("No verified source navigation binding for sideboard")
    instruction = "Go to the sideboard."
    new_alignment = {"status": "success", "instruction": instruction,
        "scene_reference": alignment.get("scene_reference"),
        "context_bindings": {"sideboard": copy.deepcopy(binding)},
        "provenance": "Reused successful source binding; no new LLM or similarity scoring"}
    new_plan = {"status": "proposal_needs_environment_validation", "instruction": instruction,
        "plan_origin": "derived_navigation_test_fixture_not_llm_generated",
        "subtask_plans": [{"subtask_id": "go_to_sideboard", "steps": [copy.deepcopy(step)]}]}
    return new_alignment, new_plan, node


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-plan-run", type=Path,
        default=SCENE / "vlm_scenegraph_skill_regression/scene201_verified_20260921_123303")
    args = p.parse_args()
    alignment = read_json(args.source_plan_run / "object_alignment.json")
    plan = read_json(args.source_plan_run / "skill_plan_final.json")
    graph = Path(alignment["scene_reference"]) / "scene_graph.json"
    aligned, proposed, node = fixture(alignment, plan, read_json(graph))
    root = SCENE / "sideboard_navigation_task" / datetime.now().strftime("scene201_%Y%m%d_%H%M%S_%f")
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / "object_alignment.json", aligned)
    write_json(root / "skill_plan_fixture.json", proposed)
    rows = []
    for seed in (201, 202, 203):
        case = root / f"seed_{seed}"
        case.mkdir()
        print(f"START sideboard seed={seed} output={case}", flush=True)
        cmd = [sys.executable, str(Path(__file__).with_name("run_navigation_demo.py")),
            "--alignment-json", str(root / "object_alignment.json"),
            "--skill-plan-json", str(root / "skill_plan_fixture.json"),
            "--scene-graph-json", str(graph),
            "--stage-config", str(SCENE / "scene_package/hf_house_201/hf_house_201.stage_config.json"),
            "--navmesh", str(SCENE / "scene_package/hf_house_201/hf_house_201.navmesh"),
            "--seed", str(seed), "--output", str(case / "navigation_demo.json"),
            "--log-file", str(case / "navigation_demo.log")]
        time_file = case / "resource_usage.txt"
        if Path("/usr/bin/time").is_file():
            cmd = ["/usr/bin/time", "-v", "-o", str(time_file), *cmd]
        with (case / "console.log").open("w", encoding="utf-8") as output:
            proc = subprocess.run(cmd, stdout=output, stderr=subprocess.STDOUT)
        result = read_json(case / "navigation_demo.json") if (case / "navigation_demo.json").exists() else {}
        navs = result.get("navigation_results", [])
        distance = (horizontal_distance(navs[0]["end"], cg_to_habitat(node["bbox_center"], "x_z_minus_y"))
                    if navs and "end" in navs[0] else None)
        passed = (proc.returncode == 0 and result.get("status") == "navigation_demo_passed"
                  and len(navs) == 1 and navs[0].get("status") == "arrived"
                  and distance is not None and distance <= 1.5)
        peak = None
        if time_file.exists():
            for line in time_file.read_text().splitlines():
                if "Maximum resident set size (kbytes):" in line:
                    peak = round(int(line.rsplit(":", 1)[1]) / 1048576, 3)
        episode = {"schema_version": "1.0", "instruction": proposed["instruction"],
            "record_type": "navigation_only_proxy_training_episode", "training_eligible": passed,
            "status": "goal_observed_in_proxy" if passed else "task_failed",
            "plan_origin": proposed["plan_origin"], "source_plan_run": str(args.source_plan_run),
            "object_id": node["id"], "seed": seed, "start": result.get("start"),
            "navigation_results": navs, "target_horizontal_distance_m": distance,
            "proximity_threshold_m": 1.5, "peak_rss_gib": peak,
            "physical_robot_execution_verified": False, "scene_graph_modified": False}
        write_json(case / "task_level_episode.json", episode)
        rows.append({"seed": seed, "passed": passed, "exit_code": proc.returncode,
                     "output": str(case), "peak_rss_gib": peak, "target_distance_m": distance})
        print(f"END seed={seed} passed={passed} exit_code={proc.returncode}", flush=True)
    summary = {"status": "passed" if all(r["passed"] for r in rows) else "failed",
        "instruction": proposed["instruction"], "plan_origin": proposed["plan_origin"],
        "success_rate": sum(r["passed"] for r in rows) / len(rows), "cases": rows,
        "scope": "Second task execution fixture; not end-to-end LLM evaluation"}
    write_json(root / "regression_summary.json", summary)
    print(f"summary={root / 'regression_summary.json'}", flush=True)
    return 0 if summary["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
