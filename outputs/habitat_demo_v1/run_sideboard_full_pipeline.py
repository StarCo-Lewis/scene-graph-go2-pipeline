#!/usr/bin/env python3
"""Fresh LLM planning through Habitat navigation, using an existing verified map."""
import argparse
import os
import subprocess
import shutil
from datetime import datetime
from pathlib import Path
from run_navigation_demo import read_json, write_json, flatten_steps, targets, cg_to_habitat, horizontal_distance

SCENE = Path("/mnt/d/NebulaAI/output/batches/scene201_vith_full_20260915_181750/seq_0001__hf_house_201")
INSTRUCTION = "Go to the sideboard."


def validate_resume(alignment, decomposition, scene_dir, instruction=INSTRUCTION):
    if alignment.get("status") != "success" or decomposition.get("instruction") != instruction:
        raise ValueError("Resume input must be a successful alignment for this exact instruction")
    if Path(alignment.get("scene_reference", "")).resolve() != scene_dir.resolve():
        raise ValueError("Resume scene reference mismatch")
    if alignment.get("task_decomposition") != decomposition:
        raise ValueError("Saved decomposition differs from alignment decomposition")
    if not decomposition.get("goal_state"):
        raise ValueError("Resume decomposition has no goal; rerun decomposition instead")


def build_execution_contract(mode):
    from run_manipulation_proxy import EXPECTED_SKILLS
    sequence = EXPECTED_SKILLS if mode == "kettle_to_sideboard" else ["navigate_to_object"]
    return {"adapter_mode": mode, "supported_skill_ids": sorted(set(sequence)),
            "supported_ordered_sequence": list(sequence),
            "planning_capabilities": {"validate_supplied_on_in_edges": True,
                "unknown_relation_policy": "Do not infer unobserved initial relations"},
            "planning_context": "Objects are localized by object alignment before execution. Do not add separate locate/search/identify/inspect subtasks merely to ground objects. Group actionable subtasks so the supported sequence can be assigned in order without empty subtasks.",
            "limitations": "No runtime visual inspection or safety/temperature measurement. Grasp is assumed only after validated approach. Never claim inspection or physical grasp success.",
            "unsupported_requirement_policy": "Do not remove explicit user safety/inspection requirements. If the instruction requires an unsupported action, report it unresolved rather than substituting or claiming success."}


def scoped_skill_library(library, contract):
    allowed = set(contract["supported_skill_ids"])
    skills = [s for s in library["skills"] if s["skill_id"] in allowed]
    if {s["skill_id"] for s in skills} != allowed:
        raise ValueError("Executor capabilities reference missing library skills")
    return {**library, "skills": skills, "execution_contract": contract}


def validate_adapter(alignment, plan, nodes, instruction=INSTRUCTION, target_id=13):
    if alignment.get("status") != "success" or plan.get("instruction") != instruction:
        raise ValueError("Fresh alignment or instruction validation failed")
    steps = flatten_steps(plan)
    if len(steps) != 1 or steps[0].get("skill_id") != "navigate_to_object":
        raise ValueError("Navigation adapter requires exactly one navigation skill; plan will not be silently edited")
    items = targets(alignment, plan, nodes)
    if len(items) != 1 or items[0]["element_id"] != target_id:
        raise ValueError(f"Binding did not resolve to required verified scene instance {target_id}")
    return items[0]


def validate_memory_target(state, nodes, target_id):
    if state.get("memory_type") != "incremental_object_state":
        raise ValueError("Unsupported memory state")
    entity = state.get("entities", {}).get(str(target_id), {})
    node = next((n for n in nodes if n.get("id") == target_id), None)
    if not entity.get("active") or node is None:
        raise ValueError("NO object can be used in this task")
    if node != entity.get("node"):
        raise ValueError("Scene snapshot does not match memory state")
    moves = [row for row in state.get("event_log", [])
             if row.get("event", {}).get("object_id") == target_id and row["event"].get("operation") == "move"]
    if not moves:
        raise ValueError("Follow-up consumption test requires an actual moved-object event")
    original = moves[0]["before"]["node"]["bbox_center"]
    current = node["bbox_center"]
    if current == original:
        raise ValueError("Object position did not change in this memory test")
    return {"object_id": target_id, "memory_version": state["version"],
            "original_bbox_center": original, "current_bbox_center": current,
            "scope": "Updated semantic planning target; simulator state not synchronized"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--planning-python", default="/home/lewis/miniforge3/envs/conceptgraph/bin/python")
    p.add_argument("--habitat-python", default="/home/lewis/miniforge3/envs/activesplat_ros311/bin/python")
    p.add_argument("--scene-dir", type=Path,
                   default=SCENE / "vlm_scenegraph_regression/scene201_verified_20260921_113316/scene")
    p.add_argument("--resume-alignment-run", type=Path,
                   help="Reuse only this instruction's successful decomposition/alignment; regenerate all downstream stages")
    p.add_argument("--instruction", default=INSTRUCTION)
    p.add_argument("--matching-mode", choices=("clip", "llm_graph"), default="clip")
    p.add_argument("--expected-target-id", type=int, default=13)
    p.add_argument("--planning-only", action="store_true",
                   help="Validate planning without loading an unsynchronized simulator")
    p.add_argument("--adapter-mode", choices=("navigation", "kettle_to_sideboard"), default="navigation")
    p.add_argument("--output-dir", type=Path, help="New isolated run directory")
    p.add_argument("--memory-state", type=Path,
                   help="Verify that planning uses the moved target from this memory checkpoint")
    args = p.parse_args()
    if not os.environ.get("DASHSCOPE_API_KEY"):
        p.error("DASHSCOPE_API_KEY required; use the shell launcher")
    outputs = Path(__file__).resolve().parent.parent
    library = outputs / "quadruped_atomic_skill_library.v0.1.json"
    root = args.output_dir or SCENE / ("memory_followup_planning" if args.memory_state else "sideboard_full_pipeline") / datetime.now().strftime("scene201_%Y%m%d_%H%M%S_%f")
    root.mkdir(parents=True, exist_ok=False)
    summary = {"schema_version": "1.0", "status": "running", "instruction": args.instruction,
        "scene_reference": str(args.scene_dir), "stages": [],
        "scope": "Fresh LLM decomposition, object/skill grounding, redundancy review and proxy navigation; perception map reused",
        "reused_task_plan": False, "physical_robot_execution_verified": False}
    summary_path = root / "pipeline_summary.json"
    write_json(summary_path, summary)
    stage = "preflight"
    def execute(name, command):
        nonlocal stage
        stage = name
        print(f"START stage={name}", flush=True)
        with (root / f"{name}_console.log").open("w", encoding="utf-8") as log:
            proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, encoding="utf-8", errors="replace")
            for line in proc.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            code = proc.wait()
        summary["stages"].append({"stage": name, "exit_code": code})
        write_json(summary_path, summary)
        if code:
            raise ValueError(f"Stage {name} failed with exit_code={code}; downstream stages skipped")
    a = root / "object_alignment.json"
    proposal = root / "skill_plan.json"
    final = root / "skill_plan_final.json"
    common = ["--alignment-json", str(a), "--skill-library", str(library)]
    try:
        for path in (Path(args.planning_python), Path(args.habitat_python), library,
                     args.scene_dir / "scene_graph.json", args.scene_dir / "scene_graph_edges.json"):
            if not path.is_file():
                raise ValueError(f"Required input missing: {path}")
        contract = build_execution_contract(args.adapter_mode)
        scoped_library = scoped_skill_library(read_json(library), contract)
        scoped_path = root / "execution_skill_library.json"
        contract_path = root / "execution_contract.json"
        write_json(scoped_path, scoped_library)
        write_json(contract_path, contract)
        summary["execution_contract"] = contract
        common = ["--alignment-json", str(a), "--skill-library", str(scoped_path)]
        if not args.planning_only and (args.instruction != INSTRUCTION or args.expected_target_id != 13):
            raise ValueError("This execution adapter only supports the original sideboard task; use planning-only for new memory targets")
        if args.memory_state:
            if not args.planning_only:
                raise ValueError("Memory scene is not synchronized with Habitat; use --planning-only")
            summary["memory_target_check"] = validate_memory_target(read_json(args.memory_state),
                read_json(args.scene_dir / "scene_graph.json"), args.expected_target_id)
            summary["scope"] = "Fresh LLM follow-up planning using updated scene memory; no simulation execution"
        if args.resume_alignment_run:
            stage = "resume_alignment_validation"
            previous = args.resume_alignment_run
            validate_resume(read_json(previous / "object_alignment.json"),
                            read_json(previous / "task_decomposition.json"), args.scene_dir, args.instruction)
            shutil.copy2(previous / "object_alignment.json", a)
            shutil.copy2(previous / "task_decomposition.json", root / "task_decomposition.json")
            summary["resumed_alignment_from"] = str(previous)
            summary["stages"].append({"stage": "llm_decomposition_and_object_alignment",
                "status": "reused_successful_checkpoint", "source": str(previous)})
            print(f"RESUME successful alignment from {previous}; downstream LLM stages will run again", flush=True)
            write_json(summary_path, summary)
        else:
            execute("llm_decomposition_and_object_alignment", [args.planning_python,
                str(outputs / "object_alignment_v1/align_objects.py"),
                "--scene-dir", str(args.scene_dir), "--instruction", args.instruction,
                "--matching-mode", args.matching_mode,
                "--execution-contract", str(contract_path),
                "--model", "qwen3.8-flash", "--decomposition-output", str(root / "task_decomposition.json"),
                "--output", str(a), "--log-file", str(root / "alignment.log")])
        execute("llm_skill_matching", [args.planning_python,
            str(outputs / "skill_matching_v1/match_skills.py"), *common,
            "--model", "qwen3.8-flash", "--output", str(proposal),
            "--log-file", str(root / "skill_matching.log")])
        execute("symbolic_and_redundancy_check", [args.planning_python,
            str(outputs / "skill_matching_v1/verify_plan.py"), *common,
            "--skill-plan-json", str(proposal), "--scene-dir", str(args.scene_dir),
            "--output", str(root / "symbolic_review.json"),
            "--redundancy-output", str(root / "redundancy_review.json")])
        execute("llm_redundancy_review", [args.planning_python,
            str(outputs / "skill_matching_v1/optimize_plan.py"), *common,
            "--skill-plan-json", str(proposal), "--scene-dir", str(args.scene_dir),
            "--model", "qwen3.8-flash", "--output-dir", str(root)])
        stage = "final_plan_adapter_gate"
        decision = read_json(root / "optimization_decision.json")
        if "llm_reason" not in decision:
            raise ValueError("LLM redundancy review failed or was invalid; original-retained fallback is not a completed LLM review")
        if read_json(root / "symbolic_review_final.json").get("status") != "conditionally_valid_symbolic_plan":
            raise ValueError("Final symbolic validation failed")
        aligned, planned = read_json(a), read_json(final)
        if args.adapter_mode == "kettle_to_sideboard":
            if not args.planning_only:
                raise ValueError("Manipulation planning adapter requires --planning-only")
            from run_manipulation_proxy import EXPECTED_SKILLS
            steps = flatten_steps(planned)
            if [s.get("skill_id") for s in steps] != EXPECTED_SKILLS:
                raise ValueError("Proxy requires the exact supported seven-skill sequence; no silent rewriting")
            items = targets(aligned, planned, read_json(args.scene_dir / "scene_graph.json"))
            if [i["element_id"] for i in items] != [67, 13]:
                raise ValueError("Manipulation must bind kettle 67 and sideboard 13")
            if steps[2]["arguments"]["target"]["element_id"] != 67 or steps[6]["arguments"]["surface"]["element_id"] != 13:
                raise ValueError("Grasp or placement target mismatch")
            item = items[0]
        else:
            item = validate_adapter(aligned, planned, read_json(args.scene_dir / "scene_graph.json"),
                                    args.instruction, args.expected_target_id)
        if args.planning_only:
            write_json(root / "planning_target.json", {"instruction": args.instruction,
                "scene_reference": str(args.scene_dir), "target": item,
                "habitat_target_position_candidate": cg_to_habitat(item["cg_center"], "x_z_minus_y"),
                "execution_status": "not_executed", "scene_graph_modified": False,
                "memory_source": str(args.memory_state) if args.memory_state else None})
            summary.update(status="planning_passed", execution_status="not_executed",
                           training_eligible=False, redundancy_decision=decision.get("status"),
                           target_scene_graph_position=item["cg_center"])
            write_json(summary_path, summary)
            print(f"status=planning_passed summary={summary_path}", flush=True)
            return 0
        command = [args.habitat_python, str(outputs / "habitat_demo_v1/run_navigation_demo.py"),
            "--alignment-json", str(a), "--skill-plan-json", str(final),
            "--scene-graph-json", str(args.scene_dir / "scene_graph.json"),
            "--stage-config", str(SCENE / "scene_package/hf_house_201/hf_house_201.stage_config.json"),
            "--navmesh", str(SCENE / "scene_package/hf_house_201/hf_house_201.navmesh"),
            "--output", str(root / "navigation_demo.json"), "--log-file", str(root / "navigation_demo.log")]
        if Path("/usr/bin/time").is_file():
            command = ["/usr/bin/time", "-v", "-o", str(root / "resource_usage.txt"), *command]
        execute("habitat_navigation", command)
        stage = "observed_goal_validation"
        result = read_json(root / "navigation_demo.json")
        nav = result.get("navigation_results", [])
        distance = horizontal_distance(nav[0]["end"], cg_to_habitat(item["cg_center"], "x_z_minus_y")) if len(nav) == 1 else None
        passed = (result.get("status") == "navigation_demo_passed" and len(nav) == 1
                  and nav[0].get("status") == "arrived" and distance is not None and distance <= 1.5)
        peak = None
        if (root / "resource_usage.txt").exists():
            for line in (root / "resource_usage.txt").read_text().splitlines():
                if "Maximum resident set size (kbytes):" in line:
                    peak = round(int(line.rsplit(":", 1)[1]) / 1048576, 3)
        episode = {"record_type": "fresh_llm_navigation_proxy_episode", "instruction": INSTRUCTION,
            "training_eligible": passed, "goal_observed_in_proxy": passed,
            "object_id": item["element_id"], "navigation_results": nav,
            "target_horizontal_distance_m": distance, "proximity_threshold_m": 1.5,
            "peak_rss_gib": peak, "plan_origin": "fresh_llm_pipeline",
            "physical_robot_execution_verified": False, "scene_graph_modified": False}
        write_json(root / "task_level_episode.json", episode)
        summary.update(status="passed" if passed else "failed", training_eligible=passed,
                       target_horizontal_distance_m=distance, peak_rss_gib=peak,
                       redundancy_decision=decision.get("status"))
    except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
        summary.update(status="failed", failed_stage=stage, training_eligible=False,
                       error=str(exc))
        print(f"ERROR stage={stage}: {exc}", flush=True)
    write_json(summary_path, summary)
    print(f"status={summary['status']} summary={summary_path}", flush=True)
    return 0 if summary["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
