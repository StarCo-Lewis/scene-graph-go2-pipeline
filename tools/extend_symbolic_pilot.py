"""Prepare/run 18 additional symbolic tasks without altering the frozen 12-task baseline."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from embodied_pipeline.data_collection import symbolic_pilot as pilot

VERSION = "symbolic_pilot_extension18_v1"


def additions():
    # English instructions; rule-authored references are never supplied to the model.
    return [
        (0, "Navigate to chair ID 2.", [2], "single_other_instance", None),
        (0, "Go to the yellow die ID 3 first, then visit chair ID 1.", [3, 1], "ordered_two", None),
        (0, "Visit chair ID 1 before going to chair ID 2.", [1, 2], "same_category_order", None),
        (0, "Start by visiting the yellow die ID 3, continue to chair ID 2, and finish at chair ID 1.", [3, 2, 1], "ordered_three", None),
        (0, "Visit chair ID 2, then chair ID 1, and finally go to the yellow die ID 3.", [2, 1, 3], "ordered_three", None),
        (0, "Visit a chair first, then go to the yellow die.", None, "ambiguous_intermediate", "Ask which chair, ID 1 or 2; the die is identifiable as ID 3. Do not choose a chair arbitrarily."),
        (1, "Navigate to the counter.", [12], "unique_category", None),
        (1, "Go to kitchen cabinet ID 9.", [9], "single_id", None),
        (1, "Visit sink ID 14 first, then finish at refrigerator ID 2.", [14, 2], "ordered_two", None),
        (1, "Visit refrigerator ID 2, then refrigerator ID 3.", [2, 3], "same_category_order", None),
        (1, "Visit kitchen cabinet ID 9, then counter ID 12, and finally sink ID 14.", [9, 12, 14], "ordered_three", None),
        (1, "Go to refrigerator ID 3, then kitchen cabinet ID 6, then refrigerator ID 2, and finish at sink ID 14.", [3, 6, 2, 14], "ordered_four", None),
        (1, "Visit kitchen cabinets ID 20, ID 19, ID 15, and ID 5, in that order.", [20, 19, 15, 5], "same_category_order_four", None),
        (1, "Visit sink ID 14, counter ID 12, kitchen cabinet ID 6, refrigerator ID 3, and kitchen cabinet ID 9, in that order. Do not move any objects.", [14, 12, 6, 3, 9], "ordered_five_preservation", None),
        (1, "Visit refrigerator ID 2, then go to a kitchen cabinet.", None, "ambiguous_final", "The refrigerator is ID 2; ask which active cabinet is intended. Do not arbitrarily choose a cabinet."),
        (1, "Go to the microwave.", None, "missing_category", "No microwave is labeled in the supplied graph. Request identification/additional evidence; do not substitute a refrigerator or invent an object."),
        (1, "Go to the kitchen cabinet nearest to my current location.", None, "unknown_speaker_pose", "No speaker pose is supplied. Ask for the speaker location or a target ID; do not equate camera origin with speaker position."),
        (1, "Go to the kitchen cabinet in the next room.", None, "unknown_room_topology", "Room membership, speaker room and room adjacency are not supplied. Ask for a target ID or the missing room information; do not infer next-room topology."),
    ]


def own_hash():
    return pilot.digest(Path(__file__))


def prepare(base, root):
    base, root = Path(base).resolve(), Path(root).resolve()
    pilot.check(base)
    old_summary = pilot.read(base / "results/pilot_summary.json")
    if old_summary["status"] != "completed_unreviewed" or old_summary["finished_tasks"] != 12:
        raise ValueError("The original 12-task run must be complete")
    old_tasks = pilot.read(base / "tasks.json")
    old_refs = pilot.read(base / "references_not_model_input.json")
    root.mkdir(parents=True, exist_ok=False)
    manifest = pilot.read(base / "PILOT_MANIFEST.json")
    for entry in manifest["files"]:
        if entry["path"].startswith("scenes/"):
            dest = root / entry["path"]
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(pilot.within(base, entry["path"]), dest)
    tasks, refs, counters = [], [], [6, 6]
    for scene_number, instruction, visits, kind, clarification in additions():
        scene = pilot.SCENES[scene_number]
        folder = root / "scenes" / scene
        _, nodes = pilot.validate_scene(folder)
        active = {n["id"] for n in nodes if n.get("active", True)}
        if visits and not set(visits) <= active:
            raise ValueError("Unknown/inactive reference target")
        counters[scene_number] += 1
        record_id = scene + "_" + str(91000000 + counters[scene_number])
        tasks.append({"record_id": record_id, "scene_id": scene, "instruction": instruction,
                      "input_scope": "symbolic_source_frame_v1",
                      "prepared_scene_sha256": {n: pilot.digest(folder / n) for n in pilot.NAMES}})
        refs.append({"record_id": record_id, "task_type": kind,
                     "expected_outcome": "ordered_symbolic_plan" if visits else "clarification_or_grounded_refusal",
                     "ordered_target_ids": visits, "terminal_target_id": visits[-1] if visits else None,
                     "steps": [{"step": i+1, "target_id": t, "depends_on": [i] if i else []}
                               for i, t in enumerate(visits or [])],
                     "clarification_criteria": clarification,
                     "preserve_objects": kind == "ordered_five_preservation",
                     "reference_source": "Rule-authored from supplied graph/metadata; not model output or physical ground truth",
                     "review_note": "Unknown pose/room data must remain unknown; only the last visit is a terminal near goal. A transport error is not a valid refusal."})
    all_tasks = old_tasks + tasks
    if len({t["record_id"] for t in all_tasks}) != 30 or len({(t["scene_id"], t["instruction"].casefold()) for t in all_tasks}) != 30:
        raise ValueError("Duplicate ID/instruction")
    sequences = [(t["scene_id"], tuple(r["ordered_target_ids"])) for t, r in zip(all_tasks, old_refs + refs)
                 if r["ordered_target_ids"]]
    if len(set(sequences)) != len(sequences):
        raise ValueError("Repeated positive target sequence")
    pilot.save(root / "tasks.json", tasks)
    pilot.save(root / "references_not_model_input.json", refs)
    pilot.save(root / "combined_30_input_index.json", {"status": "18_not_run", "tasks": all_tasks,
        "baseline_root": str(base), "baseline_summary_sha256": pilot.digest(base / "results/pilot_summary.json"),
        "analysis_policy": "30 unique tasks; preserve raw failures/clarifications; stage scores may be unobserved. Do not count model requests as samples or claim robot execution."})
    pilot.save(root / "BASELINE_MANIFEST.json", manifest)
    pilot.save(root / "EXTENSION_MANIFEST.json", {"version": VERSION, "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
        "task_count": 18, "combined_task_count": 30, "max_requests": 54, "model": "qwen3.8-flash",
        "budget_cny": 10, "budget_enforcement": "No hard currency cap; reconcile provider billing separately",
        "known_baseline_issue": "Intermediate visit incorrectly retained as terminal goal in one original task; planner unchanged for this extension",
        "robot_executed": False, "files": pilot.inventory(root), "runner_sha256": own_hash(),
        "source_revision_sha256": pilot.source_revision()})
    return check(root)


def check(root):
    root = Path(root).resolve()
    manifest = pilot.read(root / "EXTENSION_MANIFEST.json")
    if manifest["version"] != VERSION or manifest["runner_sha256"] != own_hash():
        raise ValueError("Extension version/runner changed")
    pilot.verify_files(root, manifest["files"])
    if pilot.source_revision() != manifest["source_revision_sha256"]:
        raise ValueError("Planning source/config changed since preparation")
    tasks = pilot.read(root / "tasks.json")
    if len(tasks) != 18 or len({t["record_id"] for t in tasks}) != 18:
        raise ValueError("Expected 18 unique tasks")
    for task in tasks:
        folder = root / "scenes" / task["scene_id"]
        pilot.validate_scene(folder)
        if task["prepared_scene_sha256"] != {n: pilot.digest(folder / n) for n in pilot.NAMES}:
            raise ValueError("Changed scene binding")
    return {"status": "offline_verified_not_run", "tasks": 18, "max_requests": 54, "robot_executed": False}


def run(root, acknowledge):
    root = Path(root).resolve()
    check(root)
    if not acknowledge or not os.environ.get("DASHSCOPE_API_KEY", "").strip():
        raise ValueError("Requires --acknowledge-paid-api and DASHSCOPE_API_KEY")
    results = root / "results"
    results.mkdir(exist_ok=False)
    args = argparse.Namespace(model="qwen3.8-flash", base_url="https://dashscope.aliyuncs.com/compatible-mode/v1", request_timeout=180)
    rows, start = [], time.monotonic()
    started = datetime.now(timezone.utc).isoformat()
    tasks = pilot.read(root / "tasks.json")
    def write_summary(status):
        pilot.save(results / "pilot_summary.json", {"version": VERSION, "status": status,
            "started_at_utc": started, "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "finished_tasks": len(rows), "expected_tasks": 18, "rows": rows,
            "duration_seconds": time.monotonic()-start, "robot_executed": False,
            "semantic_acceptance": "pending joint review with original 12 tasks", "cost_cny": None,
            "cost_note": "Use saved per-stage usage and isolated provider billing; missing usage is unknown, not zero"})
    write_summary("running")
    try:
        for task in tasks:
            print("START", task["record_id"], flush=True)
            try:
                row = pilot.worker.run_task(args, task, root / "scenes" / task["scene_id"], results / task["record_id"], None)
            except Exception as exc:
                row = {"record_id": task["record_id"], "status": "worker_exception", "error_type": type(exc).__name__, "error": str(exc)}
            rows.append(row)
            write_summary("completed_unreviewed" if len(rows) == 18 else "running")
            print("END", task["record_id"], row["status"], flush=True)
    except KeyboardInterrupt:
        write_summary("interrupted_no_auto_retry")
        raise
    return {"status": "completed_unreviewed", "summary": str(results / "pilot_summary.json")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "check", "run"))
    parser.add_argument("--base", type=Path)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--acknowledge-paid-api", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        if args.base is None:
            parser.error("prepare requires --base")
        result = prepare(args.base, args.root)
    elif args.command == "check":
        result = check(args.root)
    else:
        result = run(args.root, args.acknowledge_paid_api)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
