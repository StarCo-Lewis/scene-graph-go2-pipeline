"""Frozen 12-task symbolic pilot. No robot entrypoint; prepare/check never call APIs."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

from .._legacy import ROOT, load
from . import plan_artifact
from ..orchestration import task_collection_worker as worker

VERSION = "symbolic_source_frame_pilot_20261031_v1"
SCENES = ("tum_open3d_frame001_v1", "scannet_scene0488_gt_frames0_40_80_v1")
NAMES = ("scene_graph.json", "scene_graph_edges.json", "scene_metadata.json")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def within(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("Missing/outside input: " + str(relative))
    return path


def verify_files(root, entries):
    for entry in entries:
        path = within(root, entry["path"])
        if digest(path) != entry["sha256"] or path.stat().st_size != entry["bytes"]:
            raise ValueError("Input hash/size mismatch: " + entry["path"])


def inventory(root):
    return [{"path": p.relative_to(root).as_posix(), "bytes": p.stat().st_size,
             "sha256": digest(p)} for p in sorted(root.rglob("*")) if p.is_file()]


def source_revision():
    paths = list((ROOT / "embodied_pipeline").rglob("*.py"))
    for folder in ("object_alignment_v1", "skill_matching_v1", "llm_audit_v1", "scene_memory_v1", "habitat_demo_v1"):
        paths.extend((ROOT / "outputs" / folder).glob("*.py"))
    paths += [worker.LIBRARY, ROOT / "outputs/skill_matching_v1/skill_semantic_profiles.json"]
    return {p.relative_to(ROOT).as_posix(): digest(p) for p in sorted(set(paths))}


def specs(scene_id):
    # References are rule-authored, independent of generated model answers.
    if scene_id == SCENES[0]:
        return [
            ("Navigate to chair ID 1.", [1], "single_id"),
            ("Go to the yellow die.", [3], "unique_category"),
            ("First visit chair ID 2, then go to the yellow die ID 3.", [2, 3], "ordered_two"),
            ("Visit chair ID 1, then the yellow die ID 3, and finish at chair ID 2.", [1, 3, 2], "ordered_three"),
            ("Go to the chair.", None, "ambiguous_category"),
            ("Navigate to object ID 999.", None, "missing_id"),
        ]
    return [
        ("Navigate to refrigerator ID 2.", [2], "single_id"),
        ("Go to the sink.", [14], "unique_category"),
        ("Visit refrigerator ID 3 first, then go to sink ID 14.", [3, 14], "ordered_two"),
        ("Visit kitchen cabinet ID 5, then refrigerator ID 2, and finish at sink ID 14.", [5, 2, 14], "ordered_three"),
        ("Go to the refrigerator.", None, "ambiguous_category"),
        ("Navigate to object ID 999.", None, "missing_id"),
    ]


def validate_scene(folder):
    metadata = read(folder / NAMES[2])
    nodes = load("scene_input_interface").normalize_scene_input(
        read(folder / NAMES[0]), "Validate the supplied symbolic scene.")["nodes"]
    edges = read(folder / NAMES[1])
    if metadata.get("scene_id") != folder.name or not metadata.get("coordinate_frame"):
        raise ValueError("Scene identity/frame missing or mismatched")
    if metadata.get("length_unit") != "m" or edges != []:
        raise ValueError("This pilot is scoped to the two supplied metre/unknown-edge scenes")
    if not (folder / "modality_index.json").is_file():
        raise ValueError("Missing modality index")
    index = read(folder / "modality_index.json")
    references = []
    for key in ("rgb", "depth", "masks", "annotation", "source_snapshot", "calibration_manifest",
                "mapping", "object_id_mapping", "class_reference"):
        if key in index:
            references.append(index[key])
    for key in ("point_files", "instance_files", "rgb_images"):
        if key in index:
            if not isinstance(index[key], list):
                raise ValueError("Invalid modality array: " + key)
            references.extend(index[key])
    if not references:
        raise ValueError("No referenced modal assets")
    for relative in references:
        if not isinstance(relative, str):
            raise ValueError("Invalid modality reference")
        within(folder, relative)
    return metadata, nodes


def prepare(package, output):
    package, output = Path(package).resolve(), Path(output).resolve()
    manifest = read(package / "MANIFEST_SHA256.json")
    verify_files(package, manifest["files"])
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(package / "MANIFEST_SHA256.json", output / "SOURCE_PACKAGE_MANIFEST.json")
    records, references = [], []
    for scene_id in SCENES:
        source = package / "scenes" / scene_id
        folder = output / "scenes" / scene_id
        # Copy only files authenticated by the supplier manifest, never run source scripts.
        prefix = "scenes/" + scene_id + "/"
        for entry in manifest["files"]:
            if entry["path"].startswith(prefix):
                dest = output / entry["path"]
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(within(package, entry["path"]), dest)
        metadata, nodes = validate_scene(folder)
        active = {n["id"] for n in nodes if n.get("active", True)}
        for number, (instruction, visits, kind) in enumerate(specs(scene_id), 1):
            if visits and not set(visits) <= active:
                raise ValueError("Reference points to absent/inactive object")
            record_id = scene_id + "_" + str(91000000 + number)
            records.append({"record_id": record_id, "scene_id": scene_id,
                "instruction": instruction, "input_scope": "symbolic_source_frame_v1",
                "prepared_scene_sha256": {name: digest(folder / name) for name in NAMES}})
            references.append({"record_id": record_id, "task_type": kind,
                "expected_outcome": "ordered_symbolic_plan" if visits else "clarification_or_grounded_refusal",
                "ordered_target_ids": visits, "terminal_target_id": visits[-1] if visits else None,
                "steps": [{"step": i+1, "target_id": target, "depends_on": [i] if i else []}
                          for i, target in enumerate(visits or [])],
                "reference_source": "Rule-authored from supplied node labels and IDs; not model output or physical ground truth",
                "scene_graph_sha256": digest(folder / NAMES[0]),
                "review_note": "For missing/ambiguous target, arbitrary target substitution is incorrect. Do not score transport failure as refusal."})
    save(output / "tasks.json", records)
    save(output / "references_not_model_input.json", references)
    save(output / "PILOT_MANIFEST.json", {"version": VERSION,
        "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_capture_time": "Retain supplied metadata; preparation time is not scene acquisition time",
        "task_count": len(records), "model": "qwen3.8-flash",
        "max_requests": 36, "budget_cny": 10,
        "budget_enforcement": "Request count only; currency ceiling requires provider-side control",
        "input_mode": "text scene graph + source metadata; modal files retained for audit, not sent as images",
        "execution_ready": False, "robot_executed": False,
        "files": inventory(output), "source_revision_sha256": source_revision()})
    return check(output)


def check(root):
    root = Path(root).resolve()
    manifest = read(root / "PILOT_MANIFEST.json")
    if manifest["version"] != VERSION:
        raise ValueError("Unknown pilot version")
    verify_files(root, manifest["files"])
    for name, expected in manifest["source_revision_sha256"].items():
        if digest(within(ROOT, name)) != expected:
            raise ValueError("Pipeline source changed since prepare: " + name)
    tasks = read(root / "tasks.json")
    if len(tasks) != 12 or len({t["record_id"] for t in tasks}) != 12:
        raise ValueError("Expected 12 unique pilot tasks")
    for task in tasks:
        folder = root / "scenes" / task["scene_id"]
        validate_scene(folder)
        if task["prepared_scene_sha256"] != {name: digest(folder / name) for name in NAMES}:
            raise ValueError("Task preparation bindings differ")
    return {"status": "offline_inputs_verified_not_model_test", "task_count": 12,
            "max_requests": 36, "model": manifest["model"], "robot_executed": False}


def tally(rows, results):
    """Separate delivered files from complete plans; a failure envelope never counts as success."""
    counts = Counter(engineering_plan_files=0, complete_plans=0, worker_exceptions=0)
    for row in rows:
        path = Path(results) / row["record_id"] / "engineering_plan.json"
        counts["engineering_plan_files"] += path.is_file()
        counts["complete_plans"] += (plan_artifact.summary_planning_completed(row)
                                     and plan_artifact.has_complete(path))
        counts["worker_exceptions"] += row.get("status") == "worker_exception"
    counts["not_complete"] = len(rows) - counts["complete_plans"]
    return dict(counts)


def run(root, acknowledge, semantic_audit_mode="annotate"):
    root = Path(root).resolve()
    check(root)
    if not acknowledge:
        raise ValueError("Run requires --acknowledge-paid-api; review 10 CNY budget/provider limits first")
    if not os.environ.get("DASHSCOPE_API_KEY", "").strip():
        raise ValueError("DASHSCOPE_API_KEY is missing; no request made")
    # Exclusive run directory also prevents concurrent/repeated paid submissions.
    results = root / "results"
    results.mkdir(exist_ok=False)
    args = argparse.Namespace(model="qwen3.8-flash",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1", request_timeout=180,
        semantic_audit_mode=semantic_audit_mode)
    tasks = read(root / "tasks.json")  # Independent references are never read by this path.
    rows, start = [], time.monotonic()
    for task in tasks:
        print("START", task["record_id"], flush=True)
        try:
            row = worker.run_task(args, task, root / "scenes" / task["scene_id"],
                                  results / task["record_id"], None)
        except Exception as exc:
            row = {"record_id": task["record_id"], "status": "worker_exception",
                   "error_type": type(exc).__name__, "error": str(exc),
                   "planning_completed": False, "engineering_json_generated": False,
                   "engineering_plan_file_written": (results / task["record_id"] / "engineering_plan.json").is_file()}
        rows.append(row)
        outcomes = tally(rows, results)
        save(results / "pilot_summary.json", {"version": VERSION,
            "status": "completed_unreviewed" if len(rows) == len(tasks) else "running",
            "status_meaning": "All tasks were attempted once; not a success count. See outcome_counts.",
            "finished_tasks": len(rows), "expected_tasks": len(tasks), "rows": rows,
            "outcome_counts": outcomes,
            "duration_seconds": time.monotonic()-start, "robot_executed": False,
            "semantic_acceptance": "pending review against separate references",
            "cost_cny": None, "cost_note": "Reconcile response usage with provider billing; missing usage is unknown, not zero"})
        print("END", task["record_id"], row["status"], flush=True)
    return {"status": "completed_unreviewed", "outcome_counts": tally(rows, results),
            "summary": str(results / "pilot_summary.json")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "check", "run"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--package", type=Path)
    parser.add_argument("--acknowledge-paid-api", action="store_true")
    parser.add_argument("--semantic-audit-mode", choices=("annotate", "block", "off"),
                        default="annotate",
                        help="annotate (default) always produces the engineering JSON and labels "
                             "quality; block is an opt-in dry run that stops a flagged task before "
                             "paid skill binding")
    args = parser.parse_args()
    if args.command == "prepare":
        if args.package is None:
            parser.error("prepare requires --package")
        result = prepare(args.package, args.root)
    elif args.command == "check":
        result = check(args.root)
    else:
        result = run(args.root, args.acknowledge_paid_api, args.semantic_audit_mode)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
