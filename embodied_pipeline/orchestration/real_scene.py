"""Recorded real scene -> existing CG models -> audited planning scene import."""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from .._legacy import ROOT
from ..perception.real_rgbd import audit, sha, write, validate_prepared
from ..perception.conceptgraphs_scene import export_scene, import_scene, external

SOURCE = Path("/mnt/d/NebulaAI/input/D3-619_地图_v1_2026-10-21/D3-619_v1")
CG = Path("/mnt/d/NebulaAI/code/concept-graphs/concept-graphs-main")
ADAPTER = Path("/mnt/d/NebulaAI/code/nebula_pipeline")
MODELS = Path("/mnt/d/NebulaAI/models")


def run_stage(name, command, root, env, cwd):
    print(f"[REAL SCENE] START {name}", flush=True)
    log = root / "logs" / (name + ".log")
    log.parent.mkdir(exist_ok=True)
    print(f"[REAL SCENE] Progress log: {log}", flush=True)
    started = time.monotonic()
    with log.open("w", encoding="utf-8") as stream:
        result = subprocess.run([str(v) for v in command], cwd=cwd, env=env,
                                stdout=stream, stderr=subprocess.STDOUT)
    record = dict(stage=name, exit_code=result.returncode, duration_seconds=time.monotonic()-started,
                  log=str(log), command=[str(v) for v in command])
    write(root / "logs" / (name + ".json"), record)
    if result.returncode:
        raise RuntimeError(f"{name} failed; inspect {log}")
    print(f"[REAL SCENE] END {name}; log={log}", flush=True)
    return record


def run(args):
    if sys.platform != "linux":
        raise RuntimeError("Run the installed CUDA ConceptGraphs stack manually in WSL; this command never launches WSL")
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this interpreter; use the installed conceptgraph environment")
    if not os.environ.get("DASHSCOPE_API_KEY"):
        raise ValueError("Configure DASHSCOPE_API_KEY in the same shell before starting semantic graph generation")
    root = args.output_dir.resolve() if args.output_dir else ROOT / "outputs/real_scene_runs" / datetime.now().strftime("d3_619_%Y%m%d_%H%M%S_%f")
    if root.exists():
        raise FileExistsError("Use a new output directory; previous runs are never overwritten: " + str(root))
    if root == args.source.resolve() or args.source.resolve() in root.parents:
        raise ValueError("Run output must be outside the read-only source")
    required = [args.cg_root / "conceptgraph/slam/streamlined_mapping.py",
        args.adapter_root / "run_conceptgraph_class_agnostic_detections.py",
        args.adapter_root / "run_unmodified_conceptgraph_mapping.py",
        args.adapter_root / "run_qwen_conceptgraph_captions.py",
        args.adapter_root / "build_qwen_conceptgraph_baseline.py", args.sam_checkpoint]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    mapping_source = required[0].read_text(encoding="utf-8")
    # The installed version maps with dataset.poses (absolute SLAM C2W), even
    # though GradSLAMDataset also computes transformed_poses by default.
    if "unt_pose = dataset.poses[idx]" not in mapping_source or "adjusted_pose = unt_pose" not in mapping_source:
        raise ValueError("Installed mapping pose convention changed; inspect before generating world-frame geometry")
    env = os.environ.copy()
    for name in ("PYTHONHOME", "AMENT_PREFIX_PATH", "CMAKE_PREFIX_PATH", "ROS_PACKAGE_PATH", "PKG_CONFIG_PATH"):
        env.pop(name, None)
    env.update(PYTHONPATH=os.pathsep.join([str(ROOT), str(args.cg_root)]), PYTHONNOUSERSITE="1",
        HF_HOME=str(args.model_root / "conceptgraphs/huggingface"),
        HUGGINGFACE_HUB_CACHE=str(args.model_root / "conceptgraphs/huggingface/hub"),
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONUNBUFFERED="1")
    report = dict(schema_version="recorded_real_scene_run_v1", status="preparing", source=str(args.source.resolve()),
        interpreter=sys.executable, cuda_device=torch.cuda.get_device_name(), stride=args.stride,
        semantic_model=args.semantic_model, stages=[], source_code_hashes={str(p): sha(p) for p in required[:-1]},
        checkpoint=dict(path=str(args.sam_checkpoint), sha256=sha(args.sam_checkpoint)),
        sam_model_type=args.sam_model_type, coordinate_policy="Original Z-up OpenCV C2W; no axis flip or re-anchoring",
        integration_target="real_robot", execution_requested=False, simulation_required=False,
        room_topology_inferred=False, simulation_executed=False, physical_grasp_verified=False)
    try:
        if args.prepared_input:
            prepared = args.prepared_input.resolve()
            prepared_audit = validate_prepared(prepared)
            root.mkdir(parents=True, exist_ok=False)
            report["prepared_input_audit"] = dict(path=str(prepared.parent / "input_audit.json"),
                sha256=sha(prepared.parent / "input_audit.json"), frame_count=prepared_audit["frame_count"])
        else:
            audit(args.source, root, prepare=True)
            prepared = root / "prepared"
        write(root / "run_summary.json", report)
        variant = "real_classagnostic_" + args.sam_model_type + datetime.now().strftime("_%Y%m%d_%H%M%S_%f")
        report["detection_variant"] = variant
        report["prepared_input"] = str(prepared)
        python = sys.executable
        report["status"] = "running"
        stages = [
            ("detections", [python, args.adapter_root / "run_conceptgraph_class_agnostic_detections.py",
                "--prepared-input", prepared, "--sam-checkpoint", args.sam_checkpoint,
                "--sam-model-type", args.sam_model_type, "--stride", args.stride,
                "--exp-suffix", variant, "--clip-batch-size", "4", "--device", "cuda"],
                args.model_root / "conceptgraphs"),
            ("mapping", [python, args.adapter_root / "run_unmodified_conceptgraph_mapping.py",
                "--prepared-input", prepared, "--cg-root", args.cg_root,
                "--detection-folder", f"exp_{variant}/detections", "--variant", variant,
                "--save-suffix", "overlap", "--stride", args.stride, "--class-agnostic"], args.cg_root)]
        for name, command, cwd in stages:
            report["current_stage"] = name
            write(root / "run_summary.json", report)
            report["stages"].append(run_stage(name, command, root, env, cwd))
        map_file = prepared / f"dataset/pcd_saves/full_pcd_{variant}_overlap_post.pkl.gz"
        if not map_file.is_file():
            raise FileNotFoundError(map_file)
        report.update(current_stage="captions", map_file=str(map_file), conceptgraphs_mapping_completed=True)
        write(root / "run_summary.json", report)
        # This isolated worker makes English output explicit and retains HTTP
        # evidence using the existing transport auditor.
        report["stages"].append(run_stage("captions", [python, "-m",
            "embodied_pipeline.orchestration.real_scene", "caption-worker",
            "--adapter-root", args.adapter_root, "--map-file", map_file,
            "--output-dir", root / "captions", "--semantic-model", args.semantic_model,
            "--base-url", args.base_url], root, env, ROOT))
        report["current_stage"] = "relations_and_export"
        write(root / "run_summary.json", report)
        report["scene"] = export_scene(map_file, root / "captions", root / "scene", args.adapter_root,
            model=args.semantic_model, base_url=args.base_url)
        report["pipeline"] = import_scene(root / "scene", root / "pipeline")
        report["status"] = "graph_generated_and_imported_not_executed"
    except BaseException as exc:
        report.update(status="failed_or_interrupted", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        if root.exists():
            report["source_code_unchanged"] = all(sha(p) == value for p, value in report["source_code_hashes"].items())
            write(root / "run_summary.json", report)
            print(f"status={report['status']} summary={root / 'run_summary.json'}", flush=True)
    return 0


def caption_worker(args):
    from ..perception.caption_contract import request_caption
    module = external(args.adapter_root / "run_qwen_conceptgraph_captions.py", "real_caption_adapter")
    def request(options, key, content):
        return request_caption(options, key, content, args.output_dir)
    module.call_qwen = request
    original = sys.argv
    try:
        sys.argv = ["real_caption_worker", "--map-file", str(args.map_file), "--output-dir", str(args.output_dir),
            "--model", args.semantic_model, "--base-url", args.base_url,
            "--min-detections", "1", "--min-points", "1", "--minimum-crop-area", "1"]
        module.main()
    finally:
        sys.argv = original


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("audit", "prepare", "run"):
        p = sub.add_parser(name)
        p.add_argument("--source", type=Path, default=SOURCE)
        p.add_argument("--output-dir", type=Path, required=name != "run")
        if name == "run":
            p.add_argument("--prepared-input", type=Path,
                           help="Previously prepared/audited dataset; verify every prepared file before using")
            p.add_argument("--cg-root", type=Path, default=CG)
            p.add_argument("--adapter-root", type=Path, default=ADAPTER)
            p.add_argument("--model-root", type=Path, default=MODELS)
            p.add_argument("--sam-checkpoint", type=Path)
            p.add_argument("--sam-model-type", choices=("vit_b", "vit_h"), default="vit_h")
            p.add_argument("--stride", type=int, default=1)
            p.add_argument("--semantic-model", default="qwen3.8-flash")
            p.add_argument("--base-url", default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    p = sub.add_parser("resume", help="Continue captions and export from a failed caption-stage run")
    p.add_argument("--from-summary", type=Path, required=True)
    p.add_argument("--reuse-captions-from", type=Path,
                   help="Failed semantic continuation summary with verified successful caption records")
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--base-url", default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    p = sub.add_parser("rebuild-graph", help="Rebuild only upstream relations/export from completed captions")
    p.add_argument("--from-summary", type=Path, required=True)
    p.add_argument("--output-dir", type=Path)
    p = sub.add_parser("graph-worker", help=argparse.SUPPRESS)
    p.add_argument("--map-file", type=Path, required=True)
    p.add_argument("--captions-dir", type=Path, required=True)
    p.add_argument("--adapter-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--semantic-model", required=True)
    p.add_argument("--base-url", required=True)
    p = sub.add_parser("import")
    p.add_argument("--scene-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p = sub.add_parser("caption-worker", help=argparse.SUPPRESS)
    p.add_argument("--adapter-root", type=Path, required=True)
    p.add_argument("--map-file", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--semantic-model", required=True)
    p.add_argument("--base-url", required=True)
    args = parser.parse_args(argv)
    if args.command in ("audit", "prepare"):
        result = audit(args.source, args.output_dir, prepare=args.command == "prepare")
        print(f"status={result['status']} audit={args.output_dir / 'input_audit.json'}")
    elif args.command == "run":
        if args.stride < 1:
            parser.error("stride must be positive")
        if args.sam_checkpoint is None:
            args.sam_checkpoint = args.model_root / "segment_anything" / (
                "sam_vit_h_4b8939.pth" if args.sam_model_type == "vit_h" else "sam_vit_b_01ec64.pth")
        return run(args)
    elif args.command == "resume":
        from .real_scene_resume import resume
        return resume(args)
    elif args.command == "rebuild-graph":
        from .real_scene_rebuild import rebuild
        return rebuild(args)
    elif args.command == "graph-worker":
        export_scene(args.map_file, args.captions_dir, args.output_dir / "scene", args.adapter_root,
                     model=args.semantic_model, base_url=args.base_url)
        import_scene(args.output_dir / "scene", args.output_dir / "pipeline")
    elif args.command == "import":
        print(json.dumps(import_scene(args.scene_dir, args.output_dir), indent=2))
    else:
        caption_worker(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
