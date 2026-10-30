"""Continue semantic export from a completed recorded-scene mapping stage."""

import gzip
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from .._legacy import ROOT
from ..perception.real_rgbd import sha, write, validate_prepared
from ..perception.conceptgraphs_scene import export_scene, import_scene
from ..perception.caption_cache import reuse_captions


def inspect_checkpoint(summary_path):
    """Validate the old run without modifying its files or importing its pickle."""
    summary_path = Path(summary_path).resolve()
    report = json.loads(summary_path.read_text(encoding="utf-8"))
    if (report.get("schema_version") != "recorded_real_scene_run_v1"
            or report.get("status") != "failed_or_interrupted"
            or report.get("current_stage") != "captions"
            or report.get("conceptgraphs_mapping_completed") is not True):
        raise ValueError("Expected a failed caption-stage run with completed mapping")
    for name in ("detections", "mapping"):
        stages = [s for s in report.get("stages", []) if s.get("stage") == name]
        saved = json.loads((summary_path.parent / "logs" / (name + ".json")).read_text())
        if len(stages) != 1 or stages[0] != saved or saved.get("exit_code") != 0:
            raise ValueError("Missing or inconsistent successful stage evidence: " + name)
    prepared = Path(report["prepared_input"]).resolve()
    audit_path = Path(report["prepared_input_audit"]["path"])
    if audit_path.resolve() != prepared.parent / "input_audit.json":
        raise ValueError("Prepared audit path mismatch")
    if sha(audit_path) != report["prepared_input_audit"]["sha256"]:
        raise ValueError("Prepared audit changed since mapping")
    validate_prepared(prepared)
    map_file = Path(report["map_file"]).resolve()
    expected = prepared / "dataset/pcd_saves" / (
        "full_pcd_" + report["detection_variant"] + "_overlap_post.pkl.gz")
    if map_file != expected.resolve() or not map_file.is_file():
        raise ValueError("Map path does not match completed mapping")
    # Check gzip integrity before any paid requests. Unpickling remains in the
    # normal caption/export code, using only this locally generated artifact.
    with gzip.open(map_file, "rb") as stream:
        while stream.read(1024 * 1024):
            pass
    hashes = report.get("source_code_hashes")
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError("Missing upstream source hashes")
    for path, digest in hashes.items():
        if not Path(path).is_file() or sha(path) != digest:
            raise ValueError("Upstream source changed since mapping: " + path)
    evidence = dict(summary=str(summary_path), summary_sha256=sha(summary_path),
        map_file=str(map_file), map_sha256=sha(map_file),
        map_hash_scope="Measured at continuation; original run did not record a map hash",
        prepared_input_verified=True, gzip_integrity_verified=True,
        upstream_source_hashes_verified=True)
    return report, evidence


def resume(args):
    from .real_scene import run_stage
    if sys.platform != "linux":
        raise RuntimeError("Run this continuation manually in the existing WSL environment")
    if not os.environ.get("DASHSCOPE_API_KEY"):
        raise ValueError("DASHSCOPE_API_KEY is required")
    print("[REAL SCENE] Checking completed mapping and prepared input...", flush=True)
    old, checkpoint = inspect_checkpoint(args.from_summary)
    root = (args.output_dir.resolve() if args.output_dir else ROOT / "outputs/real_scene_runs" /
            datetime.now().strftime("d3_619_semantics_%Y%m%d_%H%M%S_%f"))
    # A continuation must not write inside any source/checkpoint directory.
    protected = [Path(old["source"]).resolve(), Path(old["prepared_input"]).resolve(),
                 Path(checkpoint["summary"]).parent]
    if getattr(args, "reuse_captions_from", None):
        protected.append(args.reuse_captions_from.resolve().parent)
    if any(root == p or p in root.parents or root in p.parents for p in protected):
        raise ValueError("Continuation output overlaps a source/checkpoint directory")
    if root.exists():
        raise FileExistsError("Use a new continuation output directory: " + str(root))
    adapter_paths = [Path(p) for p in old["source_code_hashes"]
                     if Path(p).name == "run_qwen_conceptgraph_captions.py"]
    if len(adapter_paths) != 1:
        raise ValueError("Expected exactly one recorded caption adapter")
    adapter_root = adapter_paths[0].parent
    model = old["semantic_model"]
    map_file = Path(checkpoint["map_file"])
    env = os.environ.copy()
    for key in ("PYTHONHOME", "AMENT_PREFIX_PATH", "CMAKE_PREFIX_PATH", "ROS_PACKAGE_PATH", "PKG_CONFIG_PATH"):
        env.pop(key, None)
    env.update(PYTHONPATH=str(ROOT), PYTHONNOUSERSITE="1", PYTHONUNBUFFERED="1")
    root.mkdir(parents=True, exist_ok=False)
    report = dict(schema_version="real_scene_semantic_continuation_v1", status="running",
        resumed_from=checkpoint, semantic_model=model, base_url=args.base_url,
        stages=[], reused_stages=["detections", "mapping"], current_stage="captions",
        code_sha256={str(Path(__file__).resolve()): sha(__file__),
                     str(Path(__file__).with_name("real_scene.py").resolve()):
                         sha(Path(__file__).with_name("real_scene.py"))},
        source_code_hashes=old["source_code_hashes"], integration_target="real_robot",
        simulation_executed=False, execution_requested=False)
    print(f"[REAL SCENE] Continuation output: {root}", flush=True)
    print("[REAL SCENE] Reusing completed detections and mapping", flush=True)
    try:
        write(root / "run_summary.json", report)
        for name in ("caption_contract.py", "caption_cache.py"):
            path = Path(__file__).parent.parent / "perception" / name
            report["code_sha256"][str(path.resolve())] = sha(path)
        if getattr(args, "reuse_captions_from", None):
            report["caption_reuse"] = reuse_captions(args.reuse_captions_from, root / "captions",
                checkpoint, model, args.base_url, old["source_code_hashes"])
            print(f"[REAL SCENE] Reusing {report['caption_reuse']['reused_count']} audited captions", flush=True)
        write(root / "run_summary.json", report)
        report["stages"].append(run_stage("captions", [sys.executable, "-m",
            "embodied_pipeline.orchestration.real_scene", "caption-worker",
            "--adapter-root", adapter_root, "--map-file", map_file,
            "--output-dir", root / "captions", "--semantic-model", model,
            "--base-url", args.base_url], root, env, ROOT))
        report["current_stage"] = "relations_and_export"
        write(root / "run_summary.json", report)
        print("[REAL SCENE] START relations_and_export", flush=True)
        report["scene"] = export_scene(map_file, root / "captions", root / "scene", adapter_root,
                                        model=model, base_url=args.base_url)
        report["pipeline"] = import_scene(root / "scene", root / "pipeline")
        report["status"] = "graph_generated_and_imported_not_executed"
    except BaseException as exc:
        report.update(status="failed_or_interrupted", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        report["map_unchanged"] = map_file.is_file() and sha(map_file) == checkpoint["map_sha256"]
        report["upstream_source_unchanged"] = all(Path(p).is_file() and sha(p) == h
                                                for p, h in old["source_code_hashes"].items())
        if not report["map_unchanged"] or not report["upstream_source_unchanged"]:
            report["status"] = "integrity_error"
        write(root / "run_summary.json", report)
        print(f"status={report['status']} summary={root / 'run_summary.json'}", flush=True)
    if report["status"] == "integrity_error":
        raise RuntimeError("Continuation inputs changed during processing")
    return 0
