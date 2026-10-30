"""Rebuild only the upstream relation graph from completed, audited captions."""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

from .._legacy import ROOT
from ..perception.real_rgbd import sha, write
from ..perception.caption_contract import validate_caption


def inspect_completed(summary_path):
    summary = Path(summary_path).resolve()
    old = json.loads(summary.read_text(encoding="utf-8"))
    if (old.get("status") != "graph_generated_and_imported_not_executed"
            or old.get("map_unchanged") is not True or old.get("upstream_source_unchanged") is not True):
        raise ValueError("Expected a completed, integrity-checked semantic continuation")
    captions = summary.parent / "captions"
    manifest_path = captions / "qwen_caption_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    map_file = Path(old["resumed_from"]["map_file"]).resolve()
    if sha(map_file) != old["resumed_from"]["map_sha256"]:
        raise ValueError("Source map changed since the completed run")
    model, base_url = old["semantic_model"], old["base_url"]
    if ((captions / "STATUS").read_text().strip() != "SUCCESS"
            or manifest.get("model") != model or manifest.get("base_url") != base_url
            or Path(manifest["map_file"]).resolve() != map_file
            or manifest["source_object_count"] != manifest["completed_object_count"]
            or manifest["eligible_object_count"] != manifest["source_object_count"]):
        raise ValueError("Caption manifest/model/map/coverage mismatch")
    hashes = {str(summary): sha(summary), str(manifest_path): sha(manifest_path),
              str(map_file): sha(map_file)}
    source_hashes = old["source_code_hashes"]
    for path, digest in source_hashes.items():
        if sha(path) != digest:
            raise ValueError("Upstream source changed: " + path)
    adapter_paths = [Path(p) for p in source_hashes if Path(p).name == "build_qwen_conceptgraph_baseline.py"]
    if len(adapter_paths) != 1:
        raise ValueError("Expected exactly one recorded upstream graph adapter")
    responses = {}
    for path in (captions / "llm_response_audit").glob("*_http_*.json"):
        event = json.loads(path.read_text(encoding="utf-8"))
        if event.get("http_status") == 200:
            response = json.loads(event["response_body"])
            if response.get("id"):
                responses[response["id"]] = (response, path)
    records = {}
    for path in (captions / "objects").glob("object_*.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        index = record.get("id")
        if (type(index) is not int or index in records or path.name != f"object_{index:03d}.json"
                or record.get("status") != "SUCCESS" or record.get("model") != model):
            raise ValueError("Invalid source caption: " + str(path))
        value = validate_caption(record.get("qwen"))
        if record.get("captions") != [value["caption"]] or record.get("request_id") not in responses:
            raise ValueError("Caption text/raw-response evidence missing: " + str(path))
        response, raw_path = responses[record["request_id"]]
        choice = response["choices"][0]
        if (validate_caption(json.loads(choice["message"]["content"]), choice.get("finish_reason")) != value
                or record.get("usage") != response.get("usage")):
            raise ValueError("Caption/raw-response disagreement: " + str(path))
        records[index] = record
        hashes[str(path)] = sha(path)
        hashes[str(raw_path)] = sha(raw_path)
    if set(records) != set(range(manifest["source_object_count"])):
        raise ValueError("Source captions do not cover every upstream map node")
    hashes.update(source_hashes)
    return dict(summary=str(summary), map_file=str(map_file), captions=str(captions),
        adapter_root=str(adapter_paths[0].parent), model=model, base_url=base_url,
        caption_count=len(records), input_hashes=hashes)


def rebuild(args):
    from .real_scene import run_stage
    if sys.platform != "linux":
        raise RuntimeError("Run this graph-only rebuild manually in WSL")
    if not os.environ.get("DASHSCOPE_API_KEY"):
        raise ValueError("DASHSCOPE_API_KEY is required for new relation requests")
    print("[REAL SCENE] Verifying completed map and caption response evidence...", flush=True)
    source = inspect_completed(args.from_summary)
    root = args.output_dir.resolve() if args.output_dir else ROOT / "outputs/real_scene_runs" / datetime.now().strftime("d3_619_graph_%Y%m%d_%H%M%S_%f")
    protected = [Path(source["summary"]).parent, Path(source["map_file"]).parent, Path(source["adapter_root"])]
    if any(root == p or root in p.parents or p in root.parents for p in protected):
        raise ValueError("New output overlaps existing inputs")
    root.mkdir(parents=True, exist_ok=False)
    report = dict(schema_version="real_scene_graph_rebuild_v1", status="running", source=source,
        reused_stages=["detections", "mapping", "captions"], reused_caption_count=source["caption_count"],
        node_policy="retain_all_upstream_nodes_no_additional_semantic_filter",
        stages=[], simulation_executed=False, execution_requested=False,
        code_sha256={str(p): sha(p) for p in (
            Path(__file__).resolve(), ROOT / "embodied_pipeline/orchestration/real_scene.py",
            ROOT / "embodied_pipeline/perception/conceptgraphs_upstream.py",
            ROOT / "embodied_pipeline/perception/conceptgraphs_scene.py")})
    env = os.environ.copy()
    for name in ("PYTHONHOME", "AMENT_PREFIX_PATH", "CMAKE_PREFIX_PATH", "ROS_PACKAGE_PATH", "PKG_CONFIG_PATH"):
        env.pop(name, None)
    env.update(PYTHONPATH=str(ROOT), PYTHONNOUSERSITE="1", PYTHONUNBUFFERED="1")
    print(f"[REAL SCENE] Graph-only output: {root}", flush=True)
    print(f"[REAL SCENE] Reusing {source['caption_count']} captions; no detection, mapping or caption calls", flush=True)
    try:
        write(root / "run_summary.json", report)
        report["stages"].append(run_stage("graph", [sys.executable, "-B", "-m",
            "embodied_pipeline.orchestration.real_scene", "graph-worker", "--map-file", source["map_file"],
            "--captions-dir", source["captions"], "--output-dir", root,
            "--adapter-root", source["adapter_root"], "--semantic-model", source["model"],
            "--base-url", source["base_url"]], root, env, ROOT))
        report["scene"] = json.loads((root / "scene/scene_metadata.json").read_text())
        report["pipeline"] = json.loads((root / "pipeline/pipeline_scene.json").read_text())
        report["status"] = "graph_generated_and_imported_not_executed"
    except BaseException as exc:
        report.update(status="failed_or_interrupted", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        report["inputs_unchanged"] = all(Path(p).is_file() and sha(p) == h for p, h in source["input_hashes"].items())
        if not report["inputs_unchanged"]:
            report["status"] = "integrity_error"
        write(root / "run_summary.json", report)
        print(f"status={report['status']} summary={root / 'run_summary.json'}", flush=True)
    if report["status"] == "integrity_error":
        raise RuntimeError("Graph rebuild inputs changed during processing")
    return 0
