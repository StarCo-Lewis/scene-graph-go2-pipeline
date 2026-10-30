"""Run the existing Qwen baseline; preserve its graph and adapt geometry explicitly."""

import gzip
import io
import json
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

from .._legacy import load
from .real_rgbd import WORLD, sha, write
from .conceptgraphs_scene import external


def attach_transport_audit(module, output):
    """Observe original adapter requests without altering its payload or retry policy."""
    original_urlopen = module.urllib.request.urlopen
    audit_module = load("response_audit")
    def observed_urlopen(request, *args, **kwargs):
        audit = audit_module.ResponseAudit(output, "upstream_qwen_relation", request=request)
        print("[UPSTREAM QWEN] Requesting relation; original adapter prompt and retries", flush=True)
        try:
            with original_urlopen(request, *args, **kwargs) as response:
                raw = response.read()
                audit.received(1, raw, getattr(response, "status", 200))
            return io.BytesIO(raw)
        except Exception as exc:
            # Preserve HTTP error bodies for diagnostics without consuming the
            # stream that the original adapter's exception handler will read.
            if isinstance(exc, module.urllib.error.HTTPError):
                raw = exc.read()
                audit.received(1, raw, exc.code)
                raise module.urllib.error.HTTPError(exc.url, exc.code, exc.msg, exc.hdrs, io.BytesIO(raw)) from exc
            else:
                audit.transport_error(1, exc)
            raise
    # Replace only this loaded adapter's namespace, not global urllib functions.
    module.urllib = SimpleNamespace(request=SimpleNamespace(
        Request=module.urllib.request.Request, urlopen=observed_urlopen), error=module.urllib.error)


def adapt_baseline(map_file, captions, output, geometry_module):
    """Preserve upstream IDs, text and edges; use source-map AABBs at full precision."""
    output, captions = Path(output), Path(captions)
    upstream = output / "upstream_export"
    if (upstream / "STATUS").read_text().strip() != "SUCCESS":
        raise ValueError("Upstream graph export did not complete")
    nodes = json.loads((upstream / "scene_graph.json").read_text(encoding="utf-8"))
    edges = json.loads((upstream / "scene_graph_edges.json").read_text(encoding="utf-8"))
    manifest = json.loads((upstream / "baseline_manifest.json").read_text(encoding="utf-8"))
    with gzip.open(map_file, "rb") as stream:
        raw = pickle.load(stream)
    objects = raw["objects"] if isinstance(raw, dict) else raw
    if (not isinstance(objects, list) or not objects or len(nodes) != len(objects)
            or [n.get("id") for n in nodes] != list(range(len(objects)))
            or manifest["node_count"] != len(nodes) or manifest["semantic_edge_count"] != len(edges)):
        raise ValueError("Upstream map/node/edge coverage mismatch")
    changes = []
    for node, obj in zip(nodes, objects):
        record = json.loads((captions / "objects" / f"object_{node['id']:03d}.json").read_text(encoding="utf-8"))
        if record.get("status") != "SUCCESS" or record.get("id") != node["id"]:
            raise ValueError("Unsuccessful or misindexed source caption")
        for key in ("caption", "object_tag", "possible_tags"):
            if node.get(key) != record["qwen"].get(key):
                raise ValueError("Upstream caption changed during export: " + key)
        if "active" in node:
            raise ValueError("Unexpected upstream active policy; review upstream version")
        geometry = geometry_module.geometry(obj)
        before = {key: node[key] for key in ("bbox_center", "bbox_extent")}
        node["bbox_center"] = geometry["center"].astype(float).tolist()
        node["bbox_extent"] = geometry["extent"].astype(float).tolist()
        changes.append(dict(id=node["id"], upstream_rounded=before,
            source_map_full_precision={key: node[key] for key in before}))
    # True zero/invalid dimensions are rejected; never epsilon-pad or drop nodes.
    load("scene_input_interface").normalize_scene_input(nodes, "Validate the upstream real scene.")
    load("llm_graph_binding").validate_edges(edges, nodes)
    write(output / "scene_graph.json", nodes)
    # Keep the upstream edge file byte-for-byte, including reasons and endpoint IDs.
    (output / "scene_graph_edges.json").write_bytes((upstream / "scene_graph_edges.json").read_bytes())
    write(output / "geometry_precision_adapter.json", dict(
        policy="Source-map AABB full precision; no node, label, edge, axis or unit changes",
        source_map_sha256=sha(map_file), changes=changes))
    for node, obj in zip(nodes, objects):
        obj["caption_dict"] = dict(id=node["id"], response=dict(summary=node["caption"],
            object_tag=node["object_tag"], possible_tags=node["possible_tags"]))
    (output / "map").mkdir()
    with gzip.open(output / "map/scene_map_cfslam_pruned.pkl.gz", "wb") as stream:
        pickle.dump(objects, stream)
    metadata = dict(schema_version="real_scene_import_v1", scene_id="D3-619_v1",
        coordinate_frame=WORLD, unit="metre", up_axis="z", camera_frame="OpenCV",
        node_count=len(nodes), active_node_count=len(nodes), inactive_node_count=0,
        active_count_meaning="All retained upstream nodes; NOT a manipulability claim",
        node_policy="retain_all_upstream_nodes_no_additional_semantic_filter",
        edge_count=len(edges), relation_candidate_policy="Existing Qwen baseline overlap MST on all upstream nodes",
        relation_prompt_policy="Existing Qwen replacement adapter, unchanged",
        model_substitution=manifest["model_substitution"], edges_are_verified=False,
        missing_relation_means="unknown", map_file=str(Path(map_file).resolve()), source_map_sha256=sha(map_file),
        captions_dir=str(captions.resolve()), upstream_export="upstream_export",
        upstream_graph_sha256=sha(upstream / "scene_graph.json"),
        upstream_edges_sha256=sha(upstream / "scene_graph_edges.json"),
        geometry_adapter="geometry_precision_adapter.json", navigation_graph_available=False,
        lidar_grid_is_separate=True, integration_target="real_robot", simulation_required=False,
        physical_execution_available=False, training_eligible=False)
    write(output / "scene_metadata.json", metadata)
    return metadata


def export_upstream(map_file, captions, output, adapter_root, *, model, base_url):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    adapter_path = Path(adapter_root) / "build_qwen_conceptgraph_baseline.py"
    module = external(adapter_path, "real_upstream_graph_adapter")
    attach_transport_audit(module, output)
    original_argv = sys.argv
    try:
        sys.argv = [str(adapter_path), "--map-file", str(map_file), "--captions-dir", str(captions),
            "--output-dir", str(output / "upstream_export"), "--model", model, "--base-url", base_url]
        module.main()
    finally:
        sys.argv = original_argv
    return adapt_baseline(map_file, captions, output, module)
