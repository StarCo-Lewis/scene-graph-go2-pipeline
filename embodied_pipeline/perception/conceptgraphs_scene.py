"""Export local ConceptGraphs objects into the planning scene interface."""

import importlib.util
import json
import os
import pickle
import gzip
import urllib.request
from pathlib import Path

from .._legacy import load
from .real_rgbd import WORLD, write, sha

RELATIONS = {"a on b", "b on a", "a in b", "b in a", "none of these"}
RELATION_PROMPT = """Assess the CURRENT spatial relation of two mapped regions.
Return only English JSON: {"object_relation":"...","reason":"..."}.
Allowed object_relation: a on b, b on a, a in b, b in a, none of these.
a is object1 and b is object2. Coordinates are metres in a right-handed world:
x forward, y left, z UP. Extents are full axis-aligned box dimensions.
Use supplied metric geometry and the observed-region descriptions. Never infer
that an object is on/in another merely because those categories commonly have
that relation. Overlapping boxes/point clouds alone do not prove support or
containment. If available evidence is insufficient, use none of these and say
that the relation is unknown. Do not infer contents, graspability, accessibility,
room connectivity, user location or current robot state. Reason <= 40 words.
The result is a semantic/geometric hypothesis, never verified physical contact.
"""


def external(path, name):
    spec = importlib.util.spec_from_file_location(name, Path(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def relation_query(query, folder, *, model, base_url):
    key = os.environ.get("DASHSCOPE_API_KEY")
    if not key:
        raise ValueError("DASHSCOPE_API_KEY is required for semantic relations")
    request = urllib.request.Request(base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(dict(model=model, enable_thinking=False, temperature=0,
            response_format={"type": "json_object"}, messages=[
                {"role": "system", "content": RELATION_PROMPT},
                {"role": "user", "content": json.dumps(query, ensure_ascii=True)}])).encode("utf-8"),
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"}, method="POST")
    audit = load("response_audit").ResponseAudit(folder, "real_scene_relation", request=request)
    envelope = load("llm_transport").request_json(request, timeout=180, attempts=2, audit=audit)
    content = envelope["choices"][0]["message"]["content"]
    try:
        if envelope["choices"][0].get("finish_reason") == "length":
            raise ValueError("Truncated relation response")
        result = json.loads(content)
        if (set(result) != {"object_relation", "reason"} or result["object_relation"] not in RELATIONS
                or not isinstance(result["reason"], str) or not result["reason"].strip()):
            raise ValueError("Invalid relation output")
        audit.parsed(content)
    except (ValueError, TypeError) as exc:
        audit.parsed(content, "rejected", exc)
        raise
    return dict(result, usage=envelope.get("usage"), response_id=envelope.get("id"),
                semantic_ground_truth=False, verified=False)


def export_scene(map_file, captions, output, adapter_root, *, model, base_url):
    """Default: consume the existing upstream Qwen baseline without extra filtering."""
    from .conceptgraphs_upstream import export_upstream
    return export_upstream(map_file, captions, output, adapter_root, model=model, base_url=base_url)


def export_filtered_scene_legacy(map_file, captions, output, adapter_root, *, model, base_url, query=relation_query):
    """Historical extra-filter policy, retained for regression only; not a default entry."""
    import numpy as np
    output, captions = Path(output), Path(captions)
    output.mkdir(parents=True, exist_ok=False)
    with gzip.open(map_file, "rb") as stream:
        raw = pickle.load(stream)
    objects = raw["objects"] if isinstance(raw, dict) else raw
    if not isinstance(objects, list) or not objects:
        raise ValueError("ConceptGraphs produced no object map")
    geometry_module = external(Path(adapter_root) / "build_qwen_conceptgraph_baseline.py", "real_cg_geometry")
    geometries = [geometry_module.geometry(obj) for obj in objects]
    records = {}
    for path in (captions / "objects").glob("object_*.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        index = record["id"]
        if type(index) is not int or not 0 <= index < len(objects) or index in records:
            raise ValueError("Invalid or duplicate caption object ID")
        records[index] = record
    if set(records) != set(range(len(objects))):
        raise ValueError("Caption records do not cover all mapped objects")
    nodes = []
    for index, geo in enumerate(geometries):
        record = records[index]
        semantic = record.get("qwen") or {}
        label, caption = semantic.get("object_tag", "unknown region"), semantic.get("caption", "Unidentified mapped region")
        valid = (record.get("status") == "SUCCESS" and isinstance(label, str) and bool(label.strip())
            and isinstance(caption, str) and bool(caption.strip())
            and semantic.get("entity_type") not in (None, "unknown", "object_part")
            and semantic.get("is_complete_object") is True
            and type(semantic.get("confidence")) in (int, float) and .5 <= semantic["confidence"] <= 1)
        if not isinstance(label, str) or not isinstance(caption, str):
            raise ValueError("Invalid caption text")
        node = dict(id=index, object_tag=label, caption=caption,
            possible_tags=semantic.get("possible_tags", []), bbox_center=geo["center"].astype(float).tolist(),
            bbox_extent=geo["extent"].astype(float).tolist(), active=valid,
            semantic_status="model_hypothesis" if valid else "uncertain_or_incomplete_region",
            entity_type=semantic.get("entity_type", "unknown"), confidence=semantic.get("confidence"),
            coordinate_frame=WORLD, provenance=dict(source="ConceptGraphs_RGBD_SAM_CLIP_Qwen",
                caption_evidence=f"../captions/objects/object_{index:03d}.json",
                physically_verified=False, training_eligible=False))
        nodes.append(node)
    # Full precision is intentional: rounding a small extent to 0.0 would
    # corrupt the interface and erase small real-world objects.
    load("scene_input_interface").normalize_scene_input(nodes, "Validate the recorded real scene.")
    write(output / "scene_graph.json", nodes)
    active = [i for i, node in enumerate(nodes) if node["active"]]
    if not active:
        raise ValueError("No mapped region has an eligible semantic label; inspect caption evidence")
    overlap = geometry_module.compute_overlap_matrix([geometries[i] for i in active], .025)
    _, components, candidates = geometry_module.build_upstream_mst(overlap, .01)
    components = [[active[i] for i in group] for group in components]
    write(output / "relation_candidates.json", [dict(source=active[c["object1_id"]],
        target=active[c["object2_id"]], overlap=c["overlap"]) for c in candidates])
    edges, decisions = [], []
    for n, candidate in enumerate(candidates, 1):
        a, b = active[candidate["object1_id"]], active[candidate["object2_id"]]
        data = dict(object1={k: nodes[a][k] for k in ("id", "object_tag", "caption", "bbox_center", "bbox_extent")},
                    object2={k: nodes[b][k] for k in ("id", "object_tag", "caption", "bbox_center", "bbox_extent")},
                    overlap=candidate["overlap"], coordinate_frame=WORLD)
        result = query(data, output, model=model, base_url=base_url)
        decision = dict(source=a, target=b, **result)
        decisions.append(decision)
        write(output / "relation_decisions.partial.json", decisions)
        if result["object_relation"] != "none of these":
            edges.append(dict(source=a, target=b, relation=result["object_relation"],
                reason=result["reason"], verified=False, evidence_kind="semantic_geometry_hypothesis",
                evidence="relation_decisions.json"))
        print(f"[REAL SCENE RELATIONS] {n}/{len(candidates)}", flush=True)
    load("llm_graph_binding").validate_edges(edges, [n for n in nodes if n["active"]])
    write(output / "scene_graph_edges.json", edges)
    write(output / "relation_decisions.json", decisions)
    write(output / "map_components.json", components)
    # Preserve feature arrays and stable map indices for optional CLIP consumers.
    for i, obj in enumerate(objects):
        obj["caption_dict"] = dict(id=i, response=dict(summary=nodes[i]["caption"],
            object_tag=nodes[i]["object_tag"], possible_tags=nodes[i]["possible_tags"]))
    (output / "map").mkdir()
    with gzip.open(output / "map/scene_map_cfslam_pruned.pkl.gz", "wb") as stream:
        pickle.dump(objects, stream)
    metadata = dict(schema_version="real_scene_import_v1", scene_id="D3-619_v1",
        coordinate_frame=WORLD, unit="metre", up_axis="z", camera_frame="OpenCV",
        node_count=len(nodes), active_node_count=len(active), inactive_node_count=len(nodes)-len(active),
        edge_count=len(edges), relation_candidate_policy="ConceptGraphs point-overlap MST; not all pairs",
        edges_are_verified=False, missing_relation_means="unknown", map_file=str(Path(map_file).resolve()),
        source_map_sha256=sha(map_file), navigation_graph_available=False,
        lidar_grid_is_separate=True, integration_target="real_robot",
        simulation_required=False, physical_execution_available=False, training_eligible=False)
    write(output / "scene_metadata.json", metadata)
    return metadata


def load_real_scene(scene_dir):
    """Read the actual full graph; never substitute a synthetic scene fixture."""
    root = Path(scene_dir).resolve()
    metadata = json.loads((root / "scene_metadata.json").read_text(encoding="utf-8"))
    if metadata.get("schema_version") != "real_scene_import_v1" or metadata.get("coordinate_frame") != WORLD:
        raise ValueError("Unsupported real-scene metadata/coordinate frame")
    nodes = json.loads((root / "scene_graph.json").read_text(encoding="utf-8"))
    edges = json.loads((root / "scene_graph_edges.json").read_text(encoding="utf-8"))
    nodes = load("scene_input_interface").normalize_scene_input(nodes, "Load the real scene graph.")["nodes"]
    active = [n for n in nodes if n.get("active", True)]
    load("llm_graph_binding").validate_edges(edges, active)
    if not active or len(nodes) != metadata["node_count"] or len(edges) != metadata["edge_count"]:
        raise ValueError("Empty or inconsistent real-scene export")
    return dict(nodes=nodes, edges=edges, metadata=metadata)


def import_scene(scene_dir, output):
    bundle = load_real_scene(scene_dir)
    root, output = Path(scene_dir).resolve(), Path(output).resolve()
    if output == root or root in output.parents or output in root.parents:
        raise ValueError("Import report must be outside the scene export")
    output.mkdir(parents=True, exist_ok=False)
    report = dict(status="graph_interface_validated_not_executed", scene_dir=str(root),
        graph_sha256=sha(root / "scene_graph.json"), edges_sha256=sha(root / "scene_graph_edges.json"),
        node_count=len(bundle["nodes"]), active_node_count=sum(n.get("active", True) for n in bundle["nodes"]),
        edge_count=len(bundle["edges"]), coordinate_frame=WORLD,
        object_binding_mode="llm_graph", perception_loader="embodied_pipeline.perception.load_real_scene",
        integration_target="real_robot", simulation_required=False,
        coordinate_contract="coordinate_contract.json",
        inference_boundary="Supplied semantic relations are hypotheses; legacy binding checks only consistency with supplied edges",
        planning_state_created=False, execution_ready=False,
        required_before_action_rollout=["Actual robot pose in a declared coordinate frame", "Measured gripper/held state",
            "User location and instruction where required", "Robot localization/TF into the SLAM map frame",
            "Connected real-robot skill execution backend"],
        simulation_executed=False, physical_grasp_verified=False, training_eligible=False)
    write(output / "pipeline_scene.json", report)
    write(output / "coordinate_contract.json", dict(schema_version="real_robot_map_frame_v1",
        coordinate_frame=WORLD, unit="metre", handedness="right", axes=dict(x="forward", y="left", z="up"),
        origin="Ground projection of the robot base centre at mapping start, as declared by the source",
        camera_pose_convention="OpenCV camera-to-world; no OpenGL flip",
        axis_conversion_applied=False, simulation_coordinate_conversion_required=False,
        ros_map_frame_id=None, live_robot_tf_registered=False,
        real_robot_requirement="Declare the actual ROS map frame and register current robot/sensor poses to this SLAM map"))
    return report
