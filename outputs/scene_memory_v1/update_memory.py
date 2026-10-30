#!/usr/bin/env python3
"""Evidence-gated scene memory. Original map files are never overwritten."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=True, indent=2, allow_nan=False) + "\n", encoding="ascii")


def digest(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def vector(value):
    if not isinstance(value, list) or len(value) != 3 or any(
            type(v) not in (int, float) or not math.isfinite(v) for v in value):
        raise ValueError("Position must be three finite numbers in metres")
    return [float(v) for v in value]


def timestamp(value):
    if not isinstance(value, str):
        raise ValueError("observed_at must be an ISO timestamp with timezone")
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        raise ValueError("observed_at must include timezone")
    return dt


def initialize(nodes, edges, scene):
    if not isinstance(nodes, list) or not nodes or not isinstance(edges, list):
        raise ValueError("Expected nonempty node list and edge list")
    ids = [n.get("id") for n in nodes]
    if any(type(i) is not int for i in ids) or len(set(ids)) != len(ids):
        raise ValueError("Scene node IDs must be unique integers")
    for edge in edges:
        if edge.get("source") not in ids or edge.get("target") not in ids:
            raise ValueError("Scene relation refers to an absent node")
    return {"schema_version": "1.0", "memory_type": "incremental_object_state",
            "base_scene": str(scene), "base_graph_digest": digest({"nodes": nodes, "edges": edges}),
            "coordinate_frame": "conceptgraphs_world", "version": 0,
            "entities": {str(n["id"]): {"node": copy.deepcopy(n), "active": True,
                "last_observed_at": None} for n in nodes},
            "edges": copy.deepcopy(edges), "processed_events": {}, "event_log": []}


def apply_events(previous, events):
    """Atomic in-memory batch; caller persists only after all events validate."""
    if not isinstance(events, list):
        raise ValueError("Events must be an array")
    state = copy.deepcopy(previous)
    if state.get("memory_type") != "incremental_object_state":
        raise ValueError("Not a supported memory state")
    for event in events:
        if not isinstance(event, dict):
            raise ValueError("Each event must be an object")
        event_id = event.get("event_id")
        if not isinstance(event_id, str) or not event_id:
            raise ValueError("event_id required")
        fingerprint = digest(event)
        prior = state["processed_events"].get(event_id)
        if prior:
            if prior != fingerprint:
                raise ValueError("Same event ID used for different content")
            continue
        node_id = event.get("object_id")
        if type(node_id) is not int or str(node_id) not in state["entities"]:
            raise ValueError("NO object can be used in this task")
        entity = state["entities"][str(node_id)]
        if not entity["active"]:
            raise ValueError("Cannot change an inactive object")
        evidence = event.get("evidence", {})
        if (event.get("action_status") != "completed" or evidence.get("verified") is not True
                or evidence.get("source") not in ("simulator_observation", "verified_post_action_observation", "operator_confirmation")
                or not isinstance(evidence.get("reference"), str) or not evidence["reference"]
                or not isinstance(event.get("action_id"), str) or not event["action_id"]):
            raise ValueError("Verified completed-action evidence required; predictions/assumed grasp are not map observations")
        observed_at = timestamp(event.get("observed_at"))
        if entity["last_observed_at"] and observed_at <= timestamp(entity["last_observed_at"]):
            raise ValueError("Stale or ambiguous equal-time object update")
        before = copy.deepcopy(entity)
        operation = event.get("operation")
        support = None
        if operation == "move":
            if event.get("coordinate_frame") != state["coordinate_frame"]:
                raise ValueError("Position coordinate-frame mismatch")
            position = vector(event.get("position"))
            support = event.get("support_object_id")
            if support is not None and (type(support) is not int or support == node_id or
                    not state["entities"].get(str(support), {}).get("active")):
                raise ValueError("Support object must be a distinct active known node")
            # Do not pretend that supported/contained children remain at old positions.
            for edge in state["edges"]:
                a, b, relation = edge["source"], edge["target"], edge.get("relation")
                carries_children = ((b == node_id and a != node_id and relation in ("a on b", "a in b"))
                                    or (a == node_id and b != node_id and relation in ("b on a", "b in a")))
                if carries_children:
                    raise ValueError("Container/support group movement needs child-state observations; unsupported in V1")
            entity["node"]["bbox_center"] = position
            # Old-location surface voxels/metadata are not valid after movement.
            entity["node"].pop("visual_memory", None)
            entity["node"]["memory_pose"] = {"event_id": event_id,
                "source": evidence["source"], "observed_at": event["observed_at"],
                "appearance_features": "inherited_not_reobserved"}
        elif operation == "delete":
            if event.get("reason") not in ("consumed_confirmed", "removed_confirmed"):
                raise ValueError("Delete requires confirmed consumption/removal; cutting or missed detection is not deletion")
            entity["active"] = False
            entity["tombstone"] = {"event_id": event_id, "reason": event["reason"]}
        else:
            raise ValueError("Only move and delete are supported")
        invalidated = [e for e in state["edges"] if node_id in (e["source"], e["target"])]
        state["edges"] = [e for e in state["edges"] if node_id not in (e["source"], e["target"])]
        if support is not None:
            state["edges"].append({"source": node_id, "target": support, "relation": "a on b",
                "reason": "Verified post-action support", "memory_event_id": event_id})
        entity["last_observed_at"] = event["observed_at"]
        state["version"] += 1
        state["processed_events"][event_id] = fingerprint
        state["event_log"].append({"version": state["version"], "event": copy.deepcopy(event),
            "before": before, "after": copy.deepcopy(entity), "invalidated_edges": invalidated})
    return state


def path_key(value):
    value = str(value).replace("\\", "/")
    if len(value) > 2 and value[1] == ":":
        value = "/mnt/" + value[0].lower() + value[2:]
    return value.rstrip("/")


def proxy_events(proxy, nodes, reference, observed_at):
    """Convert verified final placement, not assumed grasp, to one move event."""
    if proxy.get("test_fault", "none") != "none":
        raise ValueError("Fault-injected runs are not map observation sources")
    if proxy.get("status") != "proxy_instance_support_passed":
        return []
    ids = proxy.get("scene_graph_ids", {})
    kettle, board = ids.get("kettle"), ids.get("sideboard")
    node = next((n for n in nodes if n["id"] == kettle), None)
    if node is None or not any(n["id"] == board for n in nodes):
        raise ValueError("NO object can be used in this task")
    steps = proxy.get("steps", [])
    initial = next((s for s in steps if s.get("skill_id") == "instantiate_object"), {})
    cabinet = next((s for s in steps if s.get("skill_id") == "instantiate_sideboard"), {})
    placement = next((s for s in reversed(steps) if s.get("skill_id") == "place_object_on_surface"), {})
    if (placement.get("status") != "proxy_instance_support_passed" or
            placement.get("stable_last_second") is not True or
            placement.get("direct_sideboard_contact_frames_last_second", 0) < 20 or
            initial.get("cg_element_id") != kettle or cabinet.get("cg_element_id") != board or
            placement.get("sideboard_habitat_object_id") != cabinet.get("habitat_object_id") or
            type(initial.get("habitat_object_id")) is not int or type(cabinet.get("habitat_object_id")) is not int):
        raise ValueError("Placement instance/support evidence incomplete")
    old_com = vector(initial.get("position"))
    new_com = vector(placement.get("settled_position"))
    origin = vector(node.get("bbox_center"))
    # Transfer a translation delta, not the raw COM, to the graph bbox centroid.
    # Habitat -> CG delta: (dx, -dz, dy). No rotation was estimated.
    delta = [new_com[i] - old_com[i] for i in range(3)]
    position = [origin[0] + delta[0], origin[1] - delta[2], origin[2] + delta[1]]
    return [{"event_id": "proxy-placement-" + digest(proxy), "action_id": "place_object_on_surface",
        "operation": "move", "object_id": kettle, "action_status": "completed",
        "coordinate_frame": "conceptgraphs_world", "position": position,
        "support_object_id": board, "observed_at": observed_at,
        "evidence": {"source": "simulator_observation", "verified": True,
            "reference": str(reference), "proxy_digest": digest(proxy),
            "position_method": "translation_delta_from_proxy_COM_to_original_bbox_centroid",
            "orientation_update": "not_estimated"}}]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    inputs = p.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--scene-dir", type=Path)
    inputs.add_argument("--previous-state", type=Path)
    events = p.add_mutually_exclusive_group()
    events.add_argument("--events-json", type=Path)
    events.add_argument("--proxy-json", type=Path)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    try:
        if args.output_dir.exists():
            raise ValueError("Output directory exists; choose a new snapshot directory")
        if args.previous_state:
            state = read(args.previous_state)
            scene = Path(state["base_scene"])
        else:
            scene = args.scene_dir.resolve()
            state = initialize(read(scene / "scene_graph.json"), read(scene / "scene_graph_edges.json"), scene)
        base_nodes, base_edges = read(scene / "scene_graph.json"), read(scene / "scene_graph_edges.json")
        if digest({"nodes": base_nodes, "edges": base_edges}) != state["base_graph_digest"]:
            raise ValueError("Base graph changed; cannot continue this memory")
        batch = read(args.events_json) if args.events_json else []
        if args.proxy_json:
            proxy = read(args.proxy_json)
            if path_key(proxy.get("input", {}).get("scene_graph", "")) != path_key(scene / "scene_graph.json"):
                raise ValueError("Proxy scene graph differs from memory base scene")
            observed_at = datetime.fromtimestamp(args.proxy_json.stat().st_mtime, timezone.utc).isoformat()
            batch = proxy_events(proxy, base_nodes, args.proxy_json.resolve(), observed_at)
        new_state = apply_events(state, batch)
        map_file = scene / "map/scene_map_cfslam_pruned.pkl.gz"
        if not map_file.is_file():
            raise ValueError("Inherited CLIP feature map is missing")
        # Save an isolated aligner-compatible snapshot, keeping feature IDs stable.
        args.output_dir.mkdir(parents=True, exist_ok=False)
        snapshot = args.output_dir / "scene"
        (snapshot / "map").mkdir(parents=True)
        active = [e["node"] for e in new_state["entities"].values() if e["active"]]
        write(args.output_dir / "memory_state.json", new_state)
        write(args.output_dir / "applied_events.json", batch)
        write(snapshot / "scene_graph.json", active)
        write(snapshot / "scene_graph_edges.json", new_state["edges"])
        shutil.copy2(map_file, snapshot / "map" / map_file.name)
        manifest = {"status": "memory_snapshot_ready", "version": new_state["version"],
            "active_node_ids": [n["id"] for n in active], "base_scene": str(scene),
            "source_files_modified": False, "feature_map_geometry": "original_not_reconstructed",
            "scope": "Object-state memory, not visual fusion or a reconstructed physical scene",
            "position_policy": "Translation-only update; rotation/extent/child movement not inferred"}
        write(args.output_dir / "memory_summary.json", manifest)
        print(json.dumps(dict(manifest, output_dir=str(args.output_dir)), ensure_ascii=True, indent=2))
        return 0
    except (ValueError, KeyError, TypeError, OSError) as exc:
        print(json.dumps({"status": "error", "message": str(exc)}, ensure_ascii=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
