"""Read-only graph audits and byte-faithful, node-linked modality packaging.

No model calls, graph completion, room inference, or robot execution occurs here.
Absent pairwise relations are unknown, not evidence that an export is incomplete.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil


RELATIONS = {"a on b", "b on a", "a in b", "b in a"}
UNKNOWN = {"unknown", "none", "null", "n/a", "unrecognized", "unidentified"}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def local_path(raw, base=None):
    """Map recorded /mnt/<drive> paths on Windows only; never invoke WSL."""
    raw = str(raw)
    if os.name == "nt" and re.match(r"^/mnt/[A-Za-z]/", raw):
        raw = raw[5].upper() + ":/" + raw[7:]
    elif os.name != "nt" and re.match(r"^[A-Za-z]:[/\\]", raw):
        raw = "/mnt/" + raw[0].lower() + "/" + raw[3:].replace("\\", "/")
    path = Path(raw)
    return path if path.is_absolute() or base is None else Path(base) / path


def audit_graph(scene_dir):
    """Check exported data integrity without declaring semantic truth or completion."""
    scene_dir = Path(scene_dir)
    result = {"schema_version": "graph_resource_audit_v1", "scene_dir": str(scene_dir),
              "node_count": 0, "edge_count": 0, "blocking_issues": [], "warnings": [],
              "unknown_semantic_node_ids": [], "zero_extent_node_ids": [],
              "semantic_correctness_evaluated": False,
              "absent_relation_means": "unknown", "graph_modified": False}
    values = {}
    for name, key in (("scene_graph.json", "graph"), ("scene_graph_edges.json", "edges")):
        path = scene_dir / name
        result[key + "_path"] = str(path)
        if not path.is_file():
            result["blocking_issues"].append({"code": "missing_file", "file": name})
            continue
        result[key + "_sha256"] = sha256(path)
        try:
            value = _json(path)
            if not isinstance(value, list):
                raise ValueError("Expected a JSON array")
            values[key] = value
        except (ValueError, OSError) as exc:
            result["blocking_issues"].append({"code": "invalid_json_array", "file": name,
                                               "detail": str(exc)})
    graph, edges = values.get("graph", []), values.get("edges", [])
    result["node_count"], result["edge_count"] = len(graph), len(edges)
    ids = set()
    for row_index, node in enumerate(graph):
        if not isinstance(node, dict):
            result["blocking_issues"].append({"code": "invalid_node", "row_index": row_index})
            continue
        node_id = node.get("id")
        if type(node_id) is not int or node_id < 0:
            result["blocking_issues"].append({"code": "invalid_node_id", "row_index": row_index})
        elif node_id in ids:
            result["blocking_issues"].append({"code": "duplicate_node_id", "node_id": node_id})
        else:
            ids.add(node_id)
        for field in ("object_tag", "caption"):
            if not isinstance(node.get(field), str) or not node[field].strip():
                result["blocking_issues"].append({"code": "missing_semantic_field", "node_id": node_id,
                                                   "field": field})
        if str(node.get("object_tag", "")).strip().lower() in UNKNOWN:
            result["unknown_semantic_node_ids"].append(node_id)
        for field in ("bbox_center", "bbox_extent"):
            vector = node.get(field)
            valid = (isinstance(vector, list) and len(vector) == 3 and
                     all(type(x) in (int, float) and math.isfinite(x) for x in vector))
            if not valid or (field == "bbox_extent" and any(x < 0 for x in vector)):
                result["blocking_issues"].append({"code": "invalid_bbox", "node_id": node_id,
                                                   "field": field})
            elif field == "bbox_extent" and any(x == 0 for x in vector):
                result["zero_extent_node_ids"].append(node_id)
    seen_edges, endpoint_ids, relation_counts = set(), set(), Counter()
    for index, edge in enumerate(edges):
        if not isinstance(edge, dict):
            result["blocking_issues"].append({"code": "invalid_edge", "row_index": index})
            continue
        source, target, relation = edge.get("source"), edge.get("target"), edge.get("relation")
        if type(source) is not int or type(target) is not int or source not in ids or target not in ids:
            result["blocking_issues"].append({"code": "invalid_edge_endpoints", "row_index": index})
            continue
        endpoint_ids.update((source, target))
        if source == target:
            result["blocking_issues"].append({"code": "self_reference", "row_index": index})
        if not isinstance(relation, str) or relation not in RELATIONS:
            result["blocking_issues"].append({"code": "unsupported_relation", "row_index": index,
                                               "relation": relation})
            continue
        relation_counts[relation] += 1
        triple = (source, target, relation)
        if triple in seen_edges:
            result["warnings"].append({"code": "duplicate_edge", "row_index": index})
        seen_edges.add(triple)
    result["relation_counts"] = dict(sorted(relation_counts.items()))
    result["nodes_without_exported_relations"] = sorted(ids - endpoint_ids)
    for field, code in (("unknown_semantic_node_ids", "genuine_unknown_semantics"),
                        ("zero_extent_node_ids", "zero_extent_geometry_needs_review")):
        if result[field]:
            result["warnings"].append({"code": code, "node_ids": result[field]})
    if "graph" in values and not graph:
        result["blocking_issues"].append({"code": "empty_graph"})
    if "edges" in values and not edges:
        result["warnings"].append({"code": "empty_edges_not_proof_of_missing_relations"})
    result["status"] = "blocked" if result["blocking_issues"] else "structurally_readable_semantics_unverified"
    return result


def _copy_checked(source, destination):
    source, destination = Path(source), Path(destination)
    source_hash = sha256(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if sha256(destination) != source_hash:
            raise ValueError("Refusing to overwrite differing existing bundle file: " + str(destination))
    else:
        shutil.copy2(source, destination)
    copied_hash = sha256(destination)
    if copied_hash != source_hash:
        raise ValueError("Copied bytes differ: " + str(destination))
    return source_hash, copied_hash


def _caption_directory(scene_dir, explicit=None):
    if explicit:
        return local_path(explicit)
    candidates = ((scene_dir / "scene_metadata.json", "captions_dir"),
                  (scene_dir / "baseline_manifest.json", "captions_dir"),
                  (scene_dir / "upstream_export" / "baseline_manifest.json", "captions_dir"))
    for file, field in candidates:
        if file.is_file():
            raw = _json(file).get(field)
            if raw:
                return local_path(raw, file.parent)
    summary = scene_dir.parent / "run_summary.json"
    if summary.is_file():
        raw = _json(summary).get("source", {}).get("captions")
        if raw:
            return local_path(raw, summary.parent)
    return None


def collect_modalities(scene_id, scene_dir, bundle_scene_dir, *, captions_dir=None):
    """Copy original graphs, provenance metadata and <=1 existing image per node.

    The selected image must be linked by a node's visual_evidence or its same-ID
    source caption record. There is no filename-nearness/semantic-name matching.
    Full source metadata preserves unselected paths; the index says what was
    actually bundled. A source hash mismatch is a gap, never silently accepted.
    """
    scene_dir, bundle_scene_dir = Path(scene_dir), Path(bundle_scene_dir)
    audit = audit_graph(scene_dir)
    if audit["blocking_issues"]:
        raise ValueError("Graph export has blocking integrity problems: " + json.dumps(audit["blocking_issues"]))
    graph = _json(scene_dir / "scene_graph.json")
    result = {"schema_version": "node_modalities_index_v1", "scene_id": scene_id,
              "selection_policy": "first_available_provenanced_view_per_node",
              "graph_sha256": audit["graph_sha256"], "edges_sha256": audit["edges_sha256"],
              "node_resources": [], "metadata_resources": [], "missing_node_ids": [],
              "source_gaps": [], "semantics_independently_verified": False,
              "raw_rgbd_complete": False, "graph_modified": False}

    def copy_metadata(source, relative):
        digest, copied = _copy_checked(source, bundle_scene_dir / relative)
        result["metadata_resources"].append({"source_path": str(source), "bundle_path": relative,
                                               "source_sha256": digest, "copy_sha256": copied})

    for name in ("scene_graph.json", "scene_graph_edges.json"):
        copy_metadata(scene_dir / name, name)
    metadata = [scene_dir / name for name in ("scene_metadata.json", "baseline_manifest.json",
                                              "geometry_manifest.json", "geometry_precision_adapter.json",
                                              "compatibility_export.json")]
    metadata += [scene_dir / "upstream_export" / "baseline_manifest.json",
                 scene_dir.parent / "evidence_manifest.json",
                 scene_dir.parent / "graph_generation_summary.json",
                 scene_dir.parent / "relation_decisions.json"]
    for path in metadata:
        if path.is_file():
            prefix = "upstream_" if path.parent.name == "upstream_export" else ""
            copy_metadata(path, "modalities/metadata/" + prefix + path.name)
    evidence_file = scene_dir.parent / "evidence_manifest.json"
    evidence = _json(evidence_file) if evidence_file.is_file() else {}
    inventory = {row["id"]: row for row in evidence.get("inventory", []) if isinstance(row, dict) and "id" in row}
    caption_dir = _caption_directory(scene_dir, captions_dir)
    if caption_dir and (caption_dir / "qwen_caption_manifest.json").is_file():
        copy_metadata(caption_dir / "qwen_caption_manifest.json", "modalities/metadata/qwen_caption_manifest.json")
    for node in graph:
        node_id = node["id"]
        candidates = []
        evidence_views = {v["path"]: v for v in inventory.get(node_id, {}).get("render_views", []) if "path" in v}
        for raw in node.get("visual_evidence", {}).get("views", []):
            if isinstance(raw, str):
                candidates.append({"source": local_path(raw, scene_dir.parent),
                                   "expected_sha256": evidence_views.get(raw, {}).get("sha256"),
                                   "link_evidence": {"type": "graph_node_visual_evidence", "node_id": node_id,
                                                     "source_reference": raw}})
        if caption_dir:
            caption_file = caption_dir / "objects" / ("object_%03d.json" % node_id)
            if caption_file.is_file():
                source_caption = _json(caption_file)
                if source_caption.get("id") != node_id:
                    result["source_gaps"].append({"code": "caption_id_mismatch", "node_id": node_id,
                                                  "source_path": str(caption_file)})
                else:
                    caption_relative = "modalities/metadata/" + caption_file.name
                    copy_metadata(caption_file, caption_relative)
                    for view_index, view in enumerate(source_caption.get("selected_views", [])):
                        for key in ("crop", "image"):
                            if view.get(key):
                                candidates.append({"source": local_path(view[key], caption_dir),
                                                   "link_evidence": {"type": "same_id_caption_selected_view",
                                                     "node_id": node_id, "caption_file": caption_relative,
                                                     "view_index": view_index, "field": key,
                                                     "original_image": view.get("image")}})
            else:
                result["source_gaps"].append({"code": "missing_caption_record", "node_id": node_id,
                                              "source_path": str(caption_file)})
        selected = None
        for candidate in candidates:
            source = candidate["source"]
            if not source.is_file():
                result["source_gaps"].append({"code": "missing_source_image", "node_id": node_id,
                                              "source_path": str(source)})
                continue
            digest = sha256(source)
            expected = candidate.get("expected_sha256")
            if expected and digest != expected.lower():
                result["source_gaps"].append({"code": "source_hash_mismatch", "node_id": node_id,
                                              "source_path": str(source), "expected_sha256": expected,
                                              "actual_sha256": digest})
                continue
            relative = "modalities/images/node_%08d%s" % (node_id, source.suffix.lower())
            digest, copied = _copy_checked(source, bundle_scene_dir / relative)
            selected = {"node_id": node_id, "source_path": str(source), "bundle_path": relative,
                        "source_sha256": digest, "copy_sha256": copied, "exists": True,
                        "link_evidence": candidate["link_evidence"]}
            break
        if selected is None:
            result["missing_node_ids"].append(node_id)
            selected = {"node_id": node_id, "source_path": None, "bundle_path": None,
                        "source_sha256": None, "copy_sha256": None, "exists": False,
                        "gap_reason": "no_available_provenanced_image"}
        result["node_resources"].append(selected)
    result["node_count"] = len(graph)
    result["resource_count"] = sum(row["exists"] for row in result["node_resources"])
    result["status"] = "complete_selected_node_views" if not result["missing_node_ids"] else "incomplete_selected_node_views"
    _write_json(bundle_scene_dir / "graph_resource_audit.json", audit)
    _write_json(bundle_scene_dir / "modalities" / "index.json", result)
    return result
