"""Deterministic graph-grounded candidate intents, never model answers or truth.

This module performs no API, ROS, WSL, simulation or filesystem writes.  The
caller must keep returned reference artifacts OUTSIDE model inputs.  A candidate
is a proposed task, not an independently verified feasible/valid training item.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone, timedelta
from itertools import chain
from pathlib import Path
import re


VERSION = "graph_grounded_task_templates_v3"
ROOT = Path(__file__).resolve().parents[2]
QUOTAS = {
    "navigate": 25,
    "single_transport": 35,
    "pickup_hold": 20,
    "protected_transport": 30,
    "ordered_two_delivery": 30,
    "ordered_three_delivery": 35,
    "ordered_visit": 25,
}
ROOM_FILES = {
    "scene200": ROOT / "outputs/vlm_full_scene_v1/scenarios/scene200_symbolic_template.json",
    "scene202": ROOT / "outputs/vlm_full_scene_v1/scenarios/scene202_crossroom_6.json",
}
PRIORITY_MOVERS = {
    "scene200": [124, 152, 230, 236],
    "scene202": [121, 86, 123, 125, 118],
    "scene201": [67, 102, 93, 94],
    "d3_619": [8, 50, 110, 176, 180, 181, 106, 205],
}
PRIORITY_SURFACES = {
    "scene200": [110, 133, 165, 136],
    "scene202": [101, 69, 2, 303, 307],
    "scene201": [13, 21, 24, 75, 97, 113],
    "d3_619": [85, 179, 130, 188, 212, 255, 351],
}
# Classification applies ONLY to candidate authoring; source graphs are untouched.
# No graspability / safe standing-space claim follows from these lexical filters.
BAD = re.compile(r"\b(fragment|section|segment|corner|leg|legs|backrest|back|torso|person|human|humanoid|robot|unknown|panel|outline|gap|edge|trim|rail|seat|housing|body|stand-like|handle|base)\b|^wall$|^ceiling$", re.I)
MOVABLE = re.compile(r"\b(book|notebook|bottle|mouse|mug|cup|kettle|smartphone|pillow|cushion|box|tape dispenser|roll of .*tape|ornament|towel|blanket|folded cloth|container)\b", re.I)
SURFACE = re.compile(r"\b(desk|table|table top|tabletop|table surface|workbench|pedestal|cabinet|sideboard|shelf)$", re.I)
LANDMARK = re.compile(r"\b(chair|sofa|couch|desk|table|table top|tabletop|table surface|workbench|pedestal|cabinet|sideboard|shelf|clock|whiteboard|monitor|television|door|planter|coat rack|artwork|rug|office machine|speaker|heater|radiator)$", re.I)
SOFT_SUPPORT = re.compile(r"\b(sofa|couch|chair|ottoman)$", re.I)
TEXTILE = re.compile(r"\b(pillow|cushion|blanket|towel|folded cloth)\b", re.I)
PAIR_RULE_VERSION = "declared_up_axis_aabb_two_quarter_turns_v2_not_physical_feasibility"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()).hexdigest()


def normalize_instruction(text):
    """Spacing/case/punctuation normalization, not semantic equivalence."""
    return " ".join(re.findall(r"[a-z0-9]+", text.casefold()))


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _sorted_pool(nodes, pattern, priority=(), portable=False):
    result = []
    for node in nodes:
        tag = str(node.get("object_tag", "")).replace("_", " ")
        head = tag.split(" with ", 1)[0]
        extent = node.get("bbox_extent", [])
        if (node.get("active", True) is False or BAD.search(head)
                or not pattern.search(head) or len(extent) != 3
                or any(type(x) not in (int, float) or not math.isfinite(x) or x < 0 for x in extent)):
            continue
        # Deliberately reject sofa-sized "pillows" and other likely merged regions.
        if portable and (max(extent) > 0.85 or "planter" in head):
            continue
        result.append(node)
    rank = {identity: i for i, identity in enumerate(priority)}
    return sorted(result, key=lambda n: (rank.get(n["id"], len(rank)), n["id"]))


def _rooms(scene_id, by_id):
    """Use explicit existing author declarations, never invent room topology."""
    path = ROOM_FILES.get(scene_id)
    if not path or not path.is_file():
        return {}, {"status": "unknown", "room_connectivity_supplied": False}
    source = _read(path)
    mapping = {int(k): v for k, v in source.get("control_node_rooms", {}).items() if int(k) in by_id}
    return mapping, {
        "status": "author_assigned_not_perception_truth",
        "source_path": str(path), "sha256": sha256(path),
        "source_statement": source["room_assignment_source"],
        "room_connectivity_supplied": False,
    }


def _support_facts(edges):
    result = {}
    for index, edge in enumerate(edges):
        relation = edge.get("relation")
        if relation == "a on b":
            child, parent = edge.get("source"), edge.get("target")
        elif relation == "b on a":
            child, parent = edge.get("target"), edge.get("source")
        else:
            continue
        if child == parent:
            continue
        result.setdefault(child, []).append({
            "object_id": child, "support_id": parent, "edge_index": index,
            "source": "supplied_graph_edge_not_physical_truth", "relation": "on",
        })
    return result


def _support_conflicts(facts):
    """Flag source hypotheses, including cycles, without editing/removing edges."""
    adjacency = {obj: {f["support_id"] for f in values} for obj, values in facts.items()}
    conflicts = {}
    for obj, destinations in adjacency.items():
        issues = ["multiple_supplied_supports_not_disambiguated"] if len(destinations) > 1 else []
        pending, visited = list(destinations), set()
        while pending:
            current = pending.pop()
            if current == obj:
                issues.append("cycle_in_supplied_on_relations")
                break
            if current not in visited:
                visited.add(current)
                pending.extend(adjacency.get(current, ()))
        if issues:
            conflicts[obj] = issues
    return conflicts


def _geometry_contract(scene_dir):
    path = scene_dir / "scene_metadata.json"
    if path.is_file():
        metadata = _read(path)
        up = str(metadata.get("up_axis", "")).lower()
        unit = str(metadata.get("unit", "")).lower()
        if up in ("x", "y", "z") and unit in ("m", "metre", "meter", "metres", "meters"):
            indices = tuple(i for i, name in enumerate(("x", "y", "z")) if name != up)
            return indices, {"status": "explicit_up_axis_and_metre_unit", "up_axis": up,
                             "horizontal_axes": ["xyz"[i] for i in indices],
                             "source_path": str(path), "sha256": sha256(path)}
    return None, {"status": "geometry_prefilter_unknown", "up_axis": None, "horizontal_axes": None,
                  "reason": "No consumed scene_metadata.json declares both up_axis and metre units; no XY/XZ assumption.",
                  "source_path": str(path) if path.is_file() else None}


def _fits_horizontal(object_node, surface_node, axes=(0, 1)):
    """Conservative bbox footprint prefilter, not contact/reachability validation."""
    ox, oy = [object_node["bbox_extent"][i] for i in axes]
    sx, sy = [surface_node["bbox_extent"][i] for i in axes]
    return (ox <= sx and oy <= sy) or (oy <= sx and ox <= sy)


def _allowed_placement(object_node, surface_node, axes=(0, 1)):
    if SOFT_SUPPORT.search(surface_node["object_tag"]):
        if not TEXTILE.search(object_node["object_tag"]):
            return False
    return axes is None or _fits_horizontal(object_node, surface_node, axes)


def _complete_saved_caption(raw):
    return (raw.get("status") == "SUCCESS"
            and raw.get("qwen", {}).get("is_complete_object") is True
            and raw["qwen"].get("entity_type") in ("object", "furniture"))


def _caption_evidence(scene_dir):
    """Read saved real-scene evidence, never recaption or change graph activity."""
    summary = scene_dir.parent / "run_summary.json"
    if not summary.is_file():
        return {}, []
    caption_dir = _read(summary).get("source", {}).get("captions")
    if not caption_dir:
        return {}, []
    if re.match(r"^/mnt/[a-z]/", caption_dir) and str(ROOT).find(":") == 1:
        caption_dir = caption_dir[5].upper() + ":/" + caption_dir[7:]
    objects = Path(caption_dir) / "objects"
    result, resources = {}, []
    if objects.is_dir():
        for path in sorted(objects.glob("object_*.json")):
            raw = _read(path)
            if type(raw.get("id")) is int:
                result[raw["id"]] = raw
                resources.append({"kind": "caption_evidence", "object_id": raw["id"],
                                  "path": str(path), "sha256": sha256(path)})
    return result, resources


def _candidates(family, movers, surfaces, landmarks, facts, horizontal_axes=None):
    mids, sids, lids = [[n["id"] for n in pool] for pool in (movers, surfaces, landmarks)]
    by_id = {n["id"]: n for pool in (movers, surfaces, landmarks) for n in pool}
    pairs = []
    for offset in range(len(sids)):
        for index, m in enumerate(mids):
            s = sids[(index + offset) % len(sids)]
            if (m != s and s not in {f["support_id"] for f in facts.get(m, [])}
                    and _allowed_placement(by_id[m], by_id[s], horizontal_axes)):
                pairs.append((m, s))
    if family == "navigate":
        for identity in lids:
            yield [("visit", identity, None)], []
    elif family == "pickup_hold":
        for offset in range(len(lids)):
            for index, m in enumerate(mids):
                destination = lids[(index + offset) % len(lids)]
                if m != destination:
                    yield [("hold_at", m, destination)], []
    elif family == "single_transport":
        for m, s in pairs:
            yield [("deliver", m, s)], []
    elif family == "protected_transport":
        for m, s in pairs:
            for protected in mids:
                if protected not in (m, s):
                    yield [("deliver", m, s)], [protected]
    elif family in ("ordered_two_delivery", "ordered_three_delivery"):
        length = 2 if family == "ordered_two_delivery" else 3
        # Round-robin across first deliveries, not thousands of paraphrases of one item.
        for offset in range(1, len(pairs)):
            for start, pair in enumerate(pairs):
                group = [pairs[(start + offset * i) % len(pairs)] for i in range(length)]
                if len({x[0] for x in group}) == length and len({x[1] for x in group}) == length:
                    yield [("deliver", m, s) for m, s in group], []
    elif family == "ordered_visit":
        # No repeated waypoint loop; different order is meaningful, grouped for review.
        for offset in range(1, len(lids)):
            for start in range(len(lids)):
                group = [lids[(start + offset * i) % len(lids)] for i in range(3)]
                if len(set(group)) == 3:
                    yield [("visit", x, None) for x in group], []


def _text(ops, protect, by_id, rooms):
    def label(identity):
        tag = str(by_id[identity]["object_tag"]).replace("_", " ")
        location = " in " + rooms[identity] if identity in rooms else ""
        return "the {} (ID {}){}".format(tag, identity, location)
    clauses = []
    for mode, obj, target in ops:
        if mode == "visit":
            clauses.append("go near " + label(obj))
        elif mode == "hold_at":
            clauses.append("take " + label(obj) + " to " + label(target)
                           + " and keep holding it there; do not put it down")
        else:
            seat = "the seat of " if re.search(r"\bchair$", by_id[target]["object_tag"], re.I) else ""
            clauses.append("move " + label(obj) + " onto " + seat + label(target) + " and release it")
    sentence = "Please " + "; then ".join(clauses) + "."
    if len(ops) > 1:
        sentence += " Complete these requests in the stated order."
    if all(o[0] == "visit" for o in ops):
        sentence += " Do not pick up or move any objects."
    if protect:
        sentence += " Leave " + ", ".join(label(p) for p in protect) + " untouched throughout."
    return sentence


def _goals(ops):
    final, milestones = [], []
    for mode, obj, target in ops:
        if mode == "visit":
            milestones.append({"predicate": "visited_near", "object_id": obj})
        elif mode == "hold_at":
            final.extend([{"predicate": "held", "object_id": obj, "expected": True},
                          {"predicate": "near", "object_id": target}])
            milestones.append({"predicate": "transported_while_held", "object_id": obj, "destination_id": target})
        else:
            final.extend([{"predicate": "on", "object_id": obj, "support_id": target},
                          {"predicate": "held", "object_id": obj, "expected": False}])
            milestones.append({"predicate": "delivered_and_released", "object_id": obj, "destination_id": target})
    if ops[-1][0] == "visit":
        final.append({"predicate": "near", "object_id": ops[-1][1]})
    return {"final_predicates": final, "ordered_milestones": milestones,
            "source": "declarative_task_specification_not_execution_result"}


def _reference(sample_id, ops, protect, goal):
    steps = []
    for index, (mode, obj, target) in enumerate(ops, 1):
        steps.append({"step_id": "r{:02d}".format(index), "intent": mode,
                      "object_id": obj, "destination_id": target,
                      "depends_on": [] if index == 1 else ["r{:02d}".format(index - 1)],
                      "preserve_objects": protect})
    return {"record_id": sample_id, "sample_id": sample_id, "source": "rule_authored_version", "version": VERSION,
            "independent_of_model_output": True, "independently_human_verified": False,
            "independent_reference_available": False,
            "review_status": "human_review_pending", "granularity": "task_intent_not_atomic_robot_commands",
            "steps": steps, "dependencies": [[s["depends_on"][0], s["step_id"]] for s in steps if s["depends_on"]],
            "goal": goal, "limitations": ["Derived from the same authored task specification, not an independent oracle.",
                "No physical affordance, reachability, collision, source truth or model correctness verification."]}


def build_scene(scene_id, scene_dir, count=200, existing_instructions=()):
    """Return (candidate_records, diagnostics, separate_rule_reference_artifacts).

    If a quota cannot be met, status is ``blocked_candidate_shortage`` and
    generated_count < requested_count. Never pad with duplicated paraphrases.
    Existing instructions are compared case/spacing/punctuation-normalized.
    """
    if scene_id not in PRIORITY_MOVERS:
        raise ValueError("Unsupported scene_id: " + str(scene_id))
    if type(count) is not int or count <= 0 or count > 200:
        raise ValueError("This reviewed catalog supports an integer count between 1 and 200")
    scene_dir = Path(scene_dir).resolve()
    graph_path, edge_path = scene_dir / "scene_graph.json", scene_dir / "scene_graph_edges.json"
    nodes, edges = _read(graph_path), _read(edge_path)
    if not isinstance(nodes, list) or not isinstance(edges, list):
        raise ValueError("Both graph and edge inputs must be lists")
    if any(type(n.get("id")) is not int for n in nodes):
        raise ValueError("Graph IDs must be integers")
    by_id = {n["id"]: n for n in nodes}
    if len(by_id) != len(nodes):
        raise ValueError("Duplicate graph node IDs")
    if any(e.get("source") not in by_id or e.get("target") not in by_id for e in edges):
        raise ValueError("Edge references an absent node")
    movers = _sorted_pool(nodes, MOVABLE, PRIORITY_MOVERS[scene_id], portable=True)
    horizontal_axes, geometry_provenance = _geometry_contract(scene_dir)
    caption_evidence, caption_resources = _caption_evidence(scene_dir) if scene_id == "d3_619" else ({}, [])
    if scene_id == "d3_619":
        movers = [n for n in movers if _complete_saved_caption(caption_evidence.get(n["id"], {}))]
    surfaces = _sorted_pool(nodes, SURFACE, PRIORITY_SURFACES[scene_id])
    if scene_id == "d3_619":
        surfaces.extend(_sorted_pool(nodes, SOFT_SUPPORT))
    surfaces = [n for n in surfaces if not re.search(r"\b(sculpture|ornament|on|figurine|model)\b", n["object_tag"], re.I)
                and (horizontal_axes is None or min(n["bbox_extent"][i] for i in horizontal_axes) >= 0.2)]
    landmarks = _sorted_pool(nodes, re.compile("(?:" + LANDMARK.pattern + ")|(?:" + MOVABLE.pattern + ")", re.I),
                             PRIORITY_SURFACES[scene_id] + PRIORITY_MOVERS[scene_id])
    if scene_id == "d3_619":
        surfaces = [n for n in surfaces if _complete_saved_caption(caption_evidence.get(n["id"], {}))]
        landmarks = [n for n in landmarks if _complete_saved_caption(caption_evidence.get(n["id"], {}))]
    rooms, room_provenance = _rooms(scene_id, by_id)
    facts = _support_facts(edges)
    support_conflicts = _support_conflicts(facts)
    graph_hash, edges_hash = sha256(graph_path), sha256(edge_path)
    skill_path = ROOT / "outputs/quadruped_atomic_skill_library.v0.1.json"
    skill_source = {"path": str(skill_path), "sha256": sha256(skill_path) if skill_path.is_file() else None,
                    "physical_capabilities_verified": False, "robot_execution_authorized": False}
    base_resources = [{"kind": "scene_graph", "path": str(graph_path), "sha256": graph_hash},
                      {"kind": "scene_graph_edges", "path": str(edge_path), "sha256": edges_hash}]
    for path, kind in [(scene_dir.parent / "evidence_manifest.json", "render_evidence_index"),
                       (scene_dir / "baseline_manifest.json", "graph_provenance"),
                       (scene_dir / "compatibility_export.json", "graph_export_provenance"),
                       (scene_dir / "upstream_export/baseline_manifest.json", "graph_provenance")]:
        if path.is_file():
            base_resources.append({"kind": kind, "path": str(path), "sha256": sha256(path)})
    records, references, normalized, semantic = [], [], set(), set()
    generated_at = datetime.now(timezone.utc)
    old = {normalize_instruction(text) for text in existing_instructions}
    quotas = {family: value * count // 200 for family, value in QUOTAS.items()}
    remainder = count - sum(quotas.values())
    for family in list(QUOTAS)[:remainder]:
        quotas[family] += 1
    counts, skipped, shortages = Counter(), Counter(), {}
    for family, quota in quotas.items():
        if not quota:
            continue
        # Prefer already annotated objects for room coverage; no geometric inference.
        known = [[n for n in pool if n["id"] in rooms] for pool in (movers, surfaces, landmarks)]
        candidates = chain(_candidates(family, *known, facts, horizontal_axes),
                           _candidates(family, movers, surfaces, landmarks, facts, horizontal_axes))
        for ops, protect in candidates:
            instruction = _text(ops, protect, by_id, rooms)
            norm = normalize_instruction(instruction)
            key = digest({"scene": scene_id, "operations": ops, "protected": sorted(protect)})
            if norm in old:
                skipped["historical_instruction_duplicate"] += 1
                continue
            if norm in normalized or key in semantic:
                skipped["intra_catalog_duplicate"] += 1
                continue
            involved = sorted({x for op in ops for x in op[1:] if x is not None} | set(protect))
            task_rooms = sorted({rooms[x] for x in involved if x in rooms})
            scope = ("cross_room" if len(task_rooms) >= 2 else "single_room") if all(x in rooms for x in involved) else "unknown"
            sample_id = "{}_{:08d}".format(scene_id, len(records) + 1)
            constraints = {"preserve_objects": protect, "preserve_throughout": bool(protect),
                           "ordered_objectives": len(ops) > 1, "one_object_at_a_time": True,
                           "no_object_manipulation": all(o[0] == "visit" for o in ops),
                           "contents_unknown_do_not_infer": True}
            goal = _goals(ops)
            record = {
                "record_id": sample_id, "sample_id": sample_id, "scene_id": scene_id,
                "instruction_generated_at_utc": generated_at.isoformat(),
                "instruction_generated_at_local": generated_at.astimezone(timezone(timedelta(hours=8))).isoformat(),
                "source_scene_capture_time": None,
                "graph_version": "sha256:" + graph_hash, "graph_sha256": graph_hash, "edges_sha256": edges_hash,
                "instruction": instruction, "instruction_language": "en",
                "task_family": family, "horizon": "long" if len(ops) >= 2 else "short",
                "meaningful_objective_count": len(ops),
                "horizon_definition": "short=one task-level objective; long=two or more ordered objectives, not atomic action count",
                "room_scope": scope, "known_room_labels": task_rooms, "room_provenance": room_provenance,
                "grounded_object_ids": involved,
                "initial_state": {"robot_pose": None, "robot_pose_source": "not_observed_not_required_for_offline_task_authoring",
                    "held_object_id": None, "held_state_known": False, "held_state_source": "unknown_not_observed",
                    "support_relations": [f for identity in involved for f in facts.get(identity, [])],
                    "support_relation_status": "conflict_pending_review" if any(x in support_conflicts for x in involved) else "unverified_source_hypotheses",
                    "support_conflicts": {str(x): support_conflicts[x] for x in involved if x in support_conflicts},
                    "unknown_support_object_ids": [op[1] for op in ops if op[0] != "visit" and op[1] not in facts],
                    "source_graph_is_unchanged": True},
                "constraints": constraints, "goal": goal, "available_skills": skill_source,
                "modality_resources": base_resources + [resource for resource in caption_resources if resource["object_id"] in involved],
                "source_permissions": {"redistribution": "unknown_pending_review", "training_use": "unknown_pending_review"},
                "assumptions": ["Task authoring treats semantic labels and supplied relations as hypotheses, not ground truth.",
                    "Movable-object graspability, payload suitability and destination free space are unverified planning assumptions.",
                    "No robot pose, room connectivity, safe approach point or collision-free path is fabricated.",
                    "This is an independent fresh planning scenario; robot pose and holding state are unknown, not inferred from prior tests."],
                "candidate_geometry_filter": {"version": PAIR_RULE_VERSION, "source": "supplied_graph_AABB",
                    "coordinate_evidence": geometry_provenance,
                    "applied": horizontal_axes is not None,
                    "minimum_surface_horizontal_extent_m": 0.2 if horizontal_axes is not None else None,
                    "tested_horizontal_orientations_degrees": [0, 90] if horizontal_axes is not None else [],
                    "multiple_deliveries_have_distinct_destinations": True,
                    "seating_destination_rule": "Only textile objects may be placed on complete sofa/chair/ottoman hypotheses; no hard-object delivery to seating.",
                    "surface_occupancy_checked": False, "physical_placement_verified": False,
                    "graspability_verified": False},
                "reference": {"source": "rule_authored_version", "version": VERSION,
                    "separate_artifact": "reference_steps.json", "independent_reference_available": False,
                    "independently_human_verified": False, "review_status": "human_review_pending"},
                "quality": {"status": "candidate_pending_review", "correctness_evaluated": False,
                    "model_output_generated": False, "training_eligible": False, "physical_execution_verified": False},
                "dedup": {"exact_instruction_sha256": hashlib.sha256(instruction.encode()).hexdigest(),
                    "normalized_instruction_sha256": hashlib.sha256(norm.encode()).hexdigest(),
                    "semantic_key": key,
                    "near_duplicate_group": digest({"scene": scene_id, "operations_unordered": sorted(ops, key=str),
                                                     "protected": sorted(protect)}),
                    "method": "normalized text plus ordered object-role intent signature; order variants grouped separately",
                    "semantic_key_is_ground_truth": False},
            }
            records.append(record)
            references.append(_reference(sample_id, ops, protect, goal))
            normalized.add(norm)
            semantic.add(key)
            counts[family] += 1
            if counts[family] >= quota:
                break
        if counts[family] < quota:
            shortages[family] = {"requested": quota, "generated": counts[family]}
    diagnostics = {
        "status": "candidate_catalog_ready_not_evaluated" if not shortages else "blocked_candidate_shortage",
        "generator_version": VERSION, "scene_id": scene_id, "requested_count": count, "generated_count": len(records),
        "family_counts": dict(counts), "family_quotas": quotas, "shortages": shortages, "skipped": dict(skipped),
        "horizon_counts": dict(Counter(r["horizon"] for r in records)),
        "room_scope_counts": dict(Counter(r["room_scope"] for r in records)),
        "room_annotation_status": room_provenance, "historical_instruction_count": len(old),
        "candidate_pools": {"movers": [n["id"] for n in movers], "surfaces": [n["id"] for n in surfaces],
                            "landmarks": [n["id"] for n in landmarks]},
        "source_node_count": len(nodes), "source_edge_count": len(edges),
        "saved_caption_evidence_count": len(caption_evidence),
        "source_support_conflicts": {str(k): v for k, v in support_conflicts.items()},
        "candidate_geometry_filter_version": PAIR_RULE_VERSION,
        "candidate_geometry_coordinate_evidence": geometry_provenance,
        "no_source_graph_filtering": True, "no_api_calls": True, "robot_executed": False,
        "references_are_separate": True, "reference_human_review_pending": True,
        "near_duplicate_groups": dict(Counter(r["dedup"]["near_duplicate_group"] for r in records)),
        "valid_training_sample_count": None,
    }
    return records, diagnostics, references
