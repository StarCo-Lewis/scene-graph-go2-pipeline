#!/usr/bin/env python3
"""Convert a Habitat proxy episode into conservative environment memory."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=True, indent=2) + "\n", encoding="ascii")


def role_bindings(alignment: dict[str, Any]) -> dict[str, int]:
    output: dict[str, int] = {}
    for role, binding in alignment.get("context_bindings", {}).items():
        ids = binding.get("element_ids", []) if isinstance(binding, dict) else []
        if len(ids) == 1 and type(ids[0]) is int:
            output[role] = ids[0]
    return output


def build(proxy: dict[str, Any], alignment: dict[str, Any], plan: dict[str, Any],
          nodes: list[dict[str, Any]], observed_at: str) -> tuple[dict[str, Any], dict[str, Any]]:
    bindings = role_bindings(alignment)
    kettle_id = bindings.get("electric_kettle")
    board_id = bindings.get("sideboard")
    known_ids = {int(node["id"]) for node in nodes}
    if kettle_id not in known_ids or board_id not in known_ids:
        raise ValueError("alignment bindings are not present in the supplied VLM scene graph")

    facts: list[dict[str, Any]] = []
    nav_steps = [step for step in proxy.get("steps", [])
                 if step.get("skill_id") == "navigate_to_object"]
    for node_id, step in zip((kettle_id, board_id), nav_steps):
        if step.get("status") == "arrived":
            facts.append({"subject": node_id, "predicate": "navigable", "value": True,
                          "source": "navigation_map", "observed_at": observed_at})

    placement = next((step for step in reversed(proxy.get("steps", []))
                      if step.get("skill_id") == "place_object_on_surface"), None)
    support_observed = bool(
        placement
        and placement.get("status") == "proxy_instance_support_passed"
        and placement.get("stable_last_second") is True
        and int(placement.get("direct_sideboard_contact_frames_last_second", 0)) >= 20
    )
    if support_observed:
        facts.extend([
            {"subject": kettle_id, "predicate": "on", "value": board_id,
             "source": "simulator_observation", "observed_at": observed_at},
            {"subject": board_id, "predicate": "support_surface", "value": True,
             "source": "simulator_observation", "observed_at": observed_at},
        ])

    kettle_instance = next((step for step in proxy.get("steps", [])
                            if step.get("skill_id") == "instantiate_object"), {})
    board_instance = next((step for step in proxy.get("steps", [])
                           if step.get("skill_id") == "instantiate_sideboard"), {})
    memory = {
        "schema_version": "1.0",
        "status": "recorded",
        "memory_type": "post_episode_environment_observation",
        "observed_at": observed_at,
        "episode_status": proxy.get("status", "unknown"),
        "scope": "Habitat rigid-object proxy episode; not physical robot-arm execution",
        "task": {
            "instruction": plan.get("instruction"),
            "subtask_count": len(plan.get("subtask_plans", [])),
        },
        "entity_bindings": [
            {"role_id": "electric_kettle", "scene_graph_id": kettle_id,
             "habitat_object_id": kettle_instance.get("habitat_object_id")},
            {"role_id": "sideboard", "scene_graph_id": board_id,
             "habitat_object_id": board_instance.get("habitat_object_id")},
        ],
        "observed_facts": facts,
        "post_execution_goal_status": (
            "observed_in_habitat_proxy" if support_observed else "not_observed"
        ),
        "placement_evidence": placement,
        "ashita_integration": {
            "spatial_update": "partially_available_from_grounded_node_positions_and_simulator_state",
            "environment_aware_task_grounding": "available_for_the_two_verified_entities",
            "hierarchy_refinement": "not_available_without_unassigned_visual_primitives_and_word_generation",
            "hierarchical_information_bottleneck": "not_run_requires_primitive_layer_and_multilevel_conditional_distributions",
        },
        "limitations": [
            "Grasp and lift used scripted attachment, so graspable and held_by are not sensor observations.",
            "The proxy agent is not the articulated quadruped-arm robot.",
            "The two-node VLM graph is not a complete primitive-level scene representation.",
        ],
    }
    snapshot = {"schema_version": "1.0", "snapshot_type": "post_execution",
                "episode_status": proxy.get("status", "unknown"), "facts": facts}
    return memory, snapshot


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proxy-json", type=Path, required=True)
    parser.add_argument("--alignment-json", type=Path, required=True)
    parser.add_argument("--skill-plan-json", type=Path, required=True)
    parser.add_argument("--scene-graph-json", type=Path, required=True)
    parser.add_argument("--memory-output", type=Path, required=True)
    parser.add_argument("--observation-output", type=Path, required=True)
    parser.add_argument("--observed-at")
    args = parser.parse_args()
    observed_at = args.observed_at or datetime.now(timezone.utc).isoformat()
    memory, snapshot = build(read_json(args.proxy_json), read_json(args.alignment_json),
                             read_json(args.skill_plan_json), read_json(args.scene_graph_json),
                             observed_at)
    write_json(args.memory_output, memory)
    write_json(args.observation_output, snapshot)
    print(json.dumps({"status": "environment_memory_written",
                      "fact_count": len(snapshot["facts"]),
                      "memory": str(args.memory_output),
                      "observation_snapshot": str(args.observation_output)},
                     ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
