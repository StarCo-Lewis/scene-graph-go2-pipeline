#!/usr/bin/env python3
"""LLM route critique with deterministic acceptance and original-plan fallback."""
from __future__ import annotations

import argparse
import copy
import json
import os
import urllib.error
import urllib.request
from pathlib import Path

from verify_plan import read_json, review, write_json, flatten_steps, review_redundancy
from action_sequence import refresh_sequence, validate_sequence, removable_navigation_ids, adapter_shape_check
from match_skills import request_json_with_retry
from response_audit import ResponseAudit, validation as record_validation


def request_order(plan: dict, audit: dict, model: str, base_url: str, key_env: str,
                  timeout=180, attempts=1, diagnostics_dir=None) -> dict:
    key = os.environ.get(key_env, "").strip()
    if not key:
        raise ValueError(f"{key_env} is not configured")
    steps = flatten_steps(plan)
    payload = {
        "model": model, "temperature": 0, "response_format": {"type": "json_object"}, "enable_thinking": False,
        "messages": [
            {"role": "system", "content": (
                "You review a robot skill proposal for route redundancy. Return JSON only: "
                "{ordered_step_ids:[...],reason:'...'}. Use only existing step IDs once each. "
                "Keep all non-navigation steps. You may delete navigate_to_object steps only. "
                "You may reorder steps within a subtask but must preserve subtask order. "
                "Keep inspections before the manipulation they safeguard. Never invent an object, "
                "skill, affordance, or execution result. Prefer the original if unsure. "
                "If a sequence_policy is supplied, preserve relative order of every retained action; "
                "only removable_navigation_step_ids may be deleted. Necessary revisits are not redundancy."
            )},
            {"role": "user", "content": json.dumps({
                "steps": [{"step_id": s["step_id"], "skill_id": s["skill_id"],
                           "arguments": s["arguments"]} for s in steps],
                "subtask_order": [p["subtask_id"] for p in plan["subtask_plans"]],
                "sequence_policy": plan.get("sequence_policy"),
                "removable_navigation_step_ids": sorted(removable_navigation_ids(plan)) if plan.get("sequence_policy") else None,
                "redundancy_audit": audit,
                **({'conversation_context': plan['conversation_context'],
                    'original_instruction': plan['instruction']} if 'conversation_context' in plan else {}),
                **({'inference_policy': plan['inference_policy']} if 'inference_policy' in plan else {}),
                **({'factual_boundary': plan['factual_boundary']} if 'factual_boundary' in plan else {}),
            }, ensure_ascii=True)},
        ],
    }
    req = urllib.request.Request(base_url.rstrip("/") + "/chat/completions",
                                 data=json.dumps(payload).encode("utf-8"),
                                 headers={"Authorization": f"Bearer {key}",
                                          "Content-Type": "application/json"}, method="POST")
    response_audit = ResponseAudit(diagnostics_dir, 'redundancy_review', request=req)
    content = None
    try:
        result = (request_json_with_retry(req, timeout=timeout, attempts=attempts, audit=response_audit)
                  if diagnostics_dir is not None else request_json_with_retry(req, timeout=timeout, attempts=attempts))
        content = result["choices"][0]["message"]["content"]
        raw_content = content
        content = content.strip()
        if content.startswith("```json"):
            content = content[7:].strip().removesuffix("```").strip()
        proposal = json.loads(content)
        response_audit.parsed(raw_content)
        return proposal
    except urllib.error.HTTPError as exc:
        raise ValueError(f"LLM HTTP {exc.code}: {exc.read().decode('utf-8', 'replace')[:300]}") from exc
    except (urllib.error.URLError, KeyError, IndexError, TypeError, AttributeError, json.JSONDecodeError) as exc:
        response_audit.parsed(content, 'rejected', exc)
        raise ValueError(f"LLM review failed: {exc}") from exc


def candidate_plan(original: dict, proposal: dict) -> dict:
    validate_sequence(original)
    if not isinstance(proposal, dict):
        raise ValueError("review proposal must be a JSON object")
    order = proposal.get("ordered_step_ids")
    if not isinstance(order, list) or not order or any(not isinstance(x, str) for x in order):
        raise ValueError("ordered_step_ids must be a nonempty string list")
    if len(order) != len(set(order)):
        raise ValueError("duplicate step IDs")
    original_steps = flatten_steps(original)
    indexed = {step["step_id"]: step for step in original_steps}
    if len(indexed) != len(original_steps) or set(order) - set(indexed):
        raise ValueError("unknown or duplicate original step ID")
    removed = set(indexed) - set(order)
    if any(indexed[step_id]["skill_id"] != "navigate_to_object" for step_id in removed):
        raise ValueError("only navigation steps may be removed")
    if original.get("sequence_policy"):
        if removed - removable_navigation_ids(original):
            raise ValueError("ordered action intent protected: only adjacent identical navigation within a subtask may be removed")
        if order != [s["step_id"] for s in original_steps if s["step_id"] not in removed]:
            raise ValueError("ordered action intent protected: action order may not change")
    groups = {step["step_id"]: i for i, plan in enumerate(original["subtask_plans"])
              for step in plan["steps"]}
    if [groups[x] for x in order] != sorted(groups[x] for x in order):
        raise ValueError("subtask order changed")
    candidate = copy.deepcopy(original)
    for i, plan in enumerate(candidate["subtask_plans"]):
        plan["steps"] = [copy.deepcopy(indexed[x]) for x in order if groups[x] == i]
    return refresh_sequence(candidate) if original.get("sequence_policy") else candidate


def unknown_count(result: dict) -> int:
    return sum(c["status"] == "unknown" for step in result["symbolic_trace"]
               for c in step["preconditions"])


def decide(original: dict, candidate: dict, baseline: dict, revised: dict, library: dict) -> dict:
    before = review_redundancy(flatten_steps(original), library)
    after = review_redundancy(flatten_steps(candidate), library)
    reasons = []
    adapter_check = adapter_shape_check(candidate, library.get("execution_contract"))
    if adapter_check["status"] == "unsupported_shape":
        reasons.append("candidate_not_supported_by_executor_contract")
    if revised["status"] != baseline["status"] or revised["status"] != "conditionally_valid_symbolic_plan":
        reasons.append("symbolic_validation_not_preserved")
    if unknown_count(revised) > unknown_count(baseline):
        reasons.append("more_unknown_preconditions")
    if any(g["status"] != "covered" for g in revised["goal_skill_alignment"] if g["required"]):
        reasons.append("required_goal_not_covered")
    if len(flatten_steps(candidate)) >= len(flatten_steps(original)) and (
        after["inter_target_transitions"] >= before["inter_target_transitions"]
    ):
        reasons.append("no_measured_route_or_step_improvement")
    return {"schema_version": "1.0", "status": "accepted" if not reasons else "original_retained",
            "reasons": reasons, "original_step_count": len(flatten_steps(original)),
            "candidate_step_count": len(flatten_steps(candidate)),
            "original_transitions": before["inter_target_transitions"],
            "candidate_transitions": after["inter_target_transitions"],
            "original_unknown_preconditions": unknown_count(baseline),
            "candidate_unknown_preconditions": unknown_count(revised),
            "physical_feasibility": "unknown", "execution_status": "not_executed"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alignment-json", type=Path, required=True)
    parser.add_argument("--skill-plan-json", type=Path, required=True)
    parser.add_argument("--skill-library", type=Path, required=True)
    parser.add_argument("--scene-dir", type=Path, required=True)
    parser.add_argument("--candidate-json", type=Path, help="Offline LLM response for testing")
    parser.add_argument("--model", default="qwen3.8-flash")
    parser.add_argument("--base-url", default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    parser.add_argument("--api-key-env", default="DASHSCOPE_API_KEY")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    alignment, original, library = (read_json(p) for p in
                                    (args.alignment_json, args.skill_plan_json, args.skill_library))
    nodes = read_json(args.scene_dir / "scene_graph.json")
    edges = read_json(args.scene_dir / "scene_graph_edges.json")
    baseline = review(alignment, original, library, nodes, edges)
    try:
        audit = review_redundancy(flatten_steps(original), library)
        proposal = read_json(args.candidate_json) if args.candidate_json else request_order(
            original, audit, args.model, args.base_url, args.api_key_env, diagnostics_dir=args.output_dir)
        write_json(args.output_dir / 'llm_redundancy_proposal.json', proposal)
        candidate = candidate_plan(original, proposal)
        if "adapter_support" in candidate:
            candidate["adapter_support"] = adapter_shape_check(candidate, library.get("execution_contract"))
        revised = review(alignment, candidate, library, nodes, edges)
        decision = decide(original, candidate, baseline, revised, library)
        record_validation(args.output_dir, 'redundancy_review', 1,
                          'accepted' if decision['status']=='accepted' else 'not_selected',
                          evidence='llm_redundancy_proposal.json',
                          context={'proposal_origin':'offline_fixture' if args.candidate_json else 'llm'})
        decision["llm_reason"] = str(proposal.get("reason", ""))[:1000]
        write_json(args.output_dir / "skill_plan_candidate.json", candidate)
        write_json(args.output_dir / "symbolic_review_candidate.json", revised)
    except (ValueError, KeyError, TypeError, OSError) as exc:
        record_validation(args.output_dir, 'redundancy_review', 1, 'rejected', exc,
                          evidence='llm_redundancy_proposal.json')
        decision = {"schema_version": "1.0", "status": "original_retained",
                    "reasons": [str(exc)], "physical_feasibility": "unknown",
                    "execution_status": "not_executed"}
        # A rejected critique is still a completed review, not an API failure.
        if isinstance(locals().get("proposal"), dict):
            decision["llm_reason"] = str(proposal.get("reason", ""))[:1000]
    if decision["status"] == "accepted":
        write_json(args.output_dir / "skill_plan_final.json", candidate)
        write_json(args.output_dir / "symbolic_review_final.json", revised)
    else:
        write_json(args.output_dir / "skill_plan_final.json", original)
        write_json(args.output_dir / "symbolic_review_final.json", baseline)
    write_json(args.output_dir / "optimization_decision.json", decision)
    print(json.dumps(decision, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
