#!/usr/bin/env python3
"""Export multi-action planning fixtures; optionally request real LLM skill proposals."""
import argparse
import copy
import logging
import os
from datetime import datetime
from pathlib import Path

from match_skills import (read_json, write_json, build_result, request_plan,
                          validate_proposal, PlanError, DEFAULT_MODEL, DEFAULT_URL)
from verify_plan import review
from optimize_plan import candidate_plan

HERE = Path(__file__).resolve().parent
ORDER = ["navigate_to_object", "position_arm_for_object", "grasp_object",
         "lift_held_object", "navigate_to_object", "position_arm_for_object", "place_object_on_surface"]


def fixtures():
    kettle = {"role_id": "kettle", "element_id": 67}
    sideboard = {"role_id": "sideboard", "element_id": 13}
    carry = [{"skill_id": skill, "arguments": {"target": copy.deepcopy(target)}}
             for skill, target in zip(ORDER[:3], [kettle] * 3)]
    carry += [{"skill_id": "lift_held_object", "arguments": {"object": kettle}},
              {"skill_id": "navigate_to_object", "arguments": {"target": sideboard}},
              {"skill_id": "position_arm_for_object", "arguments": {"target": sideboard}},
              {"skill_id": "place_object_on_surface", "arguments": {"object": kettle, "surface": sideboard}}]
    route = [{"skill_id": "navigate_to_object", "arguments": {"target": t}}
             for t in [kettle, sideboard, kettle]]
    for name, instruction, groups, descriptions, goal in [
        ("one_subtask_seven_actions", "Move the electric kettle to the sideboard.", [carry],
         ["Approach, grasp, lift, carry and place the kettle on the sideboard."],
         {"subject_role": "kettle", "relation": "on", "object_role": "sideboard"}),
        ("three_subtasks_seven_actions", "Move the electric kettle to the sideboard.",
         [carry[:2], carry[2:4], carry[4:]],
         ["Approach the kettle and position the arm.", "Grasp and lift the kettle.",
          "Carry the kettle to the sideboard, position the arm and place it there."],
         {"subject_role": "kettle", "relation": "on", "object_role": "sideboard"}),
        ("necessary_revisit_A_B_A", "Go to the electric kettle, then to the sideboard, then return to the electric kettle. Complete every visit in order.",
         [route], ["Visit kettle, visit sideboard, return to kettle, in that exact order; final robot position is near kettle."],
         {"subject_role": "robot", "relation": "near", "object_role": "kettle"}),
    ]:
        subtasks, plans = [], []
        for i, (steps, description) in enumerate(zip(groups, descriptions), 1):
            sid = f"st_{i:02d}"
            subtasks.append({"subtask_id": sid, "description": description,
                             "dependencies": [f"st_{i-1:02d}"] if i > 1 else []})
            plans.append({"subtask_id": sid, "intent": description, "steps": copy.deepcopy(steps)})
        alignment = {"status": "success", "context_bindings": {
            "kettle": {"element_ids": [67]}, "sideboard": {"element_ids": [13]}},
            "task_decomposition": {"instruction": instruction, "subtasks": subtasks,
                "initial_scene_constraints": [],
                "goal_state": [dict(goal, constraint_id="goal_01", required=True)]}}
        yield name, alignment, {"schema_version": "1.0", "subtask_plans": plans}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live-skill-llm", action="store_true",
                        help="Chargeable skill-matching API calls; decomposition/bindings remain explicit fixtures")
    parser.add_argument("--output-dir", type=Path,
                        default=HERE / "multi_action_regression" / datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    args = parser.parse_args()
    if args.live_skill_llm and not os.environ.get("DASHSCOPE_API_KEY"):
        parser.error("DASHSCOPE_API_KEY required for --live-skill-llm")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    library = read_json(HERE.parent / "quadruped_atomic_skill_library.v0.1.json")
    nodes = [{"id": 67, "bbox_center": [0., 0., 1.]}, {"id": 13, "bbox_center": [2., 0., 1.]}]
    summary = {"status": "running", "cases": [], "live_skill_llm": args.live_skill_llm,
               "scope": "Synthetic decomposition, bindings and coordinates; skill matcher and symbolic checks only",
               "simulation_executed": False, "training_eligible": False}
    for name, alignment, expected in fixtures():
        print(f"START {name}", flush=True)
        case_dir = args.output_dir / name
        write_json(case_dir / "object_alignment_fixture.json", alignment)
        write_json(case_dir / "expected_proposal.json", expected)
        scoped = copy.deepcopy(library)
        scoped["skills"] = [s for s in library["skills"] if s["skill_id"] in ORDER]
        if name == "necessary_revisit_A_B_A":
            scoped["skills"] = [s for s in scoped["skills"] if s["skill_id"] == "navigate_to_object"]
            # Expose the current single-navigation executor limit, not an invented executor.
            supported_order = ["navigate_to_object"]
        else:
            supported_order = ORDER
        scoped["execution_contract"] = {"adapter_mode": "navigation" if len(supported_order) == 1 else "kettle_to_sideboard",
            "supported_skill_ids": sorted(set(supported_order)), "supported_ordered_sequence": supported_order,
            "planning_context": "This is a planning-only fixture. Preserve all requested actions even when the executor cannot run the resulting sequence."}
        write_json(case_dir / "skill_library.json", scoped)
        result_row = {"case": name, "passed": False}
        try:
            proposal = expected
            if args.live_skill_llm:
                error, rejected = None, None
                for attempt in range(1, 4):
                    proposal = request_plan(alignment, scoped, DEFAULT_MODEL, DEFAULT_URL, "DASHSCOPE_API_KEY",
                                            validation_error=error, rejected_proposal=rejected)
                    write_json(case_dir / f"llm_skill_proposal_attempt_{attempt}.json", proposal)
                    try:
                        validate_proposal(proposal, alignment, scoped)
                        break
                    except PlanError as exc:
                        error, rejected = str(exc), proposal
                        if attempt == 3:
                            raise
            result = build_result(alignment, scoped, proposal)
            write_json(case_dir / "skill_plan.json", result)
            audit = review(alignment, result, scoped, nodes, [])
            write_json(case_dir / "symbolic_review.json", audit)
            expected_steps = [s for p in expected["subtask_plans"] for s in p["steps"]]
            actual_steps = [{"skill_id": s["skill_id"], "arguments": s["arguments"]}
                            for p in result["subtask_plans"] for s in p["steps"]]
            # Optional physical numbers are not part of this intent-order assertion.
            stripped = [{"skill_id": s["skill_id"], "arguments": {k: v for k, v in s["arguments"].items() if isinstance(v, dict)}}
                        for s in actual_steps]
            checks = {"all_requested_actions_preserved": stripped == expected_steps,
                      "symbolic_goal_covered": audit["status"] == "conditionally_valid_symbolic_plan",
                      "no_execution_claims": all(s["execution_status"] == "not_executed" for p in result["subtask_plans"] for s in p["steps"])}
            if name == "necessary_revisit_A_B_A":
                first_two = [s["step_id"] for s in result["subtask_plans"][0]["steps"][:2]]
                try:
                    candidate_plan(result, {"ordered_step_ids": first_two})
                    checks["necessary_return_cannot_be_deleted"] = False
                except ValueError:
                    checks["necessary_return_cannot_be_deleted"] = len(actual_steps) == 3
                checks["unsupported_executor_shape_reported"] = result["adapter_support"]["status"] == "unsupported_shape"
            else:
                checks["supported_executor_shape"] = result["adapter_support"]["status"] == "supported_shape"
            result_row.update(passed=all(checks.values()), checks=checks, action_count=result["action_count"],
                              subtask_action_mapping=result["subtask_action_mapping"], adapter_support=result["adapter_support"])
        except (ValueError, KeyError, OSError, TypeError) as exc:
            result_row["error"] = str(exc)
        summary["cases"].append(result_row)
        write_json(args.output_dir / "regression_summary.json", summary)
        print(f"END {name} passed={result_row['passed']}", flush=True)
    summary["status"] = "passed" if all(c["passed"] for c in summary["cases"]) else "failed"
    write_json(args.output_dir / "regression_summary.json", summary)
    print(f"status={summary['status']} summary={args.output_dir / 'regression_summary.json'}", flush=True)
    return 0 if summary["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
