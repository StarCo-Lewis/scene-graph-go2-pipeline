"""Label one continuous proxy run without claiming physical grasp success."""
from run_manipulation_proxy import EXPECTED_SKILLS


def build_episode(proxy, plan):
    policy = proxy.get("execution_policy", {})
    if policy.get("grasp_rule") != "assume_grasp_after_validated_proximity":
        raise ValueError("Explicit assumed-grasp execution policy required")
    trace = [s for s in proxy.get("steps", []) if s.get("skill_id") in EXPECTED_SKILLS]
    grasp_index = next((i for i, s in enumerate(trace)
                        if s.get("skill_id") == "grasp_object"), None)
    grasp_assumed = False
    if grasp_index is not None:
        grasp = trace[grasp_index]
        nav = trace[0] if trace else {}
        reach = grasp.get("reach_m")
        # Only explicit execution evidence permits an assumed-grasp label.
        grasp_assumed = (
            nav.get("skill_id") == "navigate_to_object"
            and nav.get("status") == "arrived"
            and grasp.get("status") == "proxy_attached"
            and grasp.get("evidence_type") == "assumed_after_navigation_and_proximity_check"
            and isinstance(reach, (int, float))
            and 0 <= reach <= policy["proximity_threshold_m"] + 0.0005
        )
    placement = trace[-1] if trace else {}
    goal = (
        proxy.get("status") == "proxy_instance_support_passed"
        and proxy.get("test_fault", "none") == "none"
        and [s.get("skill_id") for s in trace] == EXPECTED_SKILLS
        and grasp_assumed
        and all(s.get("status") == "arrived" for s in trace
                if s.get("skill_id") == "navigate_to_object")
        and placement.get("status") == "proxy_instance_support_passed"
        and placement.get("stable_last_second") is True
        and placement.get("direct_sideboard_contact_frames_last_second", 0) >= 20
    )
    return {
        "schema_version": "1.1",
        "record_type": "task_level_proxy_training_episode",
        "status": "goal_observed_in_proxy_with_assumed_grasp" if goal else "task_failed",
        "instruction": plan.get("instruction"),
        "object_bindings": proxy.get("scene_graph_ids", {}),
        "input": proxy.get("input", {}),
        "start_sampling": proxy.get("start_sampling", {}),
        "policy": policy,
        "combined_from_separate_simulation_runs": False,
        "continuous_robot_execution_verified": False,
        "grasp_assumed": grasp_assumed,
        "goal_observed_in_proxy": goal,
        "training_eligible": goal,
        "source_execution_status": proxy.get("status"),
        "test_fault": proxy.get("test_fault", "none"),
        "steps": trace,
        "resource_summary": proxy.get("resource_summary", {}),
        "limitations": [
            "Approach is proxy navigation and horizontal proximity, not collision-free arm IK.",
            "Grasp and carrying are scripted, not physical gripper execution.",
            "This episode does not update the input scene graph.",
        ],
    }
