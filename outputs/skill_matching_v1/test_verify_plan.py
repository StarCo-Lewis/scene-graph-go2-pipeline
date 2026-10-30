import importlib.util
import json
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
SPEC = importlib.util.spec_from_file_location("verify_plan", HERE / "verify_plan.py")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
LIBRARY = json.loads((HERE.parent / "quadruped_atomic_skill_library.v0.1.json").read_text(encoding="utf-8"))


def fixtures(include_place=True):
    alignment = {
        "status": "success", "context_bindings": {
            "kettle": {"element_ids": [67]}, "sideboard": {"element_ids": [113]}},
        "task_decomposition": {
            "initial_scene_constraints": [{"constraint_id": "init_01", "subject_role": "kettle",
                                           "relation": "on", "object_role": "sideboard", "required": False}],
            "goal_state": [{"constraint_id": "goal_01", "subject_role": "kettle",
                            "relation": "on", "object_role": "sideboard", "required": True}]}}
    step = {"step_id": "s1", "skill_id": "place_object_on_surface", "arguments": {
        "object": {"role_id": "kettle", "element_id": 67},
        "surface": {"role_id": "sideboard", "element_id": 113}}}
    plan = {"status": "proposal_needs_environment_validation",
            "subtask_plans": [{"steps": [step] if include_place else []}]}
    nodes = [{"id": 67, "bbox_center": [0, 0, 0]}, {"id": 113, "bbox_center": [1, 0, 0]}]
    return alignment, plan, nodes


class VerificationTests(unittest.TestCase):
    def test_unobserved_initial_claim_is_not_trusted(self):
        alignment, plan, nodes = fixtures()
        result = MODULE.review(alignment, plan, LIBRARY, nodes, [])
        check = result["observed_state"]["initial_claim_checks"][0]
        self.assertEqual(check["status"], "unknown")
        self.assertFalse(check["trusted_as_initial_fact"])

    def test_predicted_goal_is_not_executed_success(self):
        alignment, plan, nodes = fixtures()
        result = MODULE.review(alignment, plan, LIBRARY, nodes, [])
        self.assertEqual(result["goal_check"][0]["status"], "predicted_only")
        self.assertEqual(result["physical_feasibility"], "unknown")
        self.assertEqual(result["symbolic_trace"][0]["execution_status"], "not_executed")
        match = result["goal_skill_alignment"][0]
        self.assertEqual(match["status"], "covered")
        self.assertEqual(match["instances"][0]["matched_step_ids"], ["s1"])
        self.assertEqual(match["instances"][0]["library_candidates"][0]["skill_id"],
                         "place_object_on_surface")

    def test_missing_goal_effect_invalidates_plan(self):
        alignment, plan, nodes = fixtures(include_place=False)
        result = MODULE.review(alignment, plan, LIBRARY, nodes, [])
        self.assertEqual(result["status"], "invalid_symbolic_plan")
        self.assertEqual(result["goal_check"][0]["status"], "not_supported")
        self.assertEqual(result["goal_skill_alignment"][0]["instances"][0]["status"],
                         "not_selected_in_plan")

    def test_wrong_destination_does_not_match_goal(self):
        alignment, plan, nodes = fixtures()
        nodes.append({"id": 114, "bbox_center": [2, 0, 0]})
        plan["subtask_plans"][0]["steps"][0]["arguments"]["surface"]["element_id"] = 114
        result = MODULE.review(alignment, plan, LIBRARY, nodes, [])
        self.assertEqual(result["status"], "invalid_symbolic_plan")
        self.assertEqual(result["goal_skill_alignment"][0]["instances"][0]["matched_step_ids"], [])

    def test_unrepresentable_relation_is_explicit(self):
        alignment, plan, nodes = fixtures()
        alignment["task_decomposition"]["goal_state"][0]["relation"] = "under"
        result = MODULE.review(alignment, plan, LIBRARY, nodes, [])
        self.assertEqual(result["goal_skill_alignment"][0]["instances"][0]["status"],
                         "not_expressible_by_library")
        self.assertEqual(result["status"], "invalid_symbolic_plan")

    def test_empty_goals_are_not_vacuously_valid(self):
        alignment, plan, nodes = fixtures()
        alignment["task_decomposition"]["goal_state"] = []
        result = MODULE.review(alignment, plan, LIBRARY, nodes, [])
        self.assertEqual(result["status"], "invalid_symbolic_plan")

    def test_only_adjacent_identical_idempotent_steps_removed(self):
        target = {"target": {"role_id": "kettle", "element_id": 67}}
        steps = [
            {"step_id": "a", "skill_id": "navigate_to_object", "arguments": target},
            {"step_id": "b", "skill_id": "navigate_to_object", "arguments": target},
            {"step_id": "c", "skill_id": "grasp_object", "arguments": target},
            {"step_id": "d", "skill_id": "grasp_object", "arguments": target},
        ]
        kept, removed = MODULE.conservative_optimize(steps)
        self.assertEqual([s["step_id"] for s in kept], ["a", "c", "d"])
        self.assertEqual(removed[0]["step_id"], "b")

    def test_revisits_are_flagged_without_deleting_inspections(self):
        alignment, plan, nodes = fixtures()
        steps = [
            {"step_id": "n1", "skill_id": "navigate_to_object", "arguments": {
                "target": {"role_id": "kettle", "element_id": 67}}},
            {"step_id": "i1", "skill_id": "inspect_object", "arguments": {
                "target": {"role_id": "kettle", "element_id": 67}}},
            {"step_id": "n2", "skill_id": "navigate_to_object", "arguments": {
                "target": {"role_id": "sideboard", "element_id": 113}}},
            {"step_id": "n3", "skill_id": "navigate_to_object", "arguments": {
                "target": {"role_id": "kettle", "element_id": 67}}},
        ]
        audit = MODULE.review_redundancy(steps, LIBRARY)
        self.assertEqual(audit["inter_target_transitions"], 2)
        self.assertEqual(audit["navigation_revisits"][0]["step_id"], "n3")
        self.assertEqual(audit["inspection_steps_without_symbolic_consumers"], ["i1"])
        self.assertEqual(audit["automatic_non_adjacent_removals"], 0)

    def test_external_observation_can_resolve_unknown_without_claiming_execution(self):
        alignment, plan, nodes = fixtures()
        snapshot = {"schema_version": "1.0", "facts": [{
            "subject": "robot", "predicate": "stable_base", "value": True,
            "source": "robot_telemetry", "observed_at": "2026-09-19T21:00:00+08:00",
        }]}
        result = MODULE.review(alignment, plan, LIBRARY, nodes, [], snapshot)
        check = next(x for x in result["symbolic_trace"][0]["preconditions"]
                     if x["predicate"] == "stable_base")
        self.assertEqual(check["status"], "true")
        self.assertEqual(check["reason"], "robot_telemetry")

    def test_predicted_effect_cannot_enter_observation_interface(self):
        alignment, plan, nodes = fixtures()
        snapshot = {"schema_version": "1.0", "facts": [{
            "subject": 67, "predicate": "held_by", "value": "robot",
            "source": "predicted_effect", "observed_at": "2026-09-19T21:00:00+08:00",
        }]}
        with self.assertRaisesRegex(ValueError, "untrusted source"):
            MODULE.review(alignment, plan, LIBRARY, nodes, [], snapshot)

    def test_timestamped_simulator_observation_is_accepted(self):
        alignment, plan, nodes = fixtures()
        snapshot = {"schema_version": "1.0", "facts": [{
            "subject": 67, "predicate": "on", "value": 113,
            "source": "simulator_observation", "observed_at": "2026-09-21T12:00:00+08:00",
        }]}
        result = MODULE.review(alignment, plan, LIBRARY, nodes, [], snapshot)
        fact = next(x for x in result["observed_state"]["facts"]
                    if x["subject"] == "67" and x["predicate"] == "on")
        self.assertEqual(fact["value"], 113)
        self.assertEqual(fact["source"], "simulator_observation")


if __name__ == "__main__":
    unittest.main()
