import copy
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from sequential_proxy_core import SerialExecutor, ExecutionRejected, validate_step, placement_checks
from run_continuous_sequence import (build_fixture_episode, validate_episode, contract,
                                    inject_fault, PARAMETERS, OUT, finalize_execution_evidence,
                                    rigid_object_diagnostics)
from match_skills import read_json


class FakeBackend:
    """No Habitat/physics: validates dispatch and persistent runtime bookkeeping only."""
    active_ids, mapped_ids = {67, 13}, {67, 13}

    def __init__(self):
        self.near = None
        self.positions = {67: "original", 13: "board"}
        self.support = {}
        self.nav_targets = []
        self.placement_passed = True

    def snapshot(self):
        return {"near": self.near, "positions": dict(self.positions), "support": dict(self.support)}

    def navigate(self, target, held):
        self.nav_targets.append(target)
        self.near = target
        if held is not None:
            self.positions[held] = self.positions[target]
        return {"status": "arrived"}

    def require_near(self, target):
        if self.near != target:
            raise ExecutionRejected("PROXIMITY_REJECTED")

    def require_graspable_instance(self, target):
        if target != 67:
            raise ExecutionRejected("INSTANCE_NOT_SUPPORTED_FOR_PROXY_GRASP")

    def attach(self, target):
        self.support.pop(target, None)
        self.positions[target] = "held"

    def require_placeable_pair(self, target, surface):
        if (target, surface) != (67, 13):
            raise ExecutionRejected("UNSUPPORTED_OBJECT_SURFACE_PAIR")

    def lift(self, target):
        pass

    def place(self, target, surface):
        if self.placement_passed:
            self.positions[target] = self.positions[surface]
            self.support[target] = surface
        return {"status": "proxy_instance_support_passed" if self.placement_passed else "failed",
                "validation": {"failed_checks": [] if self.placement_passed else ["tail_motion_m"]}}


def fixture():
    library = read_json(OUT / "quadruped_atomic_skill_library.v0.1.json")
    library["skills"] = [s for s in library["skills"] if s["skill_id"] in PARAMETERS]
    library["execution_contract"] = contract()
    return library, build_fixture_episode(library)


class ContinuousSequenceTests(unittest.TestCase):
    def test_five_rounds_do_not_require_exact_seven_step_macro(self):
        library, episode = fixture()
        validate_episode(episode, library, [{"id": 67}, {"id": 13}])
        backend, count, boundaries = FakeBackend(), 0, []
        executor = SerialExecutor(backend)
        for row in episode["rounds"]:
            for task in row["plan"]["subtask_plans"]:
                for action in task["steps"]:
                    # Position-arm after navigation to moved kettle identifies its instance.
                    executor.execute(action, row["alignment"])
                    count += 1
            self.assertEqual(executor.held_object_id, row["expected_end"]["held_object_id"])
            boundaries.append(executor.snapshot())
        self.assertEqual(count, 16)
        self.assertEqual(backend.nav_targets[:3], [67, 13, 67])
        self.assertEqual(boundaries[1]["held_object_id"], 67)
        self.assertEqual(boundaries[2]["backend"]["support"], {67: 13})
        self.assertEqual(boundaries[3]["held_object_id"], 67)
        self.assertNotIn(67, boundaries[3]["backend"]["support"])
        self.assertEqual(boundaries[4]["backend"]["support"], {67: 13})

    def test_place_before_grasp_is_rejected_without_mutation(self):
        _, episode = fixture()
        row = inject_fault(episode, "place_without_grasp")["rounds"][0]
        executor = SerialExecutor(FakeBackend())
        original = executor.snapshot()
        with self.assertRaisesRegex(ExecutionRejected, "LIFTED_HELD_OBJECT_REQUIRED"):
            executor.execute(row["plan"]["subtask_plans"][0]["steps"][0], row["alignment"])
        self.assertEqual(executor.snapshot(), original)

    def test_duplicate_grasp_halts_while_preserving_current_held_object(self):
        _, episode = fixture()
        backend = FakeBackend()
        backend.near = 67
        executor = SerialExecutor(backend)
        row = inject_fault(episode, "duplicate_grasp")["rounds"][1]
        steps = row["plan"]["subtask_plans"][0]["steps"]
        for action in steps[:-1]:
            executor.execute(action, row["alignment"])
        with self.assertRaisesRegex(ExecutionRejected, "GRIPPER_NOT_EMPTY"):
            executor.execute(steps[-1], row["alignment"])
        self.assertEqual(executor.held_object_id, 67)

    def test_unknown_and_missing_objects_rejected_before_simulation(self):
        library, episode = fixture()
        for fault in ("missing_object", "unsupported_skill"):
            with self.subTest(fault=fault), self.assertRaises(ValueError):
                validate_episode(inject_fault(episode, fault), library, [{"id": 67}, {"id": 13}])

    def test_unknown_instance_cannot_be_grasped(self):
        _, episode = fixture()
        step = copy.deepcopy(episode["rounds"][0]["plan"]["subtask_plans"][0]["steps"][0])
        step["arguments"]["target"]["element_id"] = 5
        bindings = {"kettle": {"element_ids": [5]}}
        with self.assertRaisesRegex(ExecutionRejected, "NO object"):
            validate_step(step, bindings, {67, 13}, {67, 13})
        with self.assertRaisesRegex(ExecutionRejected, "NO_SIMULATOR_INSTANCE"):
            validate_step(step, bindings, {67, 13, 5}, {67, 13})

    def test_unsupported_numeric_parameter_is_not_silently_ignored(self):
        _, episode = fixture()
        row = episode["rounds"][0]
        action = copy.deepcopy(row["plan"]["subtask_plans"][0]["steps"][0])
        action["arguments"]["standoff_m"] = .2
        with self.assertRaisesRegex(ExecutionRejected, "SKILL_PARAMETER"):
            SerialExecutor(FakeBackend()).execute(action, row["alignment"])

    def test_failed_support_does_not_leave_fictitious_held_state(self):
        _, episode = fixture()
        backend = FakeBackend()
        backend.near = 13
        backend.placement_passed = False
        executor = SerialExecutor(backend)
        executor.held_object_id, executor.lifted, executor.prepared_target_id = 67, True, 13
        row = episode["rounds"][2]
        with self.assertRaisesRegex(ExecutionRejected, "SUPPORT_NOT_VERIFIED") as raised:
            executor.execute(row["plan"]["subtask_plans"][0]["steps"][-1], row["alignment"])
        self.assertEqual(raised.exception.details["validation"]["failed_checks"], ["tail_motion_m"])
        self.assertIsNone(executor.held_object_id)
        self.assertFalse(executor.lifted)

    def test_placement_gates_expose_failed_metric_without_relaxation(self):
        valid = placement_checks([0., 1.083, 0.], [0., 0., 0.], 1., .01, 30)
        self.assertTrue(valid["passed"])
        for final, motion, contacts, failed in [
                ([.36, 1.083, 0.], .01, 30, "horizontal_error_m"),
                ([0., 1.30, 0.], .01, 30, "com_height_error_m"),
                ([0., 1.083, 0.], .021, 30, "tail_motion_m"),
                ([0., 1.083, 0.], .01, 19, "direct_support_contact_frames")]:
            verdict = placement_checks(final, [0., 0., 0.], 1., motion, contacts)
            self.assertEqual(verdict["failed_checks"], [failed])
            self.assertFalse(verdict["passed"])
        self.assertFalse(placement_checks([0., 1.083, 0.], [0., 0., 0.], 1., float("nan"), 30)["passed"])

    def test_failed_round_still_reports_observed_continuity(self):
        _, episode = fixture()
        rounds = [{"round": row["round"], "begin_state": {"held_object_id": 67 if row["round"] in (3, 5) else None},
                   "continuity_with_previous_round": True,
                   "actions": [{"status": "completed", "after": {}} for _ in task["steps"]]}
                  for row in episode["rounds"] for task in row["plan"]["subtask_plans"]]
        rounds[-1]["actions"][-1]["status"] = "failed"
        rounds[-1]["end_state"] = {"held_object_id": None}
        summary = {"status": "failed", "rounds": rounds, "simulator_creation_count": 1}
        finalize_execution_evidence(summary)
        self.assertEqual(summary["status"], "failed")
        self.assertTrue(summary["uninterrupted_simulator_execution"])
        self.assertEqual(summary["attempted_action_count"], 16)
        self.assertEqual(summary["executed_action_count"], 15)
        self.assertEqual(summary["cross_round_held_object_continuations"], [3, 5])
        self.assertIsNone(summary["final_state"]["held_object_id"])
        rounds[-1]["continuity_with_previous_round"] = False
        finalize_execution_evidence(summary)
        self.assertFalse(summary["uninterrupted_simulator_execution"])

    def test_rigid_diagnostics_handle_magnum_shape_and_unavailable_properties(self):
        from types import SimpleNamespace
        obj = SimpleNamespace(rotation=SimpleNamespace(scalar=1., vector=[0., 0., 0.]),
                              motion_type="DYNAMIC", linear_velocity=[0., .1, 0.], angular_velocity=[0., 0., 0.])
        report = rigid_object_diagnostics(obj)
        self.assertEqual(report["rotation_wxyz"], [1., 0., 0., 0.])
        self.assertEqual(report["linear_velocity"], [0., .1, 0.])
        del obj.angular_velocity
        report = rigid_object_diagnostics(obj)
        self.assertIsNone(report["angular_velocity"])
        self.assertIn("angular_velocity", report["unavailable_fields"])

    def test_exception_retains_independent_copy_of_backend_diagnostics(self):
        result = {"validation": {"failed_checks": ["tail_motion_m"]}}
        error = ExecutionRejected("PLACEMENT_SUPPORT_NOT_VERIFIED", result)
        result["validation"]["failed_checks"].clear()
        self.assertEqual(error.details["validation"]["failed_checks"], ["tail_motion_m"])

    def test_carry_navigation_cannot_silently_substitute_for_lift(self):
        _, episode = fixture()
        executor = SerialExecutor(FakeBackend())
        executor.held_object_id, executor.lifted = 67, False
        row = episode["rounds"][2]
        with self.assertRaisesRegex(ExecutionRejected, "MUST_BE_LIFTED"):
            executor.execute(row["plan"]["subtask_plans"][0]["steps"][0], row["alignment"])

    def test_runtime_goal_evaluator_does_not_claim_unknown_relations(self):
        from run_continuous_sequence import evaluate_goals
        _, episode = fixture()
        row = copy.deepcopy(episode["rounds"][2])
        row["alignment"]["task_decomposition"]["goal_state"][0]["relation"] = "near"
        backend = FakeBackend()
        self.assertEqual(evaluate_goals(row["alignment"], backend)[0]["status"], "not_satisfied")

    def test_duplicate_step_ids_in_legacy_input_rejected_preload(self):
        library, episode = fixture()
        plan = episode["rounds"][0]["plan"]
        plan.pop("action_sequence")
        plan["subtask_plans"][0]["steps"][1]["step_id"] = plan["subtask_plans"][0]["steps"][0]["step_id"]
        with self.assertRaisesRegex(ValueError, "step IDs"):
            validate_episode(episode, library, [{"id": 67}, {"id": 13}])

    def test_negative_rejection_is_distinguished_from_case_failure(self):
        from run_continuous_regression import evaluate_case
        fixture_result = {"status": "failed", "error": "ungrounded object in navigate_to_object.target",
                          "training_eligible": False, "simulator_creation_count": 0}
        self.assertTrue(all(evaluate_case(fixture_result, 2, "missing_object").values()))
        self.assertFalse(all(evaluate_case(fixture_result, 2, "none").values()))


if __name__ == "__main__":
    unittest.main()
