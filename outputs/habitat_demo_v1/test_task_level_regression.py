import unittest
from run_task_level_regression import check_case, distinct_starts
from unittest.mock import patch, Mock
from types import SimpleNamespace
import run_navigation_demo as navigation


class RegressionTests(unittest.TestCase):
    def test_varying_start_is_positive_case(self):
        self.assertTrue(check_case("start_201", 0, {"training_eligible": True,
            "goal_observed_in_proxy": True, "grasp_assumed": True}))

    def test_duplicate_and_invalid_starts_not_counted(self):
        rows = [{"start_position": p} for p in
                ([0, 0, 0], [0.1, 0, 0], [1, 0, 0], [float("nan"), 0, 0], None)]
        self.assertEqual(distinct_starts(rows), 2)

    def test_habitat_rng_is_seeded(self):
        pf = Mock()
        pf.get_random_navigable_point.return_value = [2, 0, 0]
        pf.find_path.return_value = True
        with patch.dict("sys.modules", {"habitat_sim": SimpleNamespace(ShortestPath=SimpleNamespace)}), \
                patch.object(navigation, "habitat_vector", side_effect=lambda p: p):
            self.assertEqual(navigation.choose_start(pf, [0, 0, 0], 203, "first"), [2, 0, 0])
        pf.seed.assert_called_once_with(203)

    def test_invalid_sampling_policy_rejected(self):
        with self.assertRaises(ValueError):
            navigation.choose_start(Mock(), [0, 0, 0], 201, "unknown")

    def test_positive(self):
        self.assertTrue(check_case("repeat_1", 0, {"training_eligible": True,
            "goal_observed_in_proxy": True, "grasp_assumed": True}))

    def test_expected_failure_is_regression_pass(self):
        self.assertTrue(check_case("navigation_blocked", 2, {
            "source_execution_status": "navigation_failed", "training_eligible": False,
            "goal_observed_in_proxy": False, "grasp_assumed": False}))

    def test_error_is_not_expected_failure(self):
        self.assertFalse(check_case("navigation_blocked", 1, {
            "source_execution_status": "navigation_failed", "training_eligible": False,
            "goal_observed_in_proxy": False, "grasp_assumed": False}))

    def test_negative_cannot_claim_grasp(self):
        self.assertFalse(check_case("proximity_rejected", 2, {
            "source_execution_status": "proxy_reach_failed", "training_eligible": False,
            "goal_observed_in_proxy": False, "grasp_assumed": True}))
