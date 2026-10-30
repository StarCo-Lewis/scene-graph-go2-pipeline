import copy
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
from run_memory_followup_navigation import choose_far_start
from run_memory_far_start_regression import check_case


class FarStartTests(unittest.TestCase):
    def test_sampling_skips_near_points_and_seeds_habitat(self):
        pf = Mock()
        pf.get_random_navigable_point.side_effect = [[0.1, 0, 0], [4, 0, 0]]
        pf.find_path.return_value = True
        with patch.dict("sys.modules", {"habitat_sim": SimpleNamespace(ShortestPath=SimpleNamespace)}), \
                patch("run_memory_followup_navigation.habitat_vector", side_effect=lambda p: p):
            point = choose_far_start(pf, [0, 0, 0], [0, 1, 0], 202, 3.0)
        self.assertEqual(point, [4, 0, 0])
        pf.seed.assert_called_once_with(202)

    def test_too_close_threshold_rejected(self):
        with self.assertRaises(ValueError):
            choose_far_start(Mock(), [0, 0, 0], [0, 0, 0], 201, 0.5)

    def test_navigation_positive_has_real_movement(self):
        result = {"status": "followup_proxy_passed", "training_eligible": True,
            "start_policy": "sampled_far", "navigation": {"actions": 20},
            "initial_distance_to_new_object_m": 4.0, "distance_to_restored_object_m": 0.5,
            "final_distance_to_old_position_m": 7.0, "restored_bbox_position_error_m": 0.0,
            "memory_modified": False}
        self.assertTrue(check_case(0, result))
        for key, value in (("distance_to_restored_object_m", 7.0),
                           ("initial_distance_to_new_object_m", 0.5),
                           ("final_distance_to_old_position_m", 0.5),
                           ("restored_bbox_position_error_m", 1.0)):
            changed = dict(result, **{key: value})
            self.assertFalse(check_case(0, changed))
        changed = copy.deepcopy(result)
        changed["navigation"]["actions"] = 0
        self.assertFalse(check_case(0, changed))
