import unittest

from run_spot_grasp_target_search import (angle_degrees,
                                          classify_contact_candidate,
                                          grasp_candidate_rank,
                                          inward_offsets_for_standoff,
                                          local_grasp_targets,
                                          pregrasp_joint_waypoints)


class SpotGraspTargetHelpersTest(unittest.TestCase):
    def test_single_sided_or_deep_contact_is_not_a_grasp_candidate(self):
        self.assertEqual(classify_contact_candidate(
            [7], {6, 7}, 0.0, 0.005, False), "single_sided_contact")
        self.assertEqual(classify_contact_candidate(
            [6, 7], {6, 7}, 0.066, 0.005, False),
            "excessive_gripper_penetration")
        self.assertEqual(classify_contact_candidate(
            [6, 7], {6, 7}, 0.003, 0.005, False), "contact_candidate")

    def test_inward_offsets_follow_selected_standoff(self):
        self.assertEqual(inward_offsets_for_standoff(0.22), [0.18, 0.2, 0.22])

    def test_candidate_rank_prefers_multi_link_contact(self):
        single = {"final_gripper_kettle_link_ids": [7],
                  "closure_contact_sample_count": 13,
                  "goal_habitat_fk_error_m": 0.001}
        bilateral = {"final_gripper_kettle_link_ids": [7, 8],
                     "closure_contact_sample_count": 4,
                     "goal_habitat_fk_error_m": 0.02}
        self.assertGreater(grasp_candidate_rank(bilateral),
                           grasp_candidate_rank(single))

    def test_candidate_rank_breaks_tie_with_persistence(self):
        brief = {"final_gripper_kettle_link_ids": [7],
                 "final_gripper_max_penetration_m": 0.001,
                 "closure_contact_sample_count": 2,
                 "goal_habitat_fk_error_m": 0.001}
        persistent = {"final_gripper_kettle_link_ids": [7],
                      "final_gripper_max_penetration_m": 0.001,
                      "closure_contact_sample_count": 12,
                      "goal_habitat_fk_error_m": 0.02}
        self.assertGreater(grasp_candidate_rank(persistent),
                           grasp_candidate_rank(brief))

    def test_candidate_rank_prefers_shallower_bilateral_contact(self):
        deep = {"final_gripper_kettle_link_ids": [6, 7],
                "final_gripper_max_penetration_m": 0.03,
                "closure_contact_sample_count": 13,
                "goal_habitat_fk_error_m": 0.001}
        shallow = {"final_gripper_kettle_link_ids": [6, 7],
                   "final_gripper_max_penetration_m": 0.002,
                   "closure_contact_sample_count": 4,
                   "goal_habitat_fk_error_m": 0.02}
        self.assertGreater(grasp_candidate_rank(shallow),
                           grasp_candidate_rank(deep))

    def test_offsets_move_toward_center_and_down(self):
        values = local_grasp_targets([-1.0, 1.5, 0.0], [0.0, 1.0, 0.0],
                                     [0.1], [0.2])
        self.assertEqual(values[0]["world_target"], [-0.9, 1.3, 0.0])
        self.assertEqual(values[0]["wrist_pitch_offset_deg"], 0.0)
        self.assertEqual(values[0]["wrist_roll_offset_deg"], 0.0)

    def test_wrist_offsets_expand_candidate_grid(self):
        values = local_grasp_targets([-1.0, 1.5, 0.0], [0.0, 1.0, 0.0],
                                     [0.1], [0.2], [-30, 30], [0, 90])
        self.assertEqual(len(values), 4)
        self.assertEqual(
            [(item["wrist_pitch_offset_deg"], item["wrist_roll_offset_deg"])
             for item in values],
            [(-30.0, 0.0), (-30.0, 90.0), (30.0, 0.0), (30.0, 90.0)])

    def test_rejects_identical_horizontal_positions(self):
        with self.assertRaises(ValueError):
            local_grasp_targets([0.0, 1.5, 0.0], [0.0, 1.0, 0.0], [0.1], [0.1])

    def test_segmented_pregrasp_replays_validated_joint_solutions(self):
        pregrasp = {
            "trajectory_strategy": "lift_traverse_descend",
            "segmented_approach": {
                "status": "feasible",
                "segments": [
                    {"segment_name": "lift", "status": "feasible",
                     "joint_solution": [1, 2]},
                    {"segment_name": "descend", "status": "feasible",
                     "joint_solution": [3, 4]},
                ],
            },
        }
        self.assertEqual(pregrasp_joint_waypoints(pregrasp), [
            {"name": "lift", "joint_solution": [1.0, 2.0]},
            {"name": "descend", "joint_solution": [3.0, 4.0]},
        ])

    def test_direct_pregrasp_has_one_waypoint(self):
        self.assertEqual(pregrasp_joint_waypoints({"joint_solution": [1, 2]}), [
            {"name": "direct_pregrasp", "joint_solution": [1.0, 2.0]},
        ])

    def test_angle_degrees_for_aligned_vectors(self):
        self.assertAlmostEqual(angle_degrees([1, 0, 0], [1, 0, 0]), 0.0)

    def test_angle_rejects_zero_vector(self):
        with self.assertRaises(ValueError):
            angle_degrees([0, 0, 0], [1, 0, 0])


if __name__ == "__main__":
    unittest.main()
