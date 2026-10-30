import math
import unittest

from run_spot_base_pose_search import (add_link_diagnostics, approach_targets,
                                       candidate_base_queries, clearance_approaches,
                                       clearance_waypoints,
                                       facing_rotation, max_contact_penetration,
                                       radial_approach)


class SpotBasePoseHelpersTest(unittest.TestCase):
    def test_queries_expand_from_reference_direction(self):
        values = candidate_base_queries([0, 1, 0], [-1, 0, 0], [0.8], [0])
        self.assertAlmostEqual(values[0]["query"][0], -0.8)
        self.assertAlmostEqual(values[0]["query"][2], 0.0, places=6)

    def test_facing_rotation_points_local_x_at_target(self):
        self.assertAlmostEqual(facing_rotation([-1, 0, 0], [0, 0, 0]), 0.0)
        self.assertAlmostEqual(facing_rotation([0, 0, 1], [0, 0, 0]), math.pi / 2)

    def test_radial_approach_is_outside_box_toward_base(self):
        point = radial_approach([0, 1, 0], [0.2, 0.4, 0.4], [-1, 0, 0], 0.05)
        self.assertEqual(point, [-0.15000000000000002, 1.0, 0.0])

    def test_approach_targets_include_side_high_and_above(self):
        targets = approach_targets([0, 1, 0], [0.2, 0.4, 0.4], [-1, 0, 0], 0.05)
        self.assertEqual([item["name"] for item in targets],
                         ["radial_mid", "radial_high", "above_center"])
        self.assertEqual(targets[0]["world_target"], [-0.15000000000000002, 1.0, 0.0])
        self.assertEqual(targets[1]["world_target"], [-0.15000000000000002, 1.25, 0.0])
        self.assertEqual(targets[2]["world_target"], [0.0, 1.25, 0.0])
        for target in targets:
            self.assertEqual(target["orientation_strategy"],
                             "preserve_measured_wrist_joints")
            self.assertEqual(target["locked_arm_joint_indices"], [5, 6])

    def test_scene201_wrist_standoff_can_be_recovered_by_grasp_search(self):
        # The Scene201 runner uses a 14 cm pre-grasp standoff.  The local
        # grasp search's largest inward offset is also 14 cm, so it can still
        # reach the observed box surface after the wrist-safe base search.
        target = radial_approach([0, 1, 0], [0.2, 0.4, 0.4], [-1, 0, 0], 0.14)
        self.assertAlmostEqual(target[0], -0.24)
        self.assertAlmostEqual(target[0] + 0.14, -0.10)

    def test_clearance_search_tracks_each_standoff(self):
        approaches = list(clearance_approaches(
            [0, 1, 0], [0.2, 0.4, 0.4], [-1, 0, 0], [0.14, 0.22]))
        self.assertEqual(len(approaches), 6)
        self.assertEqual(approaches[0]["clearance_m"], 0.14)
        self.assertEqual(approaches[3]["clearance_m"], 0.22)
        self.assertAlmostEqual(approaches[3]["world_target"][0], -0.32)

    def test_clearance_waypoints_lift_traverse_and_descend(self):
        waypoints = clearance_waypoints([0, 0.5, 0], [1, 1.0, 2], 0.2)
        self.assertEqual([item["name"] for item in waypoints],
                         ["lift_from_start", "traverse_above_goal", "descend_to_goal"])
        self.assertEqual(waypoints[0]["world_target"], [0.0, 1.2, 0.0])
        self.assertEqual(waypoints[1]["world_target"], [1.0, 1.2, 2.0])
        self.assertEqual(waypoints[2]["world_target"], [1.0, 1.0, 2.0])

    def test_contact_diagnostics_add_link_name_without_mutation(self):
        source = [{"robot_link_id": 19, "other_object_id": -1}]
        result = add_link_diagnostics(source, {19: "arm0.link_fngr"})
        self.assertEqual(result[0]["robot_link_name"], "arm0.link_fngr")
        self.assertNotIn("robot_link_name", source[0])

    def test_max_contact_penetration_uses_only_negative_distance(self):
        self.assertEqual(max_contact_penetration([]), 0.0)
        self.assertEqual(max_contact_penetration([
            {"contact_distance": 0.002}, {"contact_distance": -0.004}]),
            0.004)


if __name__ == "__main__":
    unittest.main()
