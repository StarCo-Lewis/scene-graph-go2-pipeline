import unittest

from run_spot_ik_reachability import (
    approach_candidates,
    distance,
    grounded_node,
    select_spot_arm_joints,
    within_limits,
)


class SpotIkHelpersTest(unittest.TestCase):
    def test_grounded_node_follows_alignment(self):
        alignment = {"status": "success", "context_bindings": {"kettle": {"element_ids": [67]}}}
        node = grounded_node(alignment, [{"id": 13, "bbox_center": [0, 0, 0]},
                                         {"id": 67, "bbox_center": [1, 2, 3]}], "kettle")
        self.assertEqual(node["id"], 67)

    def test_approach_candidates_use_extent_and_clearance(self):
        candidates = approach_candidates([1, 2, 3], [0.2, 0.4, 0.6], 0.05)
        self.assertEqual(len(candidates), 6)
        above = next(item for item in candidates if item["name"] == "above")
        self.assertEqual(above["world_target"], [1.0, 2.25, 3.0])

    def test_joint_limit_validation(self):
        self.assertTrue(within_limits([0.0, 0.5], [-1.0, 0.0], [1.0, 1.0]))
        self.assertFalse(within_limits([0.0, 1.5], [-1.0, 0.0], [1.0, 1.0]))

    def test_distance(self):
        self.assertAlmostEqual(distance([0, 0, 0], [1, 2, 2]), 3.0)

    def test_full_spot_joint_mapping_ignores_legs(self):
        joints = [{"index": i, "link_name": f"leg_{i}", "movable": True} for i in range(12)]
        joints += [{"index": 12 + i, "link_name": f"arm0.link_{i}", "movable": True}
                   for i in range(8)]
        arm, ee = select_spot_arm_joints(joints)
        self.assertEqual([joint["index"] for joint in arm], list(range(12, 19)))
        self.assertEqual(ee["index"], 19)


if __name__ == "__main__":
    unittest.main()
