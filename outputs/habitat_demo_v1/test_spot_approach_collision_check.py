import unittest

from run_spot_approach_collision_check import classify_robot_contacts, interpolate_joint_path


class Contact:
    def __init__(self, a, b, la, lb, distance=-0.01, position=None):
        self.object_id_a, self.object_id_b = a, b
        self.link_id_a, self.link_id_b = la, lb
        self.contact_distance = distance
        if position is not None:
            self.position_on_a_in_ws = position


class SpotApproachHelpersTest(unittest.TestCase):
    def test_interpolation_respects_joint_step(self):
        path = interpolate_joint_path([0.0, 0.0], [0.2, -0.1], 0.05, 2)
        self.assertEqual(path[0], [0.0, 0.0])
        self.assertEqual(path[-1], [0.2, -0.1])
        self.assertEqual(len(path), 5)

    def test_contact_classification(self):
        result = classify_robot_contacts(
            [Contact(5, 8, 7, -1), Contact(5, -1, 3, -1), Contact(5, -1, 12, -1)],
            robot_id=5, kettle_id=8, arm_links={3, 7})
        self.assertEqual(len(result["arm_kettle"]), 1)
        self.assertEqual(len(result["arm_environment"]), 1)
        self.assertEqual(len(result["non_arm"]), 1)

    def test_contact_classification_preserves_available_position(self):
        result = classify_robot_contacts(
            [Contact(5, -1, 3, -1, position=[1, 2, 3])],
            robot_id=5, kettle_id=8, arm_links={3})
        self.assertEqual(result["arm_environment"][0]["position_on_a_ws"],
                         [1.0, 2.0, 3.0])


if __name__ == "__main__":
    unittest.main()
