import unittest

from run_spot_grasp_contact_check import (contact_link_ids, interpolate_vector,
                                          select_spot_grasp_contact_links)


class SpotGraspContactHelpersTest(unittest.TestCase):
    def test_fixed_distal_wrist_surface_counts_as_gripper(self):
        selected = select_spot_grasp_contact_links(
            {7}, {5: "arm0.link_wr0", 6: "arm0.link_wr1",
                  7: "arm0.link_fngr"})
        self.assertEqual(selected, {6, 7})

    def test_interpolate_vector_includes_endpoints(self):
        path = interpolate_vector([0.0, 1.0], [1.0, 0.0], 2)
        self.assertEqual(path, [[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]])

    def test_interpolate_vector_rejects_invalid_inputs(self):
        with self.assertRaises(ValueError):
            interpolate_vector([0.0], [0.0, 1.0], 2)
        with self.assertRaises(ValueError):
            interpolate_vector([0.0], [1.0], 0)

    def test_contact_link_ids_is_sorted_and_unique(self):
        contacts = [{"robot_link_id": 4}, {"robot_link_id": 2},
                    {"robot_link_id": 4}]
        self.assertEqual(contact_link_ids(contacts), [2, 4])


if __name__ == "__main__":
    unittest.main()
