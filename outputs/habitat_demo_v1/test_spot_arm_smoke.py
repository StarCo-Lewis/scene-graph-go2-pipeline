import unittest

from run_spot_arm_smoke import choose_safe_joint_target, max_abs_delta


class SpotArmHelpersTest(unittest.TestCase):
    def test_choose_positive_target_within_limits(self):
        target, index = choose_safe_joint_target([0.0, 0.0], [-1.0, -1.0], [1.0, 1.0], 0.12)
        self.assertEqual(index, 0)
        self.assertAlmostEqual(target[0], 0.12)
        self.assertLessEqual(target[0], 0.97)

    def test_choose_negative_when_positive_margin_is_too_small(self):
        target, index = choose_safe_joint_target([0.95], [-1.0], [1.0], 0.12)
        self.assertEqual(index, 0)
        self.assertAlmostEqual(target[0], 0.83)

    def test_rejects_invalid_joint_vectors(self):
        with self.assertRaises(ValueError):
            choose_safe_joint_target([0.0], [-1.0, -1.0], [1.0])

    def test_max_abs_delta(self):
        self.assertAlmostEqual(max_abs_delta([0.0, 1.0], [0.1, 0.8]), 0.2)


if __name__ == "__main__":
    unittest.main()
