import unittest

from run_spot_dynamic_lift import evaluate_dynamic_lift


class SpotDynamicLiftHelpersTest(unittest.TestCase):
    def test_passes_when_object_follows_gripper(self):
        result = evaluate_dynamic_lift(0.08, 0.10, 0.02, 12, False)
        self.assertTrue(result["passed"])
        self.assertEqual(result["failures"], [])

    def test_rejects_stationary_object(self):
        result = evaluate_dynamic_lift(0.0, 0.10, 0.12, 0, False)
        self.assertFalse(result["passed"])
        self.assertIn("object_did_not_rise_enough", result["failures"])
        self.assertIn("no_gripper_contact_during_lift", result["failures"])

    def test_rejects_invalid_collision(self):
        result = evaluate_dynamic_lift(0.08, 0.10, 0.02, 12, True)
        self.assertFalse(result["passed"])
        self.assertIn("invalid_arm_collision", result["failures"])


if __name__ == "__main__":
    unittest.main()
