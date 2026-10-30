import copy
import unittest
from export_proxy_training_episode import build_episode
from run_manipulation_proxy import EXPECTED_SKILLS


class EpisodeTests(unittest.TestCase):
    def setUp(self):
        self.run = {
            "status": "proxy_instance_support_passed",
            "execution_policy": {"grasp_rule": "assume_grasp_after_validated_proximity",
                                 "proximity_threshold_m": 1.5},
            "steps": [{"skill_id": s, "status": "proxy_only_not_executed"}
                      for s in EXPECTED_SKILLS],
        }
        for i in (0, 4):
            self.run["steps"][i]["status"] = "arrived"
        self.run["steps"][2].update(status="proxy_attached", reach_m=0.8,
            evidence_type="assumed_after_navigation_and_proximity_check")
        self.run["steps"][6].update(status="proxy_instance_support_passed",
            stable_last_second=True, direct_sideboard_contact_frames_last_second=60)

    def test_pass(self):
        episode = build_episode(self.run, {"instruction": "Move the kettle."})
        self.assertTrue(episode["training_eligible"])
        self.assertFalse(episode["combined_from_separate_simulation_runs"])
        self.assertFalse(episode["continuous_robot_execution_verified"])

    def test_failed_checks_never_become_positive_samples(self):
        for index, key, value in ((0, "status", "failed"), (4, "status", "failed"),
                                  (2, "reach_m", 2.0), (2, "reach_m", float("nan")),
                                  (2, "evidence_type", None),
                                  (6, "stable_last_second", False),
                                  (6, "direct_sideboard_contact_frames_last_second", 0)):
            with self.subTest(index=index, key=key):
                run = copy.deepcopy(self.run)
                run["steps"][index][key] = value
                self.assertFalse(build_episode(run, {})["training_eligible"])

    def test_navigation_failure_partial_trace(self):
        self.run["steps"] = self.run["steps"][:1]
        self.run["status"] = "navigation_failed"
        self.assertFalse(build_episode(self.run, {})["grasp_assumed"])

    def test_legacy_run_rejected(self):
        self.run.pop("execution_policy")
        with self.assertRaises(ValueError):
            build_episode(self.run, {})

    def test_injected_fault_cannot_be_positive(self):
        self.run["test_fault"] = "before_place"
        self.assertFalse(build_episode(self.run, {})["training_eligible"])


if __name__ == "__main__":
    unittest.main()
