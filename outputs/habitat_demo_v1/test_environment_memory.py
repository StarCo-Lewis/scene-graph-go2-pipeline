import importlib.util
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).parent
SPEC = importlib.util.spec_from_file_location("build_environment_memory", HERE / "build_environment_memory.py")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class EnvironmentMemoryTests(unittest.TestCase):
    def fixtures(self, passed=True):
        proxy = {"status": "proxy_instance_support_passed",
                 "steps": [
                     {"skill_id": "instantiate_sideboard", "habitat_object_id": 0},
                     {"skill_id": "instantiate_object", "habitat_object_id": 1},
                     {"skill_id": "navigate_to_object", "status": "arrived"},
                     {"skill_id": "navigate_to_object", "status": "arrived"},
                     {"skill_id": "place_object_on_surface",
                      "status": "proxy_instance_support_passed" if passed else "proxy_instance_support_failed",
                      "stable_last_second": True,
                      "direct_sideboard_contact_frames_last_second": 60},
                 ]}
        alignment = {"context_bindings": {"electric_kettle": {"element_ids": [67]},
                                           "sideboard": {"element_ids": [13]}}}
        plan = {"instruction": "Move the kettle.", "subtask_plans": [{}, {}, {}]}
        nodes = [{"id": 67}, {"id": 13}]
        return proxy, alignment, plan, nodes

    def test_verified_contact_becomes_observed_on_fact(self):
        memory, snapshot = MODULE.build(*self.fixtures(), "2026-09-21T12:00:00+08:00")
        on_fact = next(f for f in snapshot["facts"] if f["predicate"] == "on")
        self.assertEqual((on_fact["subject"], on_fact["value"]), (67, 13))
        self.assertEqual(on_fact["source"], "simulator_observation")
        self.assertEqual(memory["entity_bindings"][0]["habitat_object_id"], 1)
        self.assertEqual(memory["entity_bindings"][1]["habitat_object_id"], 0)
        self.assertEqual(memory["post_execution_goal_status"], "observed_in_habitat_proxy")
        self.assertEqual(memory["ashita_integration"]["hierarchical_information_bottleneck"],
                         "not_run_requires_primitive_layer_and_multilevel_conditional_distributions")

    def test_failed_support_does_not_create_on_fact(self):
        _, snapshot = MODULE.build(*self.fixtures(False), "2026-09-21T12:00:00+08:00")
        self.assertNotIn("on", [fact["predicate"] for fact in snapshot["facts"]])


if __name__ == "__main__":
    unittest.main()
