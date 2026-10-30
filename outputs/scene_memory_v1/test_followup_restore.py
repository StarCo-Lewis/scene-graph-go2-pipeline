import copy
import unittest
from update_memory import initialize, proxy_events, apply_events
from run_memory_followup_navigation import restore_contract


class RestoreTests(unittest.TestCase):
    def setUp(self):
        self.nodes = [{"id": 67, "bbox_center": [1, 2, 3]}, {"id": 13, "bbox_center": [4, 5, 6]}]
        self.proxy = {"status": "proxy_instance_support_passed", "input": {"scene_graph": "scene/scene_graph.json"},
            "scene_graph_ids": {"kettle": 67, "sideboard": 13}, "steps": [
                {"skill_id": "instantiate_object", "cg_element_id": 67, "habitat_object_id": 1, "position": [1, 2.9, -2]},
                {"skill_id": "instantiate_sideboard", "cg_element_id": 13, "habitat_object_id": 0, "position": [4, 0.5, -5]},
                {"skill_id": "navigate_to_object", "status": "arrived", "end": [1, 0.1, -2]},
                {"skill_id": "navigate_to_object", "status": "arrived", "end": [4, 0.1, -5]},
                {"skill_id": "place_object_on_surface", "status": "proxy_instance_support_passed",
                 "stable_last_second": True, "direct_sideboard_contact_frames_last_second": 60,
                 "sideboard_habitat_object_id": 0, "settled_position": [4, 5.9, -5]}]}
        events = proxy_events(self.proxy, self.nodes, "proxy.json", "2026-09-26T03:00:00Z")
        self.state = apply_events(initialize(self.nodes, [], "scene"), events)
        self.current = [entity["node"] for entity in self.state["entities"].values()]

    def test_restore_translation_matches_memory(self):
        result = restore_contract(self.state, self.current, self.nodes, self.proxy)
        reconstructed = [result["kettle_com"][i] + result["bbox_minus_com_offset"][i] for i in range(3)]
        for actual, expected in zip(reconstructed, result["target_bbox_habitat"]):
            self.assertAlmostEqual(actual, expected)
        self.assertEqual(result["robot_position"], [4, 0.1, -5])

    def test_altered_source_execution_is_rejected(self):
        changed = copy.deepcopy(self.proxy)
        changed["steps"][-1]["settled_position"][0] += 1
        with self.assertRaises(ValueError):
            restore_contract(self.state, self.current, self.nodes, changed)

    def test_wrong_source_digest_is_rejected(self):
        self.state["event_log"][-1]["event"]["evidence"]["proxy_digest"] = "wrong"
        with self.assertRaises(ValueError):
            restore_contract(self.state, self.current, self.nodes, self.proxy)

    def test_original_graph_cannot_be_used_as_updated_snapshot(self):
        with self.assertRaises(ValueError):
            restore_contract(self.state, self.nodes, self.nodes, self.proxy)
