import copy
import unittest
from update_memory import initialize, apply_events, proxy_events


class MemoryTests(unittest.TestCase):
    def setUp(self):
        self.nodes = [{"id": 67, "bbox_center": [1, 2, 3]}, {"id": 13, "bbox_center": [4, 5, 6]}]
        self.state = initialize(self.nodes, [], "scene")
        self.event = {"event_id": "e1", "action_id": "place", "operation": "move", "object_id": 67,
            "action_status": "completed", "observed_at": "2026-09-26T03:00:00Z",
            "coordinate_frame": "conceptgraphs_world", "position": [4, 5, 6],
            "support_object_id": 13, "evidence": {"source": "simulator_observation",
                "verified": True, "reference": "verified-episode.json"}}

    def test_move_isolated_and_relations_updated(self):
        result = apply_events(self.state, [self.event])
        self.assertEqual(result["entities"]["67"]["node"]["bbox_center"], [4, 5, 6])
        self.assertEqual(self.state["entities"]["67"]["node"]["bbox_center"], [1, 2, 3])
        self.assertEqual(result["edges"][0]["relation"], "a on b")
        self.assertEqual(result["version"], 1)

    def test_idempotent_replay(self):
        result = apply_events(self.state, [self.event])
        self.assertEqual(apply_events(result, [self.event]), result)

    def test_conflicting_id_rejected(self):
        result = apply_events(self.state, [self.event])
        different = copy.deepcopy(self.event)
        different["position"][0] = 50
        with self.assertRaises(ValueError):
            apply_events(result, [different])

    def test_delete_is_tombstone(self):
        result = apply_events(self.state, [self.event])
        deletion = dict(self.event, event_id="e2", operation="delete", reason="removed_confirmed",
                        observed_at="2026-09-26T04:00:00Z")
        result = apply_events(result, [deletion])
        self.assertFalse(result["entities"]["67"]["active"])
        self.assertEqual(result["edges"], [])
        self.assertEqual(len(result["event_log"]), 2)

    def test_unknown_object_not_invented(self):
        with self.assertRaisesRegex(ValueError, "NO object"):
            apply_events(self.state, [dict(self.event, object_id=99)])

    def test_unverified_and_assumed_grasp_rejected(self):
        for evidence in ({"source": "assumed_grasp", "verified": True, "reference": "x"},
                         {"source": "simulator_observation", "verified": False, "reference": "x"}):
            with self.subTest(evidence=evidence), self.assertRaises(ValueError):
                apply_events(self.state, [dict(self.event, evidence=evidence)])

    def test_bad_positions_rejected(self):
        for pos in ([1, 2], [1, float("nan"), 2], [True, 1, 2]):
            with self.subTest(pos=pos), self.assertRaises(ValueError):
                apply_events(self.state, [dict(self.event, position=pos)])

    def test_stale_update_rejected(self):
        result = apply_events(self.state, [self.event])
        with self.assertRaises(ValueError):
            apply_events(result, [dict(self.event, event_id="e2")])

    def test_invalid_batch_does_not_change_previous_state(self):
        original = copy.deepcopy(self.state)
        with self.assertRaises(ValueError):
            apply_events(self.state, [self.event, dict(self.event, event_id="e2", object_id=99)])
        self.assertEqual(self.state, original)

    def test_cutting_not_deletion(self):
        with self.assertRaises(ValueError):
            apply_events(self.state, [dict(self.event, operation="delete", reason="cut")])

    def test_container_movement_requires_child_observations(self):
        state = initialize(self.nodes, [{"source": 13, "target": 67, "relation": "a on b"}], "scene")
        with self.assertRaises(ValueError):
            apply_events(state, [self.event])

    def test_empty_action_is_noop(self):
        self.assertEqual(apply_events(self.state, []), self.state)

    def test_proxy_com_delta_not_raw_bbox_centre(self):
        proxy = {"status": "proxy_instance_support_passed", "scene_graph_ids": {"kettle": 67, "sideboard": 13},
            "steps": [
                {"skill_id": "instantiate_object", "cg_element_id": 67, "habitat_object_id": 1, "position": [1, 2.9, -2]},
                {"skill_id": "instantiate_sideboard", "cg_element_id": 13, "habitat_object_id": 0},
                {"skill_id": "place_object_on_surface", "status": "proxy_instance_support_passed",
                 "stable_last_second": True, "direct_sideboard_contact_frames_last_second": 60,
                 "sideboard_habitat_object_id": 0, "settled_position": [4, 5.9, -5]}]}
        events = proxy_events(proxy, self.nodes, "run.json", "2026-09-26T03:00:00Z")
        self.assertAlmostEqual(events[0]["position"][2], 6)
        self.assertEqual(events[0]["position"][:2], [4, 5])
        proxy["steps"][-1]["stable_last_second"] = False
        with self.assertRaises(ValueError):
            proxy_events(proxy, self.nodes, "run.json", "2026-09-26T03:00:00Z")
