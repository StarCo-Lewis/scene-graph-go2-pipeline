"""Offline tests for candidate authoring; no model or robot calls."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from embodied_pipeline.data_collection import catalog


class CatalogTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.nodes = [{"id": i, "object_tag": "book", "bbox_center": [i, 0, 0], "bbox_extent": [0.2, 0.1, 0.1]}
                      for i in range(1, 31)]
        self.nodes.extend({"id": i, "object_tag": "table", "bbox_center": [i, 0, 0], "bbox_extent": [1, 1, 1]}
                          for i in range(31, 36))
        self.nodes.extend([
            {"id": 90, "object_tag": "person's torso", "bbox_extent": [0.5, 0.5, 0.5]},
            {"id": 91, "object_tag": "table fragment", "bbox_extent": [0.5, 0.5, 0.5]},
            {"id": 92, "object_tag": "unknown", "bbox_extent": [0.5, 0.5, 0.5]},
            {"id": 93, "object_tag": "pillow", "bbox_extent": [2, 0.5, 0.5]},
        ])
        self.edges = [{"source": 31, "target": 1, "relation": "b on a"}]
        self.write()

    def write(self):
        (self.root / "scene_graph.json").write_text(json.dumps(self.nodes), encoding="utf-8")
        (self.root / "scene_graph_edges.json").write_text(json.dumps(self.edges), encoding="utf-8")

    def test_two_hundred_unique_deterministic_separate_references(self):
        records, diag, refs = catalog.build_scene("scene201", self.root)
        self.assertEqual(len(records), 200)
        self.assertEqual(diag["shortages"], {})
        self.assertEqual(records[0]["sample_id"], "scene201_00000001")
        self.assertEqual(records[-1]["sample_id"], "scene201_00000200")
        self.assertEqual(len({r["dedup"]["semantic_key"] for r in records}), 200)
        rerun, diag2, refs2 = catalog.build_scene("scene201", self.root)
        self.assertEqual([r["dedup"] for r in records], [r["dedup"] for r in rerun])
        self.assertEqual((diag, refs), (diag2, refs2))
        self.assertTrue(all(r["record_id"] == r["sample_id"] for r in records))
        self.assertTrue(all(r["instruction_generated_at_local"].endswith("+08:00") for r in records))
        self.assertTrue(all(r["room_scope"] == "unknown" for r in records))
        self.assertTrue(all(not r["quality"]["training_eligible"] for r in records))
        self.assertNotIn("steps", json.dumps(records[0]["goal"]))
        self.assertTrue(all(ref["independent_of_model_output"] and not ref["independently_human_verified"] for ref in refs))
        self.assertTrue(any(len(ref["dependencies"]) == 2 for ref in refs))

    def test_support_is_from_edges_and_not_invented(self):
        records, _, _ = catalog.build_scene("scene201", self.root)
        facts = [f for r in records for f in r["initial_state"]["support_relations"]]
        self.assertTrue(facts)
        self.assertEqual({(f["object_id"], f["support_id"]) for f in facts}, {(1, 31)})
        self.assertTrue(all(r["initial_state"]["robot_pose"] is None for r in records))
        self.assertTrue(any(r["initial_state"]["unknown_support_object_ids"] for r in records))

    def test_rejects_historical_duplicates_and_bad_entities(self):
        records, _, _ = catalog.build_scene("scene201", self.root)
        historical = records[0]["instruction"].upper().replace(" ", "  ")
        rerun, diag, _ = catalog.build_scene("scene201", self.root, existing_instructions=[historical])
        self.assertEqual(len(rerun), 200)
        self.assertEqual(diag["skipped"]["historical_instruction_duplicate"], 1)
        self.assertNotIn(records[0]["instruction"], {r["instruction"] for r in rerun})
        movers = diag["candidate_pools"]["movers"]
        self.assertFalse(set(movers) & {90, 91, 92, 93})
        self.assertFalse(set(diag["candidate_pools"]["landmarks"]) & {90, 91, 92})
        self.assertEqual(diag["source_node_count"], len(self.nodes))

    def test_shortage_is_not_padded(self):
        self.nodes = self.nodes[:1]
        self.edges = []
        self.write()
        records, diag, _ = catalog.build_scene("scene201", self.root)
        self.assertEqual(diag["status"], "blocked_candidate_shortage")
        self.assertLess(len(records), 200)
        self.assertTrue(diag["shortages"])

    def test_room_claims_require_explicit_mapping(self):
        room_file = self.root / "rooms.json"
        room_file.write_text(json.dumps({"control_node_rooms": {str(i): "A" if i < 20 else "B" for i in range(1, 36)},
                                        "room_assignment_source": "authored_for_test"}))
        with patch.dict(catalog.ROOM_FILES, {"scene201": room_file}):
            records, diag, _ = catalog.build_scene("scene201", self.root)
        self.assertGreater(diag["room_scope_counts"]["cross_room"], 0)
        self.assertTrue(all(r["room_provenance"]["status"] == "author_assigned_not_perception_truth" for r in records))

    def test_source_graph_references_must_exist(self):
        self.edges.append({"source": 1, "target": 9999, "relation": "a on b"})
        self.write()
        with self.assertRaisesRegex(ValueError, "absent node"):
            catalog.build_scene("scene201", self.root)

    def test_surface_head_not_table_lamp_or_ornament(self):
        def node(identity, tag):
            return {"id": identity, "object_tag": tag, "bbox_extent": [0.5, 0.5, 0.5]}
        nodes = [node(1, "table ornament"), node(2, "adjustable desk lamp"),
                 node(3, "wooden desk with black metal legs"), node(4, "cabinet door"),
                 node(5, "cabinet handle"), node(6, "table fragment")]
        pool = catalog._sorted_pool(nodes, catalog.SURFACE)
        self.assertEqual([n["id"] for n in pool], [3])

    def test_aabb_rotation_and_seating_rules(self):
        box = {"object_tag": "cardboard box", "bbox_extent": [0.3, 0.7, 0.1]}
        table = {"object_tag": "table", "bbox_extent": [0.8, 0.4, 0.1]}
        chair = {"object_tag": "chair", "bbox_extent": [0.8, 0.4, 0.1]}
        pillow = {"object_tag": "pillow", "bbox_extent": [0.3, 0.7, 0.1]}
        self.assertTrue(catalog._fits_horizontal(box, table))
        self.assertFalse(catalog._allowed_placement(box, chair))
        self.assertTrue(catalog._allowed_placement(pillow, chair))
        self.assertFalse(catalog._fits_horizontal(box, {"bbox_extent": [0.1, 0.1, 1]}))

    def test_geometry_axes_require_explicit_evidence(self):
        axes, status = catalog._geometry_contract(self.root)
        self.assertIsNone(axes)
        self.assertEqual(status["status"], "geometry_prefilter_unknown")
        (self.root / "scene_metadata.json").write_text(json.dumps({"up_axis": "y", "unit": "metre"}))
        axes, status = catalog._geometry_contract(self.root)
        self.assertEqual(axes, (0, 2))
        self.assertEqual(status["horizontal_axes"], ["x", "z"])
        obj = {"bbox_extent": [0.3, 0.2, 0.7]}
        support = {"bbox_extent": [0.8, 0.8, 0.4]}
        self.assertTrue(catalog._fits_horizontal(obj, support, axes))
        support["bbox_extent"] = [0.5, 10, 0.4]
        self.assertFalse(catalog._fits_horizontal(obj, support, axes))

    def test_cyclic_source_support_is_flagged_not_rewritten(self):
        self.edges.extend([{"source": 1, "target": 2, "relation": "a on b"},
                           {"source": 2, "target": 1, "relation": "a on b"}])
        self.write()
        records, diag, _ = catalog.build_scene("scene201", self.root)
        self.assertIn("cycle_in_supplied_on_relations", diag["source_support_conflicts"]["1"])
        affected = [r for r in records if 1 in r["grounded_object_ids"]]
        self.assertTrue(all(r["initial_state"]["support_relation_status"] == "conflict_pending_review" for r in affected))
        self.assertEqual(json.loads((self.root / "scene_graph_edges.json").read_text()), self.edges)

    def test_real_scene_manipulation_requires_saved_complete_caption(self):
        evidence = {1: {"status": "SUCCESS", "qwen": {"is_complete_object": True, "entity_type": "object"}},
                    2: {"status": "SUCCESS", "qwen": {"is_complete_object": False, "entity_type": "object"}},
                    3: {"status": "SUCCESS", "qwen": {"is_complete_object": True, "entity_type": "fragment"}}}
        with patch.object(catalog, "_caption_evidence", return_value=(evidence, [])):
            records, diag, _ = catalog.build_scene("d3_619", self.root)
        self.assertEqual(diag["candidate_pools"]["movers"], [1])
        self.assertEqual(diag["source_node_count"], len(self.nodes))
        self.assertTrue(all(not r["initial_state"]["held_state_known"] for r in records))

    def test_invalid_count(self):
        for count in (True, 0, 201, 1.5):
            with self.assertRaises(ValueError):
                catalog.build_scene("scene201", self.root, count=count)


if __name__ == "__main__":
    unittest.main()
