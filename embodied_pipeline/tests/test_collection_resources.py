"""Offline resource audit and byte-faithful packaging tests."""
import json
from pathlib import Path
import tempfile
import unittest

from embodied_pipeline.data_collection.resources import audit_graph, collect_modalities, sha256


class CollectionResourceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.scene = self.root / "sample" / "scene"
        self.scene.mkdir(parents=True)
        self.bundle = self.root / "bundle"
        self.nodes = [{"id": 1, "object_tag": "chair", "caption": "A chair.",
                       "bbox_center": [0, 1, 2], "bbox_extent": [1, 1, 1]}]
        self.save()

    def write(self, path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data), encoding="utf-8")

    def save(self, edges=None):
        self.write(self.scene / "scene_graph.json", self.nodes)
        self.write(self.scene / "scene_graph_edges.json", edges or [])

    def test_isolation_is_not_missing_export(self):
        result = audit_graph(self.scene)
        self.assertFalse(result["blocking_issues"])
        self.assertEqual(result["nodes_without_exported_relations"], [1])
        self.assertFalse(result["semantic_correctness_evaluated"])

    def test_invalid_references_and_duplicates_block(self):
        self.nodes.append(dict(self.nodes[0]))
        self.save([{"source": 1, "target": 9, "relation": "a on b"},
                   {"source": 1, "target": 1, "relation": "a on b"}])
        codes = {row["code"] for row in audit_graph(self.scene)["blocking_issues"]}
        self.assertEqual(codes, {"duplicate_node_id", "invalid_edge_endpoints", "self_reference"})

    def test_unknown_and_flat_bbox_are_unrepaired_warnings(self):
        self.nodes[0]["object_tag"] = "unknown"
        self.nodes[0]["bbox_extent"] = [0, 1, 1]
        self.save()
        result = audit_graph(self.scene)
        self.assertFalse(result["blocking_issues"])
        self.assertEqual(result["unknown_semantic_node_ids"], [1])
        self.assertEqual(result["zero_extent_node_ids"], [1])

    def test_invalid_bbox_or_blank_semantics_block(self):
        self.nodes[0]["caption"] = " "
        self.nodes[0]["bbox_center"] = None
        self.save()
        codes = {row["code"] for row in audit_graph(self.scene)["blocking_issues"]}
        self.assertEqual(codes, {"missing_semantic_field", "invalid_bbox"})

    def test_original_bytes_and_image_hash_retained(self):
        image = self.scene.parent / "instances" / "1.png"
        image.parent.mkdir()
        image.write_bytes(b"test-image-content")
        self.nodes[0]["visual_evidence"] = {"views": ["instances/1.png"]}
        self.save()
        self.write(self.scene.parent / "evidence_manifest.json", {
            "inventory": [{"id": 1, "render_views": [{"path": "instances/1.png", "sha256": sha256(image)}]}]})
        result = collect_modalities("scene200", self.scene, self.bundle)
        self.assertEqual(result["resource_count"], 1)
        row = result["node_resources"][0]
        self.assertEqual(row["source_sha256"], row["copy_sha256"])
        self.assertEqual((self.scene / "scene_graph.json").read_bytes(),
                         (self.bundle / "scene_graph.json").read_bytes())
        self.assertEqual(result, collect_modalities("scene200", self.scene, self.bundle))

    def test_bad_provenance_hash_preserves_gap(self):
        image = self.scene.parent / "1.jpg"
        image.write_bytes(b"changed-content")
        self.nodes[0]["visual_evidence"] = {"views": ["1.jpg"]}
        self.save()
        self.write(self.scene.parent / "evidence_manifest.json", {
            "inventory": [{"id": 1, "render_views": [{"path": "1.jpg", "sha256": "0" * 64}]}]})
        result = collect_modalities("s", self.scene, self.bundle)
        self.assertEqual(result["missing_node_ids"], [1])
        self.assertEqual(result["source_gaps"][0]["code"], "source_hash_mismatch")

    def test_caption_id_and_selected_view_link(self):
        captions = self.root / "captions"
        image = self.root / "crop.jpg"
        image.write_bytes(b"caption-crop")
        self.write(captions / "objects" / "object_001.json", {
            "id": 1, "selected_views": [{"crop": str(image)}]})
        result = collect_modalities("s", self.scene, self.bundle, captions_dir=captions)
        self.assertEqual(result["resource_count"], 1)
        self.assertEqual(result["node_resources"][0]["link_evidence"]["node_id"], 1)

    def test_caption_mismatch_cannot_be_bound_by_filename(self):
        captions = self.root / "captions"
        image = self.root / "crop.jpg"
        image.write_bytes(b"caption-crop")
        self.write(captions / "objects" / "object_001.json", {
            "id": 999, "selected_views": [{"crop": str(image)}]})
        result = collect_modalities("s", self.scene, self.bundle, captions_dir=captions)
        self.assertEqual(result["missing_node_ids"], [1])
        self.assertEqual(result["source_gaps"][0]["code"], "caption_id_mismatch")


if __name__ == "__main__":
    unittest.main()
