"""Offline graph-only reconstruction; no GPU, WSL or provider requests."""

import gzip
import io
import json
import pickle
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import urllib.error
import urllib.request

import numpy as np

from embodied_pipeline.perception import conceptgraphs_upstream as upstream
from embodied_pipeline.perception import conceptgraphs_scene as scene
from embodied_pipeline.perception.real_rgbd import write, sha
from embodied_pipeline.orchestration import real_scene_rebuild as rebuild
from embodied_pipeline.orchestration import real_scene
from embodied_pipeline.tests.test_caption_contract import caption, envelope


class RebuildTests(unittest.TestCase):
    def setUp(self):
        self.network = patch("urllib.request.urlopen", side_effect=AssertionError("Network forbidden"))
        self.network.start()
        self.addCleanup(self.network.stop)

    def fixture(self, root):
        map_file = root / "mapping/map.pkl.gz"
        map_file.parent.mkdir()
        objects = [dict(pcd_np=np.array([[i, 0, 0], [i+.012, .023, .034]]), clip_ft=np.array([i, 1])) for i in range(2)]
        with gzip.open(map_file, "wb") as stream:
            pickle.dump({"objects": objects}, stream)
        old = root / "old"
        captions = old / "captions"
        for i in range(2):
            value = {**caption(), "confidence": .1, "entity_type": "object_part", "is_complete_object": False}
            raw = envelope(value, f"request-{i}")
            write(captions / f"objects/object_{i:03d}.json", dict(id=i, status="SUCCESS", model="fixture",
                qwen=value, captions=[value["caption"]], usage=raw["usage"], request_id=raw["id"]))
            write(captions / f"llm_response_audit/fixture_{i}_http_1.json", dict(http_status=200, response_body=json.dumps(raw)))
        (captions / "STATUS").write_text("SUCCESS\n")
        write(captions / "qwen_caption_manifest.json", dict(model="fixture", base_url="https://invalid.test",
            map_file=str(map_file), source_object_count=2, completed_object_count=2, eligible_object_count=2))
        adapter = root / "adapter/build_qwen_conceptgraph_baseline.py"
        adapter.parent.mkdir()
        adapter.write_text("# offline adapter fixture")
        summary = old / "run_summary.json"
        write(summary, dict(status="graph_generated_and_imported_not_executed", map_unchanged=True,
            upstream_source_unchanged=True, resumed_from=dict(map_file=str(map_file), map_sha256=sha(map_file)),
            semantic_model="fixture", base_url="https://invalid.test", source_code_hashes={str(adapter): sha(adapter)}))
        return summary, map_file, captions

    def raw_export(self, output, captions):
        raw = output / "upstream_export"
        raw.mkdir(parents=True)
        nodes = []
        for i in range(2):
            value = json.loads((captions / f"objects/object_{i:03d}.json").read_text())["qwen"]
            nodes.append(dict(id=i, bbox_center=[i, 0, 0], bbox_extent=[0, 0, 0],
                              **{k: value[k] for k in ("caption", "object_tag", "possible_tags")}))
        write(raw / "scene_graph.json", nodes)
        write(raw / "scene_graph_edges.json", [dict(source=0, target=1, relation="a on b", reason="Fixture only.")])
        write(raw / "baseline_manifest.json", dict(node_count=2, semantic_edge_count=1, model_substitution="Qwen replacement"))
        (raw / "STATUS").write_text("SUCCESS\n")
        return raw

    @staticmethod
    def geometry(obj):
        low, high = obj["pcd_np"].min(axis=0), obj["pcd_np"].max(axis=0)
        return dict(center=(low+high)/2, extent=high-low)

    def test_all_parts_low_confidence_retained_ids_edges_precision_and_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, map_file, captions = self.fixture(root)
            output = root / "new/scene"
            raw = self.raw_export(output, captions)
            old_hashes = {p: sha(p) for p in raw.iterdir() if p.is_file()}
            metadata = upstream.adapt_baseline(map_file, captions, output, SimpleNamespace(geometry=self.geometry))
            bundle = scene.load_real_scene(output)
            self.assertEqual([n["id"] for n in bundle["nodes"]], [0, 1])
            self.assertTrue(all("active" not in n for n in bundle["nodes"]))
            self.assertEqual(metadata["inactive_node_count"], 0)
            self.assertAlmostEqual(bundle["nodes"][0]["bbox_extent"][0], .012)
            self.assertEqual(sha(output / "scene_graph_edges.json"), old_hashes[raw / "scene_graph_edges.json"])
            self.assertTrue(all(sha(p) == h for p, h in old_hashes.items()))
            self.assertFalse(scene.import_scene(output, root / "new/pipeline")["execution_ready"])

    def test_true_zero_dimensions_not_padded_or_dropped(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, map_file, captions = self.fixture(root)
            output = root / "new/scene"
            self.raw_export(output, captions)
            module = SimpleNamespace(geometry=lambda obj: dict(center=np.zeros(3), extent=np.array([0., .02, .03])))
            with self.assertRaisesRegex(ValueError, "positive"):
                upstream.adapt_baseline(map_file, captions, output, module)
            self.assertFalse((output / "scene_graph.json").exists())

    def test_default_export_calls_upstream_adapter_and_restores_arguments(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, map_file, captions = self.fixture(root)
            output = root / "new/scene"
            original_args = upstream.sys.argv
            calls = []
            def main():
                calls.append(list(upstream.sys.argv))
                self.raw_export(output, captions)
            module = SimpleNamespace(main=main, geometry=self.geometry, urllib=SimpleNamespace(
                request=SimpleNamespace(Request=urllib.request.Request, urlopen=urllib.request.urlopen), error=urllib.error))
            with patch.object(upstream, "external", return_value=module):
                metadata = scene.export_scene(map_file, captions, output, root / "adapter",
                                              model="fixture", base_url="https://invalid.test")
            self.assertIs(upstream.sys.argv, original_args)
            self.assertEqual(len(calls), 1)
            self.assertIn(str(captions), calls[0])
            self.assertEqual(metadata["node_count"], 2)
            self.assertEqual(metadata["relation_prompt_policy"], "Existing Qwen replacement adapter, unchanged")

    def test_completed_snapshot_validation_and_tampering(self):
        for mutation in (None, "map", "caption", "adapter"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                summary, map_file, captions = self.fixture(root)
                if mutation == "map":
                    map_file.write_bytes(b"modified")
                elif mutation == "caption":
                    path = captions / "objects/object_000.json"
                    value = json.loads(path.read_text())
                    value["qwen"]["confidence"] = .9
                    write(path, value)
                elif mutation == "adapter":
                    (root / "adapter/build_qwen_conceptgraph_baseline.py").write_text("changed")
                if mutation:
                    with self.assertRaises(ValueError):
                        rebuild.inspect_completed(summary)
                else:
                    self.assertEqual(rebuild.inspect_completed(summary)["caption_count"], 2)

    def test_graph_only_worker_and_existing_output_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary, _, _ = self.fixture(root)
            args = SimpleNamespace(from_summary=summary, output_dir=root / "new")
            def stage(name, command, target, *unused):
                self.assertEqual(name, "graph")
                self.assertIn("graph-worker", command)
                self.assertNotIn("caption-worker", command)
                write(target / "scene/scene_metadata.json", {"node_count": 2})
                write(target / "pipeline/pipeline_scene.json", {"execution_ready": False})
                return {"stage": name, "exit_code": 0}
            with patch.object(rebuild.sys, "platform", "linux"), patch.dict(rebuild.os.environ, {"DASHSCOPE_API_KEY": "offline"}), \
                    patch.object(real_scene, "run_stage", side_effect=stage) as mock:
                self.assertEqual(rebuild.rebuild(args), 0)
                report = json.loads((args.output_dir / "run_summary.json").read_text())
                self.assertTrue(report["inputs_unchanged"])
                self.assertEqual(report["reused_caption_count"], 2)
                with self.assertRaises(FileExistsError):
                    rebuild.rebuild(args)
                self.assertEqual(mock.call_count, 1)

    def test_transport_observer_preserves_payload_success_and_http_error_body(self):
        for failed in (False, True):
            with self.subTest(failed=failed), tempfile.TemporaryDirectory() as tmp:
                payload = b'{"messages":[],"model":"fixture"}'
                request = urllib.request.Request("https://invalid.test", data=payload,
                                                 headers={"Authorization": "Bearer offline-secret"})
                def urlopen(req, **kwargs):
                    self.assertIs(req, request)
                    self.assertEqual(req.data, payload)
                    if failed:
                        raise urllib.error.HTTPError(req.full_url, 429, "fixture", {}, io.BytesIO(b"retry later"))
                    return io.BytesIO(b'{"choices":[]}')
                module = SimpleNamespace(urllib=SimpleNamespace(request=SimpleNamespace(
                    Request=urllib.request.Request, urlopen=urlopen), error=urllib.error))
                upstream.attach_transport_audit(module, Path(tmp))
                if failed:
                    with self.assertRaises(urllib.error.HTTPError) as caught:
                        module.urllib.request.urlopen(request)
                    self.assertEqual(caught.exception.read(), b"retry later")
                else:
                    with module.urllib.request.urlopen(request) as response:
                        self.assertEqual(response.read(), b'{"choices":[]}')
                self.assertNotIn("offline-secret", "".join(p.read_text() for p in Path(tmp).rglob("*.json")))


if __name__ == "__main__":
    unittest.main()
