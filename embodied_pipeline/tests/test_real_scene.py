"""Recorded-data contract tests; no live models, GPU tasks or API requests."""

import copy
import gzip
import json
import os
from pathlib import Path
import pickle
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from embodied_pipeline.perception import real_rgbd as rgbd
from embodied_pipeline.perception import conceptgraphs_scene as scene
from embodied_pipeline.perception import world_frames as frames


class RealSceneTests(unittest.TestCase):
    def setUp(self):
        self.http = patch("urllib.request.urlopen", side_effect=AssertionError("Live API forbidden"))
        self.http.start()
        self.addCleanup(self.http.stop)

    def make_input(self, root):
        source = root / "input"
        rgb = source / "rgbd_conceptgraphs"
        results = rgb / "results"
        results.mkdir(parents=True)
        config = """dataset_name: replica
camera_params:
  image_height: 3
  image_width: 4
  fx: 100
  fy: 100
  cx: 2
  cy: 1
  png_depth_scale: 1000
  crop_edge: 0
"""
        (rgb / "replica_config.yaml").write_text(config)
        pose = np.eye(4)
        pose[:3, 3] = [2, 3, 1]
        np.savetxt(rgb / "traj.txt", pose.reshape(1, 16))
        rgbd.write(rgb / "manifest.json", {"frames": [{"index": 0}]})
        Image.fromarray(np.full((3, 4, 3), 130, dtype=np.uint8)).save(results / "frame000000.jpg")
        depth = np.full((3, 4), 1234, dtype=np.uint16)
        Image.fromarray(depth).save(results / "depth000000.png")
        mask = np.zeros((3, 4), dtype=np.uint8)
        mask[0, 0] = 255
        Image.fromarray(mask).save(results / "mask000000.png")
        return source, pose

    def test_depth_mask_scale_and_world_pose_are_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, pose = self.make_input(root)
            original = rgbd.sha(source / "rgbd_conceptgraphs/results/depth000000.png")
            report = rgbd.audit(source, root / "output", prepare=True)
            prepared = root / "output/prepared"
            with Image.open(prepared / "dataset/depth/000000.png") as image:
                depth = np.asarray(image)
            self.assertEqual(depth[0, 0], 0)
            self.assertEqual(depth[1, 1], 1234)
            self.assertEqual(report["camera"]["png_depth_scale"], 1000)
            np.testing.assert_allclose(np.loadtxt(prepared / "dataset/pose/000000.txt"), pose)
            self.assertEqual(rgbd.sha(source / "rgbd_conceptgraphs/results/depth000000.png"), original)
            self.assertEqual(rgbd.validate_prepared(prepared)["frame_count"], 1)
            self.assertFalse(report["conceptgraphs_executed"])
            with self.assertRaises(FileExistsError):
                rgbd.audit(source, root / "output", prepare=True)
            with self.assertRaises(ValueError):
                rgbd.audit(source, source / "bad", prepare=True)

    def test_reject_frame_mismatch_bad_pose_and_mutated_prepared_data(self):
        for kind in ("frame", "pose", "prepared"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                source, pose = self.make_input(root)
                if kind == "frame":
                    Image.new("RGB", (4, 3)).save(source / "rgbd_conceptgraphs/results/frame000001.jpg")
                elif kind == "pose":
                    pose[0, 0] = -1
                    np.savetxt(source / "rgbd_conceptgraphs/traj.txt", pose.reshape(1, 16))
                if kind != "prepared":
                    with self.assertRaises(ValueError):
                        rgbd.inspect_input(source)
                else:
                    rgbd.audit(source, root / "output", prepare=True)
                    (root / "output/prepared/dataset/pose/000000.txt").write_text("modified")
                    with self.assertRaisesRegex(ValueError, "modified"):
                        rgbd.validate_prepared(root / "output/prepared")

    def test_export_full_precision_inactive_ids_and_actual_interface(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            objects = [dict(pcd_np=np.array([[i, 0, 0], [i+.012, .023, .034]], dtype=float),
                            clip_ft=np.array([i, 1])) for i in range(3)]
            with gzip.open(root / "map.pkl.gz", "wb") as stream:
                pickle.dump({"objects": objects}, stream)
            for i in range(3):
                rgbd.write(root / f"captions/objects/object_{i:03d}.json", dict(id=i, status="SUCCESS",
                    qwen=dict(object_tag="cup" if i == 0 else "table", caption="A mapped region.",
                        possible_tags=["object"], entity_type="object" if i != 1 else "object_part",
                        is_complete_object=i != 1, confidence=.9)))
            def geometry(obj):
                low, high = obj["pcd_np"].min(axis=0), obj["pcd_np"].max(axis=0)
                return {"center": (low+high)/2, "extent": high-low}
            fake = SimpleNamespace(geometry=geometry,
                compute_overlap_matrix=lambda g, d: np.ones((len(g), len(g))),
                build_upstream_mst=lambda o, t: (None, [[0, 1]], [dict(object1_id=0, object2_id=1, overlap=.2)]))
            seen = []
            def query(value, *args, **kwargs):
                seen.append(value)
                return dict(object_relation="a on b", reason="A test fixture relation.", usage=None, verified=False)
            with patch.object(scene, "external", return_value=fake):
                report = scene.export_filtered_scene_legacy(root / "map.pkl.gz", root / "captions", root / "scene", root,
                    model="offline", base_url="https://invalid.test", query=query)
            bundle = scene.load_real_scene(root / "scene")
            self.assertEqual([n["id"] for n in bundle["nodes"]], [0, 1, 2])
            self.assertFalse(bundle["nodes"][1]["active"])
            self.assertAlmostEqual(bundle["nodes"][0]["bbox_extent"][0], .012)
            self.assertEqual((seen[0]["object1"]["id"], seen[0]["object2"]["id"]), (0, 2))
            self.assertEqual(bundle["edges"][0]["target"], 2)
            self.assertFalse(bundle["edges"][0]["verified"])
            imported = scene.import_scene(root / "scene", root / "pipeline")
            self.assertFalse(imported["execution_ready"])
            self.assertFalse(imported["planning_state_created"])
            self.assertEqual(imported["integration_target"], "real_robot")
            self.assertFalse(imported["simulation_required"])
            coordinate = json.loads((root / "pipeline/coordinate_contract.json").read_text())
            self.assertFalse(coordinate["axis_conversion_applied"])
            self.assertFalse(coordinate["simulation_coordinate_conversion_required"])
            self.assertFalse((root / "pipeline/coordinate_adapter.json").exists())
            self.assertEqual(report["node_count"], 3)
            self.assertEqual(report["active_node_count"], 2)
            path = root / "scene/scene_graph_edges.json"
            rgbd.write(path, [dict(source=0, target=1, relation="a on b")])
            with self.assertRaises(ValueError):
                scene.load_real_scene(root / "scene")

    def test_caption_and_relation_contracts_do_not_assume_physical_execution(self):
        self.assertIn("z UP", scene.RELATION_PROMPT)
        self.assertIn("none of these", scene.RELATION_PROMPT)
        self.assertIn("Never infer", scene.RELATION_PROMPT)
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "DASHSCOPE_API_KEY"):
                scene.relation_query({}, Path("unused"), model="offline", base_url="https://invalid.test")

    def test_frame_transform_matches_existing_proxy_and_round_trips(self):
        from embodied_pipeline._legacy import load
        point = [2, 3, 1]
        converted = frames.position(point, rgbd.WORLD, frames.PROXY_WORLD)
        self.assertEqual(converted, [2, 1, -3])
        self.assertEqual(converted, load("run_navigation_demo").cg_to_habitat(point, "x_z_minus_y"))
        self.assertEqual(frames.position(converted, frames.PROXY_WORLD, rgbd.WORLD), point)
        self.assertEqual(frames.extent([.2,.3,.4], rgbd.WORLD, frames.PROXY_WORLD), [.2,.4,.3])
        rotation = np.asarray(frames.contract()["world_transform"])[:3,:3]
        self.assertAlmostEqual(np.linalg.det(rotation), 1)
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(3))
        with self.assertRaises(ValueError):
            frames.position(point, "undeclared", frames.PROXY_WORLD)


if __name__ == "__main__":
    unittest.main()
