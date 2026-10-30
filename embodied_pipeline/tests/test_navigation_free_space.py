"""Static endpoint tests, including the actual G5 obstacle incident. No motion/API."""
import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from embodied_pipeline._legacy import ROOT
from embodied_pipeline.execution.navigation_free_space import (
    candidates, free_approach, point_check, validate_endpoint, scene_grid)
from embodied_pipeline.execution.navigation_points import bbox_distance
from embodied_pipeline.execution.navigation_contract import sha
from embodied_pipeline.ui.field_backend import load_scene, DEFAULT_SCENE
from embodied_pipeline.ui.navigation_controls import build_config


def empty_grid():
    return SimpleNamespace(origin=(-50.,-50.),res=.1,w=1000,h=1000,occ=bytearray(1000000),
                           sha256='synthetic_empty_grid_test_only',path=Path('synthetic_test_only.npz'))


class FreeEndpointTests(unittest.TestCase):
    def test_old_g5_occupied_and_alternative_clear_without_changing_distance(self):
        scene=load_scene(DEFAULT_SCENE);grid=scene_grid(scene)
        folder=ROOT/'outputs/real_scene_runs/tcp_navigation_20261030_145151_4606e984'
        cfg=json.loads((folder/'dispatch_config.json').read_text(encoding='utf-8'))
        old=cfg['approach_pose'];node=next(n for n in scene['nodes'] if n['id']==140)
        self.assertEqual(point_check(grid,old)['reason'],'target_cell_occupied')
        with self.assertRaisesRegex(ValueError,'栅格检查'):
            build_config(Path(cfg['engineering_plan']),cfg['engineering_plan_sha256'],scene,
                         cfg['start_pose'],old,cfg['confirmations'],2,.4)
        new=free_approach(node,cfg['start_pose'],.4,grid)
        self.assertNotEqual(new,old);self.assertTrue(point_check(grid,new)['free'])
        self.assertAlmostEqual(bbox_distance(node,new),.4)
        checked=build_config(Path(cfg['engineering_plan']),cfg['engineering_plan_sha256'],scene,
                             cfg['start_pose'],new,cfg['confirmations'],2,.4)
        self.assertEqual(checked['limits'],cfg['limits'])
        self.assertEqual(checked['navigation_point_policy']['navigation_grid_screening']['map_sha256'],grid.sha256)

    def test_no_candidate_fails_instead_of_returning_old_unsafe_goal(self):
        scene=load_scene(DEFAULT_SCENE);node=next(n for n in scene['nodes'] if n['id']==123)
        with self.assertRaisesRegex(ValueError,'均未通过'):
            free_approach(node,{'x':0,'y':0,'yaw':0},.4,scene_grid(scene))

    def test_clear_cell_is_not_enough_footprint_and_map_boundary_checked(self):
        g=empty_grid();g.occ[500*g.w+502]=1
        self.assertEqual(point_check(g,{'x':0,'y':0,'yaw':0})['reason'],'footprint_intersects_obstacle')
        self.assertFalse(point_check(g,{'x':49.9,'y':0,'yaw':0})['free'])
        self.assertTrue(point_check(g,{'x':-1,'y':0,'yaw':0})['free'])
        with self.assertRaises(ValueError):validate_endpoint(g,{'x':float('nan'),'y':0,'yaw':0})

    def test_ring_stays_outside_large_rectangular_box_at_requested_distance(self):
        n={'bbox_center':[2,1,0],'bbox_extent':[8,1,2]}
        for q in candidates(n,{'x':-8,'y':0,'yaw':0},.4):
            self.assertAlmostEqual(bbox_distance(n,q),.4,places=7)

    def test_unrelated_scene_does_not_silently_use_619_map(self):
        with self.assertRaisesRegex(ValueError,'匹配的导航栅格'):
            scene_grid({'metadata':{'scene_id':'Scene200','coordinate_frame':'scene'}})
