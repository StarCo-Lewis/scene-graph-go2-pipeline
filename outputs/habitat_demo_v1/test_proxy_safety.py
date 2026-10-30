"""Offline guard tests; not evidence of live Habitat collision success."""
import copy
import math
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import proxy_safety as safety
from sequential_proxy_core import ExecutionRejected


def q(vector, scalar):
    return NS(vector=list(vector), scalar=scalar)


class Sim:
    def __init__(self, obj):
        self.obj, self.steps, self.probes = obj, 0, []
        self.wall, self.thin_wall, self.fail_api, self.camera_wall = False, False, False, False
    def perform_discrete_collision_detection(self):
        if self.fail_api:
            self.fail_api = False
            raise RuntimeError('Bullet query unavailable')
        self.probes.append(list(self.obj.translation))
    def get_physics_contact_points(self):
        if self.wall and .4 < self.obj.translation[0] < .6:
            return [NS(object_id_a=1, object_id_b=-1, contact_distance=-.01)]
        return []
    def cast_ray(self, ray, max_distance):
        hits = []
        if self.thin_wall and abs(ray.direction[0]) > 1e-8:
            t = (.411-ray.origin[0])/ray.direction[0]
            if 0 < t <= max_distance:
                hits.append(NS(object_id=-1, point=[ray.origin[i]+t*ray.direction[i] for i in range(3)]))
        # A wall behind the robot blocks a long camera boom, not the short one.
        if self.camera_wall and ray.direction[2] > 1e-8:
            t = (.5-ray.origin[2])/ray.direction[2]
            if 0 < t <= max_distance:
                hits.append(NS(object_id=-1, point=[ray.origin[i]+t*ray.direction[i] for i in range(3)]))
        return NS(hits=hits)


def fixture():
    obj = NS(object_id=1, translation=[0., 1.35, 0.], rotation=q([0.,0.,0.],1.), motion_type='KINEMATIC',
             linear_velocity=[0.,0.,0.], angular_velocity=[0.,0.,0.], awake=False,
             collision_shape_aabb=NS(min=[-.05,-.1,-.05], max=[.05,.1,.05]))
    backend = NS(instances={67:obj}, motion=NS(DYNAMIC='DYNAMIC'), hs=NS(geo=NS(Ray=lambda:NS())), sim=Sim(obj))
    return backend, obj


class SafetyTests(unittest.TestCase):
    def setUp(self):
        self.vector = patch.object(safety, 'habitat_vector', side_effect=lambda values: list(values))
        self.vector.start()
        self.magnum = patch.dict(sys.modules, {'magnum': NS(Vector3=lambda *v:list(v), Quaternion=q)})
        self.magnum.start()
    def tearDown(self):
        self.magnum.stop(); self.vector.stop()

    def test_spawn_center_uses_recentered_collider_not_old_origin_offset(self):
        b,o=fixture()
        o.collision_shape_aabb=NS(min=[-.070491977,-.088842116,-.095406868],
                                  max=[.070491977,.088842116,.095406868])
        result=safety.collision_center_registration(o,[-2.7,1.4,-8.])
        self.assertEqual(result['translation'],[-2.7,1.4,-8.])
        self.assertEqual(result['graph_center_offset_from_rigid_origin'],[0.,0.,0.])
        self.assertAlmostEqual(result['translation'][1]+o.collision_shape_aabb.min[1],1.311157884)
        self.assertFalse(result['physical_com_measured'])

    def test_noncentered_collider_derived_offset_not_assumed_zero(self):
        b,o=fixture(); o.collision_shape_aabb=NS(min=[-.05,0.,-.05],max=[.05,.2,.05])
        result=safety.collision_center_registration(o,[0.,1.4,0.])
        self.assertAlmostEqual(result['translation'][1],1.3)
        self.assertAlmostEqual(result['graph_center_offset_from_rigid_origin'][1],.1)

    def test_spawn_center_invalid_bounds_fail_closed(self):
        b,o=fixture(); o.collision_shape_aabb=NS(min=[0.,0.,0.],max=[0.,.2,.1])
        with self.assertRaisesRegex(ValueError,'COLLISION_BOUNDS'):
            safety.collision_center_registration(o,[0.,1.4,0.])

    def test_initial_penetration_not_relabelled_as_safe_support(self):
        b,o=fixture(); original=copy.deepcopy(vars(o))
        b.sim.get_physics_contact_points=lambda:[NS(object_id_a=1,object_id_b=-1,contact_distance=-.11689,
            position_on_a_in_ws=[0.,1.2,0.],position_on_b_in_ws=[0.,1.31,0.],
            contact_normal_on_b_in_ws=[0.,1.,0.])]
        with self.assertRaises(ExecutionRejected) as caught:
            safety.checked_motion(b,67,list(o.translation),'initial_pose_validation',commit=False)
        self.assertEqual(caught.exception.details['blocked_probe_index'],0)
        self.assertIn('contact_normal_on_b_in_ws',caught.exception.details['blockers'][0])
        self.assertEqual(vars(o),original)

    def test_backend_initial_spawn_clear_report_saved(self):
        self.check_backend_registration(False)

    def test_backend_initial_spawn_reject_report_saved_before_actions(self):
        self.check_backend_registration(True)

    def check_backend_registration(self, blocked):
        from run_continuous_sequence import HabitatBackend
        b,o=fixture()
        if blocked:
            b.sim.get_physics_contact_points=lambda:[NS(object_id_a=1,object_id_b=-1,contact_distance=-.1)]
        def register(backend):
            backend.instances={67:o}
            backend.registration_report['objects']['67']=safety.collision_center_registration(o,[0.,1.35,0.])
        modules={'habitat_sim':b.hs,'habitat_sim.physics':NS(MotionType=b.motion)}
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules,modules), \
             patch.object(HabitatBackend,'_register',register):
            root=Path(directory)
            if blocked:
                with self.assertRaisesRegex(ExecutionRejected,'INITIAL_OBJECT_POSE_REJECTED'):
                    HabitatBackend(b.sim,[],{'version':0},root,1.5,500,10.)
            else:
                backend=HabitatBackend(b.sim,[],{'version':0},root,1.5,500,10.)
                self.assertFalse(backend.motion_evidence[0]['committed'])
            report=json.loads((root/'initial_asset_registration.json').read_text())
            self.assertEqual(report['status'],'initial_pose_rejected' if blocked else 'initial_pose_sampled_clear')
            self.assertEqual(b.sim.steps,0)
    def test_samples_include_endpoints_and_bounded_steps(self):
        poses = safety.motion_samples([0,0,0],[1,0,0],[1,0,0,0],[1,0,0,0])
        self.assertEqual(poses[0][0],[0,0,0]); self.assertEqual(poses[-1][0],[1,0,0])
        self.assertTrue(all(math.dist(a[0],b[0]) <= .0200001 for a,b in zip(poses,poses[1:])))
    def test_rotation_slerp_and_sign_equivalence(self):
        poses = safety.motion_samples([0,0,0],[0,0,0],[1,0,0,0],[0,0,1,0])
        self.assertEqual(len(poses),37)
        self.assertTrue(all(abs(sum(v*v for v in rot)-1)<1e-8 for _,rot in poses))
        self.assertEqual(len(safety.motion_samples([0]*3,[0]*3,[1,0,0,0],[-1,0,0,0])),2)
    def test_clear_segment_commits_without_advancing_physics(self):
        b,o=fixture(); report=safety.checked_motion(b,67,[1.,1.35,0.],'carry')
        self.assertEqual(o.translation,[1.,1.35,0.]); self.assertTrue(report['committed'])
        self.assertEqual(o.motion_type,'KINEMATIC'); self.assertEqual(b.sim.steps,0)
    def test_readonly_preflight_restores_full_pose(self):
        b,o=fixture(); original=copy.deepcopy(vars(o))
        report=safety.checked_motion(b,67,[1.,1.35,0.],'pre_navigation',commit=False)
        self.assertEqual(vars(o),original); self.assertFalse(report['committed'])
    def test_wall_rejection_restores_no_partial_translation(self):
        b,o=fixture(); b.sim.wall=True; original=copy.deepcopy(vars(o))
        with self.assertRaisesRegex(ExecutionRejected,'PATH_BLOCKED') as caught:
            safety.checked_motion(b,67,[1.,1.35,0.],'carry')
        self.assertEqual(vars(o),original); self.assertEqual(caught.exception.details['blockers'][0]['other_object_id'],-1)
    def test_thin_wall_between_samples_rejected_by_swept_rays(self):
        b,o=fixture(); b.sim.thin_wall=True
        with self.assertRaisesRegex(ExecutionRejected,'PATH_BLOCKED'):
            safety.checked_motion(b,67,[1.,1.35,0.],'carry')
        self.assertEqual(o.translation,[0.,1.35,0.])
    def test_second_route_leg_blocked_does_not_commit_first_leg(self):
        b,o=fixture(); b.sim.wall=True
        with self.assertRaises(ExecutionRejected):
            safety.checked_motion(b,67,[1.,1.35,0.],'grasp',via=([0.,1.5,0.],[1.,1.5,0.]))
        self.assertEqual(o.translation,[0.,1.35,0.])
    def test_missing_bounds_fail_closed(self):
        b,o=fixture(); del o.collision_shape_aabb
        with self.assertRaisesRegex(ExecutionRejected,'CHECK_UNAVAILABLE'):
            safety.checked_motion(b,67,[1.,1.35,0.],'carry')
    def test_collision_api_failure_is_not_clear(self):
        b,o=fixture(); b.sim.fail_api=True
        with self.assertRaisesRegex(ExecutionRejected,'CHECK_UNAVAILABLE'):
            safety.checked_motion(b,67,[1.,1.35,0.],'carry')
        self.assertEqual(o.translation,[0.,1.35,0.])
    def test_no_probe_frames_rendered(self):
        b,o=fixture(); observed=[]
        b.audit=NS(capture=lambda *a,**k:observed.append((o.motion_type,list(o.translation))))
        safety.checked_motion(b,67,[.2,1.35,0.],'carry')
        self.assertTrue(observed); self.assertTrue(all(mode=='KINEMATIC' for mode,_ in observed))
    def test_camera_boom_shortened_before_wall(self):
        b,o=fixture(); b.sim.camera_wall=True
        result=safety.camera_pose(b.sim,b.hs,[0.,0.,0.],[1.,0.,0.,0.])
        self.assertEqual(result['boom_m'],.35); self.assertLess(result['eye'][2],.5)
        self.assertLess(result['eye'][1],2.)
    def test_camera_mast_blocked_no_unsafe_picture(self):
        b,o=fixture()
        def cast(ray,max_distance):
            return NS(hits=[NS(object_id=-1,point=[ray.origin[0],ray.origin[1]+.1,ray.origin[2]])])
        b.sim.cast_ray=cast
        with self.assertRaises(safety.AuditCameraRejected):
            safety.camera_pose(b.sim,b.hs,[0.,0.,0.],[1.,0.,0.,0.])
    def test_robot_navmesh_invalid_is_rejected(self):
        b,o=fixture(); b.sim.pathfinder=NS(is_navigable=lambda p:False)
        with self.assertRaisesRegex(ExecutionRejected,'NAVMESH_POSITION_REJECTED'):
            safety.robot_segment_check(b,[0,0,0],[1,0,0])

    def test_navigation_preflight_rejects_before_sim_step(self):
        import run_navigation_demo as nav
        state=NS(position=[0.,0.,0.])
        calls=[]
        sim=NS(get_agent=lambda _:NS(get_state=lambda:state),
               make_greedy_follower=lambda **_:NS(next_action_along=lambda _:'move_forward'),
               pathfinder=NS(find_path=lambda path:True), step=lambda action:calls.append(action))
        hs=NS(ShortestPath=lambda:NS(geodesic_distance=1.))
        def reject(action): raise ExecutionRejected('PROXY_OBJECT_PATH_BLOCKED')
        with patch.dict(sys.modules,{'habitat_sim':hs}), patch.object(nav,'habitat_vector',side_effect=lambda p:p):
            with self.assertRaises(ExecutionRejected): nav.navigate(sim,[1.,0.,0.],10,.1,25,before_action=reject)
        self.assertEqual(calls,[])


if __name__=='__main__': unittest.main()
