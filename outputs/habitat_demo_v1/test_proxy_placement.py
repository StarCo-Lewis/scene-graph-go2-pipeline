"""Offline adapter tests: mocks do not count as Habitat physics validation."""
import copy
import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from proxy_placement import (POLICY, candidate_offsets, footprint_samples,
                             descent_probe_positions, blocking_contacts, yaw_quaternion,
                             yaw_rotated_bounds, is_upright_rotation, same_upright_rotation)
from run_continuous_sequence import HabitatBackend
from sequential_proxy_core import SerialExecutor, ExecutionRejected
from run_continuous_regression import evaluate_case


def quaternion(vector, scalar):
    return NS(vector=list(vector), scalar=scalar)


def contact(other, distance=-.01):
    return NS(object_id_a=1, object_id_b=other, contact_distance=distance,
              position_on_a_in_ws=[-5.8, .95, -1.5], position_on_b_in_ws=[-5.8, .95, -1.5])


class FakeSimulation:
    def __init__(self, obj):
        self.obj = obj
        self.steps = 0
        self.detected_poses = []
        self.block_mode = "none"
        self.runtime_support = 0
        self.ray_obstacle = False
        self.throw_probe = False
        self.release_start = None

    def get_agent(self, index):
        state = NS(position=[-5.8, .1, -1.7], rotation=NS(real=1., imag=[0., 0., 0.]))
        return NS(get_state=lambda: state)

    def cast_ray(self, ray, max_distance):
        # The object can occur in its own ray. It must be filtered, not the stage.
        return NS(hits=[NS(object_id=1, point=[ray.origin[0], 1.1, ray.origin[2]]),
                        NS(object_id=-1 if self.ray_obstacle else 0,
                           point=[ray.origin[0], .95, ray.origin[2]])])

    def perform_discrete_collision_detection(self):
        if self.obj.motion_type == "DYNAMIC":
            self.detected_poses.append(list(map(float, self.obj.translation)))
            if self.throw_probe:
                raise RuntimeError("probe API failed")

    def get_physics_contact_points(self):
        if self.steps:
            return [contact(self.runtime_support, 0.)]
        if self.obj.motion_type != "DYNAMIC":
            return []
        if self.block_mode == "all":
            return [contact(-1)]
        if self.block_mode == "yaw_only" and abs(float(self.obj.rotation.vector[1]) - math.sqrt(.5)) > 1e-6:
            return [contact(-1)]
        if self.block_mode == "fine_gap" and abs(float(self.obj.translation[2]) + 1.525) > 1e-6:
            return [contact(-1)]
        # Clear rays, but an obstacle intersects the whole body near its bottom.
        if self.block_mode == "center_low" and abs(float(self.obj.translation[2]) + 1.5) < 1e-6 and float(self.obj.translation[1]) < 1.08:
            return [contact(-1)]
        return []

    def step_physics(self, dt):
        if self.release_start is None:
            self.release_start = {"rotation": copy.deepcopy(self.obj.rotation),
                "linear_velocity": list(map(float, self.obj.linear_velocity)),
                "angular_velocity": list(map(float, self.obj.angular_velocity)),
                "position": list(map(float, self.obj.translation))}
        self.steps += 1
        self.obj.translation = [float(self.obj.translation[0]), 1.033, float(self.obj.translation[2])]


def backend_fixture():
    obj = NS(object_id=1, translation=[-5.27, 1.35, -1.91],
             rotation=quaternion([-.023, .071, -.479], .875),
             linear_velocity=[.1, 0., .2], angular_velocity=[0., .2, .3], awake=False,
             motion_type="KINEMATIC",
             collision_shape_aabb=NS(min=[-.06, -.1, -.09], max=[.06, .1, .09]))
    backend = object.__new__(HabitatBackend)
    backend.instances = {67: obj, 13: NS(object_id=0, translation=[-5.8, .5, -1.5],
        rotation=quaternion([0., 0., 0.], 1.), linear_velocity=[0., 0., 0.],
        angular_velocity=[0., 0., 0.], motion_type="STATIC")}
    backend.sim = FakeSimulation(obj)
    backend.motion = NS(DYNAMIC="DYNAMIC", KINEMATIC="KINEMATIC")
    backend.hs = NS(geo=NS(Ray=lambda: NS()))
    backend.nodes = {13: {"bbox_center": [-5.8, 1.5, .5], "bbox_extent": [.4, 1.2, .9]}}
    backend.settle_seconds = 2.
    backend.support = {}
    backend.active_ids = backend.mapped_ids = {67, 13}
    backend.object_offsets = {67: [0., .08619, 0.], 13: [0., 0., 0.]}
    backend.memory = {"version": 6}
    backend.reach = 1.5
    # This fixture isolates release/support tests; motion guard has separate tests.
    backend._move = lambda *args, **kwargs: {"status": "not_tested_in_release_fixture"}
    return backend, obj


def placement_action():
    return {"step_id": "place", "skill_id": "place_object_on_surface",
            "arguments": {"object": {"role_id": "kettle", "element_id": 67},
                          "surface": {"role_id": "sideboard", "element_id": 13}}}


ALIGNMENT = {"context_bindings": {"kettle": {"element_ids": [67]}, "sideboard": {"element_ids": [13]}}}
MAGNUM = NS(Vector3=lambda *values: list(values), Quaternion=quaternion)


class ProxyPlacementTests(unittest.TestCase):
    def test_offsets_are_bounded_and_fit_full_collider(self):
        lo, hi = [-.06, -.1, -.09], [.06, .1, .09]
        offsets = candidate_offsets([.4, 1.2], lo, hi)
        self.assertEqual(offsets[0], (0., 0.))
        for dx, dz in offsets:
            self.assertLessEqual(math.hypot(dx, dz), .30 + 1e-9)
            self.assertGreaterEqual(dx + lo[0], -.2 + POLICY["surface_edge_margin_m"])
            self.assertLessEqual(dz + hi[2], .6 - POLICY["surface_edge_margin_m"])
        self.assertEqual(len(footprint_samples(lo, hi)), 9)
        self.assertEqual(candidate_offsets([.1, .1], lo, hi), [])

    def test_invalid_geometry_is_not_replaced_by_guessed_bounds(self):
        with self.assertRaisesRegex(ValueError, "COLLISION_BOUNDS"):
            candidate_offsets([.4, 1.2], [float("nan"), -.1, -.1], [.1, .1, .1])

    def test_missing_bullet_bounds_reject_before_any_search_or_release(self):
        backend, obj = backend_fixture()
        del obj.collision_shape_aabb
        original = copy.deepcopy(vars(obj))
        with self.assertRaisesRegex(ExecutionRejected, "COLLISION_BOUNDS_UNAVAILABLE") as raised:
            backend._choose_placement(67, 13, quaternion([0., 0., 0.], 1.))
        self.assertFalse(raised.exception.details["release_performed"])
        self.assertEqual(vars(obj), original)
        self.assertEqual(backend.sim.steps, 0)

    def test_probe_positions_cover_short_descent(self):
        poses = descent_probe_positions(0., 0., .95, -.1)
        self.assertAlmostEqual(poses[0][1] - .1 - .95, .04)
        self.assertAlmostEqual(poses[-1][1] - .1 - .95, .005)
        self.assertTrue(all(0 < a[1] - b[1] <= .01 + 1e-9 for a, b in zip(poses, poses[1:])))

    def test_yaw_rotation_swaps_footprint_without_tilting(self):
        lo, hi = yaw_rotated_bounds([-.06, -.1, -.09], [.06, .1, .09], 90.)
        for observed, expected in zip(lo, [-.09, -.1, -.06]):
            self.assertAlmostEqual(observed, expected)
        for observed, expected in zip(hi, [.09, .1, .06]):
            self.assertAlmostEqual(observed, expected)
        for yaw in POLICY["allowed_release_yaws_deg"]:
            self.assertTrue(is_upright_rotation(yaw_quaternion(yaw)))
        self.assertFalse(is_upright_rotation([.875, -.023, .071, -.479]))
        self.assertFalse(is_upright_rotation([2., 0., 0., 0.]))
        self.assertFalse(is_upright_rotation(None))

    def test_float32_release_quaternion_is_not_rejected_by_report_comparison(self):
        expected = yaw_quaternion(90.)
        observed = [.70710676908493, 0., .70710676908493, 0.]
        self.assertTrue(same_upright_rotation(observed, expected))
        self.assertTrue(same_upright_rotation([-v for v in observed], expected))
        self.assertFalse(same_upright_rotation(observed, yaw_quaternion(0.)))

    def test_yaw_alternative_is_probed_and_used_for_actual_release(self):
        backend, obj = backend_fixture()
        backend.sim.block_mode = "yaw_only"
        with patch.dict(sys.modules, {"magnum": MAGNUM}):
            report = backend.place(67, 13)
        self.assertEqual(report["placement_preparation"]["release_yaw_deg"], 90.)
        self.assertEqual(report["surface_candidates"][0]["reason"], "volume_probe_blocked")
        released = backend.sim.release_start["rotation"]
        self.assertAlmostEqual(released.scalar, math.sqrt(.5))
        self.assertAlmostEqual(released.vector[1], math.sqrt(.5))
        self.assertEqual(report["status"], "proxy_instance_support_passed")

    def test_finer_grid_finds_gap_between_old_five_cm_candidates(self):
        backend, obj = backend_fixture()
        backend.sim.block_mode = "fine_gap"
        report = backend._choose_placement(67, 13, quaternion([0., 0., 0.], 1.))
        self.assertAlmostEqual(report["release_position"][2], -1.525)
        self.assertTrue(report["surface_candidates"][-1]["valid"])
        self.assertEqual(backend.sim.steps, 0)

    def test_stage_and_unrelated_objects_are_obstacles_not_support(self):
        self.assertEqual(blocking_contacts([contact(-1), contact(7)], 1, 0)[0]["other_object_id"], -1)
        self.assertEqual(len(blocking_contacts([contact(-1), contact(7)], 1, 0)), 2)
        self.assertEqual(blocking_contacts([contact(0, 0.)], 1, 0), [])
        self.assertEqual(blocking_contacts([contact(0, -.01)], 1, 0)[0]["reason"], "support_penetration")
        self.assertTrue(blocking_contacts([contact(-1, float("nan"))], 1, 0))
        self.assertEqual(blocking_contacts([contact(-1, .05)], 1, 0), [])

    def test_clear_rays_but_blocked_lower_volume_try_next_candidate(self):
        backend, obj = backend_fixture()
        backend.sim.block_mode = "center_low"
        original = copy.deepcopy(vars(obj))
        report = backend._choose_placement(67, 13, quaternion([0., 0., 0.], 1.))
        self.assertEqual(report["surface_candidates"][0]["reason"], "volume_probe_blocked")
        self.assertEqual(len(report["surface_candidates"][0]["rays"]), 9)
        self.assertNotEqual(report["release_position"][2], -1.5)
        self.assertEqual(vars(obj), original)
        self.assertEqual(backend.sim.steps, 0)
        self.assertEqual(backend.memory["version"], 6)

    def test_no_safe_candidate_restores_pose_velocities_and_held_state(self):
        backend, obj = backend_fixture()
        backend.sim.block_mode = "all"
        original = copy.deepcopy(vars(obj))
        executor = SerialExecutor(backend)
        executor.held_object_id, executor.lifted, executor.prepared_target_id = 67, True, 13
        with patch.dict(sys.modules, {"magnum": MAGNUM}), self.assertRaisesRegex(ExecutionRejected, "NO_CLEAR_SUPPORT") as raised:
            executor.execute(placement_action(), ALIGNMENT)
        self.assertFalse(raised.exception.details["release_performed"])
        self.assertEqual(vars(obj), original)
        self.assertEqual(backend.sim.steps, 0)
        self.assertEqual(executor.held_object_id, 67)
        self.assertTrue(executor.lifted)
        self.assertEqual(backend.support, {})

    def test_probe_exception_still_restores_object(self):
        backend, obj = backend_fixture()
        original = copy.deepcopy(vars(obj))
        backend.sim.throw_probe = True
        with self.assertRaisesRegex(RuntimeError, "probe API"):
            backend._choose_placement(67, 13, quaternion([0., 0., 0.], 1.))
        self.assertEqual(vars(obj), original)
        self.assertEqual(backend.sim.steps, 0)

    def test_stage_ray_is_never_filtered_as_self(self):
        backend, obj = backend_fixture()
        backend.sim.ray_obstacle = True
        with self.assertRaisesRegex(ExecutionRejected, "NO_CLEAR_SUPPORT") as raised:
            backend._choose_placement(67, 13, quaternion([0., 0., 0.], 1.))
        self.assertTrue(all(r["rays"][0]["first_hit_object_id"] == -1
                            for r in raised.exception.details["surface_candidates"]))

    def test_actual_release_uses_upright_pose_and_zero_velocities(self):
        backend, obj = backend_fixture()
        with patch.dict(sys.modules, {"magnum": MAGNUM}):
            report = backend.place(67, 13)
        released = backend.sim.release_start
        self.assertEqual(released["rotation"], quaternion([0., 0., 0.], 1.))
        self.assertEqual(released["linear_velocity"], [0., 0., 0.])
        self.assertEqual(released["angular_velocity"], [0., 0., 0.])
        self.assertEqual(report["status"], "proxy_instance_support_passed")
        self.assertEqual(report["direct_support_contact_frames"], 60)
        self.assertEqual(backend.support, {67: 13})
        self.assertEqual(obj.motion_type, "KINEMATIC")

    def test_stage_only_after_release_still_fails_and_clears_held_state(self):
        backend, obj = backend_fixture()
        backend.sim.runtime_support = -1
        executor = SerialExecutor(backend)
        executor.held_object_id, executor.lifted, executor.prepared_target_id = 67, True, 13
        with patch.dict(sys.modules, {"magnum": MAGNUM}), self.assertRaisesRegex(ExecutionRejected, "SUPPORT_NOT_VERIFIED") as raised:
            executor.execute(placement_action(), ALIGNMENT)
        self.assertEqual(raised.exception.details["validation"]["failed_checks"], ["direct_support_contact_frames"])
        self.assertTrue(raised.exception.details["release_performed"])
        self.assertIsNone(executor.held_object_id)
        self.assertFalse(executor.lifted)
        self.assertEqual(backend.support, {})
        self.assertEqual(backend.memory["version"], 6)

    def test_regression_requires_new_preflight_and_release_evidence(self):
        backend, obj = backend_fixture()
        with patch.dict(sys.modules, {"magnum": MAGNUM}):
            placement = dict(backend.place(67, 13), skill_id="place_object_on_surface")
        summary = {"status": "passed", "simulator_creation_count": 1, "initial_start_sampling_count": 1,
                   "uninterrupted_simulator_execution": True, "executed_action_count": 16,
                   "base_graph_unchanged": True, "cross_round_held_object_continuations": [3, 5],
                   "training_eligible": False, "rounds": [
            {"status": "passed", "actions": [{"target_id": t, "target_center": "original"} for t in (67, 13, 67)]},
            {"status": "passed", "actions": []}, {"status": "passed", "actions": [placement]},
            {"status": "passed", "actions": [{"target_center": "moved"}]},
            {"status": "passed", "actions": [copy.deepcopy(placement)]}]}
        self.assertTrue(all(evaluate_case(summary, 0, "none").values()))
        last = summary["rounds"][-1]["actions"][0]
        last["at_release"]["rotation_wxyz"] = [.875, -.023, .071, -.479]
        self.assertFalse(evaluate_case(summary, 0, "none")["upright_zero_velocity_releases"])
        last["surface_candidates"][0]["volume_probes"][0]["blocking_contacts"] = [{"other_object_id": -1}]
        self.assertFalse(evaluate_case(summary, 0, "none")["volume_checked_placements"])
        last["validation"]["checks"]["direct_support_contact_frames"] = False
        self.assertFalse(evaluate_case(summary, 0, "none")["two_support_verified_placements"])


if __name__ == "__main__":
    unittest.main()
