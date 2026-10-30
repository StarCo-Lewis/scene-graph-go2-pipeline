#!/usr/bin/env python3
"""Capture RGB-D at restored execution endpoint and fuse known-instance surface points."""
import argparse
import math
import shutil
from pathlib import Path
from run_memory_followup_navigation import restore_contract, SCENE
from run_navigation_demo import habitat_vector, peak_rss_mb
from update_memory import read, write, digest
from visual_fusion import project_pixel, fuse_frame


def upright_camera_rotation(direction):
    """Camera-to-world columns: right, up, backward; world Y is vertical."""
    def unit(v):
        length = math.sqrt(sum(x*x for x in v))
        if not math.isfinite(length) or length < 1e-8:
            raise ValueError('Degenerate camera direction or vertical look direction')
        return [x/length for x in v]
    def cross(a, b):
        return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]
    forward = unit(direction)
    right = unit(cross(forward, [0., 1., 0.]))
    up = cross(right, forward)
    return [[right[i], up[i], -forward[i]] for i in range(3)]


def scan_direction(direction, yaw_degrees):
    """Fixed world-Y yaw around the OLD target direction, never a new pose."""
    angle = math.radians(yaw_degrees)
    x, y, z = direction
    return [math.cos(angle)*x + math.sin(angle)*z, y,
            -math.sin(angle)*x + math.cos(angle)*z]


def same_physical_checkpoint(a, b):
    """Visual overlays may change metadata without changing physical state."""
    for key in ('base_graph_digest', 'coordinate_frame', 'version', 'edges'):
        if a.get(key) != b.get(key):
            return False
    if set(a['entities']) != set(b['entities']):
        return False
    return all(a['entities'][key]['active'] == b['entities'][key]['active']
        and a['entities'][key]['node']['bbox_center'] == b['entities'][key]['node']['bbox_center']
        and a['entities'][key]['last_observed_at'] == b['entities'][key]['last_observed_at'] for key in a['entities'])


def run(args):
    import numpy as np
    import quaternion
    import habitat_sim
    from habitat_sim.agent import AgentConfiguration
    from habitat_sim.physics import MotionType
    from habitat_sim.utils.common import quat_from_two_vectors
    state = read(args.memory_dir / 'memory_state.json')
    test_delta = getattr(args, 'test_translation_habitat', None)
    if test_delta is not None and (len(test_delta) != 3 or any(not math.isfinite(v) for v in test_delta)
                                  or not 0 < math.sqrt(sum(v*v for v in test_delta)) <= .75):
        raise ValueError('Controlled translation must be finite, nonzero and <=0.75m')
    original_digest = digest(state)
    nodes = read(args.memory_dir / 'scene/scene_graph.json')
    base = read(Path(state['base_scene']) / 'scene_graph.json')
    from visual_checkpoint import fixture_contract, actual_bbox_cg
    contract = fixture_contract(args.memory_dir, getattr(args, 'previous_visual_run', None))
    truth_before = actual_bbox_cg(contract)
    robot = contract['robot_position']
    if args.followup_execution:
        execution = read(args.followup_execution)
        source_state = read(Path(execution['memory_source']))
        if execution.get('status') != 'followup_proxy_passed' or not same_physical_checkpoint(source_state, state):
            raise ValueError('Visual observation endpoint is not a successful execution of this memory')
        robot = execution['navigation']['end']
    cfg = habitat_sim.SimulatorConfiguration()
    assets = SCENE / 'habitat_split_assets_v2'
    cfg.scene_id = str(assets / 'scene201_without_kettle.stage_config.json')
    cfg.create_renderer = True
    cfg.enable_physics = True
    specs = []
    for name, sensor_type in [('rgb', habitat_sim.SensorType.COLOR), ('depth', habitat_sim.SensorType.DEPTH), ('semantic', habitat_sim.SensorType.SEMANTIC)]:
        spec = habitat_sim.CameraSensorSpec()
        spec.uuid, spec.sensor_type = name, sensor_type
        spec.resolution = [256, 256]
        spec.position = [0., 0., 0.]
        spec.hfov = 90
        specs.append(spec)
    agent = AgentConfiguration()
    agent.sensor_specifications = specs
    args.output_dir.mkdir(parents=True, exist_ok=False)
    layer_path = args.memory_dir / 'visual_layer.json'
    layer = read(layer_path) if layer_path.exists() else None
    reports = []
    with habitat_sim.Simulator(habitat_sim.Configuration(cfg, [agent])) as sim:
        mgr = sim.get_object_template_manager()
        for role, node_id, position, file in [('kettle', 67, contract['kettle_com'], 'kettle_dynamic.object_config.json'),
                ('sideboard', 13, contract['sideboard_com'], 'sideboard_static.object_config.json')]:
            ids = mgr.load_configs(str(assets / file))
            if not ids:
                raise ValueError('Object template missing: ' + role)
            obj = sim.get_rigid_object_manager().add_object_by_template_handle(mgr.get_template_by_id(ids[0]).handle)
            obj.motion_type = MotionType.KINEMATIC
            obj.translation = habitat_vector(position)
            obj.semantic_id = node_id
            if role == 'kettle' and test_delta is not None:
                obj.translation = habitat_vector([position[i] + test_delta[i] for i in range(3)])
        camera_policy = getattr(args, 'camera_policy', 'upright')
        offsets = [0., .15, -.15, .30, -.30] if camera_policy != 'legacy' else [0., .15, -.15]
        views = [(offset, 0.) for offset in offsets]
        if camera_policy == 'scan':
            views += [(offset, yaw) for offset in (.15, -.15) for yaw in (-35., 35.)]
        for view_index, (lateral, yaw) in enumerate(views):
            pose = sim.get_agent(0).get_state()
            camera = np.asarray([robot[0] + lateral, robot[1] + 1.25, robot[2]], dtype=float)
            if getattr(args, 'previous_visual_run', None):
                centre = state['entities']['67']['node']['bbox_center']
                aim = [centre[0], centre[2], -centre[1]]
                aim = [aim[i]-contract['bbox_minus_com_offset'][i] for i in range(3)]
            else:
                aim = contract['kettle_com']
            direction = np.asarray(aim) - camera
            direction = scan_direction(direction, yaw)
            pose.position = camera
            pose.rotation = (quaternion.from_rotation_matrix(np.asarray(upright_camera_rotation(direction)))
                             if camera_policy != 'legacy' else
                             quat_from_two_vectors(np.asarray([0., 0., -1.]), direction))
            sim.get_agent(0).set_state(pose)
            observations = sim.get_sensor_observations()
            sensor_pose = sim.get_agent(0).get_state().sensor_states['depth']
            rotation = quaternion.as_rotation_matrix(sensor_pose.rotation).tolist()
            translation = list(map(float, sensor_pose.position))
            intrinsics = [128., 128., 128., 128.]
            prefix = args.output_dir / f'view_{view_index}'
            np.savez_compressed(str(prefix) + '.npz', rgb=observations['rgb'], depth=observations['depth'], semantic=observations['semantic'])
            from PIL import Image
            Image.fromarray(observations['rgb'][..., :3]).save(str(prefix) + '.png')
            points = {}
            for node_id in (67, 13):
                ys, xs = np.where(observations['semantic'] == node_id)
                points[str(node_id)] = []
                for y, x in zip(ys[::2], xs[::2]):
                    depth = float(observations['depth'][y, x])
                    if not math.isfinite(depth) or not .05 < depth < 10:
                        continue
                    xyz = project_pixel(int(x), int(y), depth, intrinsics, rotation, translation)
                    color = list(map(float, observations['rgb'][y, x, :3]))
                    points[str(node_id)].append(xyz + color)
            frame_id = str(prefix.resolve())
            layer = fuse_frame(layer, state, frame_id, points)
            report = {'frame_id': frame_id, 'camera_position_habitat': translation,
                'camera_rotation_matrix': rotation, 'intrinsics_fx_fy_cx_cy': intrinsics,
                'depth_convention': 'axial_metres_camera_negative_Z',
                'point_counts': {k: len(v) for k, v in points.items()},
                'association': 'simulator_semantic_id_oracle_not_VLM',
                'camera_orientation_policy': camera_policy,
                'scan_yaw_degrees': yaw,
                'camera_policy': 'endpoint_anchored_elevated_rig_with_small_lateral_offsets_not_robot_motion'}
            write(Path(str(prefix) + '.json'), report)
            reports.append(report)
    if test_delta is not None:
        # Controlled displacement is an evaluation fixture only. Camera poses are
        # unchanged and remain aimed at the OLD target; no new-target oracle aiming.
        summary = {'status': 'controlled_capture_ready', 'frames': reports,
            'test_translation_habitat': test_delta,
            'test_translation_cg': [test_delta[0], -test_delta[2], test_delta[1]],
            'actual_bbox_before_cg': truth_before,
            'actual_bbox_after_cg': [truth_before[0]+test_delta[0], truth_before[1]-test_delta[2], truth_before[2]+test_delta[1]],
            'memory_intentionally_stale': True, 'training_eligible': False,
            'process_peak_rss_gib': round(peak_rss_mb()/1024, 3),
            'source_memory_unchanged': digest(read(args.memory_dir / 'memory_state.json')) == original_digest,
            'scope': 'Known-pose controlled kinematic displacement; not real robot manipulation'}
        write(args.output_dir / 'capture_evaluation_only.json', summary)
        print(f"status=controlled_capture_ready result={args.output_dir / 'capture_evaluation_only.json'}", flush=True)
        return 0
    # Preserve the state history and original feature map; add only an audited visual overlay.
    snapshot = args.output_dir / 'fused_memory'
    (snapshot / 'scene/map').mkdir(parents=True)
    for key, entry in layer['objects'].items():
        if entry['voxels']:
            state['entities'][key]['node']['visual_memory'] = {'voxel_count': len(entry['voxels']),
                'frame_count': len(entry['frame_ids']), 'appearance_features': 'original_CLIP_inherited',
                'association': layer['association'], 'centroid_updated_from_vision': False}
    state['visual_revision'] = state.get('visual_revision', 0) + 1
    write(snapshot / 'memory_state.json', state)
    write(snapshot / 'visual_layer.json', layer)
    write(snapshot / 'scene/scene_graph.json', [e['node'] for e in state['entities'].values() if e['active']])
    write(snapshot / 'scene/scene_graph_edges.json', state['edges'])
    shutil.copy2(args.memory_dir / 'scene/map/scene_map_cfslam_pruned.pkl.gz', snapshot / 'scene/map/scene_map_cfslam_pruned.pkl.gz')
    accepted = '67' in layer['objects'] and any(r['frame_id'] in layer['objects']['67']['frame_ids'] for r in reports)
    summary = {'status': 'visual_overlay_ready' if accepted else 'insufficient_visual_evidence',
        'fused_memory': str(snapshot), 'frames': reports, 'memory_version': state['version'],
        'visual_revision': state['visual_revision'], 'source_memory_unchanged': digest(read(args.memory_dir / 'memory_state.json')) == original_digest,
        'process_peak_rss_gib': round(peak_rss_mb()/1024, 3), 'training_eligible': False,
        'scope': 'Known-object RGB-D surface voxel fusion with oracle association; not complete open-vocabulary reconstruction or visual pose estimation'}
    write(args.output_dir / 'visual_summary.json', summary)
    print(f"status={summary['status']} result={args.output_dir / 'visual_summary.json'}", flush=True)
    return 0 if accepted and summary['source_memory_unchanged'] else 2


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--memory-dir', type=Path, required=True)
    p.add_argument('--followup-execution', type=Path)
    p.add_argument('--previous-visual-run', type=Path, help='Audited prior repair checkpoint; independent fixture truth')
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--camera-policy', choices=('upright', 'legacy', 'scan'), default='upright')
    p.add_argument('--test-translation-habitat', type=float, nargs=3,
                   help='Isolated controlled displacement; keep memory and camera aiming stale; no fused snapshot produced')
    args = p.parse_args()
    try:
        raise SystemExit(run(args))
    except Exception as exc:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        write(args.output_dir / 'visual_summary.json', {'status': 'error', 'error': str(exc), 'training_eligible': False})
        raise
