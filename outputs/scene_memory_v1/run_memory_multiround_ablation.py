#!/usr/bin/env python3
"""Continuous task-level proxy: two relocations and latest-vs-stale map navigation."""
import argparse
import copy
import logging
import math
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

from run_memory_followup_navigation import restore_contract, choose_far_start, SCENE
from run_navigation_demo import (cg_to_habitat, habitat_vector, horizontal_distance,
    select_coordinates, navigate, peak_rss_mb)
from run_sideboard_full_pipeline import validate_adapter
from update_memory import read, write, digest, apply_events

HERE = Path(__file__).resolve().parent


def observed_move(state, actual_com, offset, event_id, observed_at, reference, support=None):
    """Memory input comes from simulator readback, not the requested placement."""
    bbox = [actual_com[i] + offset[i] for i in range(3)]
    event = {'event_id': event_id, 'action_id': 'proxy_place', 'operation': 'move',
        'object_id': 67, 'action_status': 'completed', 'observed_at': observed_at,
        'coordinate_frame': 'conceptgraphs_world', 'position': [bbox[0], -bbox[2], bbox[1]],
        'evidence': {'source': 'simulator_observation', 'verified': True, 'reference': reference,
                     'execution_mode': 'kinematic_task_proxy_not_physical_grasp'}}
    if support is not None:
        event['support_object_id'] = support
    return apply_events(state, [event]), event


def item_from_memory(state):
    entity = state['entities']['67']
    if not entity['active']:
        raise ValueError('NO object can be used in this task')
    return {'element_id': 67, 'cg_center': entity['node']['bbox_center']}


def seed_episode(args, seed, directory):
    import habitat_sim
    from habitat_sim.agent import AgentConfiguration, ActionSpec, ActuationSpec
    from habitat_sim.physics import MotionType
    source = read(args.memory_dir / 'memory_state.json')
    state = copy.deepcopy(source)
    nodes = read(args.memory_dir / 'scene/scene_graph.json')
    planning_target = read(args.plan_run / 'planning_target.json')
    if Path(planning_target['memory_source']).resolve() != (args.memory_dir / 'memory_state.json').resolve():
        raise ValueError('Reusable plan does not reference supplied memory')
    if read(args.plan_run / 'pipeline_summary.json').get('status') != 'planning_passed':
        raise ValueError('Reusable plan must have passed validation')
    base_nodes = read(Path(state['base_scene']) / 'scene_graph.json')
    moves = [row for row in state['event_log'] if row['event'].get('object_id') == 67
             and row['event'].get('operation') == 'move']
    proxy = read(Path(moves[-1]['event']['evidence']['reference']))
    contract = restore_contract(state, nodes, base_nodes, proxy)
    # Reuse only a validated single-navigation skill and its stable object binding.
    target = validate_adapter(read(args.plan_run / 'object_alignment.json'),
        read(args.plan_run / 'skill_plan_final.json'), nodes, 'Go to the electric kettle.', 67)
    if target['cg_center'] != item_from_memory(state)['cg_center']:
        raise ValueError('Reusable plan target differs from starting memory')
    assets = SCENE / 'habitat_split_assets_v2'
    cfg = habitat_sim.SimulatorConfiguration()
    cfg.scene_id = str(assets / 'scene201_without_kettle.stage_config.json')
    cfg.create_renderer = False
    cfg.enable_physics = True
    agent = AgentConfiguration()
    agent.action_space = {'move_forward': ActionSpec('move_forward', ActuationSpec(amount=.25)),
        'turn_left': ActionSpec('turn_left', ActuationSpec(amount=10)),
        'turn_right': ActionSpec('turn_right', ActuationSpec(amount=10))}
    rounds = []
    directory.mkdir(parents=True, exist_ok=False)
    with habitat_sim.Simulator(habitat_sim.Configuration(cfg, [agent])) as sim:
        if not sim.pathfinder.load_nav_mesh(str(SCENE / 'scene_package/hf_house_201/hf_house_201.navmesh')):
            raise ValueError('Navmesh load failed')
        mgr = sim.get_object_template_manager()
        instances = {}
        for role, position in [('sideboard', contract['sideboard_com']), ('kettle', contract['kettle_com'])]:
            file = 'sideboard_static.object_config.json' if role == 'sideboard' else 'kettle_dynamic.object_config.json'
            ids = mgr.load_configs(str(assets / file))
            if not ids:
                raise ValueError('Missing object template: ' + role)
            obj = sim.get_rigid_object_manager().add_object_by_template_handle(mgr.get_template_by_id(ids[0]).handle)
            obj.motion_type = MotionType.KINEMATIC
            obj.translation = habitat_vector(position)
            instances[role] = obj
        kettle = instances['kettle']
        offset = contract['bbox_minus_com_offset']
        actual_bbox = [float(kettle.translation[i]) + offset[i] for i in range(3)]
        if math.dist(actual_bbox, contract['target_bbox_habitat']) > 1e-4:
            raise ValueError('Restoration position mismatch')
        original_cg = next(n['bbox_center'] for n in base_nodes if n['id'] == 67)
        original_bbox = cg_to_habitat(original_cg, 'x_z_minus_y')
        original_com = [original_bbox[i] - offset[i] for i in range(3)]
        destinations = [contract['kettle_com'], original_com, contract['kettle_com']]
        previous = None
        time_base = datetime.fromisoformat(moves[-1]['event']['observed_at'].replace('Z', '+00:00'))
        for index, destination in enumerate(destinations):
            print(f'START seed={seed} round={index + 1}', flush=True)
            transfer = None
            stale = copy.deepcopy(state) if index == 0 else previous
            if index:
                previous = copy.deepcopy(state)
                stale = previous
                # Before assuming grasp, approach the currently observed object.
                _, goals = select_coordinates(sim.pathfinder, [item_from_memory(state)], 'x_z_minus_y', 2.)
                approach = navigate(sim, goals[0]['nav_goal'], 500, .6, 25)
                near = horizontal_distance(sim.get_agent(0).get_state().position, kettle.translation)
                if approach['status'] != 'arrived' or near > 1.5:
                    raise ValueError('Cannot approach kettle before assumed grasp')
                bbox_destination = [destination[i] + offset[i] for i in range(3)]
                waypoint = {'element_id': 67, 'cg_center': [bbox_destination[0], -bbox_destination[2], bbox_destination[1]]}
                _, goal = select_coordinates(sim.pathfinder, [waypoint], 'x_z_minus_y', 2.)

                def carry():
                    robot = list(map(float, sim.get_agent(0).get_state().position))
                    kettle.translation = habitat_vector([robot[0], robot[1] + 1., robot[2]])

                carry()
                travel = navigate(sim, goal[0]['nav_goal'], 500, .6, 25, on_action=carry)
                if travel['status'] != 'arrived':
                    raise ValueError('Proxy transfer navigation failed')
                kettle.translation = habitat_vector(destination)
                actual = list(map(float, kettle.translation))
                observation_path = directory / f'round_{index + 1}_observation.json'
                observation = {'object_id': 67, 'runtime_id': int(kettle.object_id), 'actual_com': actual,
                    'assumed_grasp': True, 'placement_mode': 'kinematic_waypoint_not_support_physics',
                    'approach': approach, 'carry_navigation': travel, 'pre_grasp_distance_m': near}
                write(observation_path, observation)
                state, event = observed_move(state, actual, offset, f'seed-{seed}-move-{index}',
                    (time_base + timedelta(seconds=index)).isoformat(), str(observation_path))
                # No support relation inferred: kinematic positioning is not support evidence.
                transfer = dict(observation, event=event)
            _, latest_goals = select_coordinates(sim.pathfinder, [item_from_memory(state)], 'x_z_minus_y', 2.)
            _, stale_goals = select_coordinates(sim.pathfinder, [item_from_memory(stale)], 'x_z_minus_y', 2.)
            actual = list(map(float, kettle.translation))
            start_position = choose_far_start(sim.pathfinder, latest_goals[0]['nav_goal'], actual,
                seed + index * 100, 3.)
            initial = sim.get_agent(0).get_state()
            initial.position = habitat_vector(start_position)
            initial = copy.deepcopy(initial)
            modes = {}
            latest_end_state = None
            for mode, goals in [('latest_memory', latest_goals), ('stale_snapshot', stale_goals)]:
                sim.get_agent(0).set_state(copy.deepcopy(initial))
                nav = navigate(sim, goals[0]['nav_goal'], 500, .6, 25)
                distance = horizontal_distance(sim.get_agent(0).get_state().position, kettle.translation)
                modes[mode] = {'navigation': nav, 'distance_to_actual_object_m': distance,
                    'actual_object_reached': nav['status'] == 'arrived' and distance <= 1.5}
                if mode == 'latest_memory':
                    latest_end_state = copy.deepcopy(sim.get_agent(0).get_state())
            # Continue the next transfer from the successful latest-memory endpoint.
            sim.get_agent(0).set_state(latest_end_state)
            passed = modes['latest_memory']['actual_object_reached']
            if index:
                passed = passed and modes['stale_snapshot']['navigation']['status'] == 'arrived' and not modes['stale_snapshot']['actual_object_reached']
            else:
                passed = passed and modes['stale_snapshot']['actual_object_reached']
            record = {'round': index + 1, 'passed': passed, 'stable_object_id': 67,
                'memory_version': state['version'], 'start_position': start_position,
                'transfer': transfer, 'modes': modes, 'memory_state': state}
            write(directory / f'round_{index + 1}.json', record)
            rounds.append({k: v for k, v in record.items() if k != 'memory_state'})
            print(f'END seed={seed} round={index + 1} passed={passed}', flush=True)
    return {'seed': seed, 'passed': all(r['passed'] for r in rounds), 'rounds': rounds}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--memory-dir', type=Path, default=SCENE / 'habitat_task_with_scene_memory/scene201_20260926_135458_6541/scene_memory')
    parser.add_argument('--plan-run', type=Path, default=SCENE / 'memory_followup_planning/scene201_20260926_140047_110543')
    parser.add_argument('--seeds', type=int, nargs='+', default=[201, 202, 203])
    parser.add_argument('--output-dir', type=Path, default=SCENE / 'memory_multiround_ablation' / datetime.now().strftime('scene201_%Y%m%d_%H%M%S_%f'))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(sys.stdout),
        logging.FileHandler(args.output_dir / 'execution.log', encoding='utf-8')])
    before = digest(read(args.memory_dir / 'memory_state.json'))
    summary = {'status': 'running', 'episodes': [], 'api_called': False,
        'fresh_llm_planning': False, 'training_eligible': False,
        'scope': 'Controlled memory ablation, continuous simulator within each seed; assumed grasp and kinematic placement',
        'limitations': ['No physical grasp/support verification', 'Reuses navmesh; dynamic obstacle replanning not tested',
                        'Waypoints are existing object locations, not new scene objects', 'No visual fusion or fresh semantic matching']}
    try:
        for seed in args.seeds:
            summary['episodes'].append(seed_episode(args, seed, args.output_dir / f'seed_{seed}'))
            write(args.output_dir / 'regression_summary.json', summary)
        summary['source_memory_unchanged'] = digest(read(args.memory_dir / 'memory_state.json')) == before
        summary['process_peak_rss_gib'] = round(peak_rss_mb() / 1024, 3)
        summary['status'] = 'passed' if summary['source_memory_unchanged'] and all(e['passed'] for e in summary['episodes']) else 'failed'
    except Exception as exc:
        logging.exception('Multi-round memory test failed')
        summary.update(status='error', error=str(exc))
    write(args.output_dir / 'regression_summary.json', summary)
    print(f"status={summary['status']} summary={args.output_dir / 'regression_summary.json'}", flush=True)
    return 0 if summary['status'] == 'passed' else 2


if __name__ == '__main__':
    raise SystemExit(main())
