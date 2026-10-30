#!/usr/bin/env python3
"""Fresh plans on repaired/stale maps; navigate in independently restored moved scene."""
import argparse
import copy
import logging
import math
import sys
from datetime import datetime
from pathlib import Path
from run_full_multiround import execute, SCENE, HERE
from run_visual_move_regression import validate_api_environment
from run_memory_followup_navigation import restore_contract, choose_far_start
from run_navigation_demo import habitat_vector, cg_to_habitat, select_coordinates, navigate, horizontal_distance, peak_rss_mb
from run_sideboard_full_pipeline import validate_adapter
from update_memory import read, write, digest, path_key, vector, apply_events

INSTRUCTION = 'Go to the electric kettle.'


def validate_visual_repair(old_state, repaired_state, estimate, evaluation):
    if evaluation.get('status') != 'passed' or estimate.get('status') != 'translation_estimated':
        raise ValueError('A successful visual translation regression is required')
    if repaired_state.get('base_graph_digest') != old_state.get('base_graph_digest'):
        raise ValueError('Visual repair changed the source graph identity')
    delta = vector(estimate['translation_cg'])
    old = vector(old_state['entities']['67']['node']['bbox_center'])
    current = vector(repaired_state['entities']['67']['node']['bbox_center'])
    if not repaired_state['entities']['67']['active']:
        raise ValueError('NO object can be used in this task')
    if math.dist(current, [old[i]+delta[i] for i in range(3)]) > 1e-6:
        raise ValueError('Repaired graph does not contain the estimated translation')
    if repaired_state['version'] != old_state['version']+1:
        raise ValueError('Unexpected memory version transition')
    if repaired_state['event_log'][:-1] != old_state['event_log']:
        raise ValueError('Memory repair history differs from original checkpoint')
    event = repaired_state['event_log'][-1]['event']
    if event.get('object_id') != 67 or event.get('operation') != 'move' or event.get('position') != current:
        raise ValueError('Latest visual move event is inconsistent')
    if event.get('evidence', {}).get('estimated_from') != 'VLM_SAM_RGBD_translation_registration':
        raise ValueError('Latest event is not the visual registration result')
    if apply_events(old_state, [event]) != repaired_state:
        raise ValueError('Repaired state cannot be reproduced from its audited move event')
    return {'old_position': old, 'repaired_position': current, 'estimated_delta': delta}


def load_planned_target(plan_dir, memory_dir):
    summary = read(plan_dir / 'pipeline_summary.json')
    target = read(plan_dir / 'planning_target.json')
    state = read(memory_dir / 'memory_state.json')
    if summary.get('status') != 'planning_passed' or summary.get('instruction') != INSTRUCTION:
        raise ValueError('Fresh navigation planning did not pass')
    if path_key(target.get('memory_source', '')) != path_key(memory_dir / 'memory_state.json'):
        raise ValueError('Plan references the wrong memory')
    if summary['memory_target_check']['memory_version'] != state['version']:
        raise ValueError('Map changed after planning')
    nodes = read(memory_dir / 'scene/scene_graph.json')
    if next(n for n in nodes if n['id'] == 67) != state['entities']['67']['node']:
        raise ValueError('Memory entity and exported scene differ')
    alignment = read(plan_dir / 'object_alignment.json')
    if path_key(alignment.get('scene_reference', '')) != path_key(memory_dir / 'scene'):
        raise ValueError('Alignment used a different scene')
    item = validate_adapter(alignment, read(plan_dir / 'skill_plan_final.json'), nodes, INSTRUCTION, 67)
    if item['cg_center'] != target['target']['cg_center']:
        raise ValueError('Adapter target differs from the saved plan target')
    return item


def compare_navigation(args, root, old_memory, repaired_memory, contract, actual_com, actual_bbox_cg):
    import habitat_sim
    from habitat_sim.agent import AgentConfiguration, ActionSpec, ActuationSpec
    from habitat_sim.physics import MotionType
    items = {'repaired_map': load_planned_target(root / 'repaired_planning', repaired_memory),
             'stale_map': load_planned_target(root / 'stale_planning', old_memory)}
    assets = SCENE / 'habitat_split_assets_v2'
    cfg = habitat_sim.SimulatorConfiguration()
    cfg.scene_id = str(assets / 'scene201_without_kettle.stage_config.json')
    cfg.create_renderer = False
    cfg.enable_physics = True
    agent = AgentConfiguration()
    agent.action_space = {'move_forward': ActionSpec('move_forward', ActuationSpec(amount=.25)),
        'turn_left': ActionSpec('turn_left', ActuationSpec(amount=10)),
        'turn_right': ActionSpec('turn_right', ActuationSpec(amount=10))}
    trials = []
    with habitat_sim.Simulator(habitat_sim.Configuration(cfg, [agent])) as sim:
        if not sim.pathfinder.load_nav_mesh(str(SCENE / 'scene_package/hf_house_201/hf_house_201.navmesh')):
            raise ValueError('Navmesh load failed')
        manager = sim.get_object_template_manager()
        kettle = None
        for role, position, file in [('sideboard', contract['sideboard_com'], 'sideboard_static.object_config.json'),
                                     ('kettle', actual_com, 'kettle_dynamic.object_config.json')]:
            ids = manager.load_configs(str(assets / file))
            if not ids:
                raise ValueError('Cannot load '+role)
            obj = sim.get_rigid_object_manager().add_object_by_template_handle(manager.get_template_by_id(ids[0]).handle)
            obj.motion_type = MotionType.KINEMATIC
            obj.translation = habitat_vector(position)
            if role == 'kettle':
                kettle = obj
        actual = list(map(float, kettle.translation))
        if math.dist(actual, actual_com) > 1e-5:
            raise ValueError('Simulation restoration error')
        goals = {mode: select_coordinates(sim.pathfinder, [item], 'x_z_minus_y', 2.)[1][0]['nav_goal'] for mode, item in items.items()}
        for seed in args.seeds:
            start_position = choose_far_start(sim.pathfinder, goals['repaired_map'], actual, seed, 3.)
            initial = sim.get_agent(0).get_state()
            initial.position = habitat_vector(start_position)
            modes = {}
            for mode in ('repaired_map', 'stale_map'):
                sim.get_agent(0).set_state(copy.deepcopy(initial))
                print(f'START seed={seed} map={mode}', flush=True)
                nav = navigate(sim, goals[mode], 500, .6, 25)
                distance = horizontal_distance(sim.get_agent(0).get_state().position, kettle.translation)
                target_error = math.dist(items[mode]['cg_center'], actual_bbox_cg)
                modes[mode] = {'navigation': nav, 'distance_to_actual_kettle_m': distance,
                    'navigation_task_passed': nav['status'] == 'arrived' and distance <= 1.5 and nav.get('actions', 0)>0,
                    'semantic_target_error_m': target_error, 'semantic_target_accurate_5cm': target_error <= .05,
                    'planned_cg_center': items[mode]['cg_center']}
                print(f'END seed={seed} map={mode} reached={modes[mode]["navigation_task_passed"]} target_error_m={target_error:.4f}', flush=True)
            passed = (modes['repaired_map']['navigation_task_passed'] and modes['repaired_map']['semantic_target_accurate_5cm']
                      and not modes['stale_map']['semantic_target_accurate_5cm'])
            record = {'seed': seed, 'passed': passed, 'same_start_and_orientation': True,
                'start_position': start_position, 'modes': modes, 'actual_object_position_habitat': actual,
                'truth_used_only_for_simulation_restoration_and_evaluation': True}
            trials.append(record)
            write(root / f'seed_{seed}.json', record)
    return trials


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--visual-regression-dir', type=Path, default=SCENE / 'visual_move_regression/scene201_20260926_154100_575984')
    p.add_argument('--planning-python', default='/home/lewis/miniforge3/envs/conceptgraph/bin/python')
    p.add_argument('--seeds', type=int, nargs='+', default=[201, 202, 203])
    p.add_argument('--output-dir', type=Path, default=SCENE / 'visual_repaired_navigation' / datetime.now().strftime('scene201_%Y%m%d_%H%M%S_%f'))
    args = p.parse_args()
    try:
        validate_api_environment()
    except ValueError as exc:
        p.error(str(exc))
    root = args.output_dir
    root.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(sys.stdout),
        logging.FileHandler(root / 'execution.log', encoding='utf-8')])
    summary = {'status': 'running', 'training_eligible': False, 'fresh_llm_planning_for_both_maps': True,
        'scope': 'Vision-repaired target -> fresh LLM planning -> restored moved-scene navigation; controlled translation fixture',
        'limitations': ['Truth used for simulator restoration and scoring, not planning',
            'Small displacement means stale navigation may also pass 1.5m threshold',
            'No new visual capture, no rotation, no real grasp, inherited navmesh']}
    stage = 'preflight'
    try:
        regression = args.visual_regression_dir
        evaluation = read(regression / 'regression_summary.json')
        old_memory = Path(evaluation['source_memory'])
        repaired_memory = Path(evaluation['repaired_memory'])
        old_state = read(old_memory / 'memory_state.json')
        repaired_state = read(repaired_memory / 'memory_state.json')
        estimate = read(regression / 'estimated_translation.json')
        check = validate_visual_repair(old_state, repaired_state, estimate, evaluation)
        if path_key(repaired_state['event_log'][-1]['event']['evidence']['reference']) != path_key(regression / 'estimated_translation.json'):
            raise ValueError('Visual event references a different registration result')
        hashes = {str(m): digest(read(m / 'memory_state.json')) for m in (old_memory, repaired_memory)}
        for mode, memory in [('repaired', repaired_memory), ('stale', old_memory)]:
            stage = mode+'_fresh_planning'
            execute([sys.executable, str(HERE.parent / 'habitat_demo_v1/run_sideboard_full_pipeline.py'),
                '--planning-python', args.planning_python, '--habitat-python', sys.executable,
                '--instruction', INSTRUCTION, '--expected-target-id', '67', '--planning-only',
                '--scene-dir', str(memory / 'scene'), '--memory-state', str(memory / 'memory_state.json'),
                '--output-dir', str(root / (mode+'_planning'))], root / (mode+'_planning.console.log'))
        stage = 'restore_actual_scene'
        latest = next(row['event'] for row in reversed(old_state['event_log']) if row['event']['object_id']==67 and row['event']['operation']=='move')
        from visual_checkpoint import fixture_contract
        previous = evaluation.get('previous_visual_run')
        contract = fixture_contract(old_memory, Path(previous) if previous else None)
        fixture = read(regression / 'after_capture/capture_evaluation_only.json')
        if fixture.get('status') != 'controlled_capture_ready' or fixture['test_translation_cg'] != evaluation['evaluation_translation_cg']:
            raise ValueError('Simulation fixture and held-out evaluation disagree')
        physical_delta = vector(fixture['test_translation_habitat'])
        if math.dist([physical_delta[0], -physical_delta[2], physical_delta[1]], fixture['test_translation_cg']) > 1e-6:
            raise ValueError('Fixture coordinate conversion is inconsistent')
        actual_com = [contract['kettle_com'][i]+physical_delta[i] for i in range(3)]
        actual_bbox_cg = fixture.get('actual_bbox_after_cg', [check['old_position'][i]+fixture['test_translation_cg'][i] for i in range(3)])
        stage = 'navigation_comparison'
        trials = compare_navigation(args, root, old_memory, repaired_memory, contract, actual_com, actual_bbox_cg)
        unchanged = all(digest(read(Path(path) / 'memory_state.json'))==value for path, value in hashes.items())
        summary.update(status='passed' if unchanged and trials and all(t['passed'] for t in trials) else 'failed',
            trials=trials, source_memories_unchanged=unchanged, visual_repair_check=check,
            repaired_navigation_success_count=sum(t['modes']['repaired_map']['navigation_task_passed'] for t in trials),
            stale_navigation_success_count=sum(t['modes']['stale_map']['navigation_task_passed'] for t in trials),
            process_peak_rss_gib=round(peak_rss_mb()/1024, 3))
    except Exception as exc:
        logging.exception('Visual repair navigation failed')
        summary.update(status='failed', failed_stage=stage, error=str(exc))
    write(root / 'regression_summary.json', summary)
    print(f"status={summary['status']} summary={root / 'regression_summary.json'}", flush=True)
    return 0 if summary['status']=='passed' else 2


if __name__ == '__main__':
    raise SystemExit(main())
