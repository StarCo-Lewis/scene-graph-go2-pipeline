#!/usr/bin/env python3
"""Bounded real-LLM Scene201 regression; fresh continuous simulator per case."""
import argparse
import math
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from run_live_llm_sequence import default_instructions, validate_instructions
from run_continuous_sequence import SCENE, OUT
from run_navigation_demo import read_json, write_json
from proxy_placement import POLICY, same_upright_rotation


def cases():
    combined = {'schema_version': '1.0', 'training_eligible': False, 'rounds': [
        {'round': 1, 'instruction': 'Move the electric kettle to the sideboard and place it on the sideboard.',
         'expected_end': {'held_object_id': None, 'lifted': False}},
        {'round': 2, 'instruction': 'Visit the sideboard, then visit the moved kettle, then return to the sideboard.',
         'expected_navigation_targets': [13, 67, 13],
         'expected_end': {'held_object_id': None, 'lifted': False}},
        {'round': 3, 'instruction': 'Pick up the moved kettle and lift it; keep holding it for my next instruction.',
         'expected_end': {'held_object_id': 67, 'lifted': True}},
        {'round': 4, 'instruction': 'Place the kettle you are holding on the sideboard.',
         'expected_end': {'held_object_id': None, 'lifted': False}}]}
    return [dict(name=f'default_seed_{seed}', seed=seed, instructions=default_instructions(),
                 held_continuations=[3, 5], expected_placements=2, placement_rounds=[3, 5])
            for seed in (201, 202)] + [dict(name='combined_seed_203', seed=203,
                instructions=validate_instructions(combined), held_continuations=[4],
                expected_placements=2, placement_rounds=[1, 4])]


def close_number(value, expected, tol=1e-4):
    return type(value) in (int, float) and math.isfinite(value) and abs(value - expected) <= tol


def evaluate_case(result, exit_code, case):
    rows = result.get('rounds', [])
    expected = case['instructions']['rounds']
    actions = [a for row in rows for a in row.get('actions', [])]
    nav = [a for a in actions if a.get('skill_id') == 'navigate_to_object']
    placements = [a for a in actions if a.get('skill_id') == 'place_object_on_surface']
    checks = {
        'reported_complete': exit_code == 0 and result.get('status') == 'passed',
        'requested_seed_recorded': result.get('seed') == case['seed'],
        'actually_simulated': result.get('simulation_executed') is True,
        'one_simulator': result.get('simulator_creation_count') == 1,
        'one_start': result.get('initial_start_sampling_count') == 1,
        'continuous': result.get('uninterrupted_simulator_execution') is True,
        'all_requested_rounds': len(rows) == len(expected) and all(
            r.get('round') == e['round'] and r.get('instruction') == e['instruction'] and
            r.get('status') == 'passed' for r, e in zip(rows, expected)),
        'fresh_planning_each_round': len(rows) == len(expected) and all(
            r.get('planning_evidence', {}).get('fresh_llm') is True for r in rows),
        'current_memory_versions': bool(rows) and all(r.get('planning_evidence', {}).get('input_memory_version')
            == r.get('begin_state', {}).get('backend', {}).get('memory_version') and
            type(r.get('planning_evidence', {}).get('input_memory_version')) is int for r in rows),
        'cross_round_state_equal': bool(rows) and all(r.get('continuity_with_previous_round') is True
            and r.get('begin_state') == previous.get('end_state') for previous, r in zip(rows, rows[1:])),
        'runtime_outcomes': len(rows) == len(expected) and all(r.get('expected_runtime_state_passed') is True
            and all(r.get('end_state', {}).get(k) == v for k, v in e.get('expected_end', {}).items())
            for r, e in zip(rows, expected)),
        'all_required_goals_observed': bool(rows) and all(bool(r.get('goal_checks')) and
            any(g.get('required', True) for g in r['goal_checks']) and
            all(g.get('status') == 'observed_in_proxy' for g in r['goal_checks'] if g.get('required', True))
            for r in rows),
        'all_actions_completed': bool(actions) and result.get('executed_action_count') == len(actions)
            and result.get('attempted_action_count') == len(actions) and all(a.get('status') in
                ('arrived', 'proxy_preparation_only', 'proxy_attached', 'proxy_lifted',
                 'proxy_instance_support_passed') for a in actions),
        'held_state_inherited': result.get('cross_round_held_object_continuations') == case['held_continuations']
            and all(any(r.get('round') == i and r.get('begin_state', {}).get('held_object_id') == 67
                and r.get('begin_state', {}).get('lifted') is True for r in rows) for i in case['held_continuations']),
        'source_graph_unchanged': result.get('base_graph_unchanged') is True,
        'scope_not_overclaimed': result.get('training_eligible') is False and all(
            r.get('training_eligible') is False for r in rows),
        'navigation_current_and_arrived': bool(nav) and all(a.get('status') == 'arrived'
            and a.get('target_source') == 'current_memory_confirmed_by_live_instance_readback'
            and close_number(a.get('memory_target_error_m'), 0) for a in nav),
        'explicit_visit_order_preserved': len(rows) == len(expected) and all(
            [a.get('target_id') for a in r.get('actions', []) if a.get('skill_id') == 'navigate_to_object']
            == e['expected_navigation_targets'] for r, e in zip(rows, expected) if 'expected_navigation_targets' in e),
        'expected_placement_count': len(placements) == case['expected_placements'],
        'placement_in_requested_rounds': all(any(r.get('round') == i and
            sum(a.get('skill_id') == 'place_object_on_surface' for a in r.get('actions', [])) == 1 and
            r.get('end_state', {}).get('backend', {}).get('objects', {}).get('67', {}).get('support_id') == 13
            for r in rows) for i in case['placement_rounds']),
        'final_kettle_on_board_and_released': result.get('final_state', {}).get('held_object_id', 'missing') is None
            and result.get('final_state', {}).get('lifted') is False and
            result.get('final_state', {}).get('backend', {}).get('objects', {}).get('67', {}).get('support_id') == 13,
    }
    limits = {'horizontal_error_m': .35, 'com_height_error_m': .20,
              'tail_motion_m': .02, 'direct_support_contact_frames': 20}
    checks['placement_metrics_independently_pass'] = bool(placements) and all(
        a.get('validation', {}).get('passed') is True and all(
            a.get('validation', {}).get('checks', {}).get(k) is True and
            close_number(a.get('validation', {}).get('thresholds', {}).get(k), limit, 1e-9) and
            type(a.get('validation', {}).get('metrics', {}).get(k)) in (int, float) and
            math.isfinite(a['validation']['metrics'][k]) and
            (a['validation']['metrics'][k] >= limit if k == 'direct_support_contact_frames'
             else 0 <= a['validation']['metrics'][k] <= limit)
            for k, limit in limits.items()) for a in placements)
    checks['placement_volume_and_upright_release_evidenced'] = bool(placements) and all(
        a.get('placement_preparation', {}).get('placement_policy', {}).get('name') == POLICY['name'] and
        any(c.get('valid') is True and c.get('volume_probes') and
            all(not probe.get('blocking_contacts') for probe in c['volume_probes'])
            for c in a.get('surface_candidates', [])) and
        same_upright_rotation(a.get('at_release', {}).get('rotation_wxyz'),
            a.get('placement_preparation', {}).get('release_rotation_wxyz')) and
        a.get('at_release', {}).get('linear_velocity') == [0., 0., 0.] and
        a.get('at_release', {}).get('angular_velocity') == [0., 0., 0.] for a in placements)
    moved_targets = []
    initial_center = rows[0].get('begin_state', {}).get('backend', {}).get('objects', {}).get('67', {}).get('center') if rows else None
    for previous, row in zip(rows, rows[1:]):
        state = previous.get('end_state', {}).get('backend', {}).get('objects', {}).get('67', {})
        center = state.get('center')
        if state.get('support_id') == 13 and isinstance(center, list) and isinstance(initial_center, list):
            for action in row.get('actions', []):
                target = action.get('target_center')
                if action.get('target_id') == 67 and isinstance(target, list) and len(target) == len(center) == len(initial_center) == 3:
                    moved_targets.append(math.dist(target, center) <= 1e-4 and math.dist(center, initial_center) > .1)
    checks['moved_kettle_navigation_uses_updated_position'] = bool(moved_targets) and all(moved_targets)
    return checks


def planning_statistics(root, rounds):
    stats = {'http_attempt_log_count': 0, 'decomposition_rejections': 0, 'skill_rejections': 0,
             'transport_retry_warnings': 0, 'semantic_redecompositions': 0,
             'rounds_with_multiple_actions_in_one_subtask': [], 'all_planning_reports_present': True}
    for row in rounds:
        directory = root / f"round_{row['round']}" / 'planning'
        try:
            planning = read_json(directory / 'planning_summary.json')
            if planning.get('status') != 'planning_passed' or planning.get('input_memory_version') != row.get('begin_state', {}).get('backend', {}).get('memory_version'):
                stats['all_planning_reports_present'] = False
            stats['semantic_redecompositions'] += planning.get('semantic_redecomposition_count', 0)
            plan = read_json(directory / 'skill_plan_final.json')
            if any(len(t['steps']) > 1 for t in plan['subtask_plans']):
                stats['rounds_with_multiple_actions_in_one_subtask'].append(row['round'])
        except (OSError, ValueError, KeyError, TypeError):
            stats['all_planning_reports_present'] = False
        log = directory / 'planning.console.log'
        if log.is_file():
            for line in log.read_text(encoding='utf-8', errors='replace').splitlines():
                stats['http_attempt_log_count'] += bool(re.search(r'LLM HTTP (?:request )?attempt(?:=| )\d', line))
                stats['decomposition_rejections'] += 'Rejected LLM decomposition attempt' in line
                stats['skill_rejections'] += 'Rejected proposal attempt' in line
                stats['transport_retry_warnings'] += 'Transient LLM' in line
    stats['scope'] = 'Observed child logs, not provider billing or token usage'
    return stats


def stream_case(command, log_path):
    # Inherit the terminal process group: Ctrl-C reaches child and its normal cleanup.
    child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, encoding='utf-8', errors='replace')
    try:
        with log_path.open('w', encoding='utf-8') as log:
            for line in child.stdout:
                print(line, end='', flush=True)
                log.write(line)
                log.flush()
        return child.wait()
    finally:
        if child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        child.stdout.close()


def run_suite(args, case_catalog=None, extra_case_checks=None):
    selected = [c for c in (cases() if case_catalog is None else case_catalog)
                if not args.case or c['name'] in args.case]
    if not selected:
        raise ValueError('No regression cases selected')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    summary = {'schema_version': '1.0', 'status': 'running', 'cases': [],
        'selected_cases': [c['name'] for c in selected], 'training_eligible': False,
        'scope': 'Same Scene201 two mapped instances, five proxy skills; no claim of generalization',
        'preflight_only': args.preflight_only, 'planning_origin': 'fresh_llm_each_round_live_memory',
        'ram_scope': 'Each Habitat child peak RSS; excludes planner and Windows/WSL total'}
    destination = args.output_dir / 'regression_summary.json'
    for case in selected:
        name = case['name']
        instructions_path = args.output_dir / f'{name}.instructions.json'
        write_json(instructions_path, validate_instructions(case['instructions']))
        output = args.output_dir / name
        row = {'case': name, 'seed': case['seed'], 'passed': False, 'summary': str(output / 'pipeline_summary.json')}
        command = [sys.executable, str(OUT / 'habitat_demo_v1/run_live_llm_sequence.py'),
            '--instructions-json', str(instructions_path), '--seed', str(case['seed']),
            '--scene-dir', str(args.scene_dir), '--planning-python', args.planning_python,
            '--model', args.model, '--base-url', args.base_url, '--output-dir', str(output)]
        if args.preflight_only:
            command.append('--preflight-only')
        if getattr(args, 'visual_audit', False):
            command.extend(['--visual-audit', '--audit-stride', str(args.audit_stride),
                            '--audit-max-frames', str(args.audit_max_frames)])
        print(f'START {name} seed={case["seed"]} preflight_only={args.preflight_only} output={output}', flush=True)
        try:
            code = stream_case(command, args.output_dir / f'{name}.console.log')
            result = read_json(output / 'pipeline_summary.json')
            if args.preflight_only:
                checks = {'input_preflight_only': code == 0 and result.get('status') == 'preflight_passed',
                    'no_simulator': result.get('simulator_creation_count') == 0 and result.get('simulation_executed') is False}
            else:
                checks = evaluate_case(result, code, case)
                stats = planning_statistics(output, result.get('rounds', []))
                checks['planning_artifacts_present'] = stats['all_planning_reports_present']
                row['planning_statistics'] = stats
                if extra_case_checks is not None:
                    extra = extra_case_checks(result, output, case)
                    checks.update(extra['checks'])
                    row['stress_statistics'] = extra['statistics']
            row.update(passed=all(checks.values()), checks=checks, failed_checks=[k for k, v in checks.items() if not v],
                exit_code=code, executed_action_count=result.get('executed_action_count'),
                process_peak_rss_gib=result.get('process_peak_rss_gib'), initial_robot_position=result.get('rounds', [{}])[0].get('begin_state', {}).get('backend', {}).get('robot_position') if result.get('rounds') else None)
            if result.get('error'):
                row['error'] = result['error']
            if result.get('visual_audit'):
                row['visual_audit'] = result['visual_audit']
        except KeyboardInterrupt:
            row['error'] = 'User interrupted; remaining cases not run'
            summary['cases'].append(row)
            summary.update(status='interrupted', completed_case_count=len(summary['cases']))
            write_json(destination, summary)
            raise
        except (OSError, ValueError, KeyError, TypeError) as exc:
            row['error'] = str(exc)
        summary['cases'].append(row)
        write_json(destination, summary)
        print(f'END {name} passed={row["passed"]} exit_code={row.get("exit_code")} failed_checks={row.get("failed_checks", [])}', flush=True)
    passed = sum(c['passed'] for c in summary['cases'])
    summary.update(status=('preflight_passed' if args.preflight_only else 'passed') if passed == len(selected) else 'failed',
                   passed_case_count=passed, completed_case_count=len(summary['cases']),
                   selected_case_count=len(selected), simulation_executed=not args.preflight_only and any(
                       c.get('checks', {}).get('actually_simulated') is True for c in summary['cases']))
    starts = [c['initial_robot_position'] for c in summary['cases'] if c.get('initial_robot_position') is not None]
    summary['distinct_observed_start_count'] = len({tuple(p) for p in starts})
    write_json(destination, summary)
    print(f'status={summary["status"]} summary={destination}', flush=True)
    return 0 if passed == len(selected) else 2


def main():
    from match_skills import DEFAULT_URL
    from visual_audit import add_arguments as add_audit_arguments
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case', action='append', choices=[c['name'] for c in cases()], help='Repeat to select cases; default all three')
    p.add_argument('--preflight-only', action='store_true', help='No API and no simulator')
    add_audit_arguments(p)
    p.add_argument('--scene-dir', type=Path, default=SCENE / 'vlm_scenegraph_regression/scene201_verified_20260921_113316/scene')
    p.add_argument('--planning-python', default='/home/lewis/miniforge3/envs/conceptgraph/bin/python')
    p.add_argument('--model', default='qwen3.8-flash')
    p.add_argument('--base-url', default=DEFAULT_URL)
    p.add_argument('--output-dir', type=Path, default=SCENE / 'live_llm_regression' / datetime.now().strftime('scene201_%Y%m%d_%H%M%S_%f'))
    args = p.parse_args()
    if args.audit_stride <= 0 or args.audit_max_frames <= 0:
        p.error('Audit stride and frame budget must be positive')
    if not args.preflight_only and not os.environ.get('DASHSCOPE_API_KEY', '').strip():
        p.error('DASHSCOPE_API_KEY required; no fixture fallback')
    if not args.preflight_only and not Path(args.planning_python).is_file():
        p.error('Planning Python interpreter does not exist')
    return run_suite(args)


if __name__ == '__main__':
    raise SystemExit(main())
