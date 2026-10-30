#!/usr/bin/env python3
"""Robustness suite and gated long-horizon composed workflow. Limited to known scene objects."""
import argparse
import copy
import sys
from datetime import datetime
from pathlib import Path
from run_full_multiround import execute, SCENE, HERE
from run_visual_move_regression import validate_api_environment
from estimate_visual_translation import consensus
from update_memory import read, write, digest, apply_events


def rejection_checks(estimate, state):
    rows = estimate['views'][:2]
    if len(rows) < 2:
        raise ValueError('Successful reference needs two views')
    bad = copy.deepcopy(rows)
    bad[1]['translation_cg'][0] += .20
    poor = copy.deepcopy(rows)
    for row in poor:
        row['symmetric_trimmed_rms_m'] = .2
    zero = copy.deepcopy(rows)
    for row in zero:
        row['translation_cg'] = [0., 0., 0.]
    reports = []
    for name, data in [('single_view', rows[:1]), ('conflicting_views', bad),
                       ('high_registration_residual', poor), ('zero_displacement', zero)]:
        try:
            consensus(data)
        except ValueError as exc:
            reports.append({'case': name, 'passed': True, 'reason': str(exc), 'synthetic_fault_injection': True})
        else:
            reports.append({'case': name, 'passed': False, 'synthetic_fault_injection': True})
    event = {'event_id': 'unverified-suite-event', 'action_id': 'test', 'operation': 'move', 'object_id': 67,
        'action_status': 'completed', 'observed_at': '2099-01-01T00:00:00Z', 'coordinate_frame': 'conceptgraphs_world',
        'position': [0., 0., 0.], 'evidence': {'source': 'verified_post_action_observation', 'verified': False, 'reference': 'test'}}
    before = digest(state)
    try:
        apply_events(state, [event])
    except ValueError:
        reports.append({'case': 'unverified_move_rejected_without_mutation', 'passed': digest(state)==before,
                        'synthetic_fault_injection': True})
    else:
        reports.append({'case': 'unverified_move_rejected_without_mutation', 'passed': False})
    return reports


def robustness(args, root):
    reference = args.reference_visual_run
    reference_summary = read(reference / 'regression_summary.json')
    if reference_summary.get('status') != 'passed':
        raise ValueError('Reference visual movement must have passed')
    state = read(Path(reference_summary['source_memory']) / 'memory_state.json')
    faults = rejection_checks(read(reference / 'estimated_translation.json'), state)
    summary = {'status': 'running', 'suite_version': '1.0', 'training_eligible': False,
        'positive_cases': [{'case': 'x_25cm_reference', 'status': 'passed', 'reused_evidence': True,
                            'source': str(reference), 'translation_error_m': reference_summary['translation_error_m']}],
        'negative_cases': faults, 'scope': 'Known kettle, fixed cameras, translational robustness plus synthetic rejection checks'}
    write(root / 'regression_summary.json', summary)
    for name, delta in [('x_15cm', [.15, 0., 0.]), ('diagonal_x15_z10cm', [.15, 0., .10])]:
        case_root = root / name
        print(f'START robustness={name}', flush=True)
        try:
            execute([sys.executable, str(HERE / 'run_visual_move_regression.py'),
                '--planning-python', args.planning_python, '--translation-habitat', *map(str, delta),
                '--output-dir', str(case_root)], root / (name+'.console.log'))
            result = read(case_root / 'regression_summary.json')
            summary['positive_cases'].append({'case': name, 'status': result['status'], 'reused_evidence': False,
                'translation_error_m': result.get('translation_error_m'), 'source': str(case_root),
                'source_memory_unchanged': result.get('source_memory_unchanged')})
        except Exception as exc:
            summary['positive_cases'].append({'case': name, 'status': 'failed_or_rejected', 'error': str(exc),
                'note': 'A rejected positive case is not counted as robust success; API errors are not evidence rejection'})
        write(root / 'regression_summary.json', summary)
    summary['status'] = 'passed' if all(c['status']=='passed' for c in summary['positive_cases']) and all(c['passed'] for c in faults) else 'needs_review'
    summary['positive_pass_count'] = sum(c['status']=='passed' for c in summary['positive_cases'])
    summary['negative_pass_count'] = sum(c['passed'] for c in faults)
    write(root / 'regression_summary.json', summary)
    return summary


def latest_robustness(args):
    if args.robustness_summary:
        path = args.robustness_summary
    else:
        reports = sorted((SCENE / 'chain_robustness').glob('scene201_*/regression_summary.json'), reverse=True)
        if not reports:
            raise ValueError('Run robustness mode before long mode')
        path = reports[0]  # newest run, never silently substitute an older success
    report = read(path)
    if report.get('status') != 'passed' or report.get('suite_version') != '1.0':
        raise ValueError('Latest robustness report did not pass; review it before long mode')
    return path


def validate_multiround_checkpoint(full):
    initial = read(full / 'pipeline_summary.json')
    expected = ['Move the electric kettle to the sideboard.', 'Go to the electric kettle.', 'Go to the sideboard.']
    rounds = initial.get('rounds', [])
    if (initial.get('status') != 'passed' or initial.get('base_graph_unchanged') is not True
            or len(rounds) != 3 or [r.get('instruction') for r in rounds] != expected
            or any(r.get('status') != 'passed' for r in rounds)):
        raise ValueError('Resume requires a passed, unchanged three-command checkpoint')
    memory = Path(rounds[-1]['memory'])
    state = read(memory / 'memory_state.json')
    if state.get('version') != rounds[-1]['memory_version']:
        raise ValueError('Checkpoint memory version mismatch')
    execution = full / 'round_3/execution/followup_execution.json'
    result = read(execution)
    from capture_visual_memory import same_physical_checkpoint
    if result.get('status') != 'followup_proxy_passed' or not same_physical_checkpoint(
            state, read(Path(result['memory_source']))):
        raise ValueError('Checkpoint execution does not match final memory')
    return initial


def long_workflow(args, root):
    robust_path = latest_robustness(args)
    summary = {'status': 'running', 'workflow_type': 'scripted_long_horizon_composition_not_single_LLM_global_plan',
        'robustness_summary': str(robust_path), 'training_eligible': False, 'stages': [],
        'scope': 'Initial map -> three fresh commands -> external displacement -> vision repair -> fresh planning/navigation',
        'limitations': ['Known kettle and sideboard only', 'Checkpoint restoration, not uninterrupted robot execution',
            'Controlled translation truth gates repair acceptance; known camera poses', 'No grasp physics or visual rotation recovery']}
    write(root / 'regression_summary.json', summary)
    stage = 'three_fresh_commands'
    try:
        resumed = getattr(args, 'resume_multiround', None)
        full = resumed or root / 'initial_multiround'
        if not resumed:
            execute([sys.executable, str(HERE / 'run_full_multiround.py'), '--planning-python', args.planning_python,
                '--with-vision', '--output-dir', str(full)], root / 'multiround.console.log')
        initial = validate_multiround_checkpoint(full)
        summary['stages'].append({'stage': stage, 'status': initial['status'],
            'reused_successful_checkpoint': bool(resumed), 'summary': str(full / 'pipeline_summary.json')})
        # Use the LAST round's state and observations, not an earlier checkpoint.
        memory = Path(initial['rounds'][-1]['memory'])
        state_hash = digest(read(memory / 'memory_state.json'))
        stage = 'upright_baseline_capture'
        visual = root / 'baseline_capture'
        execute([sys.executable, str(HERE / 'capture_visual_memory.py'), '--memory-dir', str(memory),
            '--followup-execution', str(full / 'round_3/execution/followup_execution.json'),
            '--camera-policy', 'upright', '--output-dir', str(visual)], root / 'baseline_capture.console.log')
        summary['stages'].append({'stage': stage, 'status': 'passed', 'summary': str(visual / 'visual_summary.json')})
        stage = 'RGB_VLM_SAM_baseline'
        before = root / 'baseline_detection'
        execute([args.planning_python, str(HERE / 'replay_vlm_rgbd.py'), '--visual-dir', str(visual),
            '--memory-dir', str(memory), '--max-views', '5',
            '--output-dir', str(before)], root / 'baseline_detection.console.log')
        summary['stages'].append({'stage': stage, 'status': 'passed', 'summary': str(before / 'regression_summary.json')})
        stage = 'external_move_and_visual_repair'
        move = root / 'visual_move'
        execute([sys.executable, str(HERE / 'run_visual_move_regression.py'), '--planning-python', args.planning_python,
            '--baseline-run', str(full), '--memory-dir', str(memory), '--before-visual-dir', str(visual),
            '--capture-followup', str(full / 'round_3/execution/followup_execution.json'),
            '--before-detections', str(before), '--output-dir', str(move)], root / 'visual_move.console.log')
        summary['stages'].append({'stage': stage, 'status': 'passed', 'summary': str(move / 'regression_summary.json')})
        stage = 'repaired_map_fresh_planning_and_navigation'
        nav = root / 'repaired_navigation'
        execute([sys.executable, str(HERE / 'run_visual_repaired_navigation.py'), '--planning-python', args.planning_python,
            '--visual-regression-dir', str(move), '--seeds', '201', '202', '203',
            '--output-dir', str(nav)], root / 'navigation.console.log')
        summary['stages'].append({'stage': stage, 'status': 'passed', 'summary': str(nav / 'regression_summary.json')})
        summary.update(status='passed', latest_pre_perturbation_memory=str(memory),
            pre_perturbation_memory_unchanged=digest(read(memory / 'memory_state.json'))==state_hash,
            final_repaired_memory=read(move / 'regression_summary.json')['repaired_memory'],
            measured_stage_peak_rss_gib={
                'baseline_SAM': read(before / 'regression_summary.json').get('process_peak_rss_gib'),
                'after_move_SAM': read(move / 'after_detection/regression_summary.json').get('process_peak_rss_gib'),
                'after_move_render': read(move / 'after_capture/capture_evaluation_only.json').get('process_peak_rss_gib'),
                'navigation': read(nav / 'regression_summary.json').get('process_peak_rss_gib')},
            resource_scope='Individual measured stage process peaks, not whole-system or planning peaks')
        if not summary['pre_perturbation_memory_unchanged']:
            summary['status'] = 'failed'
    except Exception as exc:
        summary.update(status='failed', failed_stage=stage, error=str(exc))
    write(root / 'regression_summary.json', summary)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=('robustness', 'long'), default='robustness')
    p.add_argument('--planning-python', default='/home/lewis/miniforge3/envs/conceptgraph/bin/python')
    p.add_argument('--reference-visual-run', type=Path, default=SCENE / 'visual_move_regression/scene201_20260926_154100_575984')
    p.add_argument('--robustness-summary', type=Path)
    p.add_argument('--output-dir', type=Path)
    p.add_argument('--resume-multiround', type=Path,
                   help='Passed initial_multiround directory; recapture upright RGB-D in a new run')
    args = p.parse_args()
    try:
        validate_api_environment()
        if args.mode == 'long':
            latest_robustness(args)  # fail before spending API calls
            if args.resume_multiround:
                validate_multiround_checkpoint(args.resume_multiround)
        elif args.resume_multiround:
            raise ValueError('--resume-multiround requires --mode long')
    except ValueError as exc:
        p.error(str(exc))
    root = args.output_dir or SCENE / ('chain_robustness' if args.mode=='robustness' else 'long_horizon_memory_workflow') / datetime.now().strftime('scene201_%Y%m%d_%H%M%S_%f')
    root.mkdir(parents=True, exist_ok=False)
    try:
        result = robustness(args, root) if args.mode=='robustness' else long_workflow(args, root)
    except Exception as exc:
        result = {'status': 'failed', 'error': str(exc), 'training_eligible': False}
        write(root / 'regression_summary.json', result)
    print(f"status={result['status']} summary={root / 'regression_summary.json'}", flush=True)
    return 0 if result['status']=='passed' else 2


if __name__ == '__main__':
    raise SystemExit(main())
