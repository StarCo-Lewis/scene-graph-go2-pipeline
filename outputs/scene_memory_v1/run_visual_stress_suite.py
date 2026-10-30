#!/usr/bin/env python3
"""Independent larger-displacement cases, not sequential physical memory replay."""
import argparse
import sys
from datetime import datetime
from pathlib import Path
from run_full_multiround import execute, HERE, SCENE
from run_visual_move_regression import validate_api_environment
from update_memory import read, write, digest


def source_checkpoint(source):
    summary = read(source / 'regression_summary.json')
    if summary.get('status') != 'passed':
        raise ValueError('Stress tests require a passed long-workflow source')
    memory = Path(summary['latest_pre_perturbation_memory'])
    state = read(memory / 'memory_state.json')
    before = read(source / 'baseline_detection/regression_summary.json')
    if before.get('status') != 'non_oracle_visual_evidence_ready' or before.get('source_memory_unchanged') is not True:
        raise ValueError('Source baseline must have passed RGB-only multiview validation')
    return memory, state


def case_metrics(move, nav):
    if move.get('status') != 'passed' or nav.get('status') != 'passed':
        raise ValueError('Failed stages cannot count as a successful stress case')
    return {'translation_error_m': move['translation_error_m'],
            'stale_position_error_m': move['stale_position_error_m'],
            'repaired_position_error_m': move['repaired_position_error_m'],
            'repaired_navigation_success_count': nav['repaired_navigation_success_count'],
            'stale_navigation_success_count': nav['stale_navigation_success_count'],
            'navigation_success_gain_observed': nav['repaired_navigation_success_count'] > nav['stale_navigation_success_count'],
            'source_memories_unchanged': nav['source_memories_unchanged'] and move['source_memory_unchanged']}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-run', type=Path, default=SCENE / 'long_horizon_memory_workflow/scene201_20260926_165040_820983')
    p.add_argument('--planning-python', default='/home/lewis/miniforge3/envs/conceptgraph/bin/python')
    p.add_argument('--output-dir', type=Path)
    p.add_argument('--sequence', action='store_true', help='Carry prior repair across two further physical fixture moves')
    p.add_argument('--image-occlusion', action='store_true', help='Fixed-checkpoint RGB-D dropout and visibility recovery')
    p.add_argument('--factorial-dropout', action='store_true', help='Compare RGB-only, depth-only and joint partial dropout')
    p.add_argument('--recovery-chain', action='store_true', help='Blocked update followed by fresh visible repair and navigation')
    p.add_argument('--recovery-dropout', choices=('full', 'depth'), default='full',
                   help='Depth mode tests retained category observations without a pose commit')
    p.add_argument('--rounds', type=int, choices=range(2, 7), default=2,
                   help='Sequential moves, 2..6; each round makes up to 18 visual requests')
    p.add_argument('--move-offset', type=int, choices=range(5), default=0,
                   help='Skip this many preset moves when resuming a successful sequence checkpoint')
    p.add_argument('--previous-run', type=Path, default=SCENE / 'visual_stress_suite/scene201_20260926_171248_760537/x_50cm/visual_move')
    args = p.parse_args()
    validate_api_environment()
    if args.factorial_dropout and not args.image_occlusion:
        p.error('--factorial-dropout requires --image-occlusion')
    if sum((args.sequence, args.image_occlusion, args.recovery_chain)) > 1:
        p.error('Choose only one test mode')
    if args.recovery_chain:
        from run_visual_recovery import run
        return run(args)
    if args.image_occlusion:
        if args.sequence:
            p.error('--image-occlusion and --sequence are mutually exclusive')
        from run_image_occlusion import run
        return run(args)
    if args.sequence:
        from run_visual_sequence import run
        return run(args)
    memory, state = source_checkpoint(args.source_run)
    source_hash = digest(state)
    root = args.output_dir or SCENE / 'visual_stress_suite' / datetime.now().strftime('scene201_%Y%m%d_%H%M%S_%f')
    root.mkdir(parents=True, exist_ok=False)
    checkpoint = Path(read(args.source_run / 'regression_summary.json')['stages'][0]['summary']).parent
    followup = checkpoint / 'round_3/execution/followup_execution.json'
    report = {'status': 'running', 'training_eligible': False, 'source_run': str(args.source_run),
              'cases': [], 'scope': 'Independent displacement cases sharing one pre-perturbation checkpoint; not continuous multiround movement',
              'limitations': ['Known kettle and simulator camera poses', 'Assumed grasp, no physical robot execution',
                              'Translation only, inherited navmesh', '1.5m navigation threshold may let stale maps pass',
                              'Truth is used for restoration/evaluation and repair acceptance, not translation estimation']}
    write(root / 'regression_summary.json', report)
    try:
        baseline = root / 'baseline_scan'
        detections = root / 'baseline_detection'
        execute([sys.executable, str(HERE / 'capture_visual_memory.py'), '--memory-dir', str(memory),
            '--followup-execution', str(followup), '--camera-policy', 'scan',
            '--output-dir', str(baseline)], root / 'baseline_capture.console.log')
        execute([args.planning_python, str(HERE / 'replay_vlm_rgbd.py'), '--visual-dir', str(baseline),
            '--memory-dir', str(memory), '--max-views', '9', '--output-dir', str(detections)],
            root / 'baseline_detection.console.log')
        report['camera_policy'] = 'fixed_old_pose_yaw_scan_not_new_target_oracle'
    except Exception as exc:
        report.update(status='failed', failed_stage='scan_baseline', error=str(exc))
        write(root / 'regression_summary.json', report)
        print(f"status=failed summary={root / 'regression_summary.json'}", flush=True)
        return 2
    for name, delta in [('x_50cm', [.5, 0., 0.]), ('diagonal_35cm', [.35, 0., .35])]:
        row = {'case': name, 'translation_habitat': delta, 'status': 'running'}
        stage = 'visual_repair'
        print(f'START case={name}', flush=True)
        try:
            case = root / name
            case.mkdir()
            move = case / 'visual_move'
            execute([sys.executable, str(HERE / 'run_visual_move_regression.py'),
                '--planning-python', args.planning_python, '--memory-dir', str(memory),
                '--before-visual-dir', str(baseline),
                '--before-detections', str(detections),
                '--capture-followup', str(followup),
                '--translation-habitat', *map(str, delta), '--output-dir', str(move)], case / 'visual_move.console.log')
            stage = 'fresh_planning_and_navigation'
            nav = case / 'navigation'
            execute([sys.executable, str(HERE / 'run_visual_repaired_navigation.py'),
                '--planning-python', args.planning_python, '--visual-regression-dir', str(move),
                '--seeds', '201', '202', '203', '--output-dir', str(nav)], case / 'navigation.console.log')
            row.update(case_metrics(read(move / 'regression_summary.json'), read(nav / 'regression_summary.json')))
            row['status'] = 'passed' if row['source_memories_unchanged'] else 'failed'
        except Exception as exc:
            row.update(status='failed_or_rejected', failed_stage=stage, error=str(exc))
        report['cases'].append(row)
        write(root / 'regression_summary.json', report)
        print(f"END case={name} status={row['status']}", flush=True)
    report['source_memory_unchanged'] = digest(read(memory / 'memory_state.json')) == source_hash
    report['status'] = 'passed' if all(r['status'] == 'passed' for r in report['cases']) and report['source_memory_unchanged'] else 'needs_review'
    write(root / 'regression_summary.json', report)
    print(f"status={report['status']} summary={root / 'regression_summary.json'}", flush=True)
    return 0 if report['status'] == 'passed' else 2


if __name__ == '__main__':
    raise SystemExit(main())
