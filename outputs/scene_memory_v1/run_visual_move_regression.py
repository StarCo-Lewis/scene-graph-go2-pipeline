#!/usr/bin/env python3
"""Controlled stale-memory displacement -> RGB-only inference -> isolated visual update."""
import argparse
import math
import os
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from run_full_multiround import execute, SCENE, HERE
from update_memory import read, write, digest


def validate_api_environment(cached=False):
    if not cached and not os.environ.get('DASHSCOPE_API_KEY', '').strip():
        raise ValueError('DASHSCOPE_API_KEY required before capture; launch via run_scene201_visual_move.sh')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--planning-python', default='/home/lewis/miniforge3/envs/conceptgraph/bin/python')
    p.add_argument('--baseline-run', type=Path, default=SCENE / 'full_multiround_memory/scene201_20260926_150648_443503')
    p.add_argument('--before-detections', type=Path, default=SCENE / 'vlm_rgbd_memory_pilot/scene201_20260926_152840_427492')
    p.add_argument('--memory-dir', type=Path, help='Explicit latest physical memory checkpoint')
    p.add_argument('--before-visual-dir', type=Path)
    p.add_argument('--capture-followup', type=Path)
    p.add_argument('--previous-visual-run', type=Path)
    p.add_argument('--translation-habitat', type=float, nargs=3, default=[.25, 0., 0.])
    p.add_argument('--cached-after-detections', type=Path)
    p.add_argument('--test-occlusion', choices=('none', 'partial', 'full'), default='none',
                   help='Controlled full RGB-D dropout before repair; failure must not commit memory')
    p.add_argument('--dropout-modality', choices=('rgb', 'depth', 'joint'), default='joint')
    p.add_argument('--output-dir', type=Path, default=SCENE / 'visual_move_regression' / datetime.now().strftime('scene201_%Y%m%d_%H%M%S_%f'))
    args = p.parse_args()
    try:
        validate_api_environment(bool(args.cached_after_detections))
    except ValueError as exc:
        p.error(str(exc))
    root = args.output_dir
    root.mkdir(parents=True, exist_ok=False)
    memory = args.memory_dir or args.baseline_run / 'round_1/visual/fused_memory'
    before_visual = args.before_visual_dir or args.baseline_run / 'round_2/visual'
    state = read(memory / 'memory_state.json')
    state_hash = digest(state)
    summary = {'status': 'running', 'training_eligible': False, 'source_memory': str(memory),
        'scope': 'Controlled known-kettle translation and isolated memory repair; not general robot perception',
        'oracle_delta_available_to_estimator': False}
    stage = 'capture'
    try:
        after_visual = root / 'after_capture'
        camera_policy = read(before_visual / 'view_0.json').get('camera_orientation_policy', 'legacy')
        command = [sys.executable, str(HERE / 'capture_visual_memory.py'), '--memory-dir', str(memory),
            '--camera-policy', camera_policy,
            '--test-translation-habitat', *map(str, args.translation_habitat),
            '--output-dir', str(after_visual)]
        if args.previous_visual_run:
            command += ['--previous-visual-run', str(args.previous_visual_run)]
            summary['previous_visual_run'] = str(args.previous_visual_run)
        else:
            command += ['--followup-execution', str(args.capture_followup or args.baseline_run / 'round_2/execution/followup_execution.json')]
        execute(command, root / 'capture.console.log')
        stage = 'VLM_SAM_after_observation'
        after_detections = root / 'after_detection'
        command = [args.planning_python, str(HERE / 'replay_vlm_rgbd.py'), '--visual-dir', str(after_visual),
            '--memory-dir', str(memory), '--translation-evidence', '--max-views',
            '9' if camera_policy == 'scan' else '5', '--output-dir', str(after_detections)]
        if args.test_occlusion != 'none':
            if args.cached_after_detections:
                raise ValueError('Dropout requires fresh inference, not cached detections')
            command += ['--test-occlusion', args.test_occlusion, '--dropout-modality', args.dropout_modality]
            if args.test_occlusion == 'partial':
                command += ['--occlusion-fixture', str(args.before_detections), '--occlusion-fraction', '.5']
        if args.cached_after_detections:
            command += ['--cached-detections', str(args.cached_after_detections)]
        execute(command, root / 'detection.console.log')
        stage = 'translation_estimation'
        estimate_path = root / 'estimated_translation.json'
        # Deliberately exclude capture_evaluation_only.json and the requested delta.
        execute([args.planning_python, str(HERE / 'estimate_visual_translation.py'),
            '--before-visual', str(before_visual), '--before-detections', str(args.before_detections),
            '--after-visual', str(after_visual), '--after-detections', str(after_detections),
            '--output', str(estimate_path)], root / 'registration.console.log')
        stage = 'evaluation'
        estimate = read(estimate_path)
        truth = read(after_visual / 'capture_evaluation_only.json')['test_translation_cg']
        delta = estimate['translation_cg']
        error = math.dist(delta, truth)
        if error > .05:
            raise ValueError(f'Translation error {error:.4f}m exceeds 0.05m; do not update memory')
        old = state['entities']['67']['node']['bbox_center']
        new = [old[i]+delta[i] for i in range(3)]
        timestamps = [row['event']['observed_at'] for row in state['event_log']]
        now = datetime.now(timezone.utc)
        if timestamps:
            now = max(now, max(datetime.fromisoformat(t.replace('Z', '+00:00')) for t in timestamps)+timedelta(seconds=1))
        event = {'event_id': 'visual-translation-'+digest(estimate), 'action_id': 'reconcile_visual_translation',
            'operation': 'move', 'object_id': 67, 'action_status': 'completed', 'observed_at': now.isoformat(),
            'coordinate_frame': 'conceptgraphs_world', 'position': new,
            'evidence': {'source': 'verified_post_action_observation', 'verified': True,
                'reference': str(estimate_path), 'verification_scope': 'controlled_fixture_passed_consensus_and_held_out_translation_evaluation',
                'estimated_from': 'VLM_SAM_RGBD_translation_registration', 'rotation_estimated': False}}
        events_path = root / 'visual_move_events.json'
        write(events_path, [event])
        stage = 'isolated_memory_update'
        snapshot = root / 'repaired_memory'
        execute([sys.executable, str(HERE / 'update_memory.py'), '--previous-state', str(memory / 'memory_state.json'),
            '--events-json', str(events_path), '--output-dir', str(snapshot)], root / 'memory.console.log')
        repaired = read(snapshot / 'memory_state.json')
        fixture = read(after_visual / 'capture_evaluation_only.json')
        expected = fixture.get('actual_bbox_after_cg', [old[i]+truth[i] for i in range(3)])
        if math.dist(repaired['entities']['67']['node']['bbox_center'], expected) > .05:
            raise ValueError('Cumulative position error exceeds 0.05m; snapshot not accepted')
        summary.update(status='passed', estimated_translation_cg=delta, evaluation_translation_cg=truth,
            translation_error_m=error, stale_position_error_m=math.dist(old, expected),
            repaired_position_error_m=math.dist(repaired['entities']['67']['node']['bbox_center'], expected),
            old_bbox_center=old, updated_bbox_center=new, old_memory_version=state['version'],
            new_memory_version=repaired['version'], repaired_memory=str(snapshot),
            source_memory_unchanged=digest(read(memory / 'memory_state.json')) == state_hash,
            limitations=['Camera poses still simulator-known', 'Translation only; single known instance',
                'Evaluation truth gates test-snapshot creation but never enters translation estimation',
                'Repaired map not yet compatible with old proxy checkpoint restoration; no navigation executed',
                'Support relation not inferred from visual translation'])
        if not summary['source_memory_unchanged']:
            summary['status'] = 'failed'
    except Exception as exc:
        summary.update(status='failed', failed_stage=stage, error=str(exc))
        print(f'ERROR stage={stage}: {exc}', flush=True)
    write(root / 'regression_summary.json', summary)
    print(f"status={summary['status']} summary={root / 'regression_summary.json'}", flush=True)
    return 0 if summary['status'] == 'passed' else 2


if __name__ == '__main__':
    raise SystemExit(main())
