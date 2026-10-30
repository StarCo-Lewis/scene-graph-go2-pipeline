"""Blocked visual commit -> visible input replay -> audited repair -> fresh navigation."""
import sys
from pathlib import Path
from datetime import datetime
from run_full_multiround import HERE, SCENE, execute
from update_memory import read, write, digest
from visual_checkpoint import fixture_contract


def blocked_commit_safe(failure, detection, state_before, state_after, committed):
    return (failure.get('status') == 'failed'
        and failure.get('failed_stage') == 'VLM_SAM_after_observation'
        and detection.get('status') == 'insufficient_visual_evidence'
        and len(detection.get('frames', [])) >= 2
        and all(f.get('status') != 'accepted' for f in detection['frames'])
        and not committed and digest(state_before) == digest(state_after)
        and state_after['entities']['67']['active'] is True)


def category_only_guard(detection, audit):
    observations = audit.get('observations', [])
    return (audit.get('graph_commit_performed') is False
        and sum(o.get('observation_state') == 'category_only_position_unresolved' for o in observations) >= 2
        and detection.get('category_only_view_count', 0) >= 2
        and all(o.get('position_update_allowed') is False and o.get('delete_allowed') is False for o in observations))


def recovery_dropout_arguments(mode):
    if mode not in ('full', 'depth'):
        raise ValueError('Unknown recovery dropout mode')
    return ['--test-occlusion', 'full', '--dropout-modality', 'depth' if mode == 'depth' else 'joint']


def run(args):
    previous = args.previous_run
    prior = read(previous / 'regression_summary.json')
    memory = Path(prior['repaired_memory'])
    fixture_contract(memory, previous)
    state = read(memory / 'memory_state.json')
    root = args.output_dir or SCENE / 'visual_recovery_chain' / datetime.now().strftime('scene201_%Y%m%d_%H%M%S_%f')
    root.mkdir(parents=True, exist_ok=False)
    report = {'status': 'running', 'training_eligible': False, 'source_memory': str(memory), 'stages': [],
        'scope': 'Same moved checkpoint, full input dropout then fresh visible replay and repair; not autonomous re-observation',
        'limitations': ['Synthetic whole-frame RGB/depth dropout, not physical occlusion',
                        'Visible recovery restores the same predetermined camera rig, not autonomous search',
                        'Known kettle, simulator calibration and evaluation truth, no physical grasp']}
    dropout = getattr(args, 'recovery_dropout', 'full')
    report['dropout_mode'] = dropout
    report['dropout_scope'] = ('Whole-frame depth unavailable, original RGB retained' if dropout == 'depth'
                               else 'Whole-frame RGB and depth unavailable')
    report['limitations'][0] = 'Synthetic sensor-input dropout, not physical occlusion'
    stage = 'baseline'
    try:
        baseline, before = root / 'baseline', root / 'baseline_detection'
        execute([sys.executable, str(HERE / 'capture_visual_memory.py'), '--memory-dir', str(memory),
            '--previous-visual-run', str(previous), '--camera-policy', 'scan', '--output-dir', str(baseline)], root / 'baseline_capture.console.log')
        execute([args.planning_python, str(HERE / 'replay_vlm_rgbd.py'), '--visual-dir', str(baseline),
            '--memory-dir', str(memory), '--max-views', '9', '--output-dir', str(before)], root / 'baseline_detection.console.log')
        common = [sys.executable, str(HERE / 'run_visual_move_regression.py'), '--planning-python', args.planning_python,
            '--memory-dir', str(memory), '--previous-visual-run', str(previous), '--before-visual-dir', str(baseline),
            '--before-detections', str(before), '--translation-habitat', '.10', '0', '0']
        stage = 'blocked_update'
        blocked = root / 'blocked'
        # Full-frame depth loss preserves RGB category evidence but makes any
        # 3-D candidate impossible. Partial loss may legitimately pass.
        dropout_arguments = recovery_dropout_arguments(dropout)
        try:
            execute(common + dropout_arguments + ['--output-dir', str(blocked)], root / 'blocked.console.log')
        except ValueError:
            pass  # Only the checked evidence-rejection condition below is expected.
        safe = blocked_commit_safe(read(blocked / 'regression_summary.json'),
            read(blocked / 'after_detection/regression_summary.json'), state,
            read(memory / 'memory_state.json'), (blocked / 'repaired_memory').exists() or (blocked / 'visual_move_events.json').exists())
        if not safe:
            raise ValueError('Dropout did not produce safe rejection without a memory commit')
        detection = read(blocked / 'after_detection/regression_summary.json')
        observation_log = blocked / 'after_detection/semantic_observation_log.json'
        audit = read(observation_log)
        if dropout == 'depth' and not category_only_guard(detection, audit):
            raise ValueError('Depth dropout did not demonstrate retained category-only evidence; cannot count as success')
        report['stages'].append({'stage': stage, 'status': 'passed_expected_rejection', 'memory_version': state['version'], 'memory_unchanged': True})
        report['stages'][-1].update(observation_log=str(observation_log),
            category_recognized_views=detection.get('category_recognized_view_count', 0),
            category_only_views=detection.get('category_only_view_count', 0))
        write(root / 'regression_summary.json', report)
        stage = 'visible_recovery_and_repair'
        recovered = root / 'recovered'
        execute(common + ['--output-dir', str(recovered)], root / 'recovery.console.log')
        # Rendering restores the same checkpoint and delta; no double move.
        a = read(blocked / 'after_capture/capture_evaluation_only.json')
        b = read(recovered / 'after_capture/capture_evaluation_only.json')
        for key in ('actual_bbox_before_cg', 'actual_bbox_after_cg', 'test_translation_habitat'):
            if a[key] != b[key]:
                raise ValueError('Recovery replay changed the physical checkpoint')
        repair = read(recovered / 'regression_summary.json')
        if repair['new_memory_version'] != state['version']+1:
            raise ValueError('Recovery must commit exactly one version transition')
        report['stages'].append({'stage': stage, 'status': 'passed', 'new_version': repair['new_memory_version'],
            'cumulative_position_error_m': repair['repaired_position_error_m']})
        stage = 'fresh_planning_and_navigation'
        nav = root / 'navigation'
        execute([sys.executable, str(HERE / 'run_visual_repaired_navigation.py'), '--planning-python', args.planning_python,
            '--visual-regression-dir', str(recovered), '--seeds', '201', '202', '203', '--output-dir', str(nav)], root / 'navigation.console.log')
        result = read(nav / 'regression_summary.json')
        report['stages'].append({'stage': stage, 'status': result['status'],
            'repaired_navigation_success_count': result['repaired_navigation_success_count'],
            'stale_navigation_success_count': result['stale_navigation_success_count']})
        report.update(status='passed', final_memory=repair['repaired_memory'],
            source_memory_unchanged=digest(read(memory / 'memory_state.json')) == digest(state))
        if not report['source_memory_unchanged']:
            report['status'] = 'failed'
    except Exception as exc:
        report.update(status='failed', failed_stage=stage, error=str(exc))
    write(root / 'regression_summary.json', report)
    print(f"status={report['status']} summary={root / 'regression_summary.json'}", flush=True)
    return 0 if report['status'] == 'passed' else 2
