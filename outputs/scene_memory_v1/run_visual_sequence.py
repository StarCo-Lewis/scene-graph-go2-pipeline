"""Carry repaired memory and independent fixture truth across two further moves."""
import sys
from datetime import datetime
from pathlib import Path
from run_full_multiround import HERE, SCENE, execute
from update_memory import read, write, digest
from visual_checkpoint import fixture_contract


def sequence_moves(count, offset=0):
    if not 2 <= count <= 6 or offset < 0 or offset+count > 6:
        raise ValueError('Sequence requires 2..6 moves')
    # Bounded back-and-forth motion; no gradual drift outside the camera budget.
    moves = [[.15, 0., 0.], [-.20, 0., .10], [.15, 0., -.10],
             [-.15, 0., .10], [.10, 0., -.10], [-.10, 0., 0.]]
    return moves[offset:offset+count]


def evidence_guards(previous, state):
    from run_chain_suite import rejection_checks
    before = digest(state)
    checks = rejection_checks(read(previous / 'estimated_translation.json'), state)
    if not all(c['passed'] for c in checks) or digest(state) != before:
        raise ValueError('Synthetic evidence rejection failed or mutated memory')
    return {'status': 'passed', 'checks': checks, 'memory_unchanged': True,
            'scope': 'Synthetic single-view/conflict/high-residual/zero-motion/unverified-event guards, not rendered occlusion or API-outage recovery'}


def run(args):
    previous = args.previous_run
    evaluation = read(previous / 'regression_summary.json')
    memory = Path(evaluation['repaired_memory'])
    fixture_contract(memory, previous)  # validate before spending API calls
    root = args.output_dir or SCENE / 'visual_sequential_memory' / datetime.now().strftime('scene201_%Y%m%d_%H%M%S_%f')
    root.mkdir(parents=True, exist_ok=False)
    offset = getattr(args, 'move_offset', 0)
    moves = sequence_moves(getattr(args, 'rounds', 2), offset)
    summary = {'status': 'running', 'training_eligible': False, 'initial_visual_checkpoint': str(previous),
        'requested_rounds': len(moves), 'synthetic_guard_reports': [],
        'preset_move_offset': offset,
        'rounds': [], 'scope': 'Sequential memory and fixture truth, checkpoint-restored simulation, not uninterrupted robot execution',
        'limitations': ['Known kettle; fixed-budget scan and simulator camera calibration',
                        'Fixture truth only for rendering/evaluation, not registration or LLM planning',
                        'Translation only, assumed grasp, inherited navmesh; stale navigation may pass']}
    sources = {}
    stage = 'preflight'
    try:
        for index, delta in enumerate(moves, 1):
            folder = root / f'round_{index}'
            folder.mkdir()
            state = read(memory / 'memory_state.json')
            sources[str(memory)] = digest(state)
            stage = f'round_{index}_synthetic_evidence_guards'
            guard = evidence_guards(previous, state)
            write(folder / 'synthetic_evidence_guards.json', guard)
            summary['synthetic_guard_reports'].append({'round': index, **guard})
            write(root / 'regression_summary.json', summary)
            stage = f'round_{index}_baseline'
            baseline, detections = folder / 'baseline', folder / 'baseline_detection'
            execute([sys.executable, str(HERE / 'capture_visual_memory.py'), '--memory-dir', str(memory),
                '--previous-visual-run', str(previous), '--camera-policy', 'scan', '--output-dir', str(baseline)], folder / 'capture.console.log')
            execute([args.planning_python, str(HERE / 'replay_vlm_rgbd.py'), '--visual-dir', str(baseline),
                '--memory-dir', str(memory), '--max-views', '9', '--output-dir', str(detections)], folder / 'baseline.console.log')
            stage = f'round_{index}_repair'
            move = folder / 'visual_move'
            execute([sys.executable, str(HERE / 'run_visual_move_regression.py'), '--planning-python', args.planning_python,
                '--memory-dir', str(memory), '--previous-visual-run', str(previous), '--before-visual-dir', str(baseline),
                '--before-detections', str(detections), '--translation-habitat', *map(str, delta),
                '--output-dir', str(move)], folder / 'repair.console.log')
            result = read(move / 'regression_summary.json')
            if result['old_memory_version'] != state['version'] or result['new_memory_version'] != state['version']+1:
                raise ValueError('Sequential memory version transition mismatch')
            stage = f'round_{index}_navigation'
            nav = folder / 'navigation'
            execute([sys.executable, str(HERE / 'run_visual_repaired_navigation.py'), '--planning-python', args.planning_python,
                '--visual-regression-dir', str(move), '--seeds', '201', '202', '203', '--output-dir', str(nav)], folder / 'navigation.console.log')
            navigation = read(nav / 'regression_summary.json')
            summary['rounds'].append({'round': index, 'status': 'passed', 'source_memory': str(memory),
                'preset_move_number': offset+index,
                'physical_translation_habitat': delta,
                'result_memory': result['repaired_memory'], 'old_version': result['old_memory_version'],
                'new_version': result['new_memory_version'], 'local_translation_error_m': result['translation_error_m'],
                'cumulative_position_error_m': result['repaired_position_error_m'],
                'repaired_navigation_success_count': navigation['repaired_navigation_success_count'],
                'stale_navigation_success_count': navigation['stale_navigation_success_count']})
            previous, memory = move, Path(result['repaired_memory'])
            write(root / 'regression_summary.json', summary)
        summary['source_memories_unchanged'] = all(digest(read(Path(m) / 'memory_state.json')) == h for m, h in sources.items())
        summary['status'] = 'passed' if summary['source_memories_unchanged'] else 'failed'
        summary['final_memory'] = str(memory)
    except Exception as exc:
        summary.update(status='failed', failed_stage=stage, error=str(exc))
    write(root / 'regression_summary.json', summary)
    print(f"status={summary['status']} summary={root / 'regression_summary.json'}", flush=True)
    return 0 if summary['status'] == 'passed' else 2
