#!/usr/bin/env python3
"""Isolated memory guard regression; no API, simulator or real scene deletion."""
import argparse
import copy
import subprocess
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from update_memory import read, write, digest, apply_events

ROOT = Path('/mnt/d/NebulaAI/output/batches/scene201_vith_full_20260915_181750/seq_0001__hf_house_201')
HERE = Path(__file__).resolve().parent


def run(args):
    state = read(args.memory_dir / 'memory_state.json')
    original = digest(state)
    base = Path(state['base_scene'])
    base_digest = digest([read(base / 'scene_graph.json'), read(base / 'scene_graph_edges.json')])
    args.output_dir.mkdir(parents=True, exist_ok=False)
    cases = []

    def check(name, condition):
        cases.append({'name': name, 'passed': bool(condition)})
        print(f'END {name} passed={bool(condition)}', flush=True)

    event = next(entry['event'] for entry in reversed(state['event_log'])
                 if entry['event']['operation'] == 'move' and entry['event']['object_id'] == 67)
    replay = apply_events(state, [event, event])
    check('duplicate_move_is_exact_noop', replay == state)
    conflicting = copy.deepcopy(event)
    conflicting['position'][0] += 1
    try:
        apply_events(state, [conflicting])
    except ValueError:
        check('conflicting_event_id_rejected', True)
    else:
        check('conflicting_event_id_rejected', False)

    deletion = {'event_id': 'isolated-test-delete-67', 'action_id': 'test_remove',
        'operation': 'delete', 'object_id': 67, 'reason': 'removed_confirmed',
        'action_status': 'completed',
        'observed_at': (datetime.fromisoformat(event['observed_at'].replace('Z', '+00:00'))
                        + timedelta(seconds=1)).isoformat(),
        'evidence': {'source': 'operator_confirmation', 'verified': True,
                     'reference': 'synthetic-test-fixture-not-real-observation'}}
    events_path = args.output_dir / 'synthetic_events.json'
    write(events_path, [deletion])
    snapshot = args.output_dir / 'deleted_snapshot'
    subprocess.run([sys.executable, str(HERE / 'update_memory.py'), '--previous-state',
        str(args.memory_dir / 'memory_state.json'), '--events-json', str(events_path),
        '--output-dir', str(snapshot)], check=True)
    deleted = read(snapshot / 'memory_state.json')
    nodes = read(snapshot / 'scene/scene_graph.json')
    check('deleted_id_absent_from_active_graph', 67 not in [n['id'] for n in nodes]
          and not deleted['entities']['67']['active'])
    check('tombstone_and_history_preserved', '67' in deleted['entities']
          and deleted['version'] == state['version'] + 1)
    check('delete_replay_is_noop', apply_events(deleted, [deletion]) == deleted)
    check('inherited_features_unchanged',
          (snapshot / 'scene/map/scene_map_cfslam_pruned.pkl.gz').read_bytes()
          == (base / 'map/scene_map_cfslam_pruned.pkl.gz').read_bytes())

    # Real aligner invocation using an existing validated task, not an LLM call.
    result_path = args.output_dir / 'deleted_object_alignment.json'
    command = [args.planning_python, str(HERE.parent / 'object_alignment_v1/align_objects.py'),
        '--scene-dir', str(snapshot / 'scene'), '--decomposition-json', str(args.decomposition),
        '--output', str(result_path), '--log-file', str(args.output_dir / 'alignment.log')]
    with (args.output_dir / 'alignment.console.log').open('w', encoding='utf-8') as stream:
        completed = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
    result = read(result_path) if result_path.exists() else {}
    check('deleted_kettle_returns_no_object', completed.returncode == 2
          and result.get('error', {}).get('message') == 'NO object can be used in this task'
          and result.get('error', {}).get('code') == 'NO_OBJECT_CAN_BE_USED_IN_THIS_TASK')
    check('source_memory_and_base_graph_unchanged',
          digest(read(args.memory_dir / 'memory_state.json')) == original
          and digest([read(base / 'scene_graph.json'), read(base / 'scene_graph_edges.json')]) == base_digest)
    summary = {'status': 'passed' if all(c['passed'] for c in cases) else 'failed',
        'cases': cases, 'scope': 'Synthetic isolated deletion and replay protection; not a real removal observation',
        'api_called': False, 'simulator_executed': False, 'training_eligible': False,
        'source_memory': str(args.memory_dir), 'output_dir': str(args.output_dir)}
    write(args.output_dir / 'regression_summary.json', summary)
    print(f"status={summary['status']} summary={args.output_dir / 'regression_summary.json'}")
    return 0 if summary['status'] == 'passed' else 2


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--memory-dir', type=Path, default=ROOT / 'habitat_task_with_scene_memory/scene201_20260926_135458_6541/scene_memory')
    parser.add_argument('--decomposition', type=Path, default=ROOT / 'memory_followup_planning/scene201_20260926_140047_110543/task_decomposition.json')
    parser.add_argument('--planning-python', default='/home/lewis/miniforge3/envs/conceptgraph/bin/python')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'memory_guard_regression' / ('scene201_' + datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')))
    raise SystemExit(run(parser.parse_args()))
