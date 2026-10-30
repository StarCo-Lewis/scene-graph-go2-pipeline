"""Synthetic event-memory + real LLM replanning; no physical execution."""
import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from run_graph_relation_test import relation_fixture

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scene_memory_v1'))
from update_memory import initialize, apply_events, digest


def move_fixture(state, object_id, support_id, position, event_id):
    event = {'event_id': event_id, 'action_id': event_id, 'operation': 'move',
             'object_id': object_id, 'support_object_id': support_id, 'position': position,
             'coordinate_frame': state['coordinate_frame'], 'action_status': 'completed',
             'observed_at': datetime.now(timezone.utc).isoformat(),
             'evidence': {'verified': True, 'source': 'operator_confirmation',
                          'reference': 'synthetic_regression_fixture:' + event_id},
             'synthetic_fixture': True}
    return apply_events(state, [event]), event


def valid_selected(alignment, plan, target):
    ids = [i for b in alignment.get('context_bindings', {}).values() for i in b.get('element_ids', [])]
    steps = [s for p in plan.get('subtask_plans', []) for s in p.get('steps', [])]
    checks = alignment.get('alignment', {}).get('relation_checks', [])
    return (alignment.get('status') == 'success' and set(ids) == {target, 2} and len(steps) == 1
            and steps[0].get('skill_id') == 'navigate_to_object'
            and steps[0].get('arguments', {}).get('target', {}).get('element_id') == target
            and any(c.get('relation') == 'on' and c.get('status') == 'observed_true' for c in checks))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    if not os.environ.get('DASHSCOPE_API_KEY'):
        parser.error('DASHSCOPE_API_KEY required')
    here = Path(__file__).resolve().parent
    original = json.loads((here / 'graph_demo/scene/scene_graph.json').read_text(encoding='utf-8'))
    nodes, edges = relation_fixture(original)
    state = initialize(nodes, edges, 'synthetic_graph_fixture')
    state['coordinate_frame'] = 'test_graph_world'
    baseline = digest(state)
    root = args.output_dir or Path('/mnt/d/NebulaAI/output/graph_memory_test') / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    root.mkdir(parents=True, exist_ok=False)
    report = {'status': 'running', 'tests': [], 'synthetic_state_events': True,
              'mode': 'real_llm_planning_with_synthetic_memory_events', 'execution_status': 'not_executed',
              'training_eligible': False, 'events': []}
    def save():
        (root / 'regression_summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    save()
    instruction = 'Go to the red coke can that is on the main table. Use only observed edges to determine which can; do not infer the relation from coordinates.'
    for index, target in enumerate([6, None, 4]):
        if index == 1:
            state, event = move_fixture(state, 6, 7, [1.3, 1.4, .82], 'move_can6_away')
            report['events'].append(event)
        elif index == 2:
            state, event = move_fixture(state, 4, 2, [2.8, 1.0, .82], 'move_can4_to_main')
            report['events'].append(event)
        folder = root / ('round_' + str(index + 1)); scene = folder / 'scene'; scene.mkdir(parents=True)
        snapshot = [e['node'] for e in state['entities'].values() if e['active']]
        def write(path, data):
            path.write_text(json.dumps(data, indent=2), encoding='utf-8')
        write(scene / 'scene_graph.json', snapshot)
        write(scene / 'scene_graph_edges.json', state['edges'])
        write(folder / 'memory_state.json', state)
        before = digest(state)
        command = [sys.executable, str(here.parent / 'habitat_demo_v1/run_sideboard_full_pipeline.py'),
                   '--planning-python', sys.executable, '--scene-dir', str(scene), '--matching-mode', 'llm_graph',
                   '--instruction', instruction, '--expected-target-id', str(target or 6),
                   '--planning-only', '--output-dir', str(folder / 'results')]
        print(f'START round_{index+1} memory_version={state["version"]} expected_target={target}', flush=True)
        with (folder / 'console.log').open('w', encoding='utf-8') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        def read(name):
            path = folder / 'results' / name
            return json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
        alignment = read('object_alignment.json'); code = alignment.get('error', {}).get('code')
        passed = (result.returncode == 0 and read('pipeline_summary.json').get('status') == 'planning_passed'
                  and valid_selected(alignment, read('skill_plan_final.json'), target)) if target else (
            result.returncode != 0 and alignment.get('status') == 'error' and code in
            ('NO_OBJECT_CAN_BE_USED_IN_THIS_TASK', 'OBJECT_BINDING_AMBIGUOUS', 'OBJECT_RELATION_UNVERIFIED'))
        unchanged = digest(state) == before
        report['tests'].append({'round': index+1, 'expected_target': target, 'passed': passed and unchanged,
                                'memory_version': state['version'], 'planning_did_not_mutate_memory': unchanged,
                                'error_code': code, 'exit_code': result.returncode})
        save(); print(f'END round_{index+1} passed={passed and unchanged} exit_code={result.returncode}', flush=True)
    report['initial_fixture_digest'] = baseline
    report['status'] = 'passed' if all(t['passed'] for t in report['tests']) else 'failed'
    save(); print('summary=' + str(root / 'regression_summary.json'), flush=True)
    return 0 if report['status'] == 'passed' else 2


if __name__ == '__main__':
    raise SystemExit(main())
