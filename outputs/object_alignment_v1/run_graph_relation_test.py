"""Real LLM relation disambiguation; no simulator execution."""
import argparse
import copy
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from run_graph_binding_regression import cases


def relation_fixture(original):
    nodes = cases(original)[1][1]
    table = copy.deepcopy(next(n for n in nodes if n['id'] == 2))
    table.update(id=7, object_tag='side table', caption='A small side table', bbox_center=[1.2, 1.4, .4])
    nodes.append(table)
    return nodes, [{'source': 6, 'target': 2, 'relation': 'a on b'},
                   {'source': 4, 'target': 7, 'relation': 'a on b'}]


def relation_selection_passed(summary, alignment, plan):
    # The contextual table role is required too; do not use the ID-only check.
    bound = [i for b in alignment.get('context_bindings', {}).values() for i in b.get('element_ids', [])]
    steps = [s for p in plan.get('subtask_plans', []) for s in p.get('steps', [])]
    checks = alignment.get('alignment', {}).get('relation_checks', [])
    return (summary.get('status') == 'planning_passed' and alignment.get('status') == 'success'
            and set(bound) == {2, 6} and len(steps) == 1
            and steps[0].get('skill_id') == 'navigate_to_object'
            and steps[0].get('arguments', {}).get('target', {}).get('element_id') == 6
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
    root = args.output_dir or Path('/mnt/d/NebulaAI/output/graph_relation_test') / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    root.mkdir(parents=True, exist_ok=False)
    report = {'status': 'running', 'tests': [], 'mode': 'real_llm', 'execution_status': 'not_executed', 'training_eligible': False}
    def save():
        (root / 'regression_summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    save()
    instruction = 'Go to the red coke can that is on the main table. Use only observed edges to determine which can; do not infer the relation from coordinates.'
    for name, selected_edges in [('relation_present', edges), ('relation_missing', [])]:
        folder = root / name; scene = folder / 'scene'; scene.mkdir(parents=True)
        (scene / 'scene_graph.json').write_text(json.dumps(nodes, indent=2), encoding='utf-8')
        (scene / 'scene_graph_edges.json').write_text(json.dumps(selected_edges, indent=2), encoding='utf-8')
        command = [sys.executable, str(here.parent / 'habitat_demo_v1/run_sideboard_full_pipeline.py'),
                   '--planning-python', sys.executable, '--scene-dir', str(scene), '--matching-mode', 'llm_graph',
                   '--instruction', instruction, '--expected-target-id', '6', '--planning-only', '--output-dir', str(folder / 'results')]
        print('START ' + name, flush=True)
        with (folder / 'console.log').open('w', encoding='utf-8') as log:
            completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        def read(filename):
            path = folder / 'results' / filename
            return json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
        alignment = read('object_alignment.json')
        code = alignment.get('error', {}).get('code')
        passed = (completed.returncode == 0 and relation_selection_passed(read('pipeline_summary.json'), alignment, read('skill_plan_final.json'))) if selected_edges else (
            completed.returncode != 0 and alignment.get('status') == 'error' and code in ('OBJECT_BINDING_AMBIGUOUS', 'OBJECT_RELATION_UNVERIFIED'))
        report['tests'].append({'name': name, 'passed': passed, 'exit_code': completed.returncode,
                                'error_code': code, 'output_dir': str(folder)})
        save(); print(f'END {name} passed={passed} exit_code={completed.returncode}', flush=True)
    report['status'] = 'passed' if all(t['passed'] for t in report['tests']) else 'failed'
    save(); print('summary=' + str(root / 'regression_summary.json'), flush=True)
    return 0 if report['status'] == 'passed' else 2


if __name__ == '__main__':
    raise SystemExit(main())
