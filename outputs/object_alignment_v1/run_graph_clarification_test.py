"""Clarified duplicate-instance task through the existing planning pipeline."""
import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from run_graph_binding_regression import cases


def check_result(summary, alignment, plan):
    ids = [i for binding in alignment.get('context_bindings', {}).values() for i in binding.get('element_ids', [])]
    steps = [step for subtask in plan.get('subtask_plans', []) for step in subtask.get('steps', [])]
    return (summary.get('status') == 'planning_passed' and alignment.get('status') == 'success'
            and ids == [6] and len(steps) == 1
            and steps[0].get('skill_id') == 'navigate_to_object'
            and steps[0].get('arguments', {}).get('target', {}).get('element_id') == 6)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    if not os.environ.get('DASHSCOPE_API_KEY'):
        parser.error('DASHSCOPE_API_KEY required')
    here = Path(__file__).resolve().parent
    original = json.loads((here / 'graph_demo/scene/scene_graph.json').read_text(encoding='utf-8'))
    nodes = cases(original)[1][1]
    root = args.output_dir or Path('/mnt/d/NebulaAI/output/graph_clarification_test') / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    root.mkdir(parents=True, exist_ok=False)
    scene = root / 'scene'; scene.mkdir()
    (scene / 'scene_graph.json').write_text(json.dumps(nodes, indent=2), encoding='utf-8')
    (scene / 'scene_graph_edges.json').write_text('[]', encoding='utf-8')
    command = [sys.executable, str(here.parent / 'habitat_demo_v1/run_sideboard_full_pipeline.py'),
               '--planning-python', sys.executable, '--scene-dir', str(scene),
               '--matching-mode', 'llm_graph', '--instruction', 'Go to the red coke can with object ID 6.',
               '--expected-target-id', '6', '--planning-only', '--output-dir', str(root / 'results')]
    print('START clarified_can target_id=6', flush=True)
    with (root / 'console.log').open('w', encoding='utf-8') as log:
        completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    def read(name):
        path = root / 'results' / name
        return json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
    passed = completed.returncode == 0 and check_result(read('pipeline_summary.json'), read('object_alignment.json'), read('skill_plan_final.json'))
    report = {'status': 'passed' if passed else 'failed', 'passed': passed, 'exit_code': completed.returncode,
              'expected_target_id': 6, 'mode': 'real_llm', 'execution_status': 'not_executed',
              'training_eligible': False, 'results': str(root / 'results')}
    (root / 'regression_summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f'END clarified_can passed={passed} exit_code={completed.returncode}', flush=True)
    print('summary=' + str(root / 'regression_summary.json'), flush=True)
    return 0 if passed else 2


if __name__ == '__main__':
    raise SystemExit(main())
