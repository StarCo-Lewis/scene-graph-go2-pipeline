"""Real-API negative graph binding tests; no simulator or physical execution."""
import argparse
import copy
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def cases(nodes):
    duplicate = copy.deepcopy(nodes)
    can = copy.deepcopy(next(n for n in nodes if n['id'] == 4))
    can['id'] = 6
    can['bbox_center'] = [1.2, 1.4, .82]
    duplicate.append(can)
    return [
        ('missing_apple', copy.deepcopy(nodes), 'Go to an apple.', 'NO_OBJECT_CAN_BE_USED_IN_THIS_TASK'),
        ('ambiguous_can', duplicate, 'Go to the red coke can. No preference for either instance is given.', 'OBJECT_BINDING_AMBIGUOUS')]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    if not os.environ.get('DASHSCOPE_API_KEY'):
        parser.error('DASHSCOPE_API_KEY required')
    here = Path(__file__).resolve().parent
    nodes = json.loads((here / 'graph_demo/scene/scene_graph.json').read_text(encoding='utf-8'))
    root = args.output_dir or Path('/mnt/d/NebulaAI/output/graph_binding_regression') / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    root.mkdir(parents=True, exist_ok=False)
    summary = {'status': 'running', 'mode': 'real_llm', 'tests': [],
               'execution_status': 'not_executed', 'training_eligible': False}
    def save():
        (root / 'regression_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    save()
    for name, inventory, instruction, expected in cases(nodes):
        folder = root / name
        scene = folder / 'scene'
        scene.mkdir(parents=True)
        (scene / 'scene_graph.json').write_text(json.dumps(inventory, indent=2), encoding='utf-8')
        (scene / 'scene_graph_edges.json').write_text('[]', encoding='utf-8')
        command = [sys.executable, str(here / 'align_objects.py'), '--scene-dir', str(scene),
                   '--matching-mode', 'llm_graph', '--instruction', instruction,
                   '--execution-contract', str(here / 'graph_demo/navigation_contract.json'),
                   '--decomposition-output', str(folder / 'task_decomposition.json'),
                   '--output', str(folder / 'object_alignment.json'), '--log-file', str(folder / 'alignment.log')]
        print('START ' + name, flush=True)
        with (folder / 'console.log').open('w', encoding='utf-8') as log:
            completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        result_file = folder / 'object_alignment.json'
        result = json.loads(result_file.read_text(encoding='utf-8')) if result_file.exists() else {}
        actual = result.get('error', {}).get('code')
        passed = completed.returncode == 2 and result.get('status') == 'error' and actual == expected
        summary['tests'].append({'name': name, 'instruction': instruction, 'expected_error': expected,
                                 'actual_error': actual, 'exit_code': completed.returncode,
                                 'passed': passed, 'output_dir': str(folder)})
        save()
        print(f'END {name} passed={passed} actual_error={actual}', flush=True)
    summary['status'] = 'passed' if all(t['passed'] for t in summary['tests']) else 'failed'
    save()
    print('summary=' + str(root / 'regression_summary.json'), flush=True)
    return 0 if summary['status'] == 'passed' else 2


if __name__ == '__main__':
    raise SystemExit(main())
