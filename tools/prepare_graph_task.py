"""Prepare an explicit graph/edges/metadata task; no API or robot calls."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--graph', type=Path, required=True)
    parser.add_argument('--edges', type=Path, required=True)
    parser.add_argument('--metadata', type=Path, required=True)
    parser.add_argument('--instruction', required=True)
    parser.add_argument('--navigation-mode', choices=['single_goal', 'ordered_goals'], default='single_goal')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if not args.instruction.strip():
        parser.error('Instruction must not be empty')
    inputs = [args.graph.resolve(), args.edges.resolve(), args.metadata.resolve()]
    for path in inputs:
        if not path.is_file():
            parser.error('Missing input: ' + str(path))
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    scene = output / 'scene'
    scene.mkdir()
    names = ['scene_graph.json', 'scene_graph_edges.json', 'scene_metadata.json']
    for source, name in zip(inputs, names):
        shutil.copyfile(source, scene / name)
    from embodied_pipeline.perception.conceptgraphs_scene import load_real_scene
    try:
        load_real_scene(scene)
    except Exception as exc:
        (output / 'PREPARATION_FAILED.txt').write_text(str(exc), encoding='utf-8')
        raise  # Preserve evidence; never issue a usable task for invalid inputs.
    task = {'schema_version': 'real_scene_navigation_instruction_v1',
            'task_id': output.name, 'scene_dir': str(scene),
            'scene_graph_sha256': hashlib.sha256((scene / names[0]).read_bytes()).hexdigest(),
            'scene_edges_sha256': hashlib.sha256((scene / names[1]).read_bytes()).hexdigest(),
            'instruction': args.instruction.strip(), 'navigation_mode': args.navigation_mode,
            'provenance': {'source_files': [str(p) for p in inputs],
                           'coordinate_conversion_applied': False, 'robot_executed': False}}
    (output / 'task.json').write_text(json.dumps(task, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Prepared task: ' + str(output / 'task.json'))
    print('No API called; no robot command sent. Files renamed only; coordinates unchanged.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
