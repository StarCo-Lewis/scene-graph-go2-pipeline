#!/usr/bin/env python3
"""Re-test a saved failed-round planning input, WITHOUT simulator or execution."""
import argparse
import copy
import logging
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from run_live_llm_sequence import LivePlanner, validate_instructions
from run_continuous_sequence import SCENE, OUT, contract, PARAMETERS, digest
from run_navigation_demo import read_json, write_json, cg_to_habitat
from match_skills import DEFAULT_URL


def validate_checkpoint(source):
    snapshot = read_json(source / 'runtime_state.json')
    memory = read_json(source / 'input_memory/memory_state.json')
    nodes = read_json(source / 'input_memory/scene/scene_graph.json')
    edges = read_json(source / 'input_memory/scene/scene_graph_edges.json')
    descriptor = read_json(source / 'round_input.json')
    validate_instructions({'rounds': [dict(descriptor, round=1)]})
    active = [e['node'] for e in memory['entities'].values() if e['active']]
    if nodes != active or edges != memory['edges'] or snapshot['backend']['memory_version'] != memory['version']:
        raise ValueError('Saved memory/scene/runtime checkpoint mismatch; no guessed repair')
    if {n['id'] for n in nodes} != {67, 13}:
        raise ValueError('Only the current two-instance Scene201 adapter is supported')
    for node in nodes:
        center = snapshot['backend']['objects'][str(node['id'])]['center']
        if math.dist(cg_to_habitat(node['bbox_center'], 'x_z_minus_y'), center) > 1e-4:
            raise ValueError('Saved graph position differs from proxy instance readback')
    return snapshot, memory, descriptor, digest({'snapshot': snapshot, 'memory': memory, 'descriptor': descriptor})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-planning-dir', type=Path, required=True)
    p.add_argument('--model', default='qwen3.8-flash')
    p.add_argument('--base-url', default=DEFAULT_URL)
    p.add_argument('--output-dir', type=Path, default=SCENE / 'live_llm_planning_replay' / datetime.now().strftime('scene201_%Y%m%d_%H%M%S_%f'))
    args = p.parse_args()
    if not os.environ.get('DASHSCOPE_API_KEY', '').strip():
        p.error('DASHSCOPE_API_KEY required')
    snapshot, memory, descriptor, before = validate_checkpoint(args.source_planning_dir)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s',
        handlers=[logging.StreamHandler(), logging.FileHandler(args.output_dir / 'console.log', encoding='utf-8')])
    summary = {'status': 'running', 'scope': 'Fresh LLM planning from saved observed checkpoint; NO execution or simulator restoration',
        'source_planning_dir': str(args.source_planning_dir), 'source_round': descriptor['round'],
        'source_digest': before, 'simulation_executed': False, 'training_eligible': False}
    try:
        library = read_json(OUT / 'quadruped_atomic_skill_library.v0.1.json')
        library['skills'] = [s for s in library['skills'] if s['skill_id'] in PARAMETERS]
        library['execution_contract'] = contract()
        reach = read_json(args.source_planning_dir / 'execution_contract.json')['proxy_reach_m']
        if not isinstance(reach, (int, float)) or isinstance(reach, bool) or not math.isfinite(reach) or not 0 < reach <= 1.5:
            raise ValueError('Invalid saved proxy reach tolerance')
        planner_args = SimpleNamespace(planning_python=sys.executable, model=args.model, base_url=args.base_url,
                                       proxy_reach_m=reach)
        planner = LivePlanner(planner_args, {'rounds': [dict(descriptor, round=1)]})
        backend = SimpleNamespace(memory=copy.deepcopy(memory))
        executor = SimpleNamespace(snapshot=lambda: copy.deepcopy(snapshot), held_object_id=snapshot['held_object_id'])
        row = planner.plan(descriptor, executor, backend, library, args.output_dir / f"round_{descriptor['round']}")
        summary.update(status='planning_replay_passed', planning_evidence=row['planning_evidence'])
    except Exception as exc:
        logging.exception('Checkpoint planning replay failed')
        summary.update(status='failed', error=str(exc))
    _, _, _, after = validate_checkpoint(args.source_planning_dir)
    summary['source_checkpoint_unchanged'] = before == after
    if before != after:
        summary.update(status='failed', error='Source checkpoint changed')
    write_json(args.output_dir / 'replay_summary.json', summary)
    logging.info('status=%s summary=%s', summary['status'], args.output_dir / 'replay_summary.json')
    return 0 if summary['status'] == 'planning_replay_passed' else 2


if __name__ == '__main__':
    raise SystemExit(main())
