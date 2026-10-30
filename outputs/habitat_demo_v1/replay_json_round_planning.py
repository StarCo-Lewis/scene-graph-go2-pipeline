"""Replan one saved hypothetical JSON round; no simulation or chain resumption."""
import hashlib
import logging
import math
import shutil
import sys
from pathlib import Path

from run_navigation_demo import read_json, write_json
from run_live_llm_sequence import execute_worker
from json_symbolic_review import review_symbolic
from run_json_long_horizon import project_scene, STATE_SOURCE


INPUT_FILES = (
    'round_input.json', 'runtime_state.json', 'execution_contract.json',
    'execution_skill_library.json', 'input_memory/symbolic_state.json',
    'input_memory/scene/scene_graph.json', 'input_memory/scene/scene_graph_edges.json',
)


def checkpoint_hashes(source):
    return {name: hashlib.sha256((source / name).read_bytes()).hexdigest() for name in INPUT_FILES}


def validate_checkpoint(source):
    before = checkpoint_hashes(source)
    state = read_json(source / 'runtime_state.json')
    descriptor = read_json(source / 'round_input.json')
    policy = read_json(source / 'execution_contract.json')
    library = read_json(source / 'execution_skill_library.json')
    nodes = read_json(source / 'input_memory/scene/scene_graph.json')
    edges = read_json(source / 'input_memory/scene/scene_graph_edges.json')
    if (state.get('state_source') != STATE_SOURCE
            or state != read_json(source / 'input_memory/symbolic_state.json')
            or policy.get('adapter_mode') != 'json_symbolic_test_only'
            or policy.get('current_runtime_state') != state
            or library.get('execution_contract') != policy):
        raise ValueError('Saved JSON checkpoint mode/state/contract mismatch; no guessed repair')
    if {n['id'] for n in nodes} != {13, 62, 67} or set(state['backend']['objects']) != {'13', '62', '67'}:
        raise ValueError('Only the three-node hypothetical JSON test is supported')
    projected, predicted_edges = project_scene(nodes, state)
    if nodes != projected or edges != predicted_edges:
        raise ValueError('Saved hypothetical graph differs from its symbolic state')
    if not isinstance(descriptor.get('instruction'), str) or not descriptor['instruction'].strip():
        raise ValueError('Saved instruction is missing')
    reach = policy.get('proxy_reach_m')
    if type(reach) not in (int, float) or not math.isfinite(reach) or not 0 < reach <= 1.5:
        raise ValueError('Invalid saved proximity tolerance')
    if before != checkpoint_hashes(source):
        raise ValueError('Checkpoint changed while reading')
    return state, descriptor, library, nodes, before


def replay(args):
    source = args.replay_planning_dir.resolve()
    state, descriptor, library, nodes, before = validate_checkpoint(source)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    root = args.output_dir / 'planning'
    summary = {'status': 'running', 'mode': 'json_symbolic_round_replay',
        'scope': 'Fresh LLM planning of ONE saved hypothetical round; not a resumed or completed chain',
        'source_planning_dir': str(source), 'source_round': descriptor['round'],
        'source_checkpoint_hashes': before, 'input_projection_revision': state['backend']['memory_version'],
        'simulation_executed': False, 'geometry_verified': False, 'training_eligible': False,
        'fresh_llm_calls': False, 'completed_chain_claim': False}
    destination = args.output_dir / 'replay_summary.json'
    try:
        for name in INPUT_FILES:
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / name, target)
        if checkpoint_hashes(root) != before:
            raise ValueError('Copied checkpoint mismatch')
        write_json(destination, summary)
        summary['fresh_llm_calls'] = True
        execute_worker([sys.executable, str(Path(__file__).with_name('run_json_long_horizon.py')),
            '--worker', '--output-dir', str(root), '--model', args.model, '--base-url', args.base_url],
            root / 'planning.console.log')
        planning = read_json(root / 'planning_summary.json')
        if (planning.get('status') != 'planning_passed'
                or planning.get('validation_mode') != 'json_symbolic_test'
                or planning.get('input_memory_version') != state['backend']['memory_version']):
            raise ValueError('Unexpected replay planning mode or revision')
        # Independently check the final plan against the unchanged saved input.
        review = review_symbolic(dict(descriptor, alignment=read_json(root / 'object_alignment.json'),
            plan=read_json(root / 'skill_plan_final.json')), library, nodes, state,
            library['execution_contract']['proxy_reach_m'])
        if review.get('status') != 'symbolically_valid_not_executed':
            raise ValueError('Unexpected review status')
        write_json(args.output_dir / 'symbolic_replay_review.json', review)
        summary.update(status='json_planning_replay_passed', action_count=len(review['symbolic_trace']))
    except (Exception, KeyboardInterrupt) as exc:
        summary.update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed', error=str(exc))
        logging.exception('JSON planning replay stopped')
    finally:
        try:
            summary['source_checkpoint_unchanged'] = checkpoint_hashes(source) == before
            summary['copied_input_unchanged'] = checkpoint_hashes(root) == before
        except OSError as exc:
            summary.update(source_checkpoint_unchanged=False, copied_input_unchanged=False,
                           checkpoint_readback_error=str(exc))
        if not summary['source_checkpoint_unchanged'] or not summary['copied_input_unchanged']:
            summary.update(status='failed', error='Replay input changed or became unreadable')
        write_json(destination, summary)
    print(f'status={summary["status"]} summary={destination}', flush=True)
    return 0 if summary['status'] == 'json_planning_replay_passed' else 2
