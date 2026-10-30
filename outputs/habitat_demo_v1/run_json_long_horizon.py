#!/usr/bin/env python3
"""Real-LLM cross-room JSON planning test; no Habitat, no verified memory writes."""
import argparse
import copy
import hashlib
import json
import logging
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

from run_continuous_sequence import OUT, SCENE
from run_navigation_demo import read_json, write_json, cg_to_habitat
from sequential_proxy_core import PARAMETERS
from run_live_llm_sequence import execute_worker
from json_symbolic_review import review_symbolic
from match_skills import DEFAULT_URL
from adapter_parameter_contract import project_library
from runtime_goal_protocol import CONTRACT as STATE_GOAL_CONTRACT, PROMPT as STATE_GOAL_PROMPT

ROOMS = {67: 'kitchen', 13: 'open_office', 62: 'kitchen'}
STATE_SOURCE = 'symbolic_hypothesis_not_observation'


def instructions():
    def row(i, text, held=False, support=None, visits=None):
        result = {'round': i, 'instruction': text, 'expected_support_id': support,
                  'expected_end': {'held_object_id': 67 if held else None, 'lifted': held}}
        if visits is not None:
            result['expected_navigation_targets'] = visits
        return result
    return [
        row(1, 'Visit the kettle (ID 67), then the open-office sideboard (ID 13), then return to the kettle (ID 67).',
            visits=[67, 13, 67]),
        row(2, 'Pick up and lift the kettle (ID 67); keep holding it.', True),
        row(3, 'Carry the held kettle (ID 67) to the open-office sideboard (ID 13). Do not place it; keep holding it.',
            True, visits=[13]),
        row(4, 'Place the held kettle (ID 67) on the open-office sideboard (ID 13).', support=13),
        row(5, 'Pick up the moved kettle (ID 67), lift it, carry it to the kitchen cabinet (ID 62), and place it on that cabinet.', support=62),
        row(6, 'Visit the open-office sideboard (ID 13), then return to the moved kettle (ID 67) in the kitchen.',
            support=62, visits=[13, 67]),
        row(7, 'Pick up and lift the kettle (ID 67) in the kitchen; keep holding it.', True),
        row(8, 'Carry the held kettle (ID 67) to the open-office sideboard (ID 13) and place it there.', support=13),
        row(9, 'Pick up the kettle (ID 67), lift it, and carry it to the kitchen cabinet (ID 62). Do not place it; keep holding it.', True),
        row(10, 'Place the held kettle (ID 67) on the kitchen cabinet (ID 62).', support=62),
        row(11, 'Pick up the kettle (ID 67), lift it, carry it to the open-office sideboard (ID 13), and place it there.', support=13),
        row(12, 'Visit the moved kettle (ID 67), then the open-office sideboard (ID 13), then return to the kettle (ID 67).',
            support=13, visits=[67, 13, 67]),
    ]


def load_nodes(args):
    base = read_json(args.base_graph)
    verified = read_json(args.verified_graph)
    by_id = {n['id']: n for n in base}
    by_id.update({n['id']: n for n in verified})
    if any(i not in by_id for i in ROOMS):
        raise ValueError('Missing upstream node 13, 62 or 67; no invented replacement')
    nodes = [copy.deepcopy(by_id[i]) for i in sorted(ROOMS)]
    for node in nodes:
        for key in ('bbox_center', 'bbox_extent'):
            v = node.get(key)
            if not isinstance(v, list) or len(v) != 3 or any(type(x) not in (int, float) or not math.isfinite(x) for x in v):
                raise ValueError(f'Invalid upstream node {node["id"]} {key}')
        if any(v <= 0 for v in node['bbox_extent']):
            raise ValueError('Positive test object extents required')
        node['json_test_context'] = {'room_hypothesis': ROOMS[node['id']],
            'room_assignment_source': 'test_author_assignment_not_perception',
            'placement_affordance_verified': False}
    return nodes


def initial_state(nodes):
    objects = {str(n['id']): {'center': cg_to_habitat(n['bbox_center'], 'x_z_minus_y'),
        'support_id': None, 'room_hypothesis': ROOMS[n['id']]} for n in nodes}
    kettle = objects['67']['center']
    return {'held_object_id': None, 'lifted': False, 'prepared_target_id': None,
        'state_source': STATE_SOURCE, 'backend': {'objects': objects,
            'robot_position': [kettle[0]-1., .1, kettle[2]], 'robot_room_hypothesis': 'kitchen',
            'memory_version': 0}}


def project_scene(nodes, state):
    projected = copy.deepcopy(nodes)
    for node in projected:
        obj = state['backend']['objects'][str(node['id'])]
        x, y, z = obj['center']
        node['bbox_center'] = [x, -z, y]
        node['position_evidence'] = {'source': STATE_SOURCE, 'verified': False,
                                     'projection_revision': state['backend']['memory_version']}
        node['json_test_context']['room_hypothesis'] = obj['room_hypothesis']
    edges = [{'source': int(i), 'target': obj['support_id'], 'relation': 'a on b',
              'evidence_source': STATE_SOURCE, 'verified': False}
             for i, obj in state['backend']['objects'].items() if obj['support_id'] is not None]
    return projected, edges


def prepare_round(root, descriptor, state, nodes, model, base_url):
    projected, edges = project_scene(nodes, state)
    write_json(root / 'input_memory/scene/scene_graph.json', projected)
    write_json(root / 'input_memory/scene/scene_graph_edges.json', edges)
    write_json(root / 'input_memory/symbolic_state.json', state)
    write_json(root / 'runtime_state.json', state)
    write_json(root / 'round_input.json', descriptor)
    policy = {'adapter_mode': 'json_symbolic_test_only', 'supported_skill_ids': list(PARAMETERS),
        'action_identity_scope': {'case_id':descriptor.get('case_id','scene201_json_symbolic'),
            'round':descriptor['round'],
            'run_id':hashlib.sha256(str(root.resolve()).encode()).hexdigest()[:16]},
        'mapped_instance_ids': sorted(ROOMS), 'instance_mapping_scope': 'symbolic IDs only; no simulator mapping',
        'fixed_order_required': False, 'max_held_objects': 1,
        'goal_protocol': copy.deepcopy(STATE_GOAL_CONTRACT),
        'current_runtime_state': state, 'expected_end_test_assertion': descriptor.get('expected_end', {}),
        'proxy_reach_m': 1.5, 'runtime_coordinate_frame': 'habitat_world_y_up',
        'planning_capabilities': {'validate_supplied_on_in_edges': True},
        'simulation_executed': False, 'geometry_verified': False, 'training_eligible': False,
        'symbolic_placeable_pairs': [[67, 13], [67, 62]],
        'planning_context': (
            'JSON-only test. All runtime positions, effects, room assignments and projected on edges '
            'are symbolic hypotheses, NOT observations or verified affordances. Preserve the instruction '
            'and explicit IDs. Use current held/lifted/prepared state. Only the five supported skill IDs '
            'and grounded target/object/surface parameters are allowed; no room/pose/height arguments. '
            'Node 13 is the test open-office destination; node 62 is the test kitchen cabinet destination. '
            'Navigate before preparation if not near. Preparation targets the object before grasp and '
            'surface before placement. Navigation clears preparation. Grasp only 67 with an empty gripper; '
            'lift before carrying; place only (67,13) or (67,62) in this symbolic test. '
            'Carry/navigation intents reference BOTH payload and destination roles, even if navigation '
            'only takes a target parameter. Keeping an already held object requires no new grasp/lift '
            'or standalone keep-holding subtask. Never navigate to a held object. Preserve explicit visits. '
            'Graph coordinates are z-up; convert [x,y,z] to runtime [x,z,-y]. No perception, inspect, '
            'search or room-entry skill is executable here. Use near goals for visits and destinations, '
            'held/lifted state goals for requested pickup/keep-holding, and on plus held=false for placement. '
            'Do not invent initial constraints from future preconditions. ' + STATE_GOAL_PROMPT)}
    library = read_json(OUT / 'quadruped_atomic_skill_library.v0.1.json')
    library['skills'] = [s for s in library['skills'] if s['skill_id'] in PARAMETERS]
    library['execution_contract'] = policy
    library = project_library(library)
    policy = library['execution_contract']
    write_json(root / 'execution_contract.json', policy)
    write_json(root / 'execution_skill_library.json', library)


def run(args):
    nodes = load_nodes(args)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    state = initial_state(nodes)
    summary = {'status': 'running', 'mode': 'json_symbolic_only', 'rounds': [],
        'simulation_executed': False, 'training_eligible': False, 'geometry_verified': False,
        'expected_round_count': len(instructions()), 'fresh_llm_calls': False,
        'scope': 'Fresh LLM each round; next state is derived from assumed skill effects, not real execution',
        'limitations': ['Three existing graph nodes, author-assigned two-room context',
            'Symbolic initial robot pose and assumed placement capability',
            'No route, navmesh, collision, physical grasp or support verification',
            'Graph relation checks only validate supplied hypothetical graph predicates'],
        'source_files': {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in (args.base_graph, args.verified_graph)}}
    write_json(args.output_dir / 'instructions.json', {'rounds': instructions(), 'simulation_executed': False})
    write_json(args.output_dir / 'source_nodes.json', nodes)
    destination = args.output_dir / 'regression_summary.json'
    if args.preflight_only:
        summary['status'] = 'input_preflight_passed'
        summary['fresh_llm_calls'] = False
        write_json(destination, summary)
        print(f'status={summary["status"]} summary={destination}', flush=True)
        return 0
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
    try:
        for descriptor in instructions():
            index = descriptor['round']
            root = args.output_dir / f'round_{index}/planning'
            prepare_round(root, descriptor, state, nodes, args.model, args.base_url)
            row = {'round': index, 'instruction': descriptor['instruction'], 'status': 'planning',
                   'begin_state': copy.deepcopy(state), 'planning_dir': str(root)}
            summary['rounds'].append(row)
            write_json(destination, summary)
            print(f'START JSON round={index} projection_revision={state["backend"]["memory_version"]}', flush=True)
            summary['fresh_llm_calls'] = True
            execute_worker([sys.executable, str(Path(__file__).resolve()), '--worker', '--output-dir', str(root),
                '--model', args.model, '--base-url', args.base_url], root / 'planning.console.log')
            planning = read_json(root / 'planning_summary.json')
            if (planning.get('status') != 'planning_passed'
                    or planning.get('input_memory_version') != state['backend']['memory_version']
                    or read_json(root / 'runtime_state.json') != state):
                raise ValueError('Planning mode, input revision or state changed')
            review = review_symbolic(dict(descriptor,
                alignment=read_json(root / 'object_alignment.json'),
                plan=read_json(root / 'skill_plan_final.json')),
                read_json(root / 'execution_skill_library.json'),
                read_json(root / 'input_memory/scene/scene_graph.json'), state, 1.5)
            if review.get('status') != 'symbolically_valid_not_executed':
                raise ValueError('Unexpected final review mode; no simulated execution accepted')
            state = review['predicted_end_state']
            state['backend']['memory_version'] = index
            row.update(status='json_symbolic_passed', end_state=copy.deepcopy(state),
                       action_count=len(review['symbolic_trace']))
            write_json(root.parent / 'symbolic_projection_after.json', state)
            write_json(destination, summary)
            print(f'END JSON round={index} status={row["status"]} held={state["held_object_id"]} '
                  f'support={state["backend"]["objects"]["67"]["support_id"]}', flush=True)
        summary.update(status='json_symbolic_passed', fresh_llm_calls=True)
    except (Exception, KeyboardInterrupt) as exc:
        summary.update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed', error=str(exc))
        summary['rounds'][-1]['status'] = 'failed'
        logging.exception('JSON-only chain stopped; later rounds skipped')
    finally:
        summary['completed_round_count'] = sum(r['status'] == 'json_symbolic_passed' for r in summary['rounds'])
        summary['action_count_in_completed_rounds'] = sum(r.get('action_count', 0) for r in summary['rounds'])
        try:
            summary['source_files_unchanged'] = all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h
                                                   for p, h in summary['source_files'].items())
        except OSError as exc:
            summary['source_files_unchanged'] = False
            summary['source_readback_error'] = str(exc)
        if not summary['source_files_unchanged']:
            summary.update(status='failed', error='Upstream graph files changed during test')
        try:
            from build_report import build
            report = build(args.output_dir, args.output_dir / 'llm_audit_aggregate')
            summary['llm_audit_counts'] = {k: report[k] for k in ('logical_request_count',
                'http_attempt_count', 'transport_failure_count', 'program_rejection_count')}
        except Exception as exc:
            summary['llm_audit_report_error'] = str(exc)
        write_json(destination, summary)
    print(f'status={summary["status"]} summary={destination}', flush=True)
    return 0 if summary['status'] == 'json_symbolic_passed' else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--preflight-only', action='store_true')
    parser.add_argument('--replay-planning-dir', type=Path,
        help='Re-test ONE saved hypothetical round into a new directory; not a resumed chain')
    parser.add_argument('--model', default='qwen3.8-flash')
    parser.add_argument('--base-url', default=DEFAULT_URL)
    parser.add_argument('--base-graph', type=Path, default=SCENE / 'final/scene_graph.json')
    parser.add_argument('--verified-graph', type=Path,
        default=SCENE / 'vlm_scenegraph_regression/scene201_verified_20260921_113316/scene/scene_graph.json')
    parser.add_argument('--output-dir', type=Path,
        default=SCENE / 'json_crossroom_long_horizon' / datetime.now().strftime('scene201_%Y%m%d_%H%M%S_%f'))
    args = parser.parse_args()
    if args.replay_planning_dir and (args.worker or args.preflight_only):
        parser.error('--replay-planning-dir cannot be combined with --worker or --preflight-only')
    if not args.preflight_only and not os.environ.get('DASHSCOPE_API_KEY', '').strip():
        parser.error('DASHSCOPE_API_KEY required')
    if args.replay_planning_dir:
        from replay_json_round_planning import replay
        logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
        return replay(args)
    if args.worker:
        from plan_live_round import plan
        worker_args = SimpleNamespace(output_dir=args.output_dir, round_input=args.output_dir / 'round_input.json',
            model=args.model, base_url=args.base_url, reach=1.5)
        return plan(worker_args, reviewer=review_symbolic)
    return run(args)


if __name__ == '__main__':
    raise SystemExit(main())
