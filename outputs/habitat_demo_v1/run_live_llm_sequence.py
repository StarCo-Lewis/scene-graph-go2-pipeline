#!/usr/bin/env python3
"""Fresh planning between rounds of ONE live Scene201 Habitat simulator."""
import argparse
import copy
import logging
import math
import os
import queue
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path
from run_continuous_sequence import (run, SCENE, contract, save_memory, digest,
                                     finalize_execution_evidence)
from run_navigation_demo import read_json, write_json, peak_rss_mb
from match_skills import DEFAULT_URL


def default_instructions():
    # Test assertions specify outcomes/visit intent, NEVER a skill sequence.
    return {'schema_version': '1.0', 'planning_origin': 'fresh_llm_each_round_live_memory',
        'training_eligible': False, 'rounds': [
        {'round': 1, 'instruction': 'Visit the kettle, visit the sideboard, then return to the kettle.',
         'expected_navigation_targets': [67, 13, 67], 'expected_end': {'held_object_id': None, 'lifted': False}},
        {'round': 2, 'instruction': 'Pick up and lift the kettle; keep holding it for the next instruction.',
         'expected_end': {'held_object_id': 67, 'lifted': True}},
        {'round': 3, 'instruction': 'Carry the held kettle to the sideboard and place it there.',
         'expected_end': {'held_object_id': None, 'lifted': False}},
        {'round': 4, 'instruction': 'Go to the moved kettle, pick it up and lift it again; keep holding it.',
         'expected_end': {'held_object_id': 67, 'lifted': True}},
        {'round': 5, 'instruction': 'Carry the held kettle to the sideboard and place it back there.',
         'expected_end': {'held_object_id': None, 'lifted': False}}]}


def validate_instructions(episode):
    rows = episode.get('rounds')
    if not isinstance(rows, list) or not 1 <= len(rows) <= 20:
        raise ValueError('Provide 1..20 instruction rounds')
    for index, row in enumerate(rows, 1):
        if (not isinstance(row, dict) or type(row.get('round')) is not int or row['round'] != index
                or set(row) - {'round', 'instruction', 'expected_end', 'expected_navigation_targets'}):
            raise ValueError('Instruction rounds must be ordered, start at 1, and contain no prewritten plan')
        instruction = row.get('instruction')
        if not isinstance(instruction, str) or not instruction.strip() or not instruction.isascii():
            raise ValueError('Nonempty English instructions required for exact instruction preservation')
        expected = row.get('expected_end', {})
        if not isinstance(expected, dict) or set(expected) - {'held_object_id', 'lifted'}:
            raise ValueError('Unsupported test assertion')
        if 'held_object_id' in expected and expected['held_object_id'] is not None and (
                type(expected['held_object_id']) is not int or expected['held_object_id'] != 67):
            raise ValueError('Only kettle 67 can be held by this adapter')
        if 'lifted' in expected and type(expected['lifted']) is not bool:
            raise ValueError('lifted assertion must be boolean')
        if 'expected_navigation_targets' in row:
            targets = row['expected_navigation_targets']
            if not isinstance(targets, list) or not targets or any(type(i) is not int or i not in (67, 13) for i in targets):
                raise ValueError('Unsupported navigation assertion')
    return episode


def execute_worker(command, log_path, timeout=1200):
    """Stream child logs, terminate/wait on bounded stage timeout or interruption."""
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding='utf-8', errors='replace', start_new_session=(os.name != 'nt'))
    messages = queue.Queue()
    def read_output():
        try:
            for line in proc.stdout:
                messages.put(line)
        finally:
            messages.put(None)
    threading.Thread(target=read_output, daemon=True).start()
    start = time.monotonic()
    try:
        with log_path.open('w', encoding='utf-8') as log:
            while True:
                if time.monotonic() - start > timeout:
                    raise TimeoutError('Live planning worker exceeded 1200s; no execution or substitute plan')
                try:
                    line = messages.get(timeout=.5)
                except queue.Empty:
                    continue
                if line is None:
                    break
                print(line, end='', flush=True)
                log.write(line)
                log.flush()
        code = proc.wait(timeout=max(1., timeout - (time.monotonic() - start)))
        if code:
            detail = ''
            result_path = log_path.parent / 'planning_summary.json'
            if result_path.is_file():
                result = read_json(result_path)
                detail = f" stage={result.get('failed_stage')} cause={result.get('error')}"
            raise ValueError(f'Live planning failed exit_code={code};{detail}; log={log_path}')
    finally:
        if proc.poll() is None:
            if os.name != 'nt':
                import signal
                os.killpg(proc.pid, signal.SIGKILL)
            else:
                proc.kill()
            proc.wait()
        proc.stdout.close()


def planning_command(args, root):
    command = [args.planning_python, str(Path(__file__).with_name('plan_live_round.py')),
        '--round-input', str(root / 'round_input.json'), '--output-dir', str(root),
        '--model', args.model, '--base-url', args.base_url, '--reach', str(args.proxy_reach_m)]
    if getattr(args, 'skill_output_format', 'nested_json') == 'matrix':
        command += ['--skill-output-format', 'matrix', '--embedding-python', str(args.embedding_python),
            '--embedding-model-dir', str(args.embedding_model_dir),
            '--retrieval-top-k', str(args.retrieval_top_k),
            '--retrieval-threshold', str(args.retrieval_threshold), '--retrieval-margin', str(args.retrieval_margin)]
    return command


class LivePlanner:
    def __init__(self, args, episode):
        self.args = args
        self.episode = validate_instructions(copy.deepcopy(episode))
        self.episode['planning_origin'] = 'fresh_llm_each_round_live_memory'

    def plan(self, descriptor, executor, backend, library, round_dir):
        root = round_dir / 'planning'
        root.mkdir(parents=True, exist_ok=False)
        save_memory(backend, root / 'input_memory')
        snapshot = executor.snapshot()
        nodes = [e['node'] for e in backend.memory['entities'].values() if e['active']]
        scene_digest = digest({'nodes': nodes, 'edges': backend.memory['edges']})
        policy = contract()
        policy.update(current_runtime_state=snapshot, expected_end_test_assertion=descriptor.get('expected_end', {}),
            runtime_coordinate_frame='habitat_world_y_up',
            planning_capabilities={'validate_supplied_on_in_edges': True},
            proxy_reach_m=self.args.proxy_reach_m,
            planning_context=(
                'Generate ordered actionable subtasks only; object binding is already a planning stage, '
                'not a locate/inspect robot skill. One subtask may have several ordered skills. '
                'Only five supported skill IDs and exact required target/object/surface parameters are executable; '
                'omit optional heights/poses. Never create empty subtask steps. '
                'Runtime snapshot positions use Habitat x/y/z with y up; graph bbox uses ConceptGraphs '
                'x/y/z with z up. Convert graph [x,y,z] to Habitat [x,z,-y] before comparing positions. '
                'Use the supplied CURRENT held/lifted/prepared state, not an empty-gripper default. '
                'In a carry intent, required_role_ids must reference the held payload AND destination; '
                'these are intent participants, not the parameter list of navigate_to_object. '
                'Maintaining an already held payload is an end-state requirement within navigation, '
                'not a new grasp/lift or standalone keep-holding action. '
                'Navigate before preparation if not already within proxy_reach_m in the horizontal plane. '
                'Preparation must target the object before grasp, and the surface before placement. '
                'Navigation clears preparation. Lift before carrying. Do not grasp an already held object '
                'or navigate to that held object. Grasp only kettle 67, place only kettle 67 onto board 13. '
                'Keep-holding is an end-state requirement, not a second lift action or standalone subtask. '
                'Historical lift again means one lift after a new grasp; only explicit lift twice or '
                'lift then lift again requests two calls. This proxy uses a fixed carry height, so '
                'requested repeat calls do not verify incremental physical lifting. '
                'Preserve explicit intermediate visits. Do not claim physical grasp or arm control. '
                'Goal protocol supports near/on and robot held/lifted for one grounded object. '
                'For pickup-and-hold emit held=true; for lift-and-hold also lifted=true. '
                'For placing use kettle on sideboard plus robot held kettle expected_value=false. '
                'near alone never proves pickup; test assertions are independent from requested goals. '
                'The runtime snapshot is authoritative proxy state, not a new scene-graph observation. '
                'Robot-near initial predicates are checked from runtime coordinates, not graph edges. '
                'Do not invent an initial robot-near predicate simply to express a future skill prerequisite. '
                'No prediction or planning-stage relation check proves navigation or support success.'))
        scoped = copy.deepcopy(library)
        scoped['execution_contract'] = policy
        from adapter_parameter_contract import project_library
        scoped = project_library(scoped)
        if getattr(self.args, 'skill_output_format', 'nested_json') == 'matrix':
            from skill_retrieval import enrich
            scoped = enrich(scoped)
        policy = scoped['execution_contract']
        write_json(root / 'runtime_state.json', snapshot)
        write_json(root / 'round_input.json', descriptor)
        write_json(root / 'execution_contract.json', policy)
        write_json(root / 'execution_skill_library.json', scoped)
        logging.info('PLAN round=%s current_memory_version=%s held=%s',
                     descriptor['round'], backend.memory['version'], executor.held_object_id)
        execute_worker(planning_command(self.args, root), root / 'planning.console.log')
        result = read_json(root / 'planning_summary.json')
        if result.get('status') != 'planning_passed' or result.get('input_memory_version') != backend.memory['version']:
            raise ValueError('Planning is incomplete or refers to a different memory version')
        if getattr(self.args, 'skill_output_format', 'nested_json') == 'matrix' and (
                result.get('skill_output_format') != 'matrix' or not result.get('skill_retrieval_policy')):
            raise ValueError('Matrix/cosine planning evidence missing; no substitute plan')
        alignment, plan = read_json(root / 'object_alignment.json'), read_json(root / 'skill_plan_final.json')
        if alignment['task_decomposition']['instruction'] != descriptor['instruction'] or plan['instruction'] != descriptor['instruction']:
            raise ValueError('Instruction mismatch: no silent rewriting')
        if Path(alignment.get('scene_reference', '')).resolve() != (root / 'input_memory/scene').resolve():
            raise ValueError('Plan bound to a different scene snapshot')
        if read_json(root / 'runtime_state.json') != snapshot or executor.snapshot() != snapshot:
            raise ValueError('Runtime state changed while planning')
        if digest({'nodes': read_json(root / 'input_memory/scene/scene_graph.json'),
                   'edges': read_json(root / 'input_memory/scene/scene_graph_edges.json')}) != scene_digest:
            raise ValueError('Planning input scene was modified')
        return dict(descriptor, alignment=alignment, plan=plan, planning_evidence={
            'fresh_llm': True, 'model': self.args.model, 'matching_mode': 'llm_graph',
            'skill_output_format': result.get('skill_output_format', 'nested_json'),
            'skill_retrieval_policy': result.get('skill_retrieval_policy'),
            'planning_wall_seconds': result.get('planning_wall_seconds'),
            'input_memory_version': backend.memory['version'], 'input_scene_digest': scene_digest,
            'runtime_snapshot': str(root / 'runtime_state.json'), 'planning_summary': str(root / 'planning_summary.json')})


def main():
    from visual_audit import add_arguments as add_audit_arguments
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--planning-python', default='/home/lewis/miniforge3/envs/conceptgraph/bin/python')
    p.add_argument('--instructions-json', type=Path)
    p.add_argument('--scene-dir', type=Path, default=SCENE / 'vlm_scenegraph_regression/scene201_verified_20260921_113316/scene')
    p.add_argument('--model', default='qwen3.8-flash')
    p.add_argument('--base-url', default=DEFAULT_URL)
    p.add_argument('--skill-output-format', choices=('nested_json', 'matrix'), default='nested_json')
    p.add_argument('--embedding-python', type=Path)
    p.add_argument('--embedding-model-dir', type=Path)
    p.add_argument('--retrieval-top-k', type=int, default=2)
    p.add_argument('--retrieval-threshold', type=float, default=.35)
    p.add_argument('--retrieval-margin', type=float, default=.05)
    p.add_argument('--seed', type=int, default=203)
    p.add_argument('--max-actions', type=int, default=500)
    p.add_argument('--proxy-reach-m', type=float, default=1.5)
    p.add_argument('--settle-seconds', type=float, default=10.)
    p.add_argument('--preflight-only', action='store_true', help='No API or simulator; input checks only')
    add_audit_arguments(p)
    p.add_argument('--output-dir', type=Path)
    args = p.parse_args()
    if args.audit_stride <= 0 or args.audit_max_frames <= 0:
        p.error('Audit stride and frame budget must be positive')
    if not math.isfinite(args.proxy_reach_m) or not 0 < args.proxy_reach_m <= 1.5 or not math.isfinite(args.settle_seconds) or args.settle_seconds < 2 or args.max_actions <= 0:
        p.error('Invalid executor parameters')
    if not args.preflight_only and (not os.environ.get('DASHSCOPE_API_KEY', '').strip() or not Path(args.planning_python).is_file()):
        p.error('DASHSCOPE_API_KEY and an existing planning Python interpreter required')
    if args.skill_output_format == 'matrix':
        if not 1 <= args.retrieval_top_k <= 5 or not 0 <= args.retrieval_threshold <= 1 or not 0 <= args.retrieval_margin <= 1:
            p.error('Invalid cosine retrieval parameters')
        if not args.preflight_only and (args.embedding_python is None or not args.embedding_python.is_file()
                or args.embedding_model_dir is None or not args.embedding_model_dir.is_dir()):
            p.error('Matrix requires existing embedding interpreter and local model directory')
    output_group = 'live_matrix_continuous_sequence' if args.skill_output_format == 'matrix' else 'live_llm_continuous_sequence'
    args.output_dir = args.output_dir or SCENE / output_group / datetime.now().strftime('scene201_%Y%m%d_%H%M%S_%f')
    episode = validate_instructions(read_json(args.instructions_json) if args.instructions_json else default_instructions())
    args.episode_json, args.test_fault = None, 'none'
    args.output_dir.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s',
        handlers=[logging.StreamHandler(), logging.FileHandler(args.output_dir / 'console.log', encoding='utf-8')])
    summary = {'schema_version': '1.0', 'diagnostics_version': 'live_llm_explicit_stage_margin_motion_guard_audit_v6',
        'skill_output_format': args.skill_output_format, 'recursive_split_budget': 0,
        'skill_proposal_limit': 2 if args.skill_output_format == 'matrix' else 3,
        'seed': args.seed,
        'status': 'running', 'rounds': [], 'training_eligible': False, 'simulation_executed': False,
        'simulator_creation_count': 0, 'initial_start_sampling_count': 0,
        'scope': 'Fresh LLM planning from live memory in one uninterrupted task-level proxy simulator',
        'limitations': ['Only five proxy skills and two Scene201 instances; assumed grasp',
            'Read-only sequence rehearsal does not prove geometry or physics',
            'Instruction outcome assertions are hand authored; plans are generated by LLM',
            'No new visual observation; no HIB or learned value function; not real-robot training data']}
    try:
        run(args, summary, LivePlanner(args, episode))
    except Exception as exc:
        logging.exception('Live sequence stopped; later rounds skipped')
        summary.update(status='failed', error=str(exc))
    finalize_execution_evidence(summary)
    try:
        current = digest({'nodes': read_json(args.scene_dir / 'scene_graph.json'),
                          'edges': read_json(args.scene_dir / 'scene_graph_edges.json')})
        summary['base_graph_unchanged'] = current == summary.get('source_graph_digest_before_execution')
        if not summary['base_graph_unchanged']:
            summary.update(status='failed', source_graph_error='Source graph changed')
    except Exception as exc:
        summary.update(status='failed', source_graph_error=str(exc), base_graph_unchanged=None)
    summary['process_peak_rss_gib'] = round(peak_rss_mb() / 1024, 3) if peak_rss_mb() else None
    summary['ram_scope'] = 'Habitat process peak RSS only; NOT Windows/WSL total or concurrent planner peak'
    from token_usage_audit import usage
    summary['token_usage'] = usage(args.output_dir)
    summary['planning_wall_seconds_total'] = sum(
        read_json(p).get('planning_wall_seconds', 0.) for p in args.output_dir.glob('round_*/planning/planning_summary.json'))
    write_json(args.output_dir / 'pipeline_summary.json', summary)
    logging.info('status=%s summary=%s', summary['status'], args.output_dir / 'pipeline_summary.json')
    return 0 if summary['status'] in ('passed', 'preflight_passed') else 2


if __name__ == '__main__':
    raise SystemExit(main())
