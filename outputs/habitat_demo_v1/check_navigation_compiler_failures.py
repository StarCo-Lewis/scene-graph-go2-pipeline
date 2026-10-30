"""Offline check of saved failed leaf inputs; no API, simulation or chain repair."""
import argparse
import hashlib
import os
from pathlib import Path

from run_navigation_demo import read_json, write_json
from replay_json_round_planning import validate_checkpoint
from navigation_compiler import compile_navigation
from match_skills import build_result
from json_symbolic_review import review_symbolic

LEAVES = {
    'malformed_keys_and_empty_action': 'trials/long_navigation_8_1_bounded_descriptions/round_6/planning/hierarchy/node_002/planning',
    'misspelled_keys_and_empty_arguments': 'trials/long_navigation_8_1_bounded_cosine/round_8/planning/hierarchy/node_003/planning',
}


def source_path(path):
    if os.name == 'nt' and not str(path).startswith('\\\\?\\'):
        return Path('\\\\?\\' + str(path.resolve()))
    return path


def hashes(folder):
    # Include all raw API failures, plans and snapshot files, not just inputs.
    return {str(p.relative_to(folder)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in folder.rglob('*') if p.is_file()}


def check(args):
    args.output_dir.mkdir(parents=True, exist_ok=False)
    results = []
    for name, relative in LEAVES.items():
        source = source_path(args.source_batch / relative)
        before = hashes(source)
        if not before:
            raise ValueError('Saved failed leaf missing: ' + str(source))
        state, descriptor, library, nodes, _ = validate_checkpoint(source)
        alignment = read_json(source/'object_alignment.json')
        proposal, routing = compile_navigation(alignment, library, descriptor['instruction'], nodes, state)
        if proposal is None:
            raise ValueError('Saved leaf not eligible: ' + routing['reason'])
        plan = build_result(alignment, library, proposal)
        plan['method'] = 'Exact-ID navigation assembly from saved validated decomposition and object bindings'
        plan['proposal_origin'] = 'compiled_validated_navigation_from_saved_alignment'
        review = review_symbolic(dict(descriptor, alignment=alignment, plan=plan), library, nodes, state, 1.5)
        unchanged = before == hashes(source)
        if not unchanged:
            raise ValueError('Historical leaf changed; offline check invalid')
        output = args.output_dir/name
        write_json(output/'compiled_proposal.json', proposal)
        write_json(output/'compiled_plan.json', plan)
        write_json(output/'symbolic_review.json', review)
        write_json(output/'routing.json', routing)
        write_json(output/'source_hashes.json', before)
        results.append({'case': name, 'status': 'saved_leaf_assembly_and_symbolic_review_passed',
                        'source': str(args.source_batch/relative), 'source_unchanged': unchanged,
                        'navigation_targets': review['navigation_targets']})
    summary = {'status': 'offline_checks_passed', 'cases': results,
               'fresh_llm_requests': 0, 'simulation_executed': False, 'training_eligible': False,
               'scope': 'Saved leaf assembly only; no historical outcome rewrite, no full eight-round success claim'}
    write_json(args.output_dir/'summary.json', summary)
    print('status=offline_checks_passed summary=' + str(args.output_dir/'summary.json'))
    return 0


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-batch', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    raise SystemExit(check(p.parse_args()))
