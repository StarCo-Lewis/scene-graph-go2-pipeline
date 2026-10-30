"""One split level, at most four coherent leaves; no deeper recursive calls."""
from hierarchical_json_planner import (plan_hierarchical, request_decision, plan_leaf,
                                      validate_partition)
from strict_prompt_v2 import SPLIT, apply_contract
from run_navigation_demo import read_json, write_json
from skill_retrieval import enrich
import hashlib


def plan_bounded(args, folder, encoder=None, max_parts=4, prepare=None):
    if not 2 <= max_parts <= 4:
        raise ValueError('max_parts must be in [2,4]')
    output_format = getattr(args, 'skill_output_format', 'nested_json')
    if output_format not in ('nested_json', 'matrix'):
        raise ValueError('Unknown skill output format')
    if output_format == 'matrix' and encoder is None:
        raise ValueError('Combined matrix experiment requires cosine encoder')
    write_json(folder/'execution_skill_library.json', enrich(read_json(folder/'execution_skill_library.json')))
    immutable = ('round_input.json', 'runtime_state.json', 'execution_skill_library.json',
                 'execution_contract.json', 'input_memory/scene/scene_graph.json',
                 'input_memory/scene/scene_graph_edges.json')
    before = {name: hashlib.sha256((folder/name).read_bytes()).hexdigest() for name in immutable}
    write_json(folder/'experiment_profile.json', {
        'profile': 'bounded_cosine_matrix' if output_format == 'matrix' else
                   ('bounded_cosine' if encoder is not None else 'bounded_descriptions'),
        'skill_output_format': output_format,
        'max_split_levels': 1, 'max_parts': max_parts, 'descriptions_enriched': True,
        'state_guards': True, 'whole_instruction_review_retained': True,
        'compile_navigation': getattr(args, 'compile_navigation', False)})

    def decide(a, directory, stage, system, payload, check):
        if stage == 'hierarchy_split':
            if payload['depth'] >= 1:
                value = {'decision': 'leaf', 'reason':
                    'Configured one-level policy: plan this coherent segment as a multi-action unit.'}
                write_json(directory/'bounded_leaf_policy.json', value)
                check(value)
                return value
            def limited(value):
                parts = validate_partition(value, payload['current_text'])
                if len(parts) > max_parts:
                    raise ValueError('BOUNDED_PART_LIMIT; keep coherent multi-action segments')
                check(value)
            return request_decision(a, directory, stage,
                system+'\n'+SPLIT+f'\nUse at most {max_parts} parts. Only ONE split level is allowed. '
                'Each child will be planned as a coherent multi-action unit; do not require further subdivision.',
                payload, limited)
        return request_decision(a, directory, stage, system, payload, check)

    def leaf(a, reviewer):
        write_json(a.output_dir/'execution_skill_library.json',
                   enrich(read_json(a.output_dir/'execution_skill_library.json')))
        apply_contract(a.output_dir)
        a.skill_proposal_limit = 2
        # Hierarchy creates a fresh namespace for every leaf. Do not let the
        # wire format silently revert to the legacy nested JSON default.
        a.skill_output_format = output_format
        a.compile_navigation = getattr(args, 'compile_navigation', False)
        if encoder is not None:
            a.skill_retriever = encoder
            a.retrieval_top_k = args.top_k
            a.retrieval_threshold = args.cosine_threshold
            a.retrieval_margin = args.cosine_margin
        return plan_leaf(a, reviewer=reviewer)

    code = plan_hierarchical(args, folder,
        {'max_depth': 1, 'max_nodes': max_parts+1, 'max_leaves': max_parts},
        decide=decide, leaf_planner=leaf, state_guards=True, prepare=prepare)
    unchanged = all((folder/name).exists() and
                    hashlib.sha256((folder/name).read_bytes()).hexdigest() == digest
                    for name, digest in before.items())
    write_json(folder/'round_input_integrity.json', {'unchanged': unchanged, 'before_sha256': before})
    if not unchanged:
        summary = read_json(folder/'planning_summary.json')
        summary.update(status='failed', failed_stage='round_input_integrity', error='Planning input mutated')
        write_json(folder/'planning_summary.json', summary)
        return 2
    return code
