"""One whole-round matrix plan: no partition calls, recursion or hierarchy fallback."""
import copy
import hashlib
from types import SimpleNamespace

from run_navigation_demo import read_json, write_json
from plan_live_round import plan
from json_symbolic_review import review_symbolic
from strict_prompt_v2 import apply_contract
from hierarchical_json_planner import request_decision, REVIEW_SYSTEM, validate_coverage
from skill_retrieval import enrich
from sequential_proxy_core import serial_state_contract


def plan_flat(args, folder, encoder):
    if encoder is None:
        raise ValueError('Flat matrix planning requires the existing cosine encoder')
    apply_contract(folder)
    library = enrich(read_json(folder/'execution_skill_library.json'))
    write_json(folder/'execution_skill_library.json', library)
    immutable = ('round_input.json','runtime_state.json','execution_contract.json',
                 'execution_skill_library.json','input_memory/scene/scene_graph.json',
                 'input_memory/scene/scene_graph_edges.json')
    hashes = {n: hashlib.sha256((folder/n).read_bytes()).hexdigest() for n in immutable}
    write_json(folder/'experiment_profile.json', {
        'profile':'flat_cosine_matrix', 'max_split_levels':0, 'max_parts':1,
        'hierarchical_calls':False, 'hierarchy_fallback':False, 'compile_navigation':False,
        'state_guard_action_rewrite':False, 'whole_instruction_review_retained':True,
        'retry_policy':'Bounded validation/transport correction, not recursive splitting',
        'state_aware_planning':read_json(folder/'execution_contract.json').get('state_aware_planning')})
    options = dict(vars(args), output_dir=folder, round_input=folder/'round_input.json',
                   reach=1.5, skill_output_format='matrix', compile_navigation=False,
                   skill_retriever=encoder, retrieval_top_k=args.top_k,
                   retrieval_threshold=args.cosine_threshold, retrieval_margin=args.cosine_margin,
                   skill_proposal_limit=2)
    code = plan(SimpleNamespace(**options), reviewer=review_symbolic)
    if code == 0:
        summary = read_json(folder/'planning_summary.json')
        try:
            descriptor = read_json(folder/'round_input.json')
            initial = read_json(folder/'runtime_state.json')
            result = read_json(folder/'symbolic_review_final.json')
            proposal = read_json(folder/'skill_plan_final.json')
            payload = {'original_instruction':descriptor['instruction'], 'initial_state':initial,
                'predicted_end_state':result['predicted_end_state'],
                'executor_state_contract':serial_state_contract(),
                'symbolic_validation_evidence':{'status':result['status'],
                    'goal_checks':result['goal_checks'], 'state_transitions':result['symbolic_trace'],
                    'scope':'Assumed effects, not independent semantic or physical truth'},
                'segments':[{'instruction':descriptor['instruction'], 'kind':'whole_round_no_split',
                    'actions':proposal['subtask_plans'], 'begin_state':initial,
                    'end_state':result['predicted_end_state']}]}
            contract = read_json(folder/'execution_contract.json')
            if 'conversation_context' in contract:
                payload['conversation_context'] = contract['conversation_context']
            if 'inference_policy' in contract:
                payload['inference_policy'] = contract['inference_policy']
            if 'factual_boundary' in contract:
                payload['factual_boundary'] = contract['factual_boundary']
            write_json(folder/'coverage_input.json', payload)
            verdict = request_decision(args, folder/'coverage', 'flat_whole_instruction_coverage',
                                      REVIEW_SYSTEM, payload, validate_coverage)
            validate_coverage(verdict)
            write_json(folder/'whole_instruction_review.json', verdict)
        except Exception as exc:
            code = 2
            summary.update(status='failed',failed_stage='flat_whole_instruction_coverage',error=str(exc))
        summary['planning_strategy'] = 'flat_cosine_matrix_no_hierarchy'
        write_json(folder/'planning_summary.json',summary)
    unchanged = all(hashlib.sha256((folder/n).read_bytes()).hexdigest()==h for n,h in hashes.items())
    write_json(folder/'round_input_integrity.json',{'unchanged':unchanged,'before_sha256':hashes})
    if not unchanged:
        summary = read_json(folder/'planning_summary.json')
        summary.update(status='failed',failed_stage='round_input_integrity',error='Planning input mutated')
        write_json(folder/'planning_summary.json',summary)
        return 2
    return code
