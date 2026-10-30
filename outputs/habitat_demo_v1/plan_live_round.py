#!/usr/bin/env python3
"""Fresh LLM decomposition, metadata binding, skill proposal and bounded critique."""
import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path
from run_continuous_sequence import OUT
from run_navigation_demo import read_json, write_json
from match_skills import request_plan, build_result, DEFAULT_URL
from optimize_plan import request_order, candidate_plan
from verify_plan import flatten_steps, review_redundancy
from live_plan_review import review_live_plan
from semantic_redundancy import SemanticRedundancyError
from response_audit import validation as record_validation
from build_report import build as build_llm_audit_report
from skill_output_normalization import normalize_and_record, VERSION as NORMALIZATION_VERSION
from action_matrix import make_codebook, request_matrix, decode_matrix
from state_aware_planning import (close_runtime_objects, route_runtime_error,
                                  RuntimeReplanRequired, enabled as state_recovery_enabled)


def decompose_and_bind(args, descriptor, summary, semantic_attempt=1, feedback=None):
    wire = getattr(args, 'alignment_wire_format', 'legacy')
    if wire not in ('matrix', 'legacy'):
        raise ValueError('Unsupported alignment wire format')
    root = args.output_dir
    attempt_root = root if semantic_attempt == 1 else root / f'semantic_replan_{semantic_attempt}'
    attempt_root.mkdir(parents=True, exist_ok=True)
    contract_path = root / 'execution_contract.json'
    if feedback is not None:
        policy = read_json(contract_path)
        policy['previous_plan_semantic_error'] = feedback
        policy['semantic_correction'] = ('Re-decompose the original instruction. Keep-holding is an end state, '
            'not another lift. Preserve explicitly requested repetitions and required visits. '
            'Do not replace the removed intent with an unrelated action or an empty subtask.')
        if feedback.get('kind') == 'runtime_precondition_redecomposition':
            policy['semantic_correction'] = feedback['correction']
        contract_path = attempt_root / 'execution_contract.json'
        write_json(contract_path, policy)
    with (attempt_root / 'alignment.console.log').open('w', encoding='utf-8') as log:
        command = [sys.executable, str(OUT / 'object_alignment_v1/align_objects.py'),
            '--scene-dir', str(root / 'input_memory/scene'), '--instruction', descriptor['instruction'],
            '--matching-mode', 'llm_graph', '--execution-contract', str(contract_path),
            '--model', args.model, '--base-url', args.base_url,
            '--request-timeout', '180', '--request-attempts', '2',
            '--validation-attempts',str(getattr(args,'alignment_validation_attempts',3)),
            '--decomposition-format', ('matrix' if wire == 'matrix' else 'flat_roles'),
            '--binding-format', ('matrix' if wire == 'matrix' else 'nested_json'),
            '--matrix-contract', getattr(args,'matrix_contract','legacy_json'),
            '--output', str(attempt_root / 'object_alignment.json'),
            '--decomposition-output', str(attempt_root / 'task_decomposition.json')]
        proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding='utf-8', errors='replace')
        for line in proc.stdout:
            print(line, end='', flush=True)
            log.write(line)
            log.flush()
        code = proc.wait()
    summary['stages'].append({'stage': 'llm_decomposition_and_object_alignment',
                             'semantic_attempt': semantic_attempt, 'exit_code': code})
    alignment = read_json(attempt_root / 'object_alignment.json')
    if code:
        summary['upstream_error'] = alignment.get('error')
        cause = alignment.get('error', {}).get('message', alignment.get('status', 'unknown'))
        raise ValueError(f'LLM decomposition/object binding failed: {cause}; no substitute plan')
    if alignment['task_decomposition']['instruction'] != descriptor['instruction']:
        raise ValueError('English instruction changed during decomposition')
    write_json(root / f'object_alignment_semantic_attempt_{semantic_attempt}.json', alignment)
    write_json(root / f'task_decomposition_semantic_attempt_{semantic_attempt}.json', alignment['task_decomposition'])
    write_json(root / 'object_alignment.json', alignment)
    write_json(root / 'task_decomposition.json', alignment['task_decomposition'])
    return alignment


def plan(args, reviewer=None):
    review = review_live_plan if reviewer is None else reviewer
    root = args.output_dir
    descriptor = read_json(args.round_input)
    snapshot = read_json(root / 'runtime_state.json')
    library = read_json(root / 'execution_skill_library.json')
    nodes = read_json(root / 'input_memory/scene/scene_graph.json')
    output_format = getattr(args, 'skill_output_format', 'nested_json')
    owned_encoder = None
    started = time.monotonic()
    summary = {'status': 'running', 'stages': [], 'fresh_llm_calls': True,
               'simulation_executed': False, 'geometry_verified': False,
               'validation_mode': 'live_proxy_rehearsal' if reviewer is None else 'json_symbolic_test',
               'model': args.model, 'matching_mode': 'llm_graph', 'training_eligible': False,
               'decomposition_wire_format': 'flat_roles',
               'format_normalization_version': NORMALIZATION_VERSION,
               'skill_output_format': output_format,
               'skill_format_normalizations': [],
               'transport_policy': {'request_timeout_seconds': 180, 'max_attempts_per_request': 2,
                                    'same_input_on_retry': True, 'retry_cost_possible': True}}
    stage = 'llm_decomposition_and_object_alignment'
    try:
        if output_format not in ('nested_json', 'matrix'):
            raise ValueError('Unsupported skill output format')
        if output_format == 'matrix':
            if getattr(args, 'compile_navigation', False):
                raise ValueError('Matrix experiment requires fresh skill requests, not navigation compilation')
            from skill_retrieval import enrich, LocalEncoder, model_fingerprint
            from adapter_parameter_contract import assert_contract
            assert_contract(library)
            library = enrich(library)
            write_json(root / 'execution_skill_library.json', library)
            if getattr(args, 'skill_retriever', None) is None:
                stage = 'local_skill_embedding_initialization'
                model_path = getattr(args, 'embedding_model_dir', None)
                interpreter = getattr(args, 'embedding_python', None)
                if model_path is None or interpreter is None or not Path(interpreter).is_file():
                    raise ValueError('Matrix requires an existing local embedding model and interpreter')
                summary['embedding_model_sha256'] = model_fingerprint(model_path)
                owned_encoder = LocalEncoder(interpreter, model_path, root / 'embedding')
                args.skill_retriever = owned_encoder
            summary['skill_retrieval_policy'] = {'top_k': getattr(args, 'retrieval_top_k', 2),
                'threshold': getattr(args, 'retrieval_threshold', .35),
                'margin': getattr(args, 'retrieval_margin', .05), 'local_embedding': True,
                'widen_on_schema_correction': True, 'fallback_to_unfiltered_json': False}
        feedback = None
        summary['semantic_redecomposition_count'] = 0
        summary['runtime_redecomposition_count'] = 0
        summary['state_aware_recovery_enabled'] = state_recovery_enabled(library)
        summary['recovery_events'] = []
        summary['initial_skill_proposal_passed'] = None
        for semantic_attempt in range(1, 3):
            stage = 'llm_decomposition_and_object_alignment'
            logging.info('START stage=%s semantic_attempt=%s/2', stage, semantic_attempt)
            alignment = decompose_and_bind(args, descriptor, summary, semantic_attempt, feedback)
            if summary['state_aware_recovery_enabled']:
                # Keep raw alignment/decomposition unchanged in their attempt artifacts.
                write_json(root / f'object_alignment_raw_semantic_{semantic_attempt}.json', alignment)
                alignment, closure = close_runtime_objects(alignment, library, snapshot, nodes)
                write_json(root / f'runtime_object_closure_semantic_{semantic_attempt}.json', closure)
                write_json(root / f'object_alignment_effective_semantic_{semantic_attempt}.json', alignment)
                write_json(root / 'object_alignment.json', alignment)
            compiled = None
            if getattr(args, 'compile_navigation', False):
                from navigation_compiler import compile_navigation
                compiled, routing = compile_navigation(alignment, library, descriptor['instruction'], nodes, snapshot)
                write_json(root / f'navigation_compiler_semantic_{semantic_attempt}.json', routing)
                summary['navigation_compiler'] = routing
            retrieval = None
            retriever = getattr(args, 'skill_retriever', None)
            if retriever is not None and compiled is None:
                stage = 'local_skill_cosine_retrieval'
                from skill_retrieval import candidates
                retrieval = candidates(alignment, library, retriever,
                    top_k=getattr(args, 'retrieval_top_k', 2), threshold=getattr(args, 'retrieval_threshold', .35),
                    margin=getattr(args, 'retrieval_margin', .05))
                write_json(root / f'skill_retrieval_semantic_{semantic_attempt}.json', retrieval)
            codebook = None
            if output_format == 'matrix':
                stage = 'action_matrix_codebook'
                if retrieval is None:
                    raise ValueError('Matrix plan requires current-round cosine candidates')
                codebook = make_codebook(alignment, library)
                write_json(root / f'action_codebook_semantic_{semantic_attempt}.json', codebook)
            stage = 'compiled_navigation_and_runtime_review' if compiled is not None else 'llm_skill_matching_and_runtime_review'
            validation_error, rejected = None, None
            semantic_failure = None
            proposal_limit = getattr(args, 'skill_proposal_limit', 2 if output_format == 'matrix' else 3)
            if proposal_limit not in (1, 2, 3):
                raise ValueError('Invalid skill proposal attempt limit')
            if compiled is not None:
                proposal_limit = 1
            for attempt in range(1, proposal_limit+1):
                logging.info('START stage=%s semantic_attempt=%s attempt=%s/%s',
                             stage, semantic_attempt, attempt, proposal_limit)
                retrieval_options = {} if retrieval is None else {
                    'retrieval': retrieval, 'widen_inventory': attempt > 1}
                request_options = dict(timeout=180, attempts=2, validation_error=validation_error,
                    diagnostics_dir=root, audit_context={'semantic_attempt': semantic_attempt,
                        'validation_attempt': attempt, 'skill_output_format': output_format}, **retrieval_options)
                if compiled is not None:
                    proposal = compiled
                elif output_format == 'matrix':
                    proposal = request_matrix(alignment, library, codebook, args.model, args.base_url,
                        'DASHSCOPE_API_KEY', matrix_contract=getattr(args,'matrix_contract','legacy_json'), **request_options)
                else:
                    proposal = request_plan(alignment, library, args.model, args.base_url, 'DASHSCOPE_API_KEY',
                        rejected_proposal=rejected, **request_options)
                raw_proposal = proposal
                prefix = 'compiled_navigation' if compiled is not None else 'llm_matrix' if output_format == 'matrix' else 'llm_skill'
                proposal_file = f'{prefix}_proposal_semantic_{semantic_attempt}_attempt_{attempt}.json'
                write_json(root / proposal_file, proposal)
                try:
                    normalization = {'changed': False, 'status': 'not_applicable_compiled'}
                    if compiled is None:
                        normalization_stem = f'llm_skill_semantic_{semantic_attempt}_attempt_{attempt}'
                        try:
                            proposal, normalization = normalize_and_record(proposal, output_format, library,
                                root, normalization_stem, proposal_file,
                                matrix_contract=getattr(args,'matrix_contract','legacy_json'))
                        finally:
                            normalization_path = root / (normalization_stem+'.normalization.json')
                            if normalization_path.exists():
                                normalization = read_json(normalization_path)
                                summary['skill_format_normalizations'].append(dict(
                                    semantic_attempt=semantic_attempt, attempt=attempt, **normalization))
                    if output_format == 'matrix':
                        proposal = decode_matrix(proposal, codebook, alignment, library)
                        write_json(root / f'decoded_matrix_semantic_{semantic_attempt}_attempt_{attempt}.json', proposal)
                    result = build_result(alignment, library, proposal)
                    write_json(root / f'skill_plan_proposal_semantic_{semantic_attempt}_attempt_{attempt}.json', result)
                    if retrieval is not None and compiled is None:
                        from skill_retrieval import check_membership
                        match = check_membership(proposal, retrieval, widened=attempt > 1)
                        write_json(root / f'skill_retrieval_membership_{semantic_attempt}_{attempt}.json', match)
                    row = dict(descriptor, alignment=alignment, plan=result)
                    baseline = review(row, library, nodes, snapshot, args.reach)
                    if summary['initial_skill_proposal_passed'] is None:
                        summary['initial_skill_proposal_passed'] = True
                    record_validation(root, 'skill_matching', attempt, 'accepted',
                                      context={'semantic_attempt': semantic_attempt,
                                               'format_normalized': normalization['changed'],
                                               'proposal_source': 'compiler' if compiled is not None else 'llm'},
                                      evidence=proposal_file)
                    break
                except SemanticRedundancyError as exc:
                    if summary['initial_skill_proposal_passed'] is None:
                        summary['initial_skill_proposal_passed'] = False
                    semantic_failure = exc
                    record_validation(root, 'skill_matching', attempt, 'rejected', exc,
                                      context={'semantic_attempt': semantic_attempt},
                                      evidence=proposal_file)
                    write_json(root / f'semantic_rejection_{semantic_attempt}.json', exc.details)
                    summary['stages'].append({'stage': stage, 'status': 'semantic_rejected',
                        'semantic_attempt': semantic_attempt, 'error': exc.details})
                    logging.warning('Rejected semantic plan: %s', exc)
                    break
                except (ValueError, KeyError, TypeError) as exc:
                    if summary['initial_skill_proposal_passed'] is None:
                        summary['initial_skill_proposal_passed'] = False
                    validation_error, rejected = str(exc), raw_proposal
                    record_validation(root, 'skill_matching', attempt, 'rejected', exc,
                                      context={'semantic_attempt': semantic_attempt},
                                      evidence=proposal_file)
                    logging.warning('Rejected proposal attempt %s: %s', attempt, exc)
                    routed = route_runtime_error(exc, library, snapshot, alignment, proposal_file)
                    if routed is not None:
                        semantic_failure = RuntimeReplanRequired(routed)
                        event = dict(semantic_attempt=semantic_attempt,proposal_attempt=attempt,
                            budget_exhausted=semantic_attempt==2,**routed)
                        summary['recovery_events'].append(event)
                        write_json(root / f'runtime_rejection_semantic_{semantic_attempt}.json', event)
                        summary['stages'].append({'stage':stage,'status':'runtime_rejected',
                            'semantic_attempt':semantic_attempt,'error':routed})
                        break
                    if attempt == proposal_limit:
                        raise
            if semantic_failure is None:
                break
            if semantic_attempt == 2:
                raise semantic_failure
            feedback = semantic_failure.details
            summary['semantic_redecomposition_count'] += 1
            if isinstance(semantic_failure, RuntimeReplanRequired):
                summary['runtime_redecomposition_count'] += 1
            logging.info('Re-decomposing once from unchanged instruction and runtime snapshot; no actions executed')
        result['proposal_origin'] = ('compiled_validated_navigation' if compiled is not None else
            'fresh_llm_matrix_decoded' if output_format == 'matrix' else 'fresh_llm_skill_proposal')
        summary['selected_semantic_attempt'] = semantic_attempt
        summary['selected_binding_proposal_path'] = ('llm_binding_proposal.json' if semantic_attempt==1
            else f'semantic_replan_{semantic_attempt}/llm_binding_proposal.json')
        result['skill_output_format'] = output_format
        result['format_normalization'] = normalization
        if compiled is not None:
            result['method'] = 'Exact-ID navigation assembly from validated LLM decomposition and object bindings'
        write_json(root / 'skill_plan.json', result)
        write_json(root / 'symbolic_review.json', baseline)
        audit = review_redundancy(flatten_steps(result), library)
        write_json(root / 'redundancy_review.json', audit)
        context = library.get('execution_contract', {}).get('conversation_context')
        if context is not None:
            result['conversation_context'] = context
        if 'inference_policy' in library.get('execution_contract', {}):
            result['inference_policy'] = library['execution_contract']['inference_policy']
        if 'factual_boundary' in library.get('execution_contract', {}):
            result['factual_boundary'] = library['execution_contract']['factual_boundary']
        summary['stages'].append({'stage': stage, 'status': 'passed', 'schema_attempts': attempt})
        stage = 'compiled_navigation_retention_review' if compiled is not None else 'llm_redundancy_review'
        logging.info('START stage=%s', stage)
        if compiled is not None:
            # Every compiled action represents a requested visit, including repeats.
            # Keep program checks and the outer whole-instruction coverage review;
            # do not ask a second LLM to remove explicit navigation requirements.
            critique = {'reason': 'Compiled explicit visits are retained without an LLM shortening request'}
        else:
            critique = request_order(result, audit, args.model, args.base_url, 'DASHSCOPE_API_KEY',
                                     timeout=180, attempts=2, diagnostics_dir=root)
            write_json(root / 'llm_redundancy_proposal.json', critique)
        # API failures propagate. A completed but invalid critique may retain a validated original.
        final, final_review = result, baseline
        decision = {'status': 'original_retained', 'llm_reason': critique.get('reason', '') if compiled is None and isinstance(critique, dict) else '',
                    'execution_status': 'not_executed'}
        try:
            if compiled is not None:
                decision['reason'] = 'No LLM critique requested; explicit ordered visits must remain'
                decision['proposal_origin'] = result['proposal_origin']
                decision['llm_critique_requested'] = False
                write_json(root / 'compiled_navigation_review.json', decision)
            else:
                candidate = candidate_plan(result, critique)
                revised = review(dict(descriptor, alignment=alignment, plan=candidate),
                                           library, nodes, snapshot, args.reach)
                write_json(root / 'skill_plan_candidate.json', candidate)
                write_json(root / 'symbolic_review_candidate.json', revised)
                if len(flatten_steps(candidate)) < len(flatten_steps(result)):
                    final, final_review = candidate, revised
                    decision['status'] = 'accepted'
                else:
                    decision['reason'] = 'No permitted redundant navigation was removed'
                record_validation(root, 'redundancy_review', 1,
                                  'accepted' if decision['status']=='accepted' else 'not_selected',
                                  evidence='llm_redundancy_proposal.json')
        except (ValueError, KeyError, TypeError) as exc:
            decision['reason'] = str(exc)
            record_validation(root, 'redundancy_review', 1, 'rejected', exc, evidence='llm_redundancy_proposal.json')
        write_json(root / 'optimization_decision.json', decision)
        write_json(root / 'skill_plan_final.json', final)
        write_json(root / 'action_registry.json', {'identity':final.get('action_identity'),
            'actions':final.get('action_index',[]), 'execution_status':'not_executed'})
        write_json(root / 'symbolic_review_final.json', final_review)
        summary['stages'].append({'stage': stage, 'status': 'not_requested_compiled_navigation' if compiled is not None else 'review_completed', 'decision': decision['status']})
        summary.update(status='planning_passed', input_memory_version=snapshot['backend']['memory_version'])
    except Exception as exc:
        summary.update(status='failed', failed_stage=stage, error=str(exc))
        logging.exception('Live round planning failed')
    finally:
        if owned_encoder is not None:
            summary['embedding_metrics'] = owned_encoder.stats()
            owned_encoder.close()
            args.skill_retriever = None
        summary['planning_wall_seconds'] = time.monotonic()-started
        from token_usage_audit import usage
        summary['token_usage'] = usage(root)
    try:
        report = build_llm_audit_report(root, root / 'llm_audit_report')
        summary['llm_response_audit'] = {'report_dir': str(root / 'llm_audit_report'),
                                       'record_count': report['record_count']}
    except Exception as exc:
        summary['llm_response_audit'] = {'status': 'report_failed', 'error': str(exc)}
    write_json(root / 'planning_summary.json', summary)
    return 0 if summary['status'] == 'planning_passed' else 2


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--round-input', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--model', default='qwen3.8-flash')
    p.add_argument('--base-url', default=DEFAULT_URL)
    p.add_argument('--reach', type=float, default=1.5)
    p.add_argument('--compile-navigation', action='store_true', help='Compile only uniquely bound explicit ID visits; retain all validators')
    p.add_argument('--skill-output-format', choices=('nested_json', 'matrix'), default='nested_json')
    p.add_argument('--alignment-wire-format', choices=('legacy', 'matrix'), default='legacy',
                   help='Use matrix for decomposition and object binding; legacy preserves the old direct entry point')
    p.add_argument('--matrix-contract',choices=('typed5_schema','typed9_schema','strict_schema','typed_json','legacy_json'),default='legacy_json')
    p.add_argument('--embedding-python', type=Path)
    p.add_argument('--embedding-model-dir', type=Path)
    p.add_argument('--retrieval-top-k', type=int, default=2)
    p.add_argument('--retrieval-threshold', type=float, default=.35)
    p.add_argument('--retrieval-margin', type=float, default=.05)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
    raise SystemExit(plan(args))
