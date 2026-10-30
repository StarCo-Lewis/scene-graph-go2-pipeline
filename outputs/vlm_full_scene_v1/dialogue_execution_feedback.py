"""Open-loop proxy rollout: no semantic gate or correction before state commits.

Strict wire decoding remains necessary to dispatch a plan. Executor preconditions
produce action failures, not planning retries. Partial effects are never rolled back.
No physical execution, inspection or independent success oracle is asserted.
"""
import copy
import logging
import time
from types import SimpleNamespace

import run_crossroom_task as pilot
from run_contextual_ambiguity import scene_fixture
from dialogue_chain_cases import compact_state,evaluate
from planning_start_contract import conversation_context
from state_aware_planning import VERSION,RULES,close_runtime_objects
from plan_live_round import decompose_and_bind
from strict_prompt_v2 import apply_contract
from skill_retrieval import enrich,candidates,check_membership
from action_matrix import make_codebook,request_matrix,decode_matrix
from match_skills import build_result
from skill_output_normalization import normalize_and_record
from response_audit import validation as record_validation
from json_symbolic_review import SymbolicBackend
from sequential_proxy_core import SerialExecutor,ExecutionRejected
from matrix_type_contract import VERSION as TYPE_CONTRACT_VERSION, check_mode

PROFILE = 'dialogue_proxy_execution_feedback_v3'


def wire_profile(args):
    matrix = getattr(args, 'alignment_wire_format', 'legacy') == 'matrix'
    mode=check_mode(getattr(args,'matrix_contract','legacy_json'))
    profile=('dialogue_proxy_execution_feedback_v5_typed_matrix' if mode!='legacy_json' else
             'dialogue_proxy_execution_feedback_v4_structural_matrix') if matrix else PROFILE
    if matrix and mode=='typed9_schema': profile='dialogue_proxy_execution_feedback_v6_typed9'
    if matrix and mode=='typed5_schema': profile='dialogue_proxy_execution_feedback_v7_typed5'
    typed=matrix and mode in ('typed5_schema','typed9_schema')
    identifier_protocol=('typed5_ids_v1' if mode=='typed5_schema' else 'typed9_ids_v1') if typed else 'legacy_local_codes'
    return dict(profile=profile,matrix_contract=mode,matrix_type_contract_version=TYPE_CONTRACT_VERSION,
                identifier_protocol=identifier_protocol,
                identifier_scope='request_plan_and_stage_codebook' if typed else 'legacy',
                generation_response_format='json_schema' if matrix and mode in ('strict_schema','typed9_schema','typed5_schema') else 'json_object',
                numeric_string_normalization=matrix and mode=='typed_json',provider_fallback=False,
                alignment_wire_format='matrix' if matrix else 'legacy',
                decomposition_output_format='matrix' if matrix else 'flat_roles',
                binding_output_format='matrix' if matrix else 'nested_json', skill_output_format='matrix')


def prepare_feedback(folder,descriptor,state,history,cfg,nodes,edges,args,compact=compact_state):
    n = descriptor['round']
    if len(history)!=n-1 or state['backend']['memory_version']!=n-1:
        raise ValueError('Feedback history/round/version mismatch')
    if history and history[-1]['end_state']!=compact(state):
        raise ValueError('Feedback state differs from last recorded result')
    local=copy.deepcopy(cfg)
    local['conversation_context']['proxy_execution_history']=copy.deepcopy(history)
    pilot.prepare_round(folder,descriptor,state,nodes,args.model,args.base_url,local,edges)
    policy=pilot.read_json(folder/'execution_contract.json')
    policy['state_aware_planning']=dict(version=VERSION,enabled=True,
        max_runtime_redecompositions=0,rules=RULES,actions_inserted_by_program=False)
    policy['memory_update_policy']=dict(mode='proxy_execution_feedback',planning_semantic_gate=False,
        runtime_error_replanning=False,stop_remaining_actions_on_failure=True,
        continue_next_user_turn=True,physical_execution=False)
    policy['matrix_output_contract']=dict(version=TYPE_CONTRACT_VERSION,
        mode=getattr(args,'matrix_contract','legacy_json'),provider_fallback=False,
        scope='Wire column types only; unchanged action, role, goal and state semantics')
    if getattr(args,'matrix_contract','legacy_json') in ('typed9_schema','typed5_schema'):
        from typed_matrix_ids import protocol,CAPACITY
        digits=5 if args.matrix_contract=='typed5_schema' else 9
        policy['matrix_output_contract'].update(identifier_protocol=protocol(digits),
            issuer='program_preallocated_slots_not_model_invented_ids',allocation_capacity=CAPACITY,
            scope='Typed wire identifiers and scoped lookup; unchanged task, action and runtime semantics')
    policy['planning_context'] += (
        ' Plan ONLY the current user turn. proxy_execution_history is feedback from a SYMBOLIC '
        'proxy, not a real robot or observations. It includes unsuccessful turns. Do not infer '
        'that a prior user request succeeded: inspect its execution_status, action_results and '
        'end_state. Current runtime state supersedes past positions. The actions field lists '
        'successful proxy actions only, not all proposed actions. Failed actions and skipped '
        'actions must not be treated as completed. No future user turns are provided. Content, '
        'drinkability and user intent uncertainties remain unverified. No execution retry or '
        'automatic action repair is performed by the runner.')
    library=pilot.read_json(folder/'execution_skill_library.json')
    library['execution_contract']=policy
    pilot.write_json(folder/'execution_contract.json',policy)
    pilot.write_json(folder/'execution_skill_library.json',library)
    pilot.write_json(folder/'experiment_profile.json',dict(**wire_profile(args),max_split_levels=0,
        hierarchical_calls=False,hierarchy_fallback=False,compile_navigation=False,
        state_guard_action_rewrite=False,semantic_planning_gate=False,
        whole_instruction_judge=False,redundancy_optimization=False,
        runtime_replanning=False,alignment_validation_attempts=1,matrix_proposals=1,
        protocol_validation_retained=True,execution_preconditions_retained=True,
        cosine_membership_blocks_execution=False))


def propose_once(args,folder,encoder):
    """One decomposition/binding/matrix path; no runtime review or model critique."""
    started=time.monotonic()
    summary=dict(status='proposal_running',stages=[],fresh_llm_calls=True,**wire_profile(args),
        simulation_executed=False,physical_grasp_verified=False,
        semantic_redecomposition_count=0,runtime_redecomposition_count=0,
        whole_instruction_review_requested=False,redundancy_review_requested=False,
        proposal_semantically_validated=False)
    plan=None
    alignment=None
    stage='decomposition_and_binding'
    try:
        apply_contract(folder)
        library=enrich(pilot.read_json(folder/'execution_skill_library.json'))
        pilot.write_json(folder/'execution_skill_library.json',library)
        state=pilot.read_json(folder/'runtime_state.json')
        nodes=pilot.read_json(folder/'input_memory/scene/scene_graph.json')
        descriptor=pilot.read_json(folder/'round_input.json')
        options=SimpleNamespace(**dict(vars(args),output_dir=folder,alignment_validation_attempts=1))
        alignment=decompose_and_bind(options,descriptor,summary)
        pilot.write_json(folder/'object_alignment_raw_semantic_1.json',alignment)
        alignment,closure=close_runtime_objects(alignment,library,state,nodes)
        pilot.write_json(folder/'runtime_object_closure_semantic_1.json',closure)
        pilot.write_json(folder/'object_alignment.json',alignment)
        stage='cosine_and_codebook'
        retrieval=candidates(alignment,library,encoder,top_k=args.top_k,
            threshold=args.cosine_threshold,margin=args.cosine_margin)
        pilot.write_json(folder/'skill_retrieval_semantic_1.json',retrieval)
        book=make_codebook(alignment,library)
        pilot.write_json(folder/'action_codebook_semantic_1.json',book)
        stage='matrix_protocol'
        raw=request_matrix(alignment,library,book,args.model,args.base_url,'DASHSCOPE_API_KEY',
            retrieval=retrieval,widen_inventory=False,timeout=180,attempts=2,
            diagnostics_dir=folder,audit_context={'semantic_attempt':1,'validation_attempt':1,
                'skill_output_format':'matrix','memory_mode':'proxy_execution_feedback'},
            matrix_contract=getattr(args,'matrix_contract','legacy_json'))
        evidence='llm_matrix_proposal_semantic_1_attempt_1.json'
        pilot.write_json(folder/evidence,raw)
        normalized,note=normalize_and_record(raw,'matrix',library,folder,
            'llm_skill_semantic_1_attempt_1',evidence,matrix_contract=getattr(args,'matrix_contract','legacy_json'))
        summary['format_normalization']=note
        decoded=decode_matrix(normalized,book,alignment,library)
        pilot.write_json(folder/'decoded_matrix_semantic_1_attempt_1.json',decoded)
        plan=build_result(alignment,library,decoded)
        plan['proposal_origin']='fresh_llm_matrix_decoded_not_prevalidated'
        pilot.write_json(folder/'skill_plan_proposed.json',plan)
        # Record cosine disagreement, but do not reject/rewrite a decodable supported action.
        membership=check_membership(decoded,retrieval,widened=True)
        membership.update(widened_inventory=False,check_only=True,execution_blocked=False)
        pilot.write_json(folder/'skill_retrieval_membership_observation.json',membership)
        record_validation(folder,'matrix_wire_protocol',1,'accepted',evidence=evidence)
        summary.update(status='proposal_ready_not_semantically_validated',action_count=plan['action_count'],
            cosine_conflicts=membership['conflicts'])
    except (OSError,RuntimeError) as exc:
        summary.update(status='infrastructure_error',failed_stage=stage,error=str(exc))
        plan=None
        logging.error('Transport/runtime infrastructure error, not a simulated action failure: %s',exc)
    except (ValueError,KeyError,TypeError) as exc:
        # A malformed/unbound plan cannot be dispatched. No substitute actions are invented.
        upstream=summary.get('upstream_error') or {}
        infrastructure=upstream.get('category')=='infrastructure_error'
        summary.update(status='infrastructure_error' if infrastructure else 'proposal_unavailable',
                       failed_stage=stage,error=str(exc))
        if stage=='matrix_protocol':
            record_validation(folder,'matrix_wire_protocol',1,'rejected',exc)
        plan=None
        logging.warning('%s without corrected proposal: %s',summary['status'],exc)
    finally:
        summary['duration_seconds']=time.monotonic()-started
        summary['token_usage']=pilot.usage(folder)
        pilot.write_json(folder/'planning_summary.json',summary)
    return summary,plan,alignment


def execute_proxy(plan,alignment,library,nodes,state,compact=compact_state):
    """Execute until the first expected action failure; retain all resulting effects."""
    backend=SymbolicBackend(state,nodes,1.5,library.get('execution_contract'))
    executor=SerialExecutor(backend)
    for key in ('held_object_id','lifted','prepared_target_id'):
        setattr(executor,key,state[key])
    actions=[s for sub in plan['subtask_plans'] for s in sub['steps']]
    trace=[]
    failure=None
    for index,step in enumerate(actions):
        before=executor.snapshot()
        before['state_source']=pilot.STATE_SOURCE
        try:
            result=executor.execute(step,alignment)
            status='proxy_action_succeeded'
            error=None
        except ExecutionRejected as exc:
            result=None
            status='proxy_action_failed'
            error=str(exc)
            failure=dict(action_index=index+1,action_id=step.get('action_id'),
                skill_id=step['skill_id'],arguments=copy.deepcopy(step['arguments']),error=error,
                details=copy.deepcopy(exc.details))
        # Capture even failure-side effects. In particular a failed support check can release.
        after=executor.snapshot()
        after['state_source']=pilot.STATE_SOURCE
        trace.append(dict(action_index=index+1,action_id=step.get('action_id'),
            skill_id=step['skill_id'],arguments=copy.deepcopy(step['arguments']),status=status,
            error=error,begin_state=compact(before),end_state=compact(after),
            backend_result=result,physical_execution=False))
        if failure is not None:
            break
    end=executor.snapshot()
    end['state_source']=pilot.STATE_SOURCE
    return dict(status='proxy_action_failed' if failure else 'proxy_actions_completed',
        trace=trace,error=failure['error'] if failure else None,failure=failure,
        proposed_action_count=len(actions),attempted_action_count=len(trace),
        successful_action_count=sum(t['status']=='proxy_action_succeeded' for t in trace),
        skipped_action_count=len(actions)-len(trace),resulting_state=end,
        simulation_executed=False,physical_grasp_verified=False,
        scope='Symbolic proxy effects, not robot observations or semantic goal validation')


def run_episode_feedback(args,episode,folder,encoder):
    cfg,nodes,edges,hashes=scene_fixture(episode['scene'])
    tracked=tuple(episode.get('tracked_object_ids',(86,121)))
    compact=lambda value: compact_state(value,tracked)
    auditor=getattr(args,'feedback_evaluator',evaluate)
    if 'user_location' in episode:
        cfg['conversation_context']['user_location']=copy.deepcopy(episode['user_location'])
        cfg['conversation_context']['user_activity']=episode['user_activity']
        conversation_context(cfg,nodes)  # Validate the declared location, not perceived human pose.
    cfg.update(case_id='feedback_'+episode['episode_id'],inference_mode=args.inference_mode,
        inference_experiment=True)
    state=pilot.initial_state(nodes,cfg)
    history=[]
    summary=dict(episode_id=episode['episode_id'],scene=episode['scene'],**wire_profile(args),
        user_location=copy.deepcopy(cfg['conversation_context']['user_location']),
        user_activity=cfg['conversation_context']['user_activity'],
        status='preparing',planned_round_count=len(episode['rounds']),rounds=[],
        processed_round_count=0,fresh_llm_calls=False,source_files=hashes,
        later_rounds_skipped=0,simulation_executed=False,physical_grasp_verified=False,
        semantic_success_rate=None,state_commit_scope='proxy_execution_feedback_not_physical')
    pilot.write_json(folder/'initial_symbolic_state.json',state)
    pilot.write_json(folder/'source_snapshot/scene_graph.json',nodes)
    pilot.write_json(folder/'source_snapshot/scene_graph_edges.json',edges)
    pilot.write_json(folder/'source_snapshot/fixture.json',cfg)
    pilot.write_json(folder/'memory_checkpoint.json',dict(state=state,history=history))
    started=time.monotonic()
    destination=folder/'episode_summary.json'
    try:
        for descriptor in episode['rounds']:
            n=descriptor['round']
            planning=folder/f'round_{n}/planning'
            prepare_feedback(planning,descriptor,state,history,cfg,nodes,edges,args,compact)
            row=dict(round=n,instruction=descriptor['instruction'],status='input_prepared',
                input_memory_version=state['backend']['memory_version'],history_round_count=len(history),
                begin_state=compact(state),planning_path=str(planning.relative_to(folder)))
            summary['rounds'].append(row)
            if args.preflight_only:
                row['status']='first_round_input_preflight_passed'
                summary['status']='preflight_completed_first_round_only'
                break
            row['status']='proposing'
            summary.update(status='running',fresh_llm_calls=True)
            pilot.write_json(destination,summary)
            logging.info('START feedback dialogue=%s round=%s',episode['episode_id'],n)
            proposal,plan,alignment=propose_once(args,planning,encoder)
            if proposal['status']=='infrastructure_error':
                raise RuntimeError(proposal['error'])
            if pilot.read_json(planning/'runtime_state.json')!=state:
                raise ValueError('Input memory mutated by proposer; experiment integrity failure')
            if plan is None:
                execution=dict(status='proposal_unavailable',trace=[],error=proposal.get('error'),
                    resulting_state=copy.deepcopy(state),proposed_action_count=0,attempted_action_count=0,
                    successful_action_count=0,skipped_action_count=0,physical_execution=False)
            else:
                execution=execute_proxy(plan,alignment,pilot.read_json(planning/'execution_skill_library.json'),
                                        pilot.read_json(planning/'input_memory/scene/scene_graph.json'),state,compact)
            pilot.write_json(planning.parent/'execution_feedback.json',execution)
            candidate=copy.deepcopy(execution['resulting_state'])
            candidate['backend']['memory_version']=n
            results=[{k:copy.deepcopy(t[k]) for k in ('action_id','skill_id','arguments','status','error')}
                     for t in execution['trace']]
            # History actions lists only successful proxy actions, never unexecuted suffixes.
            entry=dict(round=n,instruction=descriptor['instruction'],source='proxy_execution_feedback_not_physical',
                begin_state=compact(state),end_state=compact(candidate),
                actions=[dict(skill_id=t['skill_id'],arguments=copy.deepcopy(t['arguments']))
                         for t in execution['trace'] if t['status']=='proxy_action_succeeded'],
                binding_notes=[],execution_status=execution['status'],error=execution['error'],action_results=results)
            binding=planning/'llm_binding_proposal.json'
            if binding.is_file():
                entry['binding_notes']=pilot.read_json(binding).get('bindings',[])
            check_cfg=copy.deepcopy(cfg)
            check_cfg['conversation_context']['proxy_execution_history']=history+[entry]
            conversation_context(check_cfg,nodes)  # Data integrity, not goal/plan correctness.
            checkpoint=dict(state=candidate,history=history+[entry])
            pending=folder/'memory_checkpoint.pending.json'
            pilot.write_json(pending,checkpoint)
            pending.replace(folder/'memory_checkpoint.json')  # State and history commit together.
            state=candidate
            history.append(entry)
            pilot.write_json(planning.parent/'memory_after_feedback.json',state)
            row.update(status=execution['status'],error=execution['error'],
                output_memory_version=n,end_state=compact(state),
                proposed_action_count=execution['proposed_action_count'],
                attempted_action_count=execution['attempted_action_count'],
                successful_action_count=execution['successful_action_count'],
                skipped_action_count=execution['skipped_action_count'],
                goal_success='not_evaluated_during_rollout')
            pilot.write_json(destination,summary)
            logging.info('END feedback dialogue=%s round=%s status=%s; next user turn continues',
                         episode['episode_id'],n,row['status'])
        else:
            summary['status']='dialogue_rollout_completed_not_success_claim'
        # Private reference audit occurs only after rollout. Never feeds prompts or commits.
        if not args.preflight_only:
            audit_rows=[]
            for row in summary['rounds']:
                planning=folder/row['planning_path']
                if 'output_memory_version' not in row:
                    continue
                begin=pilot.read_json(planning/'runtime_state.json')
                end=pilot.read_json(planning.parent/'memory_after_feedback.json')
                proposed=planning/'skill_plan_proposed.json'
                plan=pilot.read_json(proposed) if proposed.is_file() else {'subtask_plans':[]}
                try:
                    diagnostic=auditor(row['round'],begin,end,plan,episode['episode_id'])
                    diagnostic['proposed_placement_events']=diagnostic.pop('ordered_placement_events')
                    diagnostic['scope']='Post-rollout private checks; proposed events are NOT executed events'
                    feedback=pilot.read_json(planning.parent/'execution_feedback.json')
                    diagnostic['successful_proxy_placement_events']=[
                        [t['arguments']['object']['element_id'],t['arguments']['surface']['element_id']]
                        for t in feedback['trace'] if t['skill_id']=='place_object_on_surface'
                        and t['status']=='proxy_action_succeeded']
                    # Event checks must be based on successfully applied actions, not the full proposal.
                    successful_plan={'subtask_plans':[{'steps':[t for t in feedback['trace']
                        if t['status']=='proxy_action_succeeded']}]}
                    actual=auditor(row['round'],begin,end,successful_plan,episode['episode_id'])
                    diagnostic['reference_checks']=actual['reference_checks']
                    diagnostic['reference_checks_all_passed']=actual['reference_checks_all_passed']
                    # New suites also audit visits/payloads; never count an unexecuted suffix.
                    for key in ('navigation_visits','strict_id_visit_sequence_matched'):
                        if key in actual:
                            diagnostic['proposed_'+key]=diagnostic[key]
                            diagnostic[key]=actual[key]
                except (ValueError,KeyError,TypeError) as exc:
                    diagnostic=dict(status='audit_error',error=str(exc))
                pilot.write_json(planning.parent/'evaluation/post_rollout_diagnostics.json',diagnostic)
                audit_rows.append(dict(round=row['round'],execution_status=row['status'],**diagnostic))
            pilot.write_json(folder/'evaluation/final_audit.json',dict(rows=audit_rows,
                final_state=compact(state),semantic_success_rate=None,
                scope='Authored proxy reference diagnostics, not independent intent or physical truth'))
    except KeyboardInterrupt:
        summary['status']='interrupted'
    except Exception as exc:
        # Broken infrastructure/inconsistent state is not an ordinary action failure.
        summary.update(status='harness_error',error=str(exc))
        logging.exception('Feedback rollout integrity/infrastructure error')
    finally:
        summary.update(duration_seconds=time.monotonic()-started,processed_round_count=len(history),
            successful_execution_round_count=sum(e['execution_status']=='proxy_actions_completed' for e in history),
            failed_round_count=sum(e['execution_status']!='proxy_actions_completed' for e in history),
            first_error_round=next((e['round'] for e in history if e['error']),None),
            last_committed_memory_version=state['backend']['memory_version'],
            proposed_action_count=sum(r.get('proposed_action_count',0) for r in summary['rounds']),
            attempted_action_count=sum(r.get('attempted_action_count',0) for r in summary['rounds']),
            successful_action_count=sum(r.get('successful_action_count',0) for r in summary['rounds']),
            later_rounds_skipped=len(episode['rounds'])-len(summary['rounds']))
        pilot.write_json(destination,summary)
    return summary
