"""Sequential ambiguous dialogue, symbolic state/history inheritance, no hierarchy.

Paid entry point never uses reference actions. Default proxy feedback mode commits
failure prefixes and continues the next turn. Validated V2 mode stops failed episodes.
Episodes remain independent. Diagnostic reference checks never repair plans.
"""
import argparse
import copy
import logging
import os
import time
from datetime import datetime
from pathlib import Path

import run_crossroom_task as pilot
from run_contextual_ambiguity import scene_fixture
from dialogue_chain_cases import episodes, compact_state, evaluate, CASE_VERSION
from state_aware_planning import VERSION as RECOVERY_VERSION, RULES as RECOVERY_RULES


def prepare(folder,descriptor,state,history,cfg,nodes,edges,args):
    n = descriptor['round']
    if len(history)!=n-1 or state['backend']['memory_version']!=n-1:
        raise ValueError('Dialogue round/history/state version mismatch')
    if history and history[-1]['end_state']!=compact_state(state):
        raise ValueError('Current state does not match last accepted history')
    local = copy.deepcopy(cfg)
    local['conversation_context']['symbolic_dialogue_history'] = copy.deepcopy(history)
    pilot.prepare_round(folder,descriptor,state,nodes,args.model,args.base_url,local,edges)
    # Add only interpretation guidance. No future utterance or private answers.
    policy = pilot.read_json(folder/'execution_contract.json')
    policy['state_aware_planning'] = dict(version=RECOVERY_VERSION,enabled=True,
        max_runtime_redecompositions=1,rules=RECOVERY_RULES,
        actions_inserted_by_program=False,original_rejections_retained=True)
    policy['planning_context'] += (
        ' This is a sequential user dialogue, not recursive task splitting. Plan only the current '
        'utterance. symbolic_dialogue_history contains accepted SYMBOLIC predictions, not actual '
        'robot observations or execution. Use it to resolve references and original locations. '
        'The current runtime state supersedes past object positions. Past instructions are context, '
        'not commands to repeat; their temporary holding constraints do not override a new request '
        'to place/release. Accepted predictions do not establish unknown contents or user intent. '
        'No future user requests are supplied. Do not invent missing history or success evidence.')
    library = pilot.read_json(folder/'execution_skill_library.json')
    library['execution_contract'] = policy
    pilot.write_json(folder/'execution_contract.json',policy)
    pilot.write_json(folder/'execution_skill_library.json',library)


def history_entry(descriptor,begin,end,plan,alignment):
    return dict(round=descriptor['round'],instruction=descriptor['instruction'],
        source='accepted_symbolic_prediction_not_executed',begin_state=compact_state(begin),
        end_state=compact_state(end),actions=[{'skill_id':s['skill_id'],'arguments':copy.deepcopy(s['arguments'])}
            for sub in plan['subtask_plans'] for s in sub['steps']],
        binding_notes=[{k:copy.deepcopy(b.get(k)) for k in ('role_id','status','element_ids','reason')}
                       for b in alignment.get('bindings',alignment.get('context_bindings',[]))]
            if isinstance(alignment.get('bindings',alignment.get('context_bindings',[])),list) else [])


def run_episode(args,episode,folder,encoder):
    if getattr(args,'memory_mode','validated')=='execution_feedback':
        from dialogue_execution_feedback import run_episode_feedback
        return run_episode_feedback(args,episode,folder,encoder)
    cfg,nodes,edges,hashes = scene_fixture(episode['scene'])
    cfg.update(case_id='dialogue_'+episode['episode_id'],inference_mode=args.inference_mode,
               inference_experiment=True)
    state = pilot.initial_state(nodes,cfg)
    history = []
    summary = dict(episode_id=episode['episode_id'],scene=episode['scene'],status='preparing',
        planned_round_count=len(episode['rounds']),rounds=[],state_commit_scope='symbolic_only',
        simulation_executed=False,physical_grasp_verified=False,semantic_success_rate=None,
        later_rounds_skipped=0,source_files=hashes,fresh_llm_calls=False)
    pilot.write_json(folder/'initial_symbolic_state.json',state)
    pilot.write_json(folder/'committed_symbolic_state.json',state)
    pilot.write_json(folder/'source_snapshot/scene_graph.json',nodes)
    pilot.write_json(folder/'source_snapshot/scene_graph_edges.json',edges)
    pilot.write_json(folder/'source_snapshot/fixture.json',cfg)
    destination = folder/'episode_summary.json'
    started = time.monotonic()
    try:
        for descriptor in episode['rounds']:
            n = descriptor['round']
            planning = folder/f'round_{n}/planning'
            prepare(planning,descriptor,state,history,cfg,nodes,edges,args)
            row = dict(round=n,instruction=descriptor['instruction'],status='input_prepared',
                       input_memory_version=state['backend']['memory_version'],
                       history_round_count=len(history),begin_state=compact_state(state),
                       planning_path=str(planning.relative_to(folder)))
            summary['rounds'].append(row)
            if args.preflight_only:
                row['status'] = 'first_round_input_preflight_passed'
                summary['status'] = 'preflight_completed_first_round_only'
                break
            row['status'] = 'planning'
            summary['status'] = 'running'
            summary['fresh_llm_calls'] = True
            pilot.write_json(destination,summary)
            logging.info('START dialogue=%s round=%s memory=%s',episode['episode_id'],n,n-1)
            code = pilot.plan_flat(args,planning,encoder)
            outcome = pilot.read_json(planning/'planning_summary.json')
            row.update(initial_skill_proposal_passed=outcome.get('initial_skill_proposal_passed'),
                runtime_redecomposition_count=outcome.get('runtime_redecomposition_count',0),
                recovery_events=outcome.get('recovery_events',[]),
                selected_semantic_attempt=outcome.get('selected_semantic_attempt'),
                program_rejection_count=len([v for f in planning.rglob('validation*.json')
                    if (v:=pilot.read_json(f)).get('status')=='rejected']))
            if code or outcome.get('status')!='planning_passed':
                row.update(status='planning_failed',failed_stage=outcome.get('failed_stage'),error=outcome.get('error'))
                summary.update(status='stopped_on_failed_round',first_failed_round=n,
                               later_rounds_skipped=len(episode['rounds'])-n)
                break
            if pilot.read_json(planning/'runtime_state.json')!=state:
                raise ValueError('Planner mutated input memory; refusing commit')
            plan = pilot.read_json(planning/'skill_plan_final.json')
            alignment = pilot.read_json(planning/'object_alignment.json')
            review = pilot.review_symbolic(dict(descriptor,alignment=alignment,plan=plan),
                pilot.read_json(planning/'execution_skill_library.json'),
                pilot.read_json(planning/'input_memory/scene/scene_graph.json'),state,1.5)
            candidate = copy.deepcopy(review['predicted_end_state'])
            saved = pilot.read_json(planning/'symbolic_review_final.json')
            if candidate!=saved['predicted_end_state']:
                raise ValueError('Independent symbolic replay disagrees with planner prediction')
            # Diagnostics are saved separately, never inserted into the next prompt or used to repair state.
            diagnostic = evaluate(n,state,candidate,plan,episode['episode_id'])
            pilot.write_json(planning.parent/'evaluation/private_diagnostics.json',diagnostic)
            candidate['backend']['memory_version'] = state['backend']['memory_version']+1
            binding_source = outcome.get('selected_binding_proposal_path','llm_binding_proposal.json')
            raw_binding = pilot.read_json(planning/binding_source)
            entry = history_entry(descriptor,state,candidate,plan,raw_binding)
            # Verify the entire next public context before committing either memory or history.
            next_cfg = copy.deepcopy(cfg)
            next_cfg['conversation_context']['symbolic_dialogue_history'] = history+[entry]
            pilot.conversation_context(next_cfg,nodes)
            pilot.write_json(planning.parent/'accepted_symbolic_prediction.json',candidate)
            pilot.write_json(planning.parent/'accepted_history_entry.json',entry)
            pilot.write_json(folder/'committed_symbolic_state.json',candidate)
            state = candidate
            history.append(entry)
            pilot.write_json(folder/'accepted_symbolic_history.json',history)
            row.update(status='symbolic_prediction_committed_not_executed',
                output_memory_version=state['backend']['memory_version'],end_state=compact_state(state),
                action_count=plan['action_count'],placement_count=len(diagnostic['ordered_placement_events']),
                diagnostic_reference_checks_all_passed=diagnostic['reference_checks_all_passed'],
                semantic_quality='manual_review_required',
                selected_binding_source=binding_source,
                )
            pilot.write_json(destination,summary)
            logging.info('END dialogue=%s round=%s status=%s',episode['episode_id'],n,row['status'])
        else:
            summary['status'] = 'symbolic_chain_completed_semantics_unreviewed'
    except KeyboardInterrupt:
        summary.update(status='interrupted',later_rounds_skipped=len(episode['rounds'])-len(summary['rounds']))
        if summary['rounds'] and summary['rounds'][-1]['round']==descriptor['round']:
            summary['rounds'][-1]['status'] = 'interrupted'
    except Exception as exc:
        n = descriptor['round']
        summary.update(status='stopped_on_harness_or_review_error',error=str(exc),first_failed_round=n,
                       later_rounds_skipped=len(episode['rounds'])-n)
        if summary['rounds'] and summary['rounds'][-1]['round']==n:
            summary['rounds'][-1].update(status='review_or_harness_failed',error=str(exc))
        else:
            summary['rounds'].append(dict(round=n,instruction=descriptor['instruction'],
                status='input_preparation_failed',error=str(exc)))
        logging.exception('Dialogue stopped without recovery; independent next episode remains available')
    finally:
        summary.update(duration_seconds=time.monotonic()-started,
            completed_round_count=len(history),last_committed_memory_version=state['backend']['memory_version'],
            actions_in_committed_rounds=sum(r.get('action_count',0) for r in summary['rounds']),
            placements_in_committed_rounds=sum(r.get('placement_count',0) for r in summary['rounds']))
        summary['first_diagnostic_mismatch_round'] = next((r['round'] for r in summary['rounds']
            if r.get('diagnostic_reference_checks_all_passed') is False),None)
        summary['first_skill_proposal_failure_round'] = next((r['round'] for r in summary['rounds']
            if r.get('initial_skill_proposal_passed') is False),None)
        summary['runtime_redecomposition_count'] = sum(r.get('runtime_redecomposition_count',0)
                                                      for r in summary['rounds'])
        pilot.write_json(destination,summary)
    return summary


def run(args):
    feedback_mode=getattr(args,'memory_mode','validated')=='execution_feedback'
    wire = getattr(args, 'alignment_wire_format', 'legacy')
    if wire not in ('matrix', 'legacy'):
        raise ValueError('Unsupported alignment wire format')
    if wire == 'matrix' and not feedback_mode:
        raise ValueError('Structural matrix profile requires execution_feedback; validated V2 remains legacy')
    from matrix_type_contract import check_mode
    mode=check_mode(getattr(args,'matrix_contract','legacy_json'))
    if wire!='matrix' and mode!='legacy_json':
        raise ValueError('Typed matrix contract requires matrix alignment wire format')
    available=getattr(args,'episode_provider',episodes)()
    selected = [e for e in available if not args.episode_ids or e['episode_id'] in args.episode_ids]
    if args.episode_ids and set(args.episode_ids)-{e['episode_id'] for e in available}:
        raise ValueError('Unknown episode selection')
    if not selected or args.repeat<1:
        raise ValueError('Select at least one episode and a positive repeat count')
    if not args.preflight_only and not os.environ.get('DASHSCOPE_API_KEY','').strip():
        raise ValueError('DASHSCOPE_API_KEY required; launcher accepts hidden input')
    root = args.output_dir.resolve()
    root.mkdir(parents=True,exist_ok=False)
    sources = [p for name in ('vlm_full_scene_v1','habitat_demo_v1','skill_matching_v1',
        'object_alignment_v1','llm_audit_v1') for p in (pilot.OUT/name).glob('*.py')]
    sources += [pilot.OUT/'quadruped_atomic_skill_library.v0.1.json',pilot.PROFILES]
    hashes = {str(p.resolve()):pilot.sha(p) for p in sources}
    summary = dict(status='preparing',profile=('dialogue_proxy_execution_feedback_v3' if feedback_mode
        else 'sequential_ambiguous_dialogue_v2_engineering_repair'),model=args.model,
        memory_mode='execution_feedback' if feedback_mode else 'validated',
        case_version=getattr(args,'case_version',CASE_VERSION),state_recovery_version=RECOVERY_VERSION,
        suite=getattr(args,'suite','dialogue_two_starts'),
        reworded_rounds=getattr(args,'reworded_rounds',[3]),engineering_changes=(['current_held_object_parameter_closure',
            'proxy_result_commit_including_failed_turns','no_execution_error_model_correction',
            'post_rollout_diagnostics_only'] if feedback_mode else ['current_held_object_parameter_closure',
            'gripper_capacity_error_to_one_bounded_redecomposition']),
        max_runtime_redecompositions=0 if feedback_mode else 1,
        inference_mode=args.inference_mode,strategy='flat_cosine_matrix_no_hierarchy',
        planned_episode_count=len(selected)*args.repeat,
        planned_round_count=sum(len(e['rounds']) for e in selected)*args.repeat,episodes=[],
        fresh_llm_calls=False,simulation_executed=False,physical_grasp_verified=False,
        memory_committed_to=('experiment_local_proxy_resulting_state_including_failures' if feedback_mode
                             else 'experiment_local_symbolic_predictions_only'),
        engine_snapshot_copied=False,engine_hashes=hashes,
        reference_checks_gate_commit=False,semantic_success_rate=None,
        planning_semantic_gate=not feedback_mode,runtime_error_replanning=not feedback_mode,
        evaluation_timing='post_rollout_only' if feedback_mode else 'per_round_private_diagnostic',
        historical_run_retained=('dialogue_chain_v2_20261018_141344_617173' if feedback_mode else
                                 'dialogue_chain_20261018_131654_973825'),
        historical_comparison_scope='Different engineering and wording versions; descriptive, not a simultaneous ablation.',
        comparison_note=getattr(args,'comparison_note',
            'Only first utterance differs; inherited states may diverge. Not identical-state paired trials.'))
    if feedback_mode:
        from dialogue_execution_feedback import wire_profile
        summary.update(wire_profile(args))
        if wire == 'matrix':
            summary['engineering_changes'] += ['structural_matrix_decomposition', 'object_binding_matrix',
                'strict_scalar_cells_and_explicit_codebooks_no_semantic_repair']
            if mode in ('typed9_schema','typed5_schema'):
                summary['engineering_changes'] += [
                    'program_issued_typed5_ids' if mode=='typed5_schema' else 'program_issued_typed9_ids',
                    'request_scoped_typed_lookup']
    # Full manifest is never passed to prepare(), the planner, or public dialogue history.
    pilot.write_json(root/'evaluation/episode_manifest.json',selected)
    dest = root/'dialogue_summary.json'
    pilot.write_json(dest,summary)
    encoder = None
    handler = logging.FileHandler(root/'run.console.log',encoding='utf-8')
    handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))
    logging.getLogger().addHandler(handler)
    started = time.monotonic()
    model_hash = None
    try:
        if not args.preflight_only:
            if not args.embedding_python.is_file():
                raise ValueError('Existing embedding interpreter required; no automatic installation')
            model_hash = pilot.model_fingerprint(args.embedding_model_dir)
            encoder = pilot.LocalEncoder(args.embedding_python,args.embedding_model_dir,root/'embedding')
        for episode in selected:
            for repeat in range(1,args.repeat+1):
                summary.update(status='running' if not args.preflight_only else 'preflight_running')
                pilot.write_json(dest,summary)
                folder = root/'episodes'/f'{episode["episode_id"]}_r{repeat:02}'
                row = run_episode(args,episode,folder,encoder)
                row.update(repeat=repeat,path=str(folder.relative_to(root)))
                summary['episodes'].append(row)
                summary['fresh_llm_calls'] = any(e['fresh_llm_calls'] for e in summary['episodes'])
                # Refresh the currently projected source hashes; never silently change the engine.
                for path,fingerprint in row['source_files'].items():
                    if path in hashes and hashes[path]!=fingerprint:
                        raise ValueError('Source input changed between episodes')
                    hashes[path] = fingerprint
                pilot.write_json(dest,summary)
                if row['status']=='interrupted':
                    raise KeyboardInterrupt
                if feedback_mode and row['status']=='harness_error':
                    raise RuntimeError(row.get('error','Feedback execution infrastructure error'))
        summary['status'] = 'preflight_completed' if args.preflight_only else 'completed'
    except KeyboardInterrupt:
        summary['status'] = 'interrupted'
    except Exception as exc:
        summary.update(status='harness_error',error=str(exc))
        logging.exception('Dialogue batch stopped')
    finally:
        if encoder is not None:
            pilot.write_json(root/'embedding_metrics.json',encoder.stats())
            encoder.close()
        summary.update(duration_seconds=time.monotonic()-started,
            engine_and_inputs_unchanged=all(Path(p).is_file() and pilot.sha(p)==h for p,h in hashes.items()),
            embedding_model_sha256=model_hash,
            embedding_model_unchanged=None if model_hash is None else pilot.model_fingerprint(args.embedding_model_dir)==model_hash)
        if feedback_mode:
            summary.update(processed_round_count=sum(e['processed_round_count'] for e in summary['episodes']),
                successful_execution_round_count=sum(e['successful_execution_round_count'] for e in summary['episodes']),
                failed_round_count=sum(e['failed_round_count'] for e in summary['episodes']),
                processed_episode_count=sum(e['status']=='dialogue_rollout_completed_not_success_claim'
                                            for e in summary['episodes']),
                semantic_success_rate=None)
            if any(e['status']=='harness_error' for e in summary['episodes']):
                summary['status']='completed_with_harness_errors'
        else:
            summary.update(completed_round_count=sum(e['completed_round_count'] for e in summary['episodes']),
                completed_episode_count=sum(e['status']=='symbolic_chain_completed_semantics_unreviewed'
                                             for e in summary['episodes']))
        if not summary['engine_and_inputs_unchanged'] or summary['embedding_model_unchanged'] is False:
            summary['status'] = 'invalid_inputs_or_engine'
        if not args.preflight_only:
            try:
                summary['token_usage'] = pilot.usage(root)
                u = summary['token_usage']
                summary['known_fee_cny_no_discount'] = (u['prompt_tokens']*.8+u['completion_tokens']*2.7)/1e6
                summary['fee_scope'] = 'Known provider usage only; missing/timeout usage is unknown, not zero.'
                audit = pilot.build_audit(root,root/'llm_audit_aggregate')
                summary['audit_counts'] = {k:audit[k] for k in ('logical_request_count','http_attempt_count',
                    'transport_failure_count','program_rejection_count')}
            except Exception as exc:
                summary['audit_collection_error'] = str(exc)
        pilot.write_json(dest,summary)
        logging.getLogger().removeHandler(handler)
        handler.close()
    print('status='+summary['status']+' dialogue_summary='+str(dest),flush=True)
    return 0 if summary['status'] in ('completed','preflight_completed') else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preflight-only',action='store_true')
    parser.add_argument('--episode-ids',nargs='+',choices=[e['episode_id'] for e in episodes()])
    parser.add_argument('--repeat',type=int,default=1)
    parser.add_argument('--inference-mode',choices=('baseline','bounded_inference'),default='bounded_inference')
    parser.add_argument('--memory-mode',choices=('execution_feedback','validated'),default='execution_feedback',
        help='Default: commit proxy execution results including failure prefixes; validated retains V2 gates')
    parser.add_argument('--alignment-wire-format',choices=('matrix','legacy'),
        help='Default matrix in feedback mode; legacy in validated V2 mode')
    parser.add_argument('--matrix-contract',choices=('typed5_schema','typed9_schema','strict_schema','typed_json','legacy_json'),
        help='Default typed9_schema for matrix; no automatic downgrade on provider errors')
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--model',default='qwen3.8-flash')
    parser.add_argument('--base-url',default='https://dashscope.aliyuncs.com/compatible-mode/v1')
    parser.add_argument('--embedding-python',type=Path,default=Path('/home/lewis/miniforge3/envs/skill_embed/bin/python'))
    parser.add_argument('--embedding-model-dir',type=Path,default=Path('/mnt/d/NebulaAI/models/skill_embedding_minilm_v1'))
    args = parser.parse_args()
    if args.alignment_wire_format is None:
        args.alignment_wire_format = 'matrix' if args.memory_mode == 'execution_feedback' else 'legacy'
    if args.matrix_contract is None:
        args.matrix_contract='typed9_schema' if args.alignment_wire_format=='matrix' else 'legacy_json'
    if args.output_dir is None:
        prefix=('dialogue_chain_v7_typed5_' if args.matrix_contract=='typed5_schema' else
                'dialogue_chain_v6_typed9_' if args.matrix_contract=='typed9_schema' else
                'dialogue_chain_v5_typed_matrix_' if args.matrix_contract!='legacy_json' else
                'dialogue_chain_v4_matrix_' if args.alignment_wire_format == 'matrix' else
                'dialogue_chain_v3_feedback_' if args.memory_mode=='execution_feedback' else 'dialogue_chain_v2_')
        args.output_dir=pilot.HERE/'runs'/datetime.now().strftime(prefix+'%Y%m%d_%H%M%S_%f')
    args.top_k,args.cosine_threshold,args.cosine_margin = 2,.35,.05
    logging.basicConfig(level=logging.INFO,format='%(asctime)s | %(levelname)s | %(message)s')
    return run(args)


if __name__=='__main__':
    raise SystemExit(main())
