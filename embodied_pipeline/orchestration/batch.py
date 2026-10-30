"""Legacy batch flow with only the episode dispatcher redirected to modules."""

import argparse
import copy
import logging
import os
import time
from datetime import datetime
from pathlib import Path

from .._legacy import load
from .feedback import run_episode_feedback

pilot = load("run_crossroom_task")
_legacy_dialogue = load("run_dialogue_chain")
episodes = _legacy_dialogue.episodes
CASE_VERSION = _legacy_dialogue.CASE_VERSION
RECOVERY_VERSION = _legacy_dialogue.RECOVERY_VERSION


def run_episode(args, episode, folder, encoder):
    if getattr(args, "memory_mode", "validated") == "execution_feedback":
        return run_episode_feedback(args, episode, folder, encoder)
    # Validated V2 is explicitly retained; no new reimplementation or gates.
    return _legacy_dialogue.run_episode(args, episode, folder, encoder)


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
        inference_mode=args.inference_mode,strategy=('full_inventory_no_cosine_v1' if feedback_mode else 'flat_cosine_matrix_no_hierarchy'),
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
        summary.update(candidate_policy="full_inventory_no_cosine_v1", embedding_used=False)
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
        if not args.preflight_only and not feedback_mode:
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
    from ..introspection import write_run_manifest
    code = run(args)
    write_run_manifest(args.output_dir)
    return code

