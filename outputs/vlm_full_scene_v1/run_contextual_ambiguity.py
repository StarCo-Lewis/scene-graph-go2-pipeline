"""Independent contextual utterances through the unchanged flat symbolic planner.

No clarification, task recovery, reference-plan substitution or state commit.
Existing schema/transport/proposal retries remain enabled and auditable.
"""
import argparse
import copy
import logging
import os
import time
from datetime import datetime
from pathlib import Path

import run_crossroom_task as pilot


def cases():
    return [
        ('c01', 'scene200', 'explicit_control',
         'Move the tape dispenser (ID 230) to the office table (ID 133) and leave it there.'),
        ('c02', 'scene200', 'contextual_destination', 'Bring the tape dispenser here.'),
        ('c03', 'scene200', 'implicit_object', 'I need something to stick these papers together. Bring it here.'),
        ('c04', 'scene200', 'unresolved_reference', 'Bring that thing over here.'),
        ('c05', 'scene202', 'explicit_control',
         'Move the black hardcover book (ID 86) to the office_B desk (ID 2) and leave it there.'),
        ('c06', 'scene202', 'contextual_destination', 'Bring me the black hardcover book.'),
        ('c07', 'scene202', 'implicit_object', 'I would like something to read. Bring it here.'),
        ('c08', 'scene202', 'unknown_contents_and_implicit_goal', 'I am thirsty. Get me something to drink.'),
    ]


def scene_fixture(scene):
    path = pilot.HERE/'scenarios'/('scene200_symbolic_template.json' if scene == 'scene200'
                                  else 'scene202_crossroom_6.json')
    cfg,nodes,edges,hashes = pilot.load_case(path, require_task_assertions=False)
    cfg = copy.deepcopy(cfg)
    cfg['case_id'] = scene+'_contextual_generation'
    # Remove historical task instructions and outcomes entirely from the working fixture.
    cfg.pop('rounds', None)
    room,anchor = ('office',133) if scene == 'scene200' else ('office_B',2)
    cfg['conversation_context'] = {
        'source':'scenario_assumption_not_observation',
        'user_location':{'room_name':room,'near_scene_element_id':anchor},
        'user_activity':'The user is sitting at the desk in this office and working.'}
    return cfg,nodes,edges,hashes


def run(args):
    available = getattr(args,'case_provider',cases)()
    selected = [c for c in available if not args.case_ids or c[0] in args.case_ids]
    if not selected or (args.case_ids and set(args.case_ids)-{c[0] for c in available}):
        raise ValueError('Unknown or empty case selection')
    if args.repeat < 1:
        raise ValueError('repeat must be positive')
    if not args.preflight_only:
        if not os.environ.get('DASHSCOPE_API_KEY','').strip():
            raise ValueError('DASHSCOPE_API_KEY required; use the shell launcher for a hidden prompt')
        if not args.embedding_python.is_file():
            raise ValueError('Existing embedding interpreter required; no auto-install')
        model_hash = pilot.model_fingerprint(args.embedding_model_dir)
    else:
        model_hash = None
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=False)
    source_paths = [p for dirname in ('vlm_full_scene_v1','habitat_demo_v1','skill_matching_v1',
                                      'object_alignment_v1','llm_audit_v1')
                    for p in (pilot.OUT/dirname).glob('*.py')]
    source_paths += [pilot.OUT/'quadruped_atomic_skill_library.v0.1.json',pilot.PROFILES]
    hashes = {str(p.resolve()):pilot.sha(p) for p in source_paths}
    summary = {'status':'preflight_running' if args.preflight_only else 'running',
        'profile':getattr(args,'trial_profile','contextual_ambiguity_no_new_exception_handling_v1'),
        'strategy':'flat_cosine_matrix_no_hierarchy', 'model':args.model,
        'planned_trial_count':len(selected)*args.repeat,'trials':[],
        'new_clarification_or_recovery':False,'existing_bounded_correction_retained':True,
        'simulation_executed':False,'physical_grasp_verified':False,'memory_committed':False,
        'semantic_success_rate':None,'semantic_evaluation':'manual_review_required_not_LLM_judge_truth',
        'scene_capabilities':'Existing restricted symbolic fixture capabilities, not general physical affordances',
        'engine_snapshot_copied':False,'engine_and_input_hashes':hashes,
        'embedding_model_sha256':model_hash,'fresh_llm_calls':False}
    summary['inference_mode'] = getattr(args,'inference_mode','baseline')
    pilot.write_json(root/'batch_summary.json',summary)
    encoder = None
    started = time.monotonic()
    handler = logging.FileHandler(root/'run.console.log', encoding='utf-8')
    handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))
    logging.getLogger().addHandler(handler)
    try:
        for case_id,scene,kind,instruction in selected:
            for repeat in range(1,args.repeat+1):
                trial = root/'trials'/f'{case_id}_r{repeat:02}'
                folder = trial/'round_1/planning'
                row = {'case_id':case_id,'repeat':repeat,'scene':scene,'category':kind,
                       'instruction':instruction,'status':'preparing','semantic_quality':'not_reviewed'}
                summary['trials'].append(row)
                try:
                    cfg,nodes,edges,input_hashes = scene_fixture(scene)
                    transform = getattr(args,'fixture_transform',None)
                    if transform is not None:
                        cfg = transform(case_id,cfg)
                    cfg['inference_mode'] = getattr(args,'inference_mode','baseline')
                    cfg['inference_experiment'] = getattr(args,'inference_experiment',False)
                    hashes.update(input_hashes)
                    state = pilot.initial_state(nodes,cfg)
                    descriptor = {'round':1,'instruction':instruction}
                    pilot.prepare_round(folder,descriptor,state,nodes,args.model,args.base_url,cfg,edges)
                    pilot.write_json(trial/'initial_symbolic_state.json',state)
                    pilot.write_json(trial/'source_fixture.json',cfg)
                    row['conversation_context'] = cfg['conversation_context']
                    row['planning_path'] = str(folder.relative_to(root))
                    logging.info('START contextual case=%s repeat=%s scene=%s',case_id,repeat,scene)
                    if args.preflight_only:
                        row['status'] = 'input_preflight_passed'
                    else:
                        row['status'] = 'planning'
                        summary['fresh_llm_calls'] = True
                        pilot.write_json(root/'batch_summary.json',summary)
                        standalone = getattr(args,'standalone_trial',None)
                        separate_result = standalone(args,trial,row) if standalone is not None else None
                        if separate_result is not None:
                            row.update(separate_result)
                        else:
                            if encoder is None:
                                encoder = pilot.LocalEncoder(args.embedding_python,args.embedding_model_dir,root/'embedding')
                            code = pilot.plan_flat(args,folder,encoder)
                            planning = pilot.read_json(folder/'planning_summary.json')
                            row.update(status='symbolic_plan_valid_not_executed' if code == 0 else 'planning_failed',
                                       failed_stage=planning.get('failed_stage'),error=planning.get('error'),
                                       semantic_redecomposition_count=planning.get('semantic_redecomposition_count'))
                            if (folder/'skill_plan_final.json').exists():
                                plan = pilot.read_json(folder/'skill_plan_final.json')
                                row['action_count'] = plan.get('action_count')
                        # Even valid predictions remain evidence, never the next trial's memory.
                        if pilot.read_json(folder/'runtime_state.json') != state:
                            raise ValueError('Planner mutated its input memory')
                    logging.info('END contextual case=%s repeat=%s status=%s',case_id,repeat,row['status'])
                except Exception as exc:
                    row.update(status='trial_error',error=str(exc))
                    logging.exception('Independent contextual trial failed; next trial remains independent')
                post_trial = getattr(args,'post_trial',None)
                if post_trial is not None:
                    try:
                        row['fact_probe'] = post_trial(args,trial,row)
                    except Exception as exc:
                        row['fact_probe'] = {'status':'probe_error','error':str(exc)}
                pilot.write_json(trial/'generation_summary.json',row)
                pilot.write_json(root/'batch_summary.json',summary)
        summary['status'] = 'preflight_completed' if args.preflight_only else 'completed'
    except KeyboardInterrupt:
        summary['status'] = 'interrupted'
        if summary['trials'] and summary['trials'][-1]['status'] in ('planning','preparing'):
            summary['trials'][-1]['status'] = 'interrupted'
    finally:
        if encoder is not None:
            pilot.write_json(root/'embedding_metrics.json',encoder.stats())
            encoder.close()
        summary['duration_seconds'] = time.monotonic()-started
        summary['engine_and_inputs_unchanged'] = all(Path(p).is_file() and pilot.sha(p)==h for p,h in hashes.items())
        summary['embedding_model_unchanged'] = (None if args.preflight_only else
                                                pilot.model_fingerprint(args.embedding_model_dir)==model_hash)
        summary['finished_trial_count'] = sum(r['status'] not in ('preparing','planning','interrupted') for r in summary['trials'])
        summary['symbolically_valid_trial_count'] = sum(r['status']=='symbolic_plan_valid_not_executed' for r in summary['trials'])
        summary['trial_status_counts'] = {status:sum(r['status']==status for r in summary['trials'])
                                         for status in sorted({r['status'] for r in summary['trials']})}
        if not summary['engine_and_inputs_unchanged'] or summary['embedding_model_unchanged'] is False:
            summary['status'] = 'invalid_inputs_or_engine'
        if not args.preflight_only:
            try:
                summary['token_usage'] = pilot.usage(root)
                audit = pilot.build_audit(root,root/'llm_audit_aggregate')
                summary['audit_counts'] = {k:audit[k] for k in ('logical_request_count','http_attempt_count',
                                                              'transport_failure_count','program_rejection_count')}
            except Exception as exc:
                summary['audit_collection_error'] = str(exc)
        pilot.write_json(root/'batch_summary.json',summary)
        logging.getLogger().removeHandler(handler)
        handler.close()
    print('status='+summary['status']+' summary='+str(root/'batch_summary.json'),flush=True)
    return 0 if summary['status'] in ('completed','preflight_completed') else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preflight-only',action='store_true')
    parser.add_argument('--case-ids',nargs='+',choices=[c[0] for c in cases()])
    parser.add_argument('--repeat',type=int,default=1)
    parser.add_argument('--output-dir',type=Path,default=pilot.HERE/'runs'/datetime.now().strftime('contextual_ambiguity_%Y%m%d_%H%M%S_%f'))
    parser.add_argument('--model',default='qwen3.8-flash')
    parser.add_argument('--base-url',default='https://dashscope.aliyuncs.com/compatible-mode/v1')
    parser.add_argument('--embedding-python',type=Path,default=Path('/home/lewis/miniforge3/envs/skill_embed/bin/python'))
    parser.add_argument('--embedding-model-dir',type=Path,default=Path('/mnt/d/NebulaAI/models/skill_embedding_minilm_v1'))
    args = parser.parse_args()
    args.top_k,args.cosine_threshold,args.cosine_margin = 2,.35,.05
    logging.basicConfig(level=logging.INFO,format='%(asctime)s | %(levelname)s | %(message)s')
    return run(args)


if __name__ == '__main__':
    raise SystemExit(main())
