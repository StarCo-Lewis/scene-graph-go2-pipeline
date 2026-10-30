"""New flat-planning stress cases. Independent cases continue, failed rounds do not."""
import argparse
import copy
import logging
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import run_crossroom_task as pilot
from flat_stress_cases import catalog


def run(args):
    source = pilot.HERE/'scenarios/scene202_crossroom_6.json'
    base = pilot.read_json(source)
    base['source_scene_dir'] = str((source.parent/base['source_scene_dir']).resolve())
    return run_cases(args,catalog(base),[source],
        'New task combinations on existing Scene202, not three new scenes or a causal A/B test')


def run_cases(args,entries,source_paths,scope):
    selected = [(name,cfg) for name,cfg in entries if not args.case or name in args.case]
    if not selected:
        raise ValueError('No selected task cases')
    root = args.output_dir.resolve()
    root.mkdir(parents=True,exist_ok=False)
    source_hashes={str(p.resolve()):pilot.sha(p) for p in source_paths}
    before=next(iter(source_hashes.values()))
    summary = {'status':'running','strategy':'flat_cosine_matrix_no_hierarchy',
        'preflight_only':args.preflight_only,'repeats':args.repeats,
        'new_case_names':[name for name,_ in selected], 'trials':[],
        'original_scene_config_sha256':before, 'simulation_executed':False,'training_eligible':False,
        'source_config_hashes':source_hashes,'scope':scope,
        'scene_task_counts':{scene:sum(cfg.get('scene_id','scene202')==scene for _,cfg in selected)
                             for scene in sorted({cfg.get('scene_id','scene202') for _,cfg in selected})},
        'expected_round_count':sum(len(cfg['rounds']) for _,cfg in selected),
        'correction_policy':'Existing bounded automatic validation/transport retries; no split, fallback, manual repair or failed-round replay'}
    started = time.monotonic()
    for name,cfg in selected:
        pilot.write_json(root/'cases'/f'{name}.json',cfg)
    # Validate every selected graph/config before any paid request and pin the
    # whole batch's graph inputs, not only each case's individual run window.
    for name,_ in selected:
        _,_,_,hashes=pilot.load_case(root/'cases'/f'{name}.json')
        source_hashes.update(hashes)
    summary['source_config_hashes']=dict(source_hashes)
    summary['all_selected_inputs_preflight_checked']=True
    for repeat in range(1,args.repeats+1):
        for index,(name,cfg) in enumerate(selected,1):
            # Short result paths avoid Windows MAX_PATH in offline preflight.
            # Human-readable case names remain in the manifest and case files.
            folder = root/'trials'/f'c{index:02}_r{repeat:02}'
            options = dict(vars(args),case_config=root/'cases'/f'{name}.json',output_dir=folder,
                           planning_strategy='flat',skill_output_format='matrix',compile_navigation=False,
                           top_k=2,cosine_threshold=.35,cosine_margin=.05)
            print(f'START flat-stress case={name} repeat={repeat}/{args.repeats}',flush=True)
            row = {'case':name,'scene_id':cfg.get('scene_id','scene202'),
                   'repeat':repeat,'status':'running','result_dir':str(folder)}
            summary['trials'].append(row)
            pilot.write_json(root/'batch_summary.json',summary)
            try:
                code = pilot.run(SimpleNamespace(**options))
                result = pilot.read_json(folder/'regression_summary.json')
                row.update(exit_code=code,**{k:result.get(k) for k in
                    ('status','completed_round_count','expected_round_count','action_count_in_completed_rounds',
                     'duration_seconds','known_fee_cny_no_discount','audit_counts','token_usage','error')})
                corrections=[pilot.read_json(p) for p in folder.glob('round_*/planning/planning_summary.json')]
                row['format_normalization_count']=sum(n.get('changed',False) for c in corrections
                                                       for n in c.get('skill_format_normalizations',[]))
                audits=result.get('audit_counts',{})
                corrected=bool(audits.get('program_rejection_count',0) or row['format_normalization_count'])
                transport=bool(audits.get('transport_failure_count',0))
                row['outcome']=('preflight_only' if args.preflight_only else 'final_failure' if code else
                    'passed_audit_incomplete' if not all(k in audits for k in
                        ('program_rejection_count','transport_failure_count')) else
                    'passed_after_validation_and_transport_retry' if corrected and transport else
                    'passed_after_validation_correction' if corrected else
                    'passed_after_transport_retry' if transport else 'clean_first_pass')
                if code:
                    row['failed_round'] = next((r['round'] for r in result['rounds'] if r['status']=='failed'),None)
                if result['status']=='interrupted':
                    summary['status']='interrupted'
                    pilot.write_json(root/'batch_summary.json',summary)
                    return 130
            except KeyboardInterrupt:
                row.update(status='interrupted')
                summary['status']='interrupted'
                pilot.write_json(root/'batch_summary.json',summary)
                return 130
            except Exception as exc:
                row.update(status='setup_failed',error=str(exc))
            print(f'END flat-stress case={name} repeat={repeat} status={row["status"]}',flush=True)
            pilot.write_json(root/'batch_summary.json',summary)
    summary.update(status='completed',duration_seconds=time.monotonic()-started,
        source_scene_config_unchanged=all(pilot.sha(Path(p))==h for p,h in source_hashes.items()),
        successful_trials=sum(t['status']=='json_symbolic_passed' for t in summary['trials']),
        total_trials=len(summary['trials']),
        all_expected_checks_passed=all(t['status']==('input_preflight_passed' if args.preflight_only
                                                    else 'json_symbolic_passed') for t in summary['trials']))
    if not summary['source_scene_config_unchanged']:
        summary.update(status='invalid_inputs',all_expected_checks_passed=False)
    summary['clean_first_pass_trials']=sum(t.get('outcome')=='clean_first_pass' for t in summary['trials'])
    summary['completed_round_count']=sum(t.get('completed_round_count') or 0 for t in summary['trials'])
    summary['case_pass_rate']=None if args.preflight_only else summary['successful_trials']/summary['total_trials']
    summary['clean_first_pass_rate']=None if args.preflight_only else summary['clean_first_pass_trials']/summary['total_trials']
    summary['per_scene_results']={scene:{
        'evaluated_trials':sum(t['scene_id']==scene for t in summary['trials']),
        'passed_trials':sum(t['scene_id']==scene and t['status']=='json_symbolic_passed' for t in summary['trials']),
        'clean_first_pass_trials':sum(t['scene_id']==scene and t.get('outcome')=='clean_first_pass' for t in summary['trials'])}
        for scene in summary['scene_task_counts']}
    summary['known_fee_cny_no_discount']=sum(t.get('known_fee_cny_no_discount') or 0 for t in summary['trials'])
    summary['fee_scope']='Received provider usage only; absent usage is unknown, not free. User input0.8/output2.7 CNY per million, no discount.'
    summary['fee_collection_missing_trials']=[t['case'] for t in summary['trials']
                                            if not args.preflight_only and t.get('known_fee_cny_no_discount') is None]
    pilot.write_json(root/'batch_summary.json',summary)
    print('status='+summary['status']+' summary='+str(root/'batch_summary.json'),flush=True)
    return 0 if summary['all_expected_checks_passed'] else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    names = [name for name,_ in catalog({})]
    parser.add_argument('--case',action='append',choices=names)
    parser.add_argument('--repeats',type=int,default=1)
    parser.add_argument('--output-dir',type=Path,default=pilot.HERE/'runs'/datetime.now().strftime('flat_stress_%Y%m%d_%H%M%S_%f'))
    parser.add_argument('--preflight-only',action='store_true')
    parser.add_argument('--model',default='qwen3.8-flash')
    parser.add_argument('--base-url',default='https://dashscope.aliyuncs.com/compatible-mode/v1')
    parser.add_argument('--embedding-python',type=Path,default=Path('/home/lewis/miniforge3/envs/skill_embed/bin/python'))
    parser.add_argument('--embedding-model-dir',type=Path,default=Path('/mnt/d/NebulaAI/models/skill_embedding_minilm_v1'))
    args = parser.parse_args()
    if not 1 <= args.repeats <= 3:
        parser.error('repeats must be 1..3; start with one without adaptive repairs')
    logging.basicConfig(level=logging.INFO,format='%(asctime)s | %(levelname)s | %(message)s')
    return run(args)


if __name__=='__main__': raise SystemExit(main())
