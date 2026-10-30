"""Run natural instructions first, then their bounded-inference counterparts."""
import argparse
import copy
import logging
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import run_contextual_ambiguity as experiment
import run_crossroom_task as pilot
from natural_inference_cases import cases, evaluation_notes
from inference_fact_probe import probe


def normalized_contract(contract):
    value = copy.deepcopy(contract)
    value.pop('inference_policy',None)
    value.get('action_identity_scope',{}).pop('run_id',None)
    return value


def compare_pair(root,left,right):
    a,b = root/'baseline'/left['planning_path'],root/'bounded_inference'/right['planning_path']
    identical = {}
    for name in ('round_input.json','runtime_state.json','input_memory/scene/scene_graph.json',
                 'input_memory/scene/scene_graph_edges.json'):
        identical[name] = (a/name).is_file() and (b/name).is_file() and pilot.sha(a/name)==pilot.sha(b/name)
    if not (a/'execution_contract.json').is_file() or not (b/'execution_contract.json').is_file():
        return {'available':False,'reason':'Missing public planning inputs'}
    ac,bc = pilot.read_json(a/'execution_contract.json'),pilot.read_json(b/'execution_contract.json')
    identical['contract_except_policy_and_run_identity'] = normalized_contract(ac)==normalized_contract(bc)
    al,bl = pilot.read_json(a/'execution_skill_library.json'),pilot.read_json(b/'execution_skill_library.json')
    al['execution_contract'],bl['execution_contract'] = normalized_contract(al['execution_contract']),normalized_contract(bl['execution_contract'])
    identical['library_except_policy_and_run_identity'] = al==bl
    return {'available':True,'checks':identical,'paired_inputs_match':all(identical.values()),
            'treatment_present_only_in_bounded': 'inference_policy' not in ac and
                bc.get('inference_policy',{}).get('mode')=='bounded_inference'}


def run(args):
    case_provider = getattr(args,'case_provider',cases)
    reference_provider = getattr(args,'reference_provider',evaluation_notes)
    if args.repeat < 1:
        raise ValueError('repeat must be positive')
    if args.case_ids and set(args.case_ids)-{c[0] for c in case_provider()}:
        raise ValueError('Unknown case IDs')
    root = args.output_dir.resolve()
    root.mkdir(parents=True,exist_ok=False)
    summary = {'status':'preparing','profile':getattr(args,'comparison_profile','natural_and_contents_paired_inference_v1'),
        'group_order':['baseline','bounded_inference'],'groups':{},'pairs':[],
        'semantic_success_rate':None,'simulation_executed':False,'memory_committed':False,
        'fact_probe_is_post_plan_and_never_feedback':True,'engine_snapshot_copied':False,
        'historical_eight_case_batch_is_not_this_baseline':True}
    source_paths = [p for name in ('vlm_full_scene_v1','habitat_demo_v1','skill_matching_v1',
                                 'object_alignment_v1','llm_audit_v1') for p in (pilot.OUT/name).glob('*.py')]
    source_paths += [pilot.OUT/'quadruped_atomic_skill_library.v0.1.json',pilot.PROFILES]
    summary['engine_hashes_before'] = {str(p):pilot.sha(p) for p in source_paths}
    selected = [c for c in case_provider() if not args.case_ids or c[0] in args.case_ids]
    pilot.write_json(root/'evaluation/private_reference_notes.json',
                     {c[0]:reference_provider(c[0]) for c in selected})
    pilot.write_json(root/'comparison_summary.json',summary)
    try:
        for mode in summary['group_order']:
            summary['status'] = 'running_'+mode
            pilot.write_json(root/'comparison_summary.json',summary)
            options = dict(vars(args),output_dir=root/mode,inference_mode=mode,
                           inference_experiment=True,case_provider=case_provider,
                           post_trial=getattr(args,'post_trial_override',None if args.no_fact_probe else probe))
            code = experiment.run(SimpleNamespace(**options))
            batch = pilot.read_json(root/mode/'batch_summary.json')
            summary['groups'][mode] = batch
            pilot.write_json(root/'comparison_summary.json',summary)
            if code or batch['status'] not in ('completed','preflight_completed'):
                summary['status'] = 'stopped_after_'+mode
                break
        else:
            summary['status'] = 'preflight_completed' if args.preflight_only else 'completed'
        if len(summary['groups']) == 2:
            left = {(r['case_id'],r['repeat']):r for r in summary['groups']['baseline']['trials']}
            right = {(r['case_id'],r['repeat']):r for r in summary['groups']['bounded_inference']['trials']}
            for key in sorted(left.keys() & right.keys()):
                a,b = left[key],right[key]
                integrity = compare_pair(root,a,b) if 'planning_path' in a and 'planning_path' in b else {'available':False}
                summary['pairs'].append({'case_id':key[0],'repeat':key[1],
                    'baseline_status':a['status'],'bounded_inference_status':b['status'],
                    'input_integrity':integrity,'semantic_quality':'manual_review_required',
                    'baseline_fact_probe':a.get('fact_probe'),'bounded_fact_probe':b.get('fact_probe')})
            summary['all_paired_public_inputs_match'] = bool(summary['pairs']) and all(
                p['input_integrity'].get('paired_inputs_match',False) and
                p['input_integrity'].get('treatment_present_only_in_bounded',False) for p in summary['pairs'])
    except KeyboardInterrupt:
        summary['status'] = 'interrupted'
    finally:
        summary['engine_unchanged_across_groups'] = all(Path(p).is_file() and pilot.sha(p)==h
            for p,h in summary['engine_hashes_before'].items())
        if not summary['engine_unchanged_across_groups']:
            summary['status'] = 'invalid_engine_changed'
        pilot.write_json(root/'comparison_summary.json',summary)
    print('status='+summary['status']+' summary='+str(root/'comparison_summary.json'),flush=True)
    return 0 if summary['status'] in ('completed','preflight_completed') else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preflight-only',action='store_true')
    parser.add_argument('--case-ids',nargs='+',choices=[c[0] for c in cases()])
    parser.add_argument('--repeat',type=int,default=1)
    parser.add_argument('--no-fact-probe',action='store_true')
    parser.add_argument('--output-dir',type=Path,default=pilot.HERE/'runs'/datetime.now().strftime('inference_comparison_%Y%m%d_%H%M%S_%f'))
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
