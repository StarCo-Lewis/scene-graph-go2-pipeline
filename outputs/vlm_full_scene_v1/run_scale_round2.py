"""Second-round ordinal pilot: 16 planning tasks and 4 separate state questions."""
import argparse
import logging
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import run_crossroom_task as pilot
import run_inference_comparison as comparison
from scale_round2_cases import cases, evaluation_notes, fixture_transform
from scale_round2_state_decision import standalone_trial


def run(args):
    root = args.output_dir.resolve()
    options = dict(vars(args),case_provider=cases,reference_provider=evaluation_notes,
                   fixture_transform=fixture_transform,standalone_trial=standalone_trial,
                   post_trial_override=None,no_fact_probe=True,
                   comparison_profile='scale_round2_ordinal_pilot_v1',
                   trial_profile='scale_round2_mixed_tracks_no_new_task_recovery_v1')
    code = comparison.run(SimpleNamespace(**options))
    paired = pilot.read_json(root/'comparison_summary.json')
    # This sidecar is evaluation-only and not in any planning folder.
    selected = [c for c in cases() if not args.case_ids or c[0] in args.case_ids]
    report = {'profile':'scale_round2_ordinal_pilot_v1','status':paired['status'],
        'labels':{c[0]:evaluation_notes(c[0]) for c in selected},'groups':{},
        'repeat':args.repeat,'ordinal_label_status':'proposed_not_human_adjudicated',
        'semantic_goal_success_rate':None,'hallucination_rate':None,
        'engine_snapshot_copied':False,'simulation_executed':False,
        'paired_inputs_match':paired.get('all_paired_public_inputs_match'),
        'engine_unchanged':paired['engine_unchanged_across_groups'],
        'do_not_mix_state_answers_with_symbolic_plan_passes':True}
    for mode,batch in paired['groups'].items():
        rows = []
        for row in batch['trials']:
            label = evaluation_notes(row['case_id'])
            rows.append({'case_id':row['case_id'],'repeat':row['repeat'],
                'dimension':label['target_dimension'],'level':label['target_level'],
                'track':label['track'],'base_task_id':label['base_task_id'],
                'status':row['status'],'semantic_quality':'not_reviewed',
                'planning_path':row.get('planning_path'),
                'state_label_agreement':row.get('authored_label_agreement')})
        planning = [r for r in rows if r['track']=='symbolic_planning']
        state = [r for r in rows if r['track']=='state_fact_decision']
        report['groups'][mode] = {
            'planned_planning_trials':sum(not c[0].startswith('S') for c in selected)*args.repeat,
            'planned_state_questions':sum(c[0].startswith('S') for c in selected)*args.repeat,
            'planning_trial_rows':len(planning),'state_question_rows':len(state),
            'symbolically_valid_plans':sum(r['status']=='symbolic_plan_valid_not_executed' for r in planning),
            'state_questions_answered':sum(r['status']=='state_decision_answered' for r in state),
            'preflight_rows':sum(r['status']=='input_preflight_passed' for r in rows),
            'rows':rows,'token_usage':batch.get('token_usage'),'audit_counts':batch.get('audit_counts')}
    pilot.write_json(root/'scale_round2_summary.json',report)
    print('status='+report['status']+' scale_summary='+str(root/'scale_round2_summary.json'),flush=True)
    return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preflight-only',action='store_true')
    parser.add_argument('--case-ids',nargs='+',choices=[c[0] for c in cases()])
    parser.add_argument('--repeat',type=int,default=1)
    parser.add_argument('--output-dir',type=Path,default=pilot.HERE/'runs'/datetime.now().strftime('scale_round2_%Y%m%d_%H%M%S_%f'))
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
