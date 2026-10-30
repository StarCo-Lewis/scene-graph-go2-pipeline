"""Matrix-profile new long tasks and exact historical protocol-error chain reruns."""
import argparse
import logging
from datetime import datetime
from pathlib import Path

import run_dialogue_chain as dialogue
import run_crossroom_task as pilot
from matrix_followup_cases import episodes, evaluate, CASE_VERSION, RERUN_IDS
from long_task_error_audit import build as build_error_audit

HISTORICAL=pilot.HERE/'runs/long10_20261018_153343_861238'


def historical_targets():
    """Select saved protocol rejections, not all action failures or all semantic errors."""
    batch=pilot.read_json(HISTORICAL/'dialogue_summary.json')
    if batch['status']!='completed':
        raise ValueError('Historical source batch must be completed')
    targets=[]
    for ep in batch['episodes']:
        for row in ep['rounds']:
            if row['status']!='proposal_unavailable':
                continue
            error=row.get('error','')
            if 'unreferenced_role_ids' in error:
                kind='role_usage_closure_not_JSON_syntax'
            elif 'unknown role' in error:
                kind='undefined_role_reference_not_JSON_syntax'
            elif 'contiguous original order' in error:
                kind='action_matrix_coverage_not_JSON_syntax'
            elif 'expected_value is only supported' in error:
                kind='predicate_field_protocol_not_JSON_syntax'
            elif 'nonempty role ID' in error:
                kind='missing_or_null_role_field'
            elif 'quantity/required/semantic_labels' in error:
                kind='role_field_type_or_shape'
            else:
                kind='protocol_rejection_requires_raw_review'
            targets.append(dict(episode_id=ep['episode_id'],round=row['round'],error=error,
                category=kind,source_planning_path=str(Path(ep['path'].replace('\\','/')) /
                    row['planning_path'].replace('\\','/'))))
    if {r['episode_id'] for r in targets}!=set(RERUN_IDS):
        raise ValueError('Historical rerun selection differs from the prepared seven-chain manifest')
    return targets


def run(args):
    targets=historical_targets()
    selected=[e for e in episodes() if args.group=='all' or
        (e['experiment_group']=='new_eight_turn_task')==(args.group=='new')]
    if args.episode_ids:
        requested=set(args.episode_ids)
        if requested-{e['episode_id'] for e in selected}:
            raise ValueError('Episode selection is outside the requested experiment group')
        selected=[e for e in selected if e['episode_id'] in requested]
    source_path=HISTORICAL/'dialogue_summary.json'
    source_hash=pilot.sha(source_path)
    args.memory_mode='execution_feedback'
    args.alignment_wire_format='matrix'
    args.episode_provider=lambda:selected
    args.feedback_evaluator=evaluate
    args.case_version=CASE_VERSION
    args.suite='matrix_new4_and_historical_protocol_rerun7'
    args.reworded_rounds=[]
    args.comparison_note=('New eight-turn tasks and whole-chain historical protocol reruns are reported separately. '
        'Reruns retain original utterances, declared user locations and initial scene fixtures; '
        'later actual states may diverge. Different engineering/prompt versions, not a simultaneous causal ablation. '
        'Historical protocol failures are not all JSON syntax failures. No reference actions or state resets during a chain.')
    code=dialogue.run(args)
    root=args.output_dir.resolve()
    batch=pilot.read_json(root/'dialogue_summary.json')
    groups={name:dict(episode_ids=[e['episode_id'] for e in selected if e['experiment_group']==name],
        planned_round_count=sum(len(e['rounds']) for e in selected if e['experiment_group']==name)*args.repeat)
        for name in ('historical_protocol_whole_chain_rerun','new_eight_turn_task')}
    batch.update(experiment_groups=groups,historical_source_batch=str(HISTORICAL),
        historical_source_summary_sha256=source_hash,
        historical_source_summary_unchanged=pilot.sha(source_path)==source_hash)
    if not batch['historical_source_summary_unchanged']:
        batch['status']='invalid_historical_source_changed'
        code=2
    pilot.write_json(root/'dialogue_summary.json',batch)
    pilot.write_json(root/'evaluation/followup_design.json',dict(case_version=CASE_VERSION,groups=groups,
        historical_protocol_targets=targets,historical_targets_count=len(targets),
        rerun_policy='Replay complete chains from original initial states, never isolate a turn with missing history.',
        semantic_caveat='Protocol rejection includes type, reference and action-coverage failures; not all malformed JSON.',
        comparison_note=args.comparison_note,reference_inputs_private=True,physical_execution=False))
    build_error_audit(root)
    return code


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preflight-only',action='store_true')
    parser.add_argument('--group',choices=('all','new','rerun'),default='all')
    parser.add_argument('--episode-ids',nargs='+',choices=[e['episode_id'] for e in episodes()])
    parser.add_argument('--repeat',type=int,default=1)
    parser.add_argument('--inference-mode',choices=('baseline','bounded_inference'),default='bounded_inference')
    parser.add_argument('--matrix-contract',choices=('typed5_schema','typed9_schema','strict_schema','typed_json','legacy_json'),default='typed9_schema',
        help='Typed generation by default; compatibility is explicit and audited, never automatic')
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--model',default='qwen3.8-flash')
    parser.add_argument('--base-url',default='https://dashscope.aliyuncs.com/compatible-mode/v1')
    parser.add_argument('--embedding-python',type=Path,default=Path('/home/lewis/miniforge3/envs/skill_embed/bin/python'))
    parser.add_argument('--embedding-model-dir',type=Path,default=Path('/mnt/d/NebulaAI/models/skill_embedding_minilm_v1'))
    args=parser.parse_args()
    if args.output_dir is None:
        prefix=('matrix_followup_v7_typed5_' if args.matrix_contract=='typed5_schema' else
                'matrix_followup_v6_typed9_' if args.matrix_contract=='typed9_schema' else
                'matrix_followup_v5_' if args.matrix_contract!='legacy_json' else 'matrix_followup_legacy_json_')
        args.output_dir=pilot.HERE/'runs'/datetime.now().strftime(prefix+'%Y%m%d_%H%M%S_%f')
    args.top_k,args.cosine_threshold,args.cosine_margin=2,.35,.05
    logging.basicConfig(level=logging.INFO,format='%(asctime)s | %(levelname)s | %(message)s')
    return run(args)


if __name__=='__main__':
    raise SystemExit(main())
