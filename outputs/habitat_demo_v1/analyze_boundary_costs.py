"""Read-only historical evidence audit plus offline guard replay; no API calls."""
import argparse
import os
from pathlib import Path

from run_json_long_horizon import SCENE
from run_navigation_demo import read_json, write_json
from state_boundary_guards import attach_constraints, repair_placement_approach
from token_usage_audit import usage


def analyze(source, output):
    if os.name == 'nt' and not str(source).startswith('\\\\?\\'):
        source = Path('\\\\?\\' + str(source.resolve()))
    output.mkdir(parents=True, exist_ok=False)
    batch = read_json(source/'batch_summary.json')
    rows = []
    for trial in batch['trials']:
        # Resolve by trial identity relative to given batch, not platform-specific stored paths.
        folder = source/'trials'/f'{trial["case"]}_repeat_{trial["repeat"]}_{trial["strategy"]}'
        print(f'AUDIT {folder.name}', flush=True)
        rows.append({'case':trial['case'], 'profile':trial['strategy'], 'repeat':trial['repeat'],
                     'outcome':trial['evaluation_outcome'], 'completed_rounds':trial['completed_round_count'],
                     'seconds':trial['duration_seconds'], 'usage':usage(folder)})
    groups = []
    for name in batch['case_input_hashes']:
        for arm in batch['comparison_arms']:
            group = [r for r in rows if r['case']==name and r['profile']==arm]
            groups.append({'case':name,'profile':arm, 'trials':len(group),
                'passed':sum(r['outcome'].startswith(('clean_', 'passed_')) for r in group),
                'completed_rounds':sum(r['completed_rounds'] for r in group),
                'seconds':sum(r['seconds'] for r in group),
                **{k:sum(r['usage'][k] for r in group) for k in ('prompt_tokens','completion_tokens',
                    'total_tokens','http_responses','responses_with_usage','responses_missing_usage',
                    'transport_failures_unknown_usage')}})
    root=source/'trials/extended_mixed_18_repeat_1_v2/round_16/planning/hierarchy/node_003'
    decision=read_json(root/'hierarchy_split_attempt_1.json')
    parts,events=attach_constraints(decision['parts'])
    assert len(parts)==1 and ''.join(decision['parts'])==parts[0] and events
    constraint={'historical_error':'EMPTY_LEAF_PLAN', 'raw_parts':decision['parts'],
                'repaired_parts':parts, 'events':events, 'text_preserved':True,
                'scope':'Offline partition repair; no new model plan and no full-round success claim'}
    root=source/'trials/extended_mixed_18_repeat_2_v2/round_5/planning/hierarchy/node_005/planning'
    descriptor=read_json(root/'round_input.json')
    plan,review,report=repair_placement_approach(descriptor,read_json(root/'object_alignment.json'),
        read_json(root/'skill_plan_final.json'),read_json(root/'execution_skill_library.json'),
        read_json(root/'input_memory/scene/scene_graph.json'),read_json(root/'runtime_state.json'))
    assert report['changed'] and plan['action_count']==2 and report['end_state_identical']
    write_json(output/'placement_candidate.json',plan)
    write_json(output/'placement_candidate_review.json',review)
    write_json(output/'historical_guard_checks.json',{'status':'offline_guard_checks_passed',
        'constraint_attachment':constraint,'placement':report,'fresh_llm_calls':False,
        'completed_chain_claim':False,'simulation_executed':False,'training_eligible':False})
    write_json(output/'token_usage_summary.json',{'source':str(source),'trials':rows,'groups':groups,
        'scope':'Received-response usage, not billing; timeouts unknown. Unequal completed workloads.'})
    print(f'summary={output/"token_usage_summary.json"}',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-batch',type=Path,default=SCENE/'hierarchical_json_ab/scene201_20261004_145912_513615')
    p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args()
    analyze(a.source_batch,a.output_dir)
