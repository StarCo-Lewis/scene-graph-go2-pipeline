"""Program-owned action occurrence IDs, scoped to a case/round/plan revision.

IDs label actions; they never imply completion, change state, or authorize replay.
The four-column model matrix and legacy step_id remain unchanged.
"""
import copy
import hashlib
import json

VERSION = 'case_round_action_identity_v1'


def assign_action_identity(plan,scope=None):
    result=copy.deepcopy(plan)
    scope=copy.deepcopy(scope or result.get('action_identity',{}).get('scope') or
                        {'case_id':'unscoped','round':0,'run_id':'unscoped'})
    if (not isinstance(scope,dict) or not isinstance(scope.get('case_id'),str) or
            not scope['case_id'].strip() or type(scope.get('round')) is not int or scope['round']<0 or
            not isinstance(scope.get('run_id'),str) or not scope['run_id'].strip()):
        raise ValueError('Invalid program action identity scope')
    canonical={'scope':scope,'instruction':result.get('instruction'), 'actions':[
        {'subtask_id':task['subtask_id'],'step_id':s['step_id'],
         'skill_id':s['skill_id'],'arguments':s['arguments']}
        for task in result['subtask_plans'] for s in task['steps']]}
    revision=hashlib.sha256(json.dumps(canonical,sort_keys=True,ensure_ascii=True,
                                      allow_nan=False).encode()).hexdigest()
    index=[]
    for task in result['subtask_plans']:
        for step in task['steps']:
            ordinal=len(index)+1
            step['action_id']=f"{scope['case_id']}/r{scope['round']:03d}/p{revision[:16]}/a{ordinal:04d}"
            step['action_ordinal']=ordinal
            index.append({'action_id':step['action_id'],'ordinal':ordinal,
                          'step_id':step['step_id'],'subtask_id':task['subtask_id']})
    result['action_identity']={'version':VERSION,'scope':scope,'plan_revision_sha256':revision,
        'source':'program_after_matrix_decoding_or_plan_composition',
        'semantics':'Occurrence identity within this plan revision; not a skill code, state, completion or idempotency guarantee. Changing the action list creates a new revision.'}
    result['action_index']=index
    return result


def validate_action_identity(plan):
    if 'action_identity' not in plan:
        return  # Historical plans remain readable; no retroactive edits.
    steps=[s for t in plan['subtask_plans'] for s in t['steps']]
    step_ids=[s.get('step_id') for s in steps]
    if any(not isinstance(i,str) or not i for i in step_ids) or len(set(step_ids))!=len(step_ids):
        raise ValueError('action step IDs must be unique nonempty strings')
    expected=assign_action_identity(plan)
    if plan.get('action_identity')!=expected['action_identity'] or plan.get('action_index')!=expected['action_index']:
        raise ValueError('Stale or inconsistent program action identity/index')
    for actual,wanted in zip([s for t in plan['subtask_plans'] for s in t['steps']],
                             [s for t in expected['subtask_plans'] for s in t['steps']]):
        if (actual.get('action_id'),actual.get('action_ordinal'))!=(wanted['action_id'],wanted['action_ordinal']):
            raise ValueError('Stale or inconsistent program action occurrence ID')
