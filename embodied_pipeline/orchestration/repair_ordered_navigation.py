"""Explicit offline correction of reviewed visit-vs-terminal constraints, never an LLM output rewrite."""
import argparse
import copy
from datetime import datetime, timezone
from pathlib import Path
import shutil
import zipfile

from . import real_scene_plan as planner
from .._legacy import load
from ..perception.real_rgbd import sha, write


def correct(alignment, plan, reviewed_targets):
    """Operator supplies ordered IDs; no instruction parsing or inferred destination."""
    if not reviewed_targets or any(type(i) is not int for i in reviewed_targets):
        raise ValueError('Explicit reviewed ordered integer target IDs required')
    tasks = plan.get('subtask_plans', [])
    steps = [s for t in tasks for s in t.get('steps', [])]
    if plan.get('unplanned_subtasks') or any(s.get('skill_id') != 'navigate_to_object' for s in steps):
        raise ValueError('Only complete pure-navigation plans can use this correction')
    targets = [s.get('arguments',{}).get('target',{}).get('element_id') for s in steps]
    if targets != reviewed_targets or len(tasks) != len(steps):
        raise ValueError('Reviewed order must exactly match one navigation action per visit')
    decomposition=alignment['task_decomposition']
    if [t['subtask_id'] for t in tasks] != [t['subtask_id'] for t in decomposition['subtasks']]:
        raise ValueError('Plan visits differ from decomposition')
    for i,t in enumerate(tasks):
        if i and (tasks[i-1]['subtask_id'] not in t.get('dependencies',[]) or
                  tasks[i-1]['subtask_id'] not in decomposition['subtasks'][i].get('dependencies',[])):
            raise ValueError('Ordered visit dependency missing; not repaired automatically')
    goals=decomposition.get('goal_state',[])
    if plan.get('deferred',{}).get('goal_state',[]) != goals:
        raise ValueError('Decomposition and skill plan disagree on goals')
    bindings=alignment.get('context_bindings',{})
    for s in steps:
        ref=s['arguments']['target']
        if bindings.get(ref['role_id'],{}).get('element_ids') != [ref['element_id']]:
            raise ValueError('Ambiguous or mismatched binding')
    kept=[];moved=[]
    for goal in goals:
        if goal.get('subject_role')=='robot' and goal.get('relation')=='near':
            ids=bindings.get(goal.get('object_role'),{}).get('element_ids',[])
            if len(ids)!=1 or ids[0] not in targets:
                raise ValueError('Goal is outside reviewed visits')
            if ids[0]!=targets[-1]:
                moved.append(copy.deepcopy(goal));continue
        kept.append(copy.deepcopy(goal))
    if not any(g.get('subject_role')=='robot' and g.get('relation')=='near' and
               bindings.get(g.get('object_role'),{}).get('element_ids')==[targets[-1]] and g.get('required',True) for g in kept):
        raise ValueError('Missing final destination requirement; not invented')
    a,p=copy.deepcopy(alignment),copy.deepcopy(plan)
    a['task_decomposition']['goal_state']=kept
    p['deferred']['goal_state']=copy.deepcopy(kept)
    note={'version':'reviewed_ordered_goal_correction_v1','source':'operator_authorized_offline_semantic_correction',
          'not_raw_model_output':True,'reviewed_target_order':targets,'moved_terminal_constraints':moved,
          'milestones':[{'subtask_id':t['subtask_id'],'step_id':s['step_id'],'target_element_id':target,
                         'depends_on_subtasks':t.get('dependencies',[]),'satisfaction_scope':'after_this_navigation_succeeds_not_global_final_state'}
                        for t,s,target in zip(tasks,steps,targets)],
          'final_goals':kept,'actions_and_dependencies_changed':False,'api_calls':0,'robot_executed':False}
    return a,p,note


def repair(source,destination,reviewed_targets):
    source,destination=Path(source).resolve(),Path(destination).resolve()
    if destination==source or source in destination.parents:
        raise ValueError('Use a separate output directory outside the source')
    task=planner.read(source/'task_input.json')
    if task.get('navigation_mode')!='ordered_goals':raise ValueError('Not ordered_goals')
    alignment,plan,note=correct(planner.read(source/'object_alignment.json'),planner.read(source/'skill_plan.json'),reviewed_targets)
    graph=source/'input_memory/scene'
    nodes=planner.read(graph/'scene_graph.json');edges=planner.read(graph/'scene_graph_edges.json')
    if sha(graph/'scene_graph.json')!=task['scene_graph_sha256'] or sha(graph/'scene_graph_edges.json')!=task['scene_edges_sha256']:
        raise ValueError('Source graph integrity mismatch')
    library=planner.read(source/'execution_skill_library.json')
    check=planner.check_navigation(plan,task,{'nodes':nodes},alignment)
    review=load('verify_plan').review(alignment,plan,library,nodes,edges)
    initial=review.get('observed_state',{}).get('initial_claim_checks',[])
    passed=check['passed'] and review['status']=='conditionally_valid_symbolic_plan' and not any(c.get('required',True) and c.get('status') not in ('observed_true','true') for c in initial)
    destination.mkdir(parents=True,exist_ok=False)
    with zipfile.ZipFile(destination/'original_evidence.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for item in source.rglob('*'):
            if item.is_file():archive.write(item,item.relative_to(source).as_posix())
    shutil.copytree(graph,destination/'input_memory/scene')
    note.update(source_directory=str(source),source_engineering_sha256=sha(source/'engineering_plan.json'),
                corrected_utc=datetime.now(timezone.utc).isoformat())
    envelope=planner.read(source/'engineering_plan.json')
    envelope.update(plan=plan,task_output_check=check,static_review_status=review['status'],semantic_correction=note)
    summary=planner.read(source/'planning_summary.json')
    summary.update(status='planning_json_generated_not_executed' if passed else 'planning_output_requires_review',
        static_review_status=review['status'],task_check_passed=check['passed'],review_issues=check['issues'],
        engineering_plan=str(destination/'engineering_plan.json'),fresh_llm_calls=False,offline_semantic_correction=True,
        source_run=str(source),input_files_unchanged=True,robot_executed=False,execution_ready=False,
        original_duration_seconds=summary.pop('duration_seconds',None))
    for name,data in [('engineering_plan',envelope),('planning_summary',summary),('object_alignment',alignment),
                      ('task_decomposition',alignment['task_decomposition']),('skill_plan',plan),('static_review',review),
                      ('task_output_check',check),('semantic_correction',note),('task_input',task),('execution_skill_library',library)]:
        write(destination/(name+'.json'),data)
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--reviewed-targets',required=True,type=int,nargs='+')
    a=p.parse_args();s=repair(a.source,a.output,a.reviewed_targets)
    print('status='+s['status']+' engineering_plan='+s['engineering_plan'])
    return 0 if s['status']=='planning_json_generated_not_executed' else 2


if __name__=='__main__':raise SystemExit(main())
