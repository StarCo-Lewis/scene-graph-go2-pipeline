"""Prepare a new, linked attempt for selected infrastructure failures only."""
import argparse
import copy
from datetime import datetime, timezone
from pathlib import Path
import shutil

from . import batch,natural_batch
from .batch import read,read_jsonl,write,jsonl,sha
from .plan_artifact import has_complete

VERSION='natural_infrastructure_retry_v1'
EXPECTED={'scene200':128,'scene202':112,'d3_619':90}

def failure_kind(summary):
    if summary.get('engineering_json_generated'):return None
    error=summary.get('error') or ''
    if 'Arrearage' in error:return 'account_arrearage'
    if 'TimeoutError' in error:return 'transport_timeout'
    if 'HTTP' in error:return 'http_error'
    return None

def prepare(source,output):
    source=source.resolve();output=output.resolve()
    if source==output or output.exists():raise ValueError('Retry destination must be new; originals cannot be overwritten')
    config=read(source/'collection_config.json')
    if config.get('public_context_version')!='reviewed_user_context_v1':raise ValueError('Wrong source collection')
    if list(source.glob('cases/*/worker.lock')):raise ValueError('Source still has workers; inspect before retrying')
    selected={};lineage=[]
    for scene in natural_batch.SCENES:
        batch.require_graph_unchanged(source/'cases'/scene/'graph',config['scenes'][scene])
        tasks=batch.tasks_for(source,scene);chosen=[]
        for task in tasks:
            folder=source/'cases'/scene/'tasks'/task['record_id']
            summary_path=folder/'model_run/planning_summary.json';summary=read(summary_path)
            kind=failure_kind(summary)
            if not kind:continue
            # Failed attempts carry an incomplete engineering_plan.json; only a complete plan is unexpected.
            if has_complete(folder/'model_run/engineering_plan.json'):raise ValueError('Unexpected completed plan in infrastructure retry selection')
            if task!=read(folder/'task.json'):raise ValueError('Source task copies differ')
            chosen.append(task)
            lineage.append({'record_id':task['record_id'],'scene_id':scene,'attempt_id':'infra_retry_01',
                'previous_attempt_summary':str(summary_path),'previous_summary_sha256':sha(summary_path),
                'previous_failure_kind':kind,'original_task_sha256':sha(folder/'task.json'),
                'rerun_policy':'New full three-stage attempt; no partial-stage reuse; no resampling for quality.'})
        if len(chosen)!=EXPECTED[scene]:raise ValueError('Selected count differs from user-approved 330: '+scene)
        selected[scene]=chosen
    output.mkdir(parents=True)
    fresh=copy.deepcopy(config);fresh.update(version=VERSION,source_collection=str(source),
        prepared_at_utc=datetime.now(timezone.utc).isoformat(),selection='infrastructure_failures_only',
        approved_retry_tasks=330,attempt_id='infra_retry_01',max_requests_all_scenes=990,
        account_circuit_breaker=True,scenes={},max_requests_per_scene={s:3*n for s,n in EXPECTED.items()},
        clarification_policy='Selected infrastructure failures rerun once from the start. Same typed5 protocol. Keep incomplete and clarification outputs ungraded; do not select best-of-two outputs.')
    combined=[]
    for scene,tasks in selected.items():
        old=source/'cases'/scene;target=output/'cases'/scene
        shutil.copytree(old/'graph',target/'graph')
        shutil.copytree(old/'modalities',target/'modalities')
        for name in ['modality_index.json','source_modality_index.json','input_graph_audit.json']:
            if (old/name).exists():shutil.copy2(old/name,target/name)
        references=[]
        for task in tasks:
            rid=task['record_id'];folder=target/'tasks'/rid;folder.mkdir(parents=True)
            shutil.copy2(old/'tasks'/rid/'task.json',folder/'task.json')
            shutil.copy2(old/'tasks'/rid/'reference.review_only.json',folder/'reference.review_only.json')
            references.append(read(folder/'reference.review_only.json'))
        jsonl(target/'tasks.jsonl',tasks);jsonl(target/'references.review_only.jsonl',references)
        info=copy.deepcopy(config['scenes'][scene]);info.update(count=len(tasks),tasks_sha256=sha(target/'tasks.jsonl'))
        fresh['scenes'][scene]=info;combined+=tasks
    shutil.copytree(source/'shared',output/'shared')
    jsonl(output/'dataset.inputs.jsonl',combined);jsonl(output/'retry_lineage.jsonl',lineage)
    write(output/'collection_config.json',fresh)
    batch.code_snapshot(output)
    write(output/'input_manifest.json',[{'path':p.relative_to(output).as_posix(),'sha256':sha(p)} for p in sorted((output/'cases').rglob('*')) if p.is_file()])
    natural_batch.summarize(output)
    print('Prepared 330 new attempts only; no API requests made. Output='+str(output))

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--approve-infrastructure-retry',action='store_true')
    a=p.parse_args(argv)
    if not a.approve_infrastructure_retry:p.error('--approve-infrastructure-retry required')
    prepare(a.source,a.output)
    return 0

if __name__=='__main__':raise SystemExit(main())
