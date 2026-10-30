"""Approved three-scene context-aware collection. Offline prepare; explicit run."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
import re
from pathlib import Path
import shutil

from . import batch
from .batch import read,read_jsonl,write,jsonl,sha
from .context_contract import VERSION as CONTEXT_VERSION,public_context
from .resources import audit_graph
from .._legacy import ROOT

VERSION='natural_context_collection_v1'
SCENES=('scene200','scene202','d3_619')

def prepare(args):
    source=args.source.resolve();root=args.output.resolve()
    if not args.approve_reviewed_draft:raise ValueError('User approval of this draft is required')
    if root.exists():raise FileExistsError('Prepared destination already exists; do not overwrite or regenerate')
    summary=read(source/'draft_summary.json')
    if summary.get('total_records')!=600 or summary.get('model_calls')!=0:
        raise ValueError('Expected unrun reviewed 600-task draft')
    # Validate every source artifact before freezing a paid-run input set.
    for row in read(source/'file_manifest.json'):
        p=(source/row['path']).resolve();p.relative_to(source)
        if sha(p)!=row['sha256']:raise ValueError('Reviewed draft changed: '+row['path'])
    scene_inputs={}
    for scene in SCENES:
        base=source/'cases'/scene;tasks=read_jsonl(base/'tasks.jsonl')
        refs=read_jsonl(base/'references.review_only.jsonl')
        if len(tasks)!=200 or len(refs)!=200:raise ValueError('Expected 200 per scene')
        if len({t['record_id'] for t in tasks})!=200 or any(not re.fullmatch(re.escape(scene)+r'_\d{8}',t['record_id']) for t in tasks):
            raise ValueError('Task IDs must be unique scene_id + eight digits')
        if [t['record_id'] for t in tasks]!=[r['record_id'] for r in refs]:raise ValueError('Reference IDs mismatch')
        nodes=read(base/'graph/scene_graph.json')
        for task in tasks:
            public_context(task,nodes)
            if task['scene_id']!=scene or task['collection_status']!='not_run':raise ValueError('Unexpected task state')
        audit=audit_graph(base/'graph')
        if audit['blocking_issues']:raise ValueError('Scene graph integrity error: '+scene)
        scene_inputs[scene]=(tasks,refs,audit)
    root.mkdir(parents=True)
    config={'version':VERSION,'public_context_version':CONTEXT_VERSION,'model':args.model,'base_url':args.base_url,
       'prepared_at_utc':datetime.now(timezone.utc).isoformat(),'source_review':str(source),
       'source_review_summary_sha256':sha(source/'draft_summary.json'),
       'user_approved_reviewed_draft':True,'max_requests_per_task':3,'max_requests_per_scene':600,
       'max_requests_all_scenes':1800,'monetary_budget_estimated':False,'scenes':{},
       'semantic_retries':0,'http_attempts_per_stage':1,'robot_execution':False,
       'context_projection':'instruction + allowlisted conversation_context/public_spatial_context via execution_contract; references excluded',
       'clarification_policy':'All 600 receive the same existing typed5 protocol. Do not use private expected_response to route or skip calls. Keep raw ambiguity/protocol rejection evidence ungraded; no new free-text clarification schema is introduced.'}
    combined=[]
    for scene,(tasks,refs,audit) in scene_inputs.items():
        base=source/'cases'/scene;target=root/'cases'/scene
        shutil.copytree(base/'graph',target/'graph')
        shutil.copy2(base/'source_modality_index.json',target/'source_modality_index.json')
        # Freeze the already-collected images/metadata too; no new VLM calls.
        indexes={t['modality_resource_index'] for t in tasks}
        if len(indexes)!=1:raise ValueError('Expected a shared scene modality index')
        modality_source=batch.portable_source(next(iter(indexes)))
        modality_index=read(modality_source)
        if modality_index!=read(base/'source_modality_index.json'):raise ValueError('Modality index differs from reviewed source')
        shutil.copytree(modality_source.parent/'modalities',target/'modalities')
        for resource in modality_index['node_resources']:
            if resource.get('exists'):
                image=(target/resource['bundle_path']).resolve();image.relative_to(target.resolve())
                if sha(image)!=resource['copy_sha256']:raise ValueError('Modality hash mismatch')
        write(target/'modality_index.json',modality_index)
        for t,r in zip(tasks,refs):
            t['collection_public_context_version']=CONTEXT_VERSION
            t['approved_draft_version']=t['draft_version']
            t['approved_source_task_sha256']=sha(base/'tasks'/t['record_id']/'task.json')
            t['source_modality_resource_index']=t['modality_resource_index']
            t['modality_resource_index']='cases/'+scene+'/modality_index.json'
            t['draft_review_status']=t['review_status']
            t['review_status']='user_approved_for_collection_not_independent_quality_check'
            write(target/'tasks'/t['record_id']/'task.json',t)
            write(target/'tasks'/t['record_id']/'reference.review_only.json',r)
        jsonl(target/'tasks.jsonl',tasks);jsonl(target/'references.review_only.jsonl',refs)
        combined+=tasks
        config['scenes'][scene]={'count':len(tasks),'tasks_sha256':sha(target/'tasks.jsonl'),
          'graph_sha256':sha(target/'graph/scene_graph.json'),'edges_sha256':sha(target/'graph/scene_graph_edges.json'),
          'metadata_sha256':sha(target/'graph/scene_metadata.json') if (target/'graph/scene_metadata.json').exists() else None,
          'source_review_tasks_sha256':sha(base/'tasks.jsonl')}
        write(target/'input_graph_audit.json',audit)
        shutil.copy2(base/'instructions_review.html',target/'instructions_review.html')
    jsonl(root/'dataset.inputs.jsonl',combined)
    (root/'shared').mkdir()
    shutil.copy2(ROOT/'outputs/quadruped_atomic_skill_library.v0.1.json',root/'shared/source_skill_library.json')
    shutil.copy2(source/'instructions_review.html',root/'instructions_review.html')
    write(root/'collection_config.json',config)
    batch.code_snapshot(root)
    write(root/'input_manifest.json',[{'path':str(p.relative_to(root)),'sha256':sha(p)} for p in sorted((root/'cases').rglob('*')) if p.is_file()])
    summarize(root)
    print(json.dumps({'status':'prepared_not_run','root':str(root),'tasks':600,'max_api_requests':1800,'model':args.model},ensure_ascii=False))

def summarize(root):
    root=Path(root);config=read(root/'collection_config.json');results=[];counts={}
    for scene in SCENES:
        rows=[]
        for task in batch.tasks_for(root,scene):
            folder=root/'cases'/scene/'tasks'/task['record_id']/'model_run'
            item={'record_id':task['record_id'],'scene_id':scene,'instruction':task['instruction'],
              'conversation_context':task['conversation_context'],'horizon':task['horizon'],
              'planning_summary':None,'status':'not_started','semantic_correctness_evaluated':False}
            if (folder/'planning_summary.json').is_file():
                saved=read(folder/'planning_summary.json');item.update(status=saved['status'],planning_summary=str(folder/'planning_summary.json'),
                  engineering_json_generated=saved.get('engineering_json_generated',False),
                  failed_stage=saved.get('failed_stage'),error=saved.get('error'),duration_seconds=saved.get('duration_seconds'),
                  collection_started_at_utc=saved.get('collection_started_at_utc'),collection_finished_at_utc=saved.get('collection_finished_at_utc'))
            elif (folder/'harness_error.json').is_file():item.update(status='task_harness_error_ungraded',harness_error=read(folder/'harness_error.json'))
            elif folder.exists():item['status']='started_no_final_summary_do_not_replay'
            if (folder/'engineering_plan.json').exists():
                dest=root/'engineering_json'/scene/(task['record_id']+'.json');dest.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(folder/'engineering_plan.json',dest)
                item['engineering_json']=str(dest)
            rows.append(item)
        counts[scene]=dict(Counter(x['status'] for x in rows));results+=rows
    jsonl(root/'results.ungraded.jsonl',results)
    write(root/'collection_summary.json',{'version':VERSION,'records':len(results),'scenes':counts,
       'engineering_json_count':sum('engineering_json' in r for r in results),'semantic_correctness_evaluated':False,
       'protocol_note':config['clarification_policy'],'updated_at_utc':datetime.now(timezone.utc).isoformat()})
    print(json.dumps({'scene_status_counts':counts,'engineering_json_count':sum('engineering_json' in r for r in results)},ensure_ascii=False))

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['prepare','worker','summarize'])
    p.add_argument('--output',type=Path,required=True);p.add_argument('--source',type=Path)
    p.add_argument('--scene',choices=SCENES);p.add_argument('--approve-reviewed-draft',action='store_true')
    p.add_argument('--approve-new-api-calls',action='store_true');p.add_argument('--max-api-calls',type=int)
    p.add_argument('--model',default='qwen3.8-flash');p.add_argument('--base-url',default='https://dashscope.aliyuncs.com/compatible-mode/v1')
    p.add_argument('--embedding-python',type=Path,default=Path('/home/lewis/miniforge3/envs/skill_embed/bin/python'))
    p.add_argument('--embedding-model',type=Path,default=Path('/mnt/d/NebulaAI/models/skill_embedding_minilm_v1'))
    p.add_argument('--request-timeout',type=float,default=180.);p.add_argument('--limit-per-scene',type=int)
    a=p.parse_args(argv)
    if not math.isfinite(a.request_timeout) or a.request_timeout<=0:p.error('Timeout must be positive')
    if a.limit_per_scene is not None and a.limit_per_scene<1:p.error('Limit must be positive')
    if a.command=='prepare':
        if a.source is None:p.error('--source required')
        prepare(a)
    elif a.command=='worker':
        if a.scene is None:p.error('--scene required')
        if read(a.output/'collection_config.json').get('public_context_version')!=CONTEXT_VERSION:raise ValueError('Wrong collection version')
        # Uses existing per-scene lock, CPU encoder preflight, per-task error
        # isolation and durable at-most-once directories. No automatic retry.
        return batch.run_scene(a)
    else:summarize(a.output)
    return 0

if __name__=='__main__':raise SystemExit(main())
