"""Offline audit of 74 non-infrastructure failures and a separate input-only arm."""
import argparse
import copy
from collections import Counter
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import shutil

from . import batch, natural_batch
from .batch import read, read_jsonl, write, jsonl, sha
from .context_contract import public_context
from .instruction_revision_specs import VERSION, specification, revised_text, ROOM_GAPS

EXCLUDE = {'generated','account_arrearage','transport_timeout','http_error'}

def digest_text(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def revise(task):
    """Keep reference, graph, user pose and robot unknowns out of text rewriting."""
    result = copy.deepcopy(task);rid=task['record_id'];now=datetime.now(timezone.utc).isoformat()
    result['instruction']=revised_text(rid,task['instruction'])
    result['model_input']['instruction']=result['instruction']
    # A same-item task is intentionally still contextual; change its antecedent,
    # not the main command into a standalone command.
    if specification(rid)['kind']=='nonunique_referent':
        for msg in result['conversation_context'].get('dialogue_history',[]):
            msg['text']=revised_text(rid,msg['text'])
    result['model_input']['conversation_context']=copy.deepcopy(result['conversation_context'])
    result['model_input']['context_messages']=[{'role':'user','content':m['text']} for m in result['conversation_context'].get('dialogue_history',[])]
    result.update(draft_version=VERSION,revision_of=task['draft_version'],revision_at_utc=now,
        original_instruction=task['instruction'],instruction_sha256=digest_text(result['instruction']),
        instruction_unchanged_from_v2=False,collection_status='not_run',raw_model_response=None,
        review_status='AI_input_revision_user_requested_not_independent_human_review',
        approved_draft_version=VERSION,valid_candidate=None,training_eligible=False)
    result['input_revision']={'parent_record_id':rid,'attempt_id':'input_revision_01',
        'specification':specification(rid),'causal_claim':'Hypothesis only; edited inputs change the condition. A single stochastic rerun cannot prove causation.',
        'new_independent_task':False,'reference_sent_to_model':False}
    result['reading_aid_zh']=task['reading_aid_zh']+'【修订说明：'+specification(rid)['reason']+'】'
    return result

def evidence_for(root,row):
    """Resolve saved review pointers against actual artifacts, not just prose."""
    checked=[]
    for issue in row['issues']:
        for evidence in issue.get('evidence',[]):
            path=root/evidence['file']
            value=read(path)
            for segment in evidence.get('pointer','').strip('/').split('/'):
                if not segment:continue
                key=segment.replace('~1','/').replace('~0','~')
                value=value[int(key)] if isinstance(value,list) else value[key]
            if isinstance(value,str) and (isinstance(evidence['observed'],(dict,list)) or evidence.get('decoded_content_pointer')):
                value=json.loads(value)
            for segment in evidence.get('decoded_content_pointer','').strip('/').split('/'):
                if not segment:continue
                key=segment.replace('~1','/').replace('~0','~')
                value=value[int(key)] if isinstance(value,list) else value[key]
            checked.append({'file':str(path),'pointer':evidence.get('pointer',''),
                            'decoded_content_pointer':evidence.get('decoded_content_pointer'),
                            'sha256':sha(path),'observed':value,'expected':evidence.get('expected'),
                            'prior_review_observed':evidence['observed'],
                            'prior_excerpt_exact_match':value==evidence['observed']})
    return checked

def create(root,out):
    root=root.resolve();out=out.resolve()
    if out.exists():raise FileExistsError('Use a new output folder; originals and prior audits are immutable')
    rows=[r for r in read_jsonl(root/'reports/quality_check_natural_v1/scores.jsonl') if r['failure_category'] not in EXCLUDE]
    if len(rows)!=74:raise ValueError('Expected exactly 74 non-infrastructure failures')
    config=read(root/'collection_config.json');graphs={};edges={}
    for scene in natural_batch.SCENES:
        folder=root/'cases'/scene/'graph'
        batch.require_graph_unchanged(folder,config['scenes'][scene])
        graphs[scene]=read(folder/'scene_graph.json');edges[scene]=read(folder/'scene_graph_edges.json')
    audits=[];changes=[];ready={s:[] for s in natural_batch.SCENES}
    for row in rows:
        rid=row['record_id'];scene=row['scene_id'];folder=root/'cases'/scene/'tasks'/rid
        task=read(folder/'task.json');ref=read(folder/'reference.review_only.json')
        if task['instruction']!=row['instruction']:raise ValueError('Task differs from reviewed instruction')
        spec=specification(rid);gaps=ROOM_GAPS.get(rid,[])
        evidence=evidence_for(root,row)
        model_errors=[i['reason'] for i in row['issues']]
        if not model_errors and 'protocol' in row['failure_category']:model_errors=[row['raw_error']]
        input_problem=bool(spec or gaps)
        category=('mixed_input_and_output' if input_problem and model_errors else
                  'input_evidence_insufficient' if input_problem else
                  'model_output_or_representation_contract' if model_errors else 'unresolved_review_required')
        item={'record_id':rid,'scene_id':scene,'category':category,
              'instruction':task['instruction'],'conversation_context':task['conversation_context'],
              'runtime_stop_category':row['failure_category'],'raw_error':row['raw_error'],
              'instruction_issue':spec,'public_data_gap':{'missing_room_identity':gaps} if gaps else None,
              'observable_output_errors':model_errors,'prior_review_notes':row.get('notes',[]),
              'evidence':evidence,'task_source':str(folder/'task.json'),'task_sha256':sha(folder/'task.json'),
              'reference_source':str(folder/'reference.review_only.json'),
              'reference_not_ground_truth':'Authored intended objective, not proof that the old instruction uniquely specified it.',
              'causality':'Observed mismatch is not proof of an intrinsic LLM defect or of a fixed-prompt cause; controlled ablations are still needed.',
              'decision':'preserve_original_no_instruction_rerun'}
        if gaps:item['decision']='hold_for_public_room_identity_data'
        if spec:
            new=revise(task);public_context(new,graphs[scene])
            if (new['instruction'],new['conversation_context'])==(task['instruction'],task['conversation_context']):raise ValueError('Empty revision '+rid)
            for n in spec.get('node_ids',[]):
                if n not in {x['id'] for x in graphs[scene]}:raise ValueError('Evidence node missing')
            idxs=spec.get('edge_indices',[])+([spec['edge_index']] if 'edge_index' in spec else [])
            graph_evidence=[{'file':str(root/'cases'/scene/'graph/scene_graph.json'),
                            'pointer':'/'+str(i),'observed':n} for i,n in enumerate(graphs[scene]) if n['id'] in spec.get('node_ids',[])]
            graph_evidence += [{'file':str(root/'cases'/scene/'graph/scene_graph_edges.json'),
                               'pointer':'/'+str(i),'observed':edges[scene][i]} for i in idxs]
            change={'record_id':rid,'scene_id':scene,'before_instruction':task['instruction'],
                'after_instruction':new['instruction'],'before_context':task['conversation_context'],
                'after_context':new['conversation_context'],'specification':spec,
                'graph_evidence':graph_evidence,'reference_steps_unchanged':True,
                'eligible_to_run':not gaps,'held_reason':gaps,
                'original_task_sha256':sha(folder/'task.json'),'previous_summary_sha256':sha(folder/'model_run/planning_summary.json')}
            changes.append((change,new,ref));item['decision']='revised_but_held_for_room_data' if gaps else 'input_revision_rerun_ready'
            if not gaps:ready[scene].append((new,ref,change))
        audits.append(item)
    if len(changes)!=39 or sum(map(len,ready.values()))!=34:raise ValueError('Unexpected revision selection')
    out.mkdir(parents=True)
    jsonl(out/'attribution_74.jsonl',audits)
    jsonl(out/'instruction_changes_39.jsonl',[c for c,t,r in changes])
    for c,t,r in changes:
        write(out/'revised_drafts'/t['record_id']/'task.json',t)
        write(out/'revised_drafts'/t['record_id']/'reference.review_only.json',r)
    bundle=out/'corrected_run';fresh=copy.deepcopy(config)
    fresh.update(version=VERSION,source_collection=str(root),prepared_at_utc=datetime.now(timezone.utc).isoformat(),
        scenes={},approved_retry_tasks=34,attempt_id='input_revision_01',selection='instruction_defect_only_with_public_evidence',
        account_circuit_breaker=True,max_requests_all_scenes=102,max_requests_per_scene={s:3*len(ts) for s,ts in ready.items()},
        source_review=str(out/'instruction_changes_39.jsonl'),user_approved_reviewed_draft=False,
        user_authorized_instruction_repair=True,
        clarification_policy='Input-only revision arm; original references retained privately, graph/runtime/prompts unchanged. One attempt per task; do not pick best-of-two or count revisions as new unique tasks.')
    combined=[]
    for scene,items in ready.items():
        source=root/'cases'/scene;target=bundle/'cases'/scene
        shutil.copytree(source/'graph',target/'graph');shutil.copytree(source/'modalities',target/'modalities')
        for name in ['modality_index.json','source_modality_index.json','input_graph_audit.json']:
            if (source/name).exists():shutil.copy2(source/name,target/name)
        tasks=[t for t,r,c in items];refs=[r for t,r,c in items]
        for t,r,c in items:
            p=target/'tasks'/t['record_id'];write(p/'task.json',t);write(p/'reference.review_only.json',r);write(p/'input_revision.json',c)
        jsonl(target/'tasks.jsonl',tasks);jsonl(target/'references.review_only.jsonl',refs)
        info=copy.deepcopy(config['scenes'][scene]);info.update(count=len(tasks),tasks_sha256=sha(target/'tasks.jsonl'))
        fresh['scenes'][scene]=info;combined+=tasks
    shutil.copytree(root/'shared',bundle/'shared')
    jsonl(bundle/'dataset.inputs.jsonl',combined);write(bundle/'collection_config.json',fresh)
    jsonl(bundle/'retry_lineage.jsonl',[c for c,t,r in changes if c['eligible_to_run']])
    batch.code_snapshot(bundle)
    write(bundle/'input_manifest.json',[{'path':p.relative_to(bundle).as_posix(),'sha256':sha(p)} for p in sorted((bundle/'cases').rglob('*')) if p.is_file()])
    natural_batch.summarize(bundle)
    summary={'version':VERSION,'scope':'74 original non-infrastructure failures only; not all 600 or the in-flight 330',
      'records_reviewed':74,'categories':dict(Counter(a['category'] for a in audits)),
      'instruction_revisions':39,'ready':34,'held_revisions':5,'public_room_data_gaps':13,
      'per_scene_ready':{s:len(ts) for s,ts in ready.items()},'max_new_api_requests':102,
      'paid_calls_made':0,'physical_execution':False,'input_reference_leakage':False,
      'causal_proof':False,'independent_human_review':False,
      'evidence_pointers_verified':sum(len(a['evidence']) for a in audits),
      'notes':['303 original arrearage responses including the previously generic HTTP 400; 27 timeouts are outside this semantic audit.',
               'Input edits cannot repair missing room mappings or prove model-intrinsic causes.',
               'Graph facts are supplied hypotheses, not physical ground truth; no edges or poses are invented.',
               'The full public scene and the existing fixed prompts, retrieval and validation remain unchanged.']}
    write(out/'summary.json',summary);render(out,audits,changes,summary)
    print(json.dumps(summary,ensure_ascii=False))

def render(out,audits,changes,summary):
    esc=lambda v:html.escape(str(v))
    parts=['<!doctype html><meta charset="utf-8"><title>74条失败归因与指令修订</title>',
      '<style>body{font:16px/1.65 sans-serif;max-width:1200px;margin:32px auto;padding:0 24px;color:#182235}article{border:1px solid #ccd3dd;border-radius:8px;padding:20px;margin:20px 0}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f3f5f8;padding:14px}h1,h2{line-height:1.3}summary{cursor:pointer}a{color:#1466aa}</style>',
      '<h1>74条非基础设施失败：归因与输入修订</h1>',
      '<p>这是证据归类和待验证的修订，不是已证实的模型内部因果解释。303条欠费、27条超时仍在独立330条补跑中。原数据不覆盖，修改版不计新增独立任务。</p>',
      '<p>39条存在可改进的句法或实例指代，其中34条可单独运行，5条还缺公开房间身份信息暂缓。13条存在公开房间映射缺口（与指令问题重叠）。只修用户指令/其对话前文；固定提示词、图、技能候选策略、校验器不变。没有调用API或机器人。</p>',
      '<p>新增来源限定属于任务作者明确指代，不是证明原指令已有足够信息。已有参考仅用于保留任务意图；步骤答案不会传给模型。图中关系未经过真机核实。</p>',
      '<pre>'+esc(json.dumps(summary,ensure_ascii=False,indent=2))+'</pre><h2>逐条修改（原文 → 修订）</h2>']
    for c,t,r in changes:
        parts+=['<article id="change-'+esc(c['record_id'])+'"><h3>'+esc(c['record_id'])+(' · 可运行' if c['eligible_to_run'] else ' · 暂缓：缺房间身份数据')+'</h3>',
                '<p>'+esc(c['specification']['reason'])+'</p><b>原指令</b><pre>'+esc(c['before_instruction'])+'</pre><b>修订指令</b><pre>'+esc(c['after_instruction'])+'</pre>']
        if c['before_context']!=c['after_context']:
            parts+=['<b>前文同步修改（主句中的that same item仍然保留）</b><pre>'+esc(json.dumps({'before':c['before_context'],'after':c['after_context']},ensure_ascii=False,indent=2))+'</pre>']
        parts+=['<p>原中文阅读辅助：'+esc(t['reading_aid_zh'])+'</p></article>']
    parts+=['<h2>74条逐例归因</h2>']
    for a in audits:
        parts+=['<article id="'+esc(a['record_id'])+'"><h3>'+esc(a['record_id'])+' · '+esc(a['category'])+'</h3><pre>'+esc(a['instruction'])+'</pre>',
          '<p>处理：'+esc(a['decision'])+'</p><p>输入问题：'+esc(a['instruction_issue'])+'</p><p>数据缺口：'+esc(a['public_data_gap'])+'</p>',
          '<p>可观察输出问题：'+esc('；'.join(a['observable_output_errors']) or '未发现独立输出错误；拒绝/澄清不等于模型错误。')+'</p>',
          '<details><summary>上下文、原始错误和文件证据</summary><pre>'+esc(json.dumps(a,ensure_ascii=False,indent=2))+'</pre></details></article>']
    (out/'report.html').write_text('\n'.join(parts),encoding='utf-8')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();create(a.source,a.output)

if __name__=='__main__':main()
