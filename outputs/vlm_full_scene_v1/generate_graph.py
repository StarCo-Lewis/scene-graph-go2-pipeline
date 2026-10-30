"""VLM semantics for EVERY inventoried instance, plus evidence-gated on/in edges.

Standard library + Pillow. No ConceptGraphs, CLIP object features or manual labels.
No network calls with --preflight-only. API responses (including invalid ones)
are saved without request headers or API keys.
"""
from __future__ import annotations
import argparse
import base64
import hashlib
import json
import logging
import math
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/'object_alignment_v1'))
from scene_input_interface import normalize_scene_input
from llm_graph_binding import validate_edges


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda:f.read(1024*1024),b''):
            h.update(part)
    return h.hexdigest()


def chunks(rows, count):
    return [rows[i:i+count] for i in range(0,len(rows),count)]


NODE_SYSTEM = '''Describe rendered scene instances in English using ONLY the images.
Every numbered cell shows TWO isolated views of the SAME geometric instance.
These are source-asset views with other objects hidden, not robot observations.
Numbers are stable instance IDs, never categories. Do not merge IDs or invent IDs.
Do not guess unseen objects or brand names. No fixed category vocabulary is imposed.
Walls, floors, windows, furniture, decorations and small objects must ALL be covered.
Return ONLY valid JSON: {"nodes":[{"id":integer,"status":"recognized|uncertain",
"object_tag":"English noun phrase","caption":"English visual description",
"evidence":"specific visible features","uncertainty_reason":null_or_string}]}.
Return every requested ID exactly once. Ambiguous fragments must be uncertain;
an isolated view is not proof of graspability, navigability or placement support.
Never output coordinates, affordance claims, relations or extra fields.'''

EDGE_SYSTEM = '''Evaluate visual relations in pairs of scene instances.
Each numbered cell shows TWO views of ONLY the specified pair in their UNCHANGED
source world transforms; other scene objects have been hidden. Overlay boxes are
projected AABBs, NOT segmentation masks or proof of visibility/contact.
Image view alone cannot verify physical support. Return a conservative HYPOTHESIS
only when both identified objects and the relation are clear in both views.
Use on for apparent resting on a surface; in for visibly inside a real container,
not just overlapping bounding boxes. Adjacent, behind, floating, intersecting,
hidden, ambiguous or attached wall decoration => unknown. Do not guess relations
from supplied labels or the mere existence of a candidate. Direction is source
relative to target. Missing edges remain unknown, never false.
Return ONLY valid JSON: {"decisions":[{"source":integer,"target":integer,
"relation":"on|in|unknown","evidence":"short English visible evidence",
"both_visible_in_both_views":boolean}]} with EVERY requested pair exactly once.
Do not claim physics, contact validation or robot reachability.'''

OVERVIEW_SYSTEM = '''Describe each numbered FULL SCENE view in English. These
views retain source scene geometry and occlusion. Some are global exterior views,
others overhead views of floor candidates. Do not assume each floor is a room.
Return ONLY valid JSON: {"views":[{"id":integer,"caption":"visible layout and furnishings",
"room_type_hint":"English room type or unknown","uncertain":boolean}]}.
Include every requested view exactly once. A room label is a visual hypothesis,
not ground truth. Do not invent hidden objects, routes, object IDs or coordinates.'''


def validate_overview(value, rows):
    result=value.get('views') if isinstance(value,dict) else None
    expected={r['id'] for r in rows}
    if not isinstance(result,list) or len(result)!=len(rows):
        raise ValueError('Overview must cover every requested full-scene view')
    seen=set()
    for row in result:
        if not isinstance(row,dict) or type(row.get('id')) is not int or row['id'] not in expected or row['id'] in seen:
            raise ValueError('Invalid overview view ID')
        seen.add(row['id'])
        if type(row.get('uncertain')) is not bool:
            raise ValueError('Overview uncertainty must be boolean')
        if any(not isinstance(row.get(k),str) or not row[k].strip() for k in ('caption','room_type_hint')):
            raise ValueError('Missing overview description')
    return result


def validate_nodes(value, rows):
    result = value.get('nodes') if isinstance(value,dict) else None
    if not isinstance(result,list) or len(result)!=len(rows):
        raise ValueError('nodes must cover the entire requested batch')
    expected = {r['id'] for r in rows}
    seen = set()
    for row in result:
        if not isinstance(row,dict) or type(row.get('id')) is not int or row['id'] not in expected or row['id'] in seen:
            raise ValueError('Unknown, duplicate or non-integer node ID')
        seen.add(row['id'])
        if row.get('status') not in ('recognized','uncertain'):
            raise ValueError('Invalid node status')
        for field in ('object_tag','caption','evidence'):
            if not isinstance(row.get(field),str) or not row[field].strip():
                raise ValueError('Missing node '+field)
        if row['status']=='uncertain' and (not isinstance(row.get('uncertainty_reason'),str)
                                         or not row['uncertainty_reason'].strip()):
            raise ValueError('Uncertain node requires an uncertainty reason')
    return result


def validate_decisions(value, rows):
    result = value.get('decisions') if isinstance(value,dict) else None
    if not isinstance(result,list) or len(result)!=len(rows):
        raise ValueError('decisions must cover every requested pair')
    expected = {(r['source'],r['target']) for r in rows}
    seen = set()
    for row in result:
        if not isinstance(row,dict) or any(type(row.get(k)) is not int for k in ('source','target')):
            raise ValueError('Invalid edge endpoint types')
        key = row['source'],row['target']
        if key not in expected or key in seen:
            raise ValueError('Unknown/duplicate edge pair')
        seen.add(key)
        if row.get('relation') not in ('on','in','unknown') or type(row.get('both_visible_in_both_views')) is not bool:
            raise ValueError('Invalid relation decision')
        if not isinstance(row.get('evidence'),str) or not row['evidence'].strip():
            raise ValueError('Missing relation evidence')
    return result


def sheet(root, rows, stage, destination):
    from PIL import Image,ImageDraw
    # Each tile preserves both original views, no AI-generated imagery.
    width,height = (384 if stage=='overview' else 768),420
    canvas = Image.new('RGB',(width*2,height*math.ceil(len(rows)/2)),'white')
    draw = ImageDraw.Draw(canvas)
    for index,row in enumerate(rows):
        x,y = (index%2)*width,(index//2)*height
        caption = f'VIEW {row["id"]}' if stage=='overview' else (f'INSTANCE ID {row["id"]}' if stage=='nodes'
                  else f'SOURCE {row["source"]} -> TARGET {row["target"]}')
        draw.text((x+8,y+8),caption,fill='black')
        for view_index,view in enumerate(row['render_views']):
            with Image.open(root/view['path']) as original:
                image = original.convert('RGB').resize((384,384))
            if stage=='edges':
                overlay = ImageDraw.Draw(image)
                for identity,uv in view['projected_aabb_uv'].items():
                    box = [max(0,min(383,round(v*384))) for v in uv]
                    color = 'red' if int(identity)==row['source'] else 'blue'
                    overlay.rectangle(box,outline=color,width=2)
                    overlay.text((box[0]+3,box[1]+3),identity,fill=color,stroke_width=1,stroke_fill='white')
            canvas.paste(image,(x+view_index*384,y+30))
    destination.parent.mkdir(parents=True,exist_ok=True)
    canvas.save(destination)


def redact_diagnostic(text, key):
    """Keep provider diagnostics useful without persisting echoed credentials."""
    text = str(text)
    if key:
        text = text.replace(key, '[REDACTED]')
    text = re.sub(r'(?i)\bBearer\s+[^\s"\'<>]+', 'Bearer [REDACTED]', text)
    text = re.sub(r'\bsk-[A-Za-z0-9_-]+', '[REDACTED]', text)
    text = re.sub(r'(?i)(["\']?(?:api[_-]?key|authorization|access[_-]?token|secret)["\']?\s*[:=]\s*)("[^"\n]*"|\'[^\'\n]*\'|[^\s,}\n]+)',
                  r'\1"[REDACTED]"', text)
    return text


def query(args, stage, rows, picture, output, validate, budget):
    system = {'nodes':NODE_SYSTEM,'edges':EDGE_SYSTEM,'overview':OVERVIEW_SYSTEM}[stage]
    spec = [{'id':r['id']} for r in rows] if stage in ('nodes','overview') else [
        {'source':r['source'],'target':r['target']} for r in rows]
    parameters = {'model':args.model,'temperature':0,'max_tokens':6000,
                  'response_format':{'type':'json_object'},'enable_thinking':False,'stream':False}
    fingerprint = hashlib.sha256((json.dumps(parameters,sort_keys=True)+args.base_url+system+
                                  json.dumps(spec,sort_keys=True)+digest(picture)).encode()).hexdigest()
    result_file = output/'validated.json'
    if result_file.exists():
        cached = read(result_file)
        if cached['input_sha256'] != fingerprint:
            raise ValueError('Cached VLM inputs differ; choose a fresh output directory')
        validate(cached['value'],rows)
        return cached['value']
    key = os.environ.get('DASHSCOPE_API_KEY','').strip()
    if not key:
        raise ValueError('DASHSCOPE_API_KEY required; no semantic fallback')
    feedback = None
    for attempt in range(1,args.attempts+1):
        if budget['requests'] >= args.max_requests:
            raise ValueError('Request budget exhausted; not silently dropping remaining instances')
        message = json.dumps({'requested':spec,'previous_validation_error':feedback})
        payload = dict(parameters, messages=[
                       {'role':'system','content':system}, {'role':'user','content':[
                           {'type':'text','text':message}, {'type':'image_url','image_url':{
                               'url':'data:image/png;base64,'+base64.b64encode(picture.read_bytes()).decode('ascii')}}]}])
        request = urllib.request.Request(args.base_url.rstrip('/')+'/chat/completions',
            data=json.dumps(payload).encode(),headers={'Content-Type':'application/json','Authorization':'Bearer '+key})
        audit = {'stage':stage,'attempt':attempt,'input_sha256':fingerprint,'model':args.model,
                 'requested':spec,'image_path':str(picture),'image_sha256':digest(picture),
                 'system_prompt':system,'feedback':feedback,'status':'requesting',
                 'request_parameters':parameters,'request_bytes':len(request.data)}
        path = output/f'attempt_{attempt}.json'
        # Preserve historical rejected responses when resuming this same batch.
        suffix = 0
        while path.exists():
            suffix += 1
            path = output/f'attempt_{attempt}_resume_{suffix}.json'
        save(path,audit)
        budget['requests'] += 1
        started = time.monotonic()
        logging.info('VLM %s batch=%s attempt=%d/%d',stage,output.name,attempt,args.attempts)
        try:
            with urllib.request.urlopen(request,timeout=args.timeout) as response:
                text = response.read().decode('utf-8')
            audit['raw_response'] = text
            save(path,audit)  # Save raw BEFORE parsing any content.
            raw = json.loads(text)
            content = raw['choices'][0]['message']['content']
            audit['raw_content'] = content
            audit['usage'] = raw.get('usage')
            value = json.loads(content)
            validate(value,rows)
            audit.update(status='validated',duration_seconds=time.monotonic()-started)
            save(path,audit)
            save(result_file,{'input_sha256':fingerprint,'value':value,'audit':str(path)})
            return value
        except urllib.error.HTTPError as exc:
            audit.update(status='http_error',http_status=exc.code,usage_unknown=True)
            # Preserve a bounded, redacted server body, never headers/API keys.
            try:
                diagnostic = exc.read(65537)
                audit['http_error_body_truncated'] = len(diagnostic)>65536
                body = redact_diagnostic(diagnostic[:65536].decode('utf-8',errors='replace'), key)
                audit['http_error_body_redacted'] = body
                try:
                    parsed = json.loads(body)
                    detail = parsed.get('error',parsed) if isinstance(parsed,dict) else {}
                    if isinstance(detail,dict):
                        for field in ('code','message','type','param'):
                            if field in detail:
                                audit['provider_'+field] = detail[field]
                except (ValueError,TypeError):
                    pass
            except Exception as read_error:
                audit['diagnostic_read_error_type'] = type(read_error).__name__
            finally:
                exc.close()
            reason = 'HTTP '+str(exc.code)
            detail = str(audit.get('provider_message') or audit.get('http_error_body_redacted') or '').strip()
            if detail:
                reason += ': ' + detail[:1500]
            logging.error('%s; audit=%s',reason,path)
            if exc.code not in (408,429,500,502,503,504):
                audit['duration_seconds']=time.monotonic()-started
                save(path,audit)
                raise RuntimeError(reason) from None
        except (TimeoutError,ConnectionError,urllib.error.URLError) as exc:
            audit.update(status='transport_error',error_type=type(exc).__name__,usage_unknown=True)
            # Transport retries keep the SAME input; only schema rejection
            # introduces corrective feedback on the next validation attempt.
            reason = type(exc).__name__
        except (ValueError,KeyError,IndexError,TypeError) as exc:
            feedback = str(exc)
            audit.update(status='validation_rejected',error=feedback)
        audit['duration_seconds']=time.monotonic()-started
        save(path,audit)
        if attempt==args.attempts:
            raise ValueError(f'{stage} failed after {attempt} attempts: {feedback or audit.get("error_type") or audit.get("http_status")}')
    raise AssertionError('Unreachable')


def export(root, manifest, annotations, decisions, summary):
    nodes, rejected = [], []
    for geometry in manifest['inventory']:
        semantic = annotations.get(geometry['id'])
        if not geometry['geometry_valid'] or semantic is None:
            rejected.append({'id':geometry['id'],'reason':'invalid_geometry' if not geometry['geometry_valid'] else 'not_annotated'})
            continue
        recognized = semantic['status']=='recognized'
        nodes.append({'id':geometry['id'],'object_tag':semantic['object_tag'] if recognized else 'unresolved visual instance',
                      'caption':semantic['caption'],'bbox_center':geometry['bbox_center'],
                      'bbox_extent':geometry['bbox_extent'],'active':recognized,
                      'semantic_status':semantic['status'],'evidence':semantic['evidence'],
                      'uncertainty_reason':semantic.get('uncertainty_reason'),
                      'position_evidence':{'source':'blend_world_aabb_oracle','coordinate_frame':manifest['coordinate_frame']},
                      'visual_evidence':{'source':'vlm_asset_renders','views':[v['path'] for v in geometry['render_views']],
                                         'independently_verified':False},
                      'source_mesh_names':geometry['mesh_names'],'instance_identity_verified':False,
                      'training_eligible':False})
    active = {n['id'] for n in nodes if n['active']}
    candidates = {(p['source'],p['target']):p for p in manifest['relation_candidates']}
    edges, rejected_edges = [], []
    for decision in decisions:
        source,target,relation = decision['source'],decision['target'],decision['relation']
        proposal = candidates[(source,target)]
        reason = None
        if relation=='unknown':
            reason='vlm_unknown'
        elif source not in active or target not in active:
            reason='unresolved_endpoint'
        elif not decision['both_visible_in_both_views']:
            reason='insufficient_visual_evidence'
        elif relation not in proposal['possible_relations']:
            reason='geometry_plausibility_rejected_not_a_proof_of_false_relation'
        if reason:
            rejected_edges.append(dict(decision,rejection=reason))
            continue
        edges.append({'source':source,'target':target,'relation':'a '+relation+' b',
                      'evidence_source':'vlm_pair_visual_hypothesis','evidence':decision['evidence'],
                      'views':[v['path'] for v in proposal['render_views']],
                      'verified':False,'physics_verified':False,'training_eligible':False})
    normalize_scene_input(nodes,'Validate the full-scene VLM test fixture.')
    validate_edges(edges,nodes)
    all_nodes = len(annotations)==manifest['instance_count']
    all_pairs = len(decisions)==len(manifest['relation_candidates'])
    overview_complete=summary.get('full_scene_overviews_processed',not manifest.get('overview_views'))
    complete = all_nodes and all_pairs and not rejected and overview_complete
    directory = root/('scene' if complete else 'provisional_scene')
    save(directory/'scene_graph.json',nodes)
    save(directory/'scene_graph_edges.json',edges)
    summary.update(status='full_scene_fixture_generated' if complete else 'partial_not_ready',
                   annotated_instances=len(annotations),exported_nodes=len(nodes),
                   recognized_active_nodes=len(active),uncertain_nodes=len(nodes)-len(active),
                   evaluated_relation_candidates=len(decisions),accepted_hypothesis_edges=len(edges),
                   unexportable_instances=rejected,full_inventory_processed=all_nodes,
                   full_relation_candidate_set_processed=all_pairs,downstream_scene_dir=str(directory),
                   scene_graph_sha256=digest(directory/'scene_graph.json'),
                   scene_graph_edges_sha256=digest(directory/'scene_graph_edges.json'))
    save(root/'relation_decisions.json',{'decisions':decisions,'rejected_or_unknown':rejected_edges})
    save(root/'graph_generation_summary.json',summary)
    return complete


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sample-dir',type=Path,required=True)
    p.add_argument('--model',default='qwen3.8-flash')
    p.add_argument('--base-url',default='https://dashscope.aliyuncs.com/compatible-mode/v1')
    p.add_argument('--node-batch-size',type=int,default=8)
    p.add_argument('--edge-batch-size',type=int,default=4)
    p.add_argument('--attempts',type=int,choices=(1,2,3),default=2)
    p.add_argument('--timeout',type=float,default=180)
    p.add_argument('--max-requests',type=int,default=500)
    p.add_argument('--preflight-only',action='store_true')
    p.add_argument('--api-smoke-only',action='store_true',
                   help='Validate only the first overview batch; does NOT generate a complete scene graph')
    args=p.parse_args()
    if not (1<=args.node_batch_size<=8 and 1<=args.edge_batch_size<=4 and args.max_requests>0
            and math.isfinite(args.timeout) and args.timeout>0):
        p.error('Invalid batch/request limits')
    logging.basicConfig(level=logging.INFO,format='%(asctime)s | %(levelname)s | %(message)s')
    root=args.sample_dir.resolve()
    manifest=read(root/'evidence_manifest.json')
    node_batches=chunks(manifest['inventory'],args.node_batch_size)
    edge_batches=chunks(manifest['relation_candidates'],args.edge_batch_size)
    overview_rows=[{'id':i+1,'render_views':[view]} for i,view in enumerate(manifest.get('overview_views',[]))]
    overview_batches=chunks(overview_rows,4)
    planned_overviews=2+sum(r['bbox_extent'][2]<.20 and r['bbox_extent'][0]*r['bbox_extent'][1]>2
                          and r['bbox_center'][2]<.30 for r in manifest['inventory'])
    requests=len(node_batches)+len(edge_batches)+math.ceil(planned_overviews/4)
    summary={'status':'preflight_only','mode':'full_scene_vlm_geometry_oracle_test_fixture',
             'generator_script_sha256':digest(Path(__file__)),
             'model':args.model,'source_sha256':manifest['source_sha256'],
             'evidence_manifest_sha256':digest(root/'evidence_manifest.json'),
             'instance_count':manifest['instance_count'],'source_mesh_count':manifest['mesh_count'],
             'relation_candidate_count':len(manifest['relation_candidates']),
             'planned_full_scene_overviews':planned_overviews,
             'first_attempt_request_estimate':requests,
             'bounded_request_estimate':args.attempts*requests,
             'missing_edge_means':'unknown','simulation_executed':False,'training_eligible':False,
             'semantic_ground_truth_verified':False,'perception_coverage_verified':False,
             'scope':'ALL inventoried source instances; ALL generated on/in candidates. No old graph patches.',
             'limitations':manifest['limitations']}
    save(root/'generation_preflight.json',summary)
    if args.preflight_only:
        print(json.dumps(summary,ensure_ascii=False,indent=2))
        return 0
    if manifest['status']!='evidence_complete':
        raise ValueError('Full scene evidence is incomplete; render ALL instances/pairs first')
    if len(overview_rows)!=planned_overviews:
        raise ValueError('Missing global/floor-candidate whole-scene views')
    evidence_rows=manifest['inventory']+manifest['relation_candidates']
    for view in manifest.get('overview_views',[]):
        if not (root/view['path']).is_file() or digest(root/view['path'])!=view['sha256']:
            raise ValueError('Whole-scene evidence missing/changed')
    for row in evidence_rows:
        if len(row['render_views'])!=2 or any(not (root/v['path']).is_file() for v in row['render_views']):
            raise ValueError('Missing two-view evidence; no partial coverage masquerading as complete')
        if any(digest(root/v['path'])!=v['sha256'] for v in row['render_views']):
            raise ValueError('Evidence image changed after rendering')
    # Verify original source when available on this OS. Windows path is not
    # usable in WSL; the immutable manifest/render hashes are still recorded.
    source=Path(manifest['source_blend'])
    if os.name!='nt' and re.match(r'^[A-Za-z]:[\\/]',manifest['source_blend']):
        source=Path('/mnt')/manifest['source_blend'][0].lower()/manifest['source_blend'][3:].replace('\\','/')
    summary['source_available_on_current_os']=source.is_file()
    if source.exists() and digest(source)!=manifest['source_sha256']:
        raise ValueError('Source blend changed since rendering')
    annotations,decisions,budget={},[],{'requests':0}
    if args.api_smoke_only:
        folder=root/'vlm_audit/overview/batch_0000'
        picture=folder/'contact_sheet.png'
        smoke={'status':'api_smoke_failed','model':args.model,'full_scene_generated':False,
               'simulation_executed':False,'training_eligible':False}
        try:
            sheet(root,overview_batches[0],'overview',picture)
            query(args,'overview',overview_batches[0],picture,folder,validate_overview,budget)
            smoke.update(status='api_smoke_passed',validated_batch=str(folder/'validated.json'))
        except Exception as exc:
            smoke['error']=type(exc).__name__+': '+str(exc)
            logging.error('API smoke stopped; no substitute graph: %s',smoke['error'])
        smoke['requests_this_invocation']=budget['requests']
        save(root/'api_smoke_summary.json',smoke)
        print('status='+smoke['status']+' summary='+str(root/'api_smoke_summary.json'))
        return 0 if smoke['status']=='api_smoke_passed' else 2
    error=None
    try:
        views=[]
        for index,rows in enumerate(overview_batches):
            folder=root/'vlm_audit/overview'/f'batch_{index:04d}'
            picture=folder/'contact_sheet.png'
            sheet(root,rows,'overview',picture)
            value=query(args,'overview',rows,picture,folder,validate_overview,budget)
            views.extend(value['views'])
            save(root/'scene_overview_annotations.json',{'views':views,'source_views':manifest['overview_views'],
                                                        'room_labels_verified':False})
        summary['full_scene_overviews_processed']=len(views)==planned_overviews
        for stage,batches in (('nodes',node_batches),('edges',edge_batches)):
            for index,rows in enumerate(batches):
                folder=root/'vlm_audit'/stage/f'batch_{index:04d}'
                picture=folder/'contact_sheet.png'
                sheet(root,rows,stage,picture)
                value=query(args,stage,rows,picture,folder,
                            validate_nodes if stage=='nodes' else validate_decisions,budget)
                if stage=='nodes':
                    annotations.update({n['id']:n for n in value['nodes']})
                else:
                    decisions.extend(value['decisions'])
                export(root,manifest,annotations,decisions,summary)
    except Exception as exc:
        error=type(exc).__name__+': '+str(exc)
        logging.error('Stopped; no substitute scene graph: %s',error)
    complete=export(root,manifest,annotations,decisions,summary)
    attempts=[read(path) for path in (root/'vlm_audit').rglob('attempt_*.json')]
    usage=[a.get('usage') for a in attempts if isinstance(a.get('usage'),dict)]
    inp=sum(u.get('prompt_tokens',0) for u in usage)
    out=sum(u.get('completion_tokens',0) for u in usage)
    summary.update(requests_this_invocation=budget['requests'],recorded_attempts=len(attempts),
                   error=error,known_prompt_tokens=inp,known_completion_tokens=out,
                   transport_or_http_attempts_unknown_usage=sum(a.get('usage_unknown',False) for a in attempts),
                   responses_without_usage=sum('raw_response' in a and not isinstance(a.get('usage'),dict) for a in attempts),
                   known_fee_cny_reference_rates_no_discount=(inp*.8+out*2.7)/1e6,
                   fee_note='User reference rates only: input 0.8/output 2.7 CNY per million. Unknown usage excluded, not zero.')
    save(root/'graph_generation_summary.json',summary)
    print('status='+summary['status']+' summary='+str(root/'graph_generation_summary.json'))
    return 0 if complete and error is None else 2


if __name__=='__main__':
    raise SystemExit(main())
