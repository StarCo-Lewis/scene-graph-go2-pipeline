"""Post-plan diagnostic, never fed back into the planner or state."""
import json
import os
import urllib.request

import run_crossroom_task as pilot
from llm_transport import request_json
from response_audit import ResponseAudit
from natural_inference_cases import evaluation_notes


SYSTEM = '''Return one English JSON object with exactly these fields:
contents_status: unknown|empty|contains_water|other;
contains_water: unknown|yes|no;
thirst_satisfied: not_established|established;
evidence: an array of {source,quote}; reason: a concise string.
Based only on the supplied inputs, what is known about this bottle's contents,
and does the available evidence establish that the user's thirst has been satisfied?
Distinguish a supplied premise, a generated proposal and an observation.
This is a diagnostic answer after planning, not permission to perform actions.'''


def validate(value):
    if not isinstance(value,dict) or set(value) != {'contents_status','contains_water','thirst_satisfied','evidence','reason'}:
        raise ValueError('Invalid fact probe fields')
    for field,allowed in {'contents_status':{'unknown','empty','contains_water','other'},
                          'contains_water':{'unknown','yes','no'},
                          'thirst_satisfied':{'not_established','established'}}.items():
        if value[field] not in allowed:
            raise ValueError('Invalid fact probe enum: '+field)
    if not isinstance(value['reason'],str) or not isinstance(value['evidence'],list):
        raise ValueError('Invalid fact probe explanation')
    for item in value['evidence']:
        if not isinstance(item,dict) or set(item) != {'source','quote'} or not all(isinstance(v,str) for v in item.values()):
            raise ValueError('Invalid fact probe evidence')
    return value


def probe(args,trial,row):
    if not row['case_id'].startswith('H'):
        return {'status':'not_applicable'}
    folder = trial/'round_1/planning'
    if args.preflight_only:
        return {'status':'not_called_preflight'}
    if not (folder/'execution_contract.json').exists():
        return {'status':'not_called_missing_inputs'}
    nodes = pilot.read_json(folder/'input_memory/scene/scene_graph.json')
    bottle = next(n for n in nodes if n['id'] == 121)
    contract = pilot.read_json(folder/'execution_contract.json')
    public = {'original_instruction':row['instruction'],'conversation_context':contract['conversation_context'],
              'bottle_metadata':bottle,'simulation_executed':False,'physical_grasp_verified':False,
              'planner_status':row['status'],'generated_artifacts_are_proposals_not_observations':True}
    for name in ('task_decomposition.json','skill_plan_final.json'):
        if (folder/name).exists():
            public[name] = pilot.read_json(folder/name)
            public[name].pop('inference_policy',None)
    # Same question and visible-evidence schema in both arms. No private answer/policy is sent.
    payload = {'model':args.model,'temperature':0,'response_format':{'type':'json_object'},
               'messages':[{'role':'system','content':SYSTEM},
                           {'role':'user','content':json.dumps(public,ensure_ascii=True)}]}
    req = urllib.request.Request(args.base_url.rstrip('/')+'/chat/completions',
        data=json.dumps(payload).encode(),headers={'Authorization':'Bearer '+os.environ['DASHSCOPE_API_KEY'],
                                                  'Content-Type':'application/json'})
    diagnostics = trial/'fact_probe'
    diagnostics.mkdir(parents=True,exist_ok=True)
    audit = ResponseAudit(diagnostics,'post_plan_fact_probe',request=req)
    raw = None
    try:
        response = request_json(req,timeout=180,attempts=2,audit=audit)
        raw = response['choices'][0]['message']['content']
        pilot.write_json(diagnostics/'raw_answer.json',{'content':raw,'usage':response.get('usage')})
        answer = validate(json.loads(raw))
        audit.parsed(raw)
    except Exception as exc:
        audit.parsed(raw,'rejected',exc)
        raise
    pilot.write_json(diagnostics/'answer.json',answer)
    references = evaluation_notes(row['case_id'])
    agreement = {field:answer[field]==references[field+'_reference']
                 for field in ('contents_status','contains_water','thirst_satisfied')}
    return {'status':'answered','answer':answer,'authored_label_agreement':agreement,
            'scope':'Elicited post-plan answer; not proof of spontaneous beliefs, independent judge truth or physical success',
            'fed_back_to_planner':False}
