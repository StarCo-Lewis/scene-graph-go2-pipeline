"""Configurable full-graph, real-LLM multi-object SYMBOLIC planning pilot.

No Habitat/robot execution, invented nodes, renamed IDs or oracle action plans.
--preflight-only prepares/checks inputs without paid calls or model downloads.
"""
import argparse
import copy
import hashlib
import json
import logging
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'habitat_demo_v1'))
from run_navigation_demo import read_json, write_json, cg_to_habitat
from run_continuous_sequence import OUT
from sequential_proxy_core import PARAMETERS
from json_symbolic_review import review_symbolic
from bounded_skill_planner import plan_bounded
from flat_matrix_planner import plan_flat
from adapter_parameter_contract import project_library
from runtime_goal_protocol import CONTRACT, PROMPT
from skill_retrieval import LocalEncoder, model_fingerprint, PROFILES
from token_usage_audit import usage
from build_report import build as build_audit
from scene_input_interface import normalize_scene_input
from llm_graph_binding import validate_edges
from planning_start_contract import public_round, conversation_context

STATE_SOURCE = 'symbolic_hypothesis_not_observation'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inside(node, floor):
    return all(abs(node['bbox_center'][axis]-floor['bbox_center'][axis]) <=
               floor['bbox_extent'][axis]/2+.02 for axis in (0,1))


def load_case(path, *, require_task_assertions=True):
    cfg = read_json(path)
    schemas = {'symbolic_scene_case_v1'} if require_task_assertions else {'symbolic_scene_case_v1', 'symbolic_scene_template_v1'}
    if cfg.get('schema_version') not in schemas or cfg.get('coordinate_frame') != 'source_blender_world_z_up_m':
        raise ValueError('Explicit symbolic case schema and Z-up metre frame required')
    scene = (path.parent/cfg['source_scene_dir']).resolve()
    files = [path,scene/'scene_graph.json',scene/'scene_graph_edges.json',scene.parent/'graph_generation_summary.json']
    summary = read_json(files[-1])
    if summary.get('status') != 'full_scene_fixture_generated':
        raise ValueError('Full-scene generation must be complete before this pilot')
    if sha(files[1]) != summary['scene_graph_sha256'] or sha(files[2]) != summary['scene_graph_edges_sha256']:
        raise ValueError('Source graph hashes do not match generation summary')
    nodes,edges = read_json(files[1]),read_json(files[2])
    normalize_scene_input(nodes,'Validate the complete new-scene planning fixture.')
    validate_edges(edges,nodes)
    by_id = {n['id']:n for n in nodes}
    if len(by_id) != len(nodes) or len(nodes) != summary['exported_nodes']:
        raise ValueError('Missing or duplicate source node IDs')
    for node in nodes:
        if (not all(isinstance(node.get(k),list) and len(node[k])==3 and
                    all(type(v) in (int,float) and math.isfinite(v) for v in node[k])
                    for k in ('bbox_center','bbox_extent')) or any(v<=0 for v in node['bbox_extent'])):
            raise ValueError('Invalid finite positive source geometry')
    room_names = [r['name'] for r in cfg['rooms']]
    if len(set(room_names)) != len(room_names) or len(room_names)<2:
        raise ValueError('At least two uniquely named author-assigned room regions required')
    for room in cfg['rooms']:
        if type(room['floor_id']) is not int or room['floor_id'] not in by_id:
            raise ValueError('Room region references a missing floor node')
    for identity,name in cfg['control_node_rooms'].items():
        node = by_id[int(identity)]
        room = next((r for r in cfg['rooms'] if r['name']==name),None)
        if room is None or not inside(node,by_id[room['floor_id']]) or not node.get('active',True):
            raise ValueError('Control node is inactive or outside its configured room region')
    for identity in list(cfg['named_ids'].values())+[cfg['robot_start_reference_id']]+cfg['graspable_ids']:
        if type(identity) is not int or identity not in by_id or not by_id[identity].get('active',True):
            raise ValueError('Case references a missing/inactive original ID')
    for pair in cfg['placeable_pairs']:
        if (not isinstance(pair,list) or len(pair)!=2 or pair[0] not in cfg['graspable_ids']
                or pair[1] not in by_id or not by_id[pair[1]].get('active',True) or pair[0]==pair[1]):
            raise ValueError('Invalid explicit hypothetical placement capability')
    for identity,support in cfg['initial_support_by_object'].items():
        if not any(e['source']==int(identity) and e['target']==support and e['relation']=='a on b' for e in edges):
            raise ValueError('Initial support assumption must reference an existing visual hypothesis edge')
    if cfg.get('reach_m') != 1.5:
        raise ValueError('This pilot preserves the existing hierarchy reach contract of 1.5 m')
    if not require_task_assertions:
        return cfg,nodes,edges,{str(f):sha(f) for f in files}
    if not cfg['rounds'] or [r['round'] for r in cfg['rounds']] != list(range(1,len(cfg['rounds'])+1)):
        raise ValueError('Nonempty sequential rounds required')
    for row in cfg['rounds']:
        if not isinstance(row['instruction'],str) or not row['instruction'].strip():
            raise ValueError('English original instruction required')
        visits = row['expected_navigation_targets']
        if not visits or any(type(i) is not int or i not in by_id for i in visits):
            raise ValueError('Explicit navigation assertions must reference original scene IDs')
        if row['expected_robot_room'] not in room_names:
            raise ValueError('Unknown asserted final robot room')
        if type(row['expected_placement_count']) is not int or row['expected_placement_count']<0:
            raise ValueError('Nonnegative placement-count assertion required')
        if 'expected_placement_events' in row:
            events = row['expected_placement_events']
            if (not isinstance(events,list) or len(events)!=row['expected_placement_count'] or
                    any(not isinstance(e,list) or len(e)!=2 or e not in cfg['placeable_pairs'] for e in events)):
                raise ValueError('Ordered placement assertions must use configured pairs')
        if 'expected_held_during_visits' in row:
            held = row['expected_held_during_visits']
            if (not isinstance(held,list) or len(held)!=len(visits) or
                    any(i is not None and (type(i) is not int or i not in cfg['graspable_ids']) for i in held)):
                raise ValueError('Held-during-visit assertions must match navigation count and graspable IDs')
        for identity,support in row['expected_support_by_object'].items():
            if int(identity) not in cfg['graspable_ids'] or (support is not None and [int(identity),support] not in cfg['placeable_pairs']
                                                         and cfg['initial_support_by_object'].get(identity)!=support):
                raise ValueError('Invalid expected support assertion')
    return cfg,nodes,edges,{str(f):sha(f) for f in files}


def check_event_assertions(descriptor,plan,review):
    """Independent authored event checks, not instructions supplied to the LLM."""
    steps = [s for task in plan['subtask_plans'] for s in task['steps']]
    trace = review['symbolic_trace']
    if [s['step_id'] for s in steps] != [t['step_id'] for t in trace]:
        raise ValueError('Plan/trace step identity mismatch')
    placements = [[s['arguments']['object']['element_id'],s['arguments']['surface']['element_id']]
                  for s in steps if s['skill_id']=='place_object_on_surface']
    held_visits = [t['executor_state_after']['held_object_id'] for t in trace
                   if t['skill_id']=='navigate_to_object']
    if 'expected_placement_events' in descriptor and placements!=descriptor['expected_placement_events']:
        raise ValueError('Ordered placement object/destination events mismatch')
    if 'expected_held_during_visits' in descriptor and held_visits!=descriptor['expected_held_during_visits']:
        raise ValueError('Carried object at requested visit mismatch')
    review['case_event_assertions'] = {'ordered_placement_events':placements,
        'held_during_visits':held_visits,
        'ordered_placement_events_checked':'expected_placement_events' in descriptor,
        'held_during_visits_checked':'expected_held_during_visits' in descriptor,
        'scope':'Independent authored symbolic assertions, not physical observations'}


def room_of(node, cfg, by_id):
    explicit = cfg['control_node_rooms'].get(str(node['id']))
    matches = [r['name'] for r in cfg['rooms'] if inside(node,by_id[r['floor_id']])]
    return explicit or (matches[0] if len(matches)==1 else 'unassigned_or_overlapping')


def initial_state(nodes,cfg):
    by_id = {n['id']:n for n in nodes}
    objects = {str(n['id']):{'center':cg_to_habitat(n['bbox_center'],'x_z_minus_y'),
               'support_id':cfg['initial_support_by_object'].get(str(n['id'])),
               'room_hypothesis':room_of(n,cfg,by_id)} for n in nodes if n.get('active',True)}
    start = objects[str(cfg['robot_start_reference_id'])]
    return {'held_object_id':None,'lifted':False,'prepared_target_id':None,'state_source':STATE_SOURCE,
            'backend':{'objects':objects,'robot_position':[start['center'][0],.1,start['center'][2]],
                       'robot_room_hypothesis':start['room_hypothesis'],'memory_version':0,
                       'symbolically_updated_object_ids':[]}}


def project_scene(nodes,edges,state,cfg):
    by_id = {n['id']:n for n in nodes}
    moved = set(state['backend']['symbolically_updated_object_ids'])
    projected = []
    for node in nodes:
        # All original nodes, not a five-node substitute scene. Strip bulky render
        # metadata from LLM input; preserve the full originals in source_snapshot.
        item = {k:copy.deepcopy(node[k]) for k in ('id','object_tag','caption','bbox_center','bbox_extent','active') if k in node}
        obj = state['backend']['objects'].get(str(node['id']))
        if obj:
            x,y,z = obj['center']
            item['bbox_center'] = [x,-z,y]
        item['json_test_context'] = {'room_hypothesis':obj['room_hypothesis'] if obj else room_of(node,cfg,by_id),
                                    'room_assignment_source':cfg['room_assignment_source'],
                                    'placement_affordance_verified':False}
        item['position_evidence'] = {'source':STATE_SOURCE if node['id'] in moved else 'blend_world_aabb_oracle',
                                    'projection_revision':state['backend']['memory_version'],
                                    'robot_observation_verified':False}
        projected.append(item)
    projected_edges = [copy.deepcopy(e) for e in edges if e['source'] not in moved]
    for identity in moved:
        support = state['backend']['objects'][str(identity)]['support_id']
        if support is not None:
            projected_edges.append({'source':identity,'target':support,'relation':'a on b',
                                    'evidence_source':STATE_SOURCE,'verified':False,'training_eligible':False})
    validate_edges(projected_edges,projected)
    return projected,projected_edges


def prepare_round(folder,descriptor,state,nodes,model,base_url,cfg,edges):
    projected,relations = project_scene(nodes,edges,state,cfg)
    write_json(folder/'input_memory/scene/scene_graph.json',projected)
    write_json(folder/'input_memory/scene/scene_graph_edges.json',relations)
    write_json(folder/'input_memory/symbolic_state.json',state)
    write_json(folder/'runtime_state.json',state)
    write_json(folder/'round_input.json',public_round(descriptor))
    private_assertions = {k:copy.deepcopy(v) for k,v in descriptor.items() if k.startswith('expected_')}
    if private_assertions:
        write_json(folder.parent/'evaluation/task_assertions.json',private_assertions)
    policy = {'adapter_mode':'json_symbolic_test_only','supported_skill_ids':list(PARAMETERS),
              'action_identity_scope':{'case_id':cfg['case_id'],'round':descriptor['round'],
                  'run_id':sha_identity_root(folder)},
              'mapped_instance_ids':[n['id'] for n in nodes if n.get('active',True)],
              'instance_mapping_scope':'original symbolic IDs only; no simulator mapping',
              'fixed_order_required':False,'max_held_objects':1,'proxy_reach_m':1.5,
              'runtime_coordinate_frame':'habitat_world_y_up','goal_protocol':copy.deepcopy(CONTRACT),
              'current_runtime_state':state,
              'planning_capabilities':{'validate_supplied_on_in_edges':True},
              'symbolic_placeable_pairs':cfg['placeable_pairs'],
              'symbolic_scene_capabilities':{'graspable_ids':cfg['graspable_ids'],
                                            'placeable_pairs':cfg['placeable_pairs']},
              'author_assigned_rooms':cfg['rooms'],'control_node_rooms':cfg['control_node_rooms'],
              'simulation_executed':False,'geometry_verified':False,'training_eligible':False,
              'planning_context':(
                  'FULL new-scene graph symbolic planning pilot. All original IDs are retained. '
                  'Use the explicit room mapping, current positions and held/lifted/prepared state. '
                  'Room names, robot pose, graspability, free placement space and action effects are '
                  'TEST ASSUMPTIONS, not observations or verified affordances. Only the five supplied skills '
                  'and target/object/surface parameters are executable. Grasp only IDs in '
                  'symbolic_scene_capabilities.graspable_ids, place only its placeable_pairs. '
                  'Prepare the object before grasp, lift before carry, and prepare the surface before placement. '
                  'Navigation clears preparation. Never navigate to a held object. Already holding/lifted '
                  'does not need another grasp/lift. Keep-holding and leave-other-objects-untouched are '
                  'constraints, not extra empty actions. Explicit visits and their order must survive, even '
                  'when near. Do not add an implicit approach when already near within 1.5 m horizontally. '
                  'Graph coordinates are Z-up; runtime positions are [x,z,-y]. Do not generate room, '
                  'pose, height or geometry parameters. Supplied initial edges are unverified VLM hypotheses. '
                  'No inspect, room-entry, search, container-opening or put-in-container skill is supported. '
                  'Pickup/holding use held and lifted state goals; placement uses on plus held=false. '+PROMPT)}
    context = conversation_context(cfg,nodes)
    if context is not None:
        policy['conversation_context'] = context
        policy['planning_context'] += (
            ' The conversation_context describes the user, not the robot. It is supplied scenario '
            'context, not a verified person detection or a requested destination. Interpret the original '
            'utterance using that context without inventing user coordinates, contents or capabilities. '
            'A nearby desk is a location reference, not a person node or an automatic placement command.')
    inference_mode = cfg.get('inference_mode','baseline')
    if inference_mode not in ('baseline','bounded_inference'):
        raise ValueError('Unknown inference mode')
    if cfg.get('inference_experiment',False):
        policy['factual_boundary'] = (
            'In both experiment conditions, a label such as water bottle is not evidence that water '
            'is present. Unknown contents, drinkability and user preferences must not be reported as '
            'observed facts. Respect an explicit empty/unknown premise as a supplied scenario premise, '
            'not an observation. Moving a container is not evidence that thirst has been satisfied. '
            'These factual restrictions do not prohibit forming a task interpretation or proposing '
            'an object as a candidate. Never invent scene objects, capabilities or execution evidence.')
    if inference_mode == 'bounded_inference':
        policy['inference_policy'] = {
            'mode':'bounded_inference',
            'rules':(
                'You may infer omitted ordinary action prerequisites, use the supplied user location '
                'to interpret here/me/my desk, and form a plausible task interpretation from the user need. '
                'For an underspecified symbolic fetch request, you may provisionally choose one existing '
                'category-compatible candidate supported by the declared capability inventory, even when '
                'the instruction does not identify a unique instance. Such a choice is a provisional default, '
                'not proof of the user\'s uniquely intended object. Explain the selection basis and uncertainty '
                'concisely in existing binding reason fields. Preserve explicit IDs, attributes, constraints '
                'and supplied facts and the shared factual_boundary. Missing relations are unknown; '
                'do not assert an unsupported initial relation to justify a candidate. Do not claim thirst '
                'has been satisfied merely because a container is brought nearby. No clarification, search, '
                'physical action or new recovery skill is added by this policy.')}
    library = read_json(OUT/'quadruped_atomic_skill_library.v0.1.json')
    library['skills'] = [s for s in library['skills'] if s['skill_id'] in PARAMETERS]
    library['execution_contract'] = policy
    library = project_library(library)
    write_json(folder/'execution_skill_library.json',library)
    write_json(folder/'execution_contract.json',library['execution_contract'])


def sha_identity_root(folder):
    # Hierarchical children and the final whole plan share a run/round scope,
    # while their different action content gets a distinct revision hash.
    root=next((p.parent for p in [folder]+list(folder.parents)
               if p.name.startswith('round_') and p.name[6:].isdigit()),folder)
    return hashlib.sha256(str(root.resolve()).encode()).hexdigest()[:16]


def run(args):
    strategy = getattr(args,'planning_strategy','bounded')
    if strategy not in ('bounded','flat'):
        raise ValueError('Unknown planning strategy')
    cfg,nodes,edges,hashes = load_case(args.case_config.resolve())
    for dependency in (OUT/'quadruped_atomic_skill_library.v0.1.json',PROFILES):
        hashes[str(dependency.resolve())] = sha(dependency)
    if not args.preflight_only:
        if not os.environ.get('DASHSCOPE_API_KEY','').strip():
            raise ValueError('DASHSCOPE_API_KEY required')
        if not args.embedding_python.is_file():
            raise ValueError('Existing local embedding Python required; no auto-install/download')
        embedding_hash = model_fingerprint(args.embedding_model_dir)
    else:
        embedding_hash = None
    root = args.output_dir.resolve()
    root.mkdir(parents=True,exist_ok=False)
    write_json(root/'source_snapshot/case.json',cfg)
    write_json(root/'source_snapshot/scene_graph.json',nodes)
    write_json(root/'source_snapshot/scene_graph_edges.json',edges)
    write_json(root/'source_snapshot/atomic_skill_library.json',read_json(OUT/'quadruped_atomic_skill_library.v0.1.json'))
    write_json(root/'source_snapshot/skill_semantic_profiles.json',read_json(PROFILES))
    engine_files = list(HERE.glob('*.py'))+list((HERE.parent/'habitat_demo_v1').glob('*.py'))+list((HERE.parent/'skill_matching_v1').glob('*.py'))+list((HERE.parent/'object_alignment_v1').glob('*.py'))+list((HERE.parent/'llm_audit_v1').glob('*.py'))
    code_hashes = {str(p.resolve()):sha(p) for p in engine_files}
    state = initial_state(nodes,cfg)
    summary = {'status':'input_preflight_passed' if args.preflight_only else 'running',
               'case_id':cfg['case_id'],'method':('cosine_validation_flat_matrix' if strategy=='flat'
                                                 else 'cosine_validation_bounded_hierarchy_matrix'),
               'planning_strategy':strategy,
               'all_source_node_count':len(nodes),'all_source_edge_count':len(edges),
               'expected_round_count':len(cfg['rounds']),'rounds':[],'fresh_llm_calls':False,
               'source_files':hashes,'engine_hashes':code_hashes,'engine_snapshot_copied':False,
               'embedding_model_sha256':embedding_hash,'max_split_levels':0 if strategy=='flat' else 1,
               'max_parts':1 if strategy=='flat' else 4,
               'assumptions':cfg['assumptions'],'simulation_executed':False,
               'training_eligible':False,'semantic_ground_truth_verified':False,
               'scope':'Full graph planning; author-assigned rooms and hypothetical skill effects, not physical cross-room execution'}
    destination = root/'regression_summary.json'
    write_json(root/'initial_symbolic_state.json',state)
    write_json(root/'instructions.json',{'rounds':cfg['rounds']})
    def prepare(folder,descriptor,snapshot,source_nodes,model,base_url):
        prepare_round(folder,descriptor,snapshot,source_nodes,model,base_url,cfg,edges)
    if args.preflight_only:
        prepare(root/'round_1/planning',cfg['rounds'][0],state,nodes,args.model,args.base_url)
        write_json(destination,summary)
        print('status=input_preflight_passed summary='+str(destination),flush=True)
        return 0
    started = time.monotonic()
    encoder = None
    log_handler = logging.FileHandler(root/'run.console.log',encoding='utf-8')
    log_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))
    logging.getLogger().addHandler(log_handler)
    try:
        encoder = LocalEncoder(args.embedding_python,args.embedding_model_dir,root/'embedding')
        for descriptor in cfg['rounds']:
            folder = root/f'round_{descriptor["round"]}/planning'
            prepare(folder,descriptor,state,nodes,args.model,args.base_url)
            row = {'round':descriptor['round'],'instruction':descriptor['instruction'],
                   'status':'planning','begin_state':copy.deepcopy(state)}
            summary['rounds'].append(row)
            summary['fresh_llm_calls'] = True
            write_json(destination,summary)
            logging.info('START new-scene round=%s memory_version=%s held=%s',descriptor['round'],
                         state['backend']['memory_version'],state['held_object_id'])
            code = (plan_flat(args,folder,encoder) if strategy=='flat'
                    else plan_bounded(args,folder,encoder,4,prepare=prepare))
            if code:
                failure = read_json(folder/'planning_summary.json')
                row['failed_stage'] = failure.get('failed_stage')
                raise ValueError(failure.get('error','Bounded planning failed'))
            if read_json(folder/'runtime_state.json') != state:
                raise ValueError('Planner mutated its input state')
            review = review_symbolic(dict(descriptor,alignment=read_json(folder/'object_alignment.json'),
                                           plan=read_json(folder/'skill_plan_final.json')),
                                     read_json(folder/'execution_skill_library.json'),
                                     read_json(folder/'input_memory/scene/scene_graph.json'),state,1.5)
            placements = sum(s['skill_id']=='place_object_on_surface' for s in review['symbolic_trace'])
            if placements != descriptor['expected_placement_count']:
                raise ValueError('Requested placement count mismatch')
            check_event_assertions(descriptor,read_json(folder/'skill_plan_final.json'),review)
            previous_room = state['backend'].get('robot_room_hypothesis')
            state = review['predicted_end_state']
            state['backend']['memory_version'] = descriptor['round']
            write_json(folder.parent/'symbolic_projection_after.json',state)
            write_json(folder/'symbolic_review_with_case_assertions.json',review)
            row.update(status='json_symbolic_passed',end_state=copy.deepcopy(state),
                       action_count=len(review['symbolic_trace']),placement_count=placements,
                       symbolic_room_transition_count=sum(s['skill_id']=='navigate_to_object' and
                           s['robot_room_before']!=s['robot_room_after'] for s in review['symbolic_trace']),
                       expected_room_hypothesis=descriptor['expected_robot_room'],
                       room_changed_between_round_boundaries=previous_room!=state['backend']['robot_room_hypothesis'])
            write_json(destination,summary)
            logging.info('END new-scene round=%s status=%s held=%s room=%s',descriptor['round'],
                         row['status'],state['held_object_id'],state['backend']['robot_room_hypothesis'])
        summary['status'] = 'json_symbolic_passed'
    except (Exception,KeyboardInterrupt) as exc:
        summary.update(status='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',error=str(exc))
        if summary['rounds']:
            summary['rounds'][-1].update(status='failed',error=str(exc))
        logging.exception('New-scene chain stopped; later rounds skipped, no fallback plan')
    finally:
        logging.getLogger().removeHandler(log_handler)
        log_handler.close()
        if encoder is not None:
            write_json(root/'embedding_metrics.json',encoder.stats())
            encoder.close()
        summary['duration_seconds'] = time.monotonic()-started
        summary['completed_round_count'] = sum(r['status']=='json_symbolic_passed' for r in summary['rounds'])
        summary['action_count_in_completed_rounds'] = sum(r.get('action_count',0) for r in summary['rounds'])
        summary['symbolic_room_transitions_in_completed_rounds'] = sum(r.get('symbolic_room_transition_count',0) for r in summary['rounds'])
        summary['source_files_unchanged'] = all(Path(p).is_file() and sha(p)==h for p,h in hashes.items())
        summary['engine_files_unchanged'] = all(Path(p).is_file() and sha(p)==h for p,h in code_hashes.items())
        try:
            summary['embedding_model_unchanged'] = model_fingerprint(args.embedding_model_dir)==embedding_hash
        except Exception as exc:
            summary['embedding_model_unchanged'] = False
            summary['embedding_readback_error'] = str(exc)
        if not all(summary[k] for k in ('source_files_unchanged','engine_files_unchanged','embedding_model_unchanged')):
            summary.update(status='invalid_inputs_or_engine',error='Input/model/code changed during pilot')
        try:
            summary['token_usage'] = usage(root)
            u = summary['token_usage']
            summary['known_fee_cny_no_discount'] = (u['prompt_tokens']*.8+u['completion_tokens']*2.7)/1e6
            summary['fee_note'] = 'User input0.8/output2.7 CNY per million; unknown/missing usage excluded, not zero.'
            audit = build_audit(root,root/'llm_audit_aggregate')
            summary['audit_counts'] = {k:audit[k] for k in ('logical_request_count','http_attempt_count',
                                                         'transport_failure_count','program_rejection_count')}
        except Exception as exc:
            summary['audit_collection_error'] = str(exc)
        write_json(destination,summary)
    print('status='+summary['status']+' summary='+str(destination),flush=True)
    return 0 if summary['status']=='json_symbolic_passed' else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case-config',type=Path,default=HERE/'scenarios/scene202_crossroom_6.json')
    parser.add_argument('--output-dir',type=Path,default=HERE/'runs'/datetime.now().strftime('crossroom_%Y%m%d_%H%M%S_%f'))
    parser.add_argument('--preflight-only',action='store_true')
    parser.add_argument('--planning-strategy',choices=('bounded','flat'),default='bounded')
    parser.add_argument('--embedding-python',type=Path,default=Path('/home/lewis/miniforge3/envs/skill_embed/bin/python'))
    parser.add_argument('--embedding-model-dir',type=Path,default=Path('/mnt/d/NebulaAI/models/skill_embedding_minilm_v1'))
    parser.add_argument('--model',default='qwen3.8-flash')
    parser.add_argument('--base-url',default='https://dashscope.aliyuncs.com/compatible-mode/v1')
    args = parser.parse_args()
    args.skill_output_format,args.compile_navigation = 'matrix',False
    args.top_k,args.cosine_threshold,args.cosine_margin = 2,.35,.05
    logging.basicConfig(level=logging.INFO,format='%(asctime)s | %(levelname)s | %(message)s')
    return run(args)


if __name__=='__main__':
    raise SystemExit(main())
