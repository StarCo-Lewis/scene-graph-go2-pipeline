"""Synthetic feasibility references ONLY, never supplied to live model calls."""
import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import run_two_scene_stress as suite
import run_crossroom_task as pilot
from run_flat_stress import run_cases
from json_symbolic_review import review_symbolic
from match_skills import build_result
from action_sequence import validate_sequence
from sequential_proxy_core import SerialExecutor

N='navigate_to_object'; P='position_arm_for_object'; G='grasp_object'
L='lift_held_object'; R='place_object_on_surface'


def transfer(o,s): return [(N,o),(P,o),(G,o),(L,o),(N,s),(P,s),(R,o,s)]
def pickup_carry(o,s): return [(N,o),(P,o),(G,o),(L,o),(N,s)]
def release(o,s): return [(P,s),(R,o,s)]
def visits(*ids): return [(N,i) for i in ids]


def reference_sequences():
    return [
        [[(N,121),(P,121),(G,121),(L,121),(N,69),(N,2),(P,2),(R,121,2)]
         +transfer(86,101)+transfer(121,69)+transfer(86,2)+transfer(121,101)+visits(2,69,121)],
        [pickup_carry(86,101),
         release(86,101)+pickup_carry(121,69)+visits(2),
         release(121,2)+transfer(86,69)+visits(2,101,69),
         transfer(121,101)+transfer(86,2)+pickup_carry(121,69),
         release(121,69)+transfer(86,101)+transfer(121,2)+visits(69,101,86)],
        [transfer(152,165)],
        [visits(110,133,165,133,110,165)],
        [pickup_carry(124,165)+visits(133),
         release(124,133)+transfer(152,110)+visits(165,133)],
        [transfer(152,110)+transfer(124,165)+transfer(230,133)+visits(110,165,133)],
        [transfer(152,110)+transfer(124,133)+visits(230),
         transfer(124,110)+transfer(152,133)+visits(230,133)],
        [pickup_carry(230,110)+visits(133),
         release(230,133)+transfer(152,110)+pickup_carry(124,165),
         release(124,165)+[(N,152),(P,152),(G,152),(L,152),(N,133),(N,165),(P,165),(R,152,165)]+pickup_carry(230,110),
         release(230,110)+transfer(124,133)+pickup_carry(152,110),
         release(152,110)+transfer(124,165)+transfer(230,133)+visits(152),
         transfer(152,133)+transfer(124,110)+transfer(230,165)+visits(133,110,165)]
    ]


def fixture(folder,cfg,desc,state,nodes,edges,actions):
    pilot.prepare_round(folder,desc,state,nodes,'offline','https://invalid.test',cfg,edges)
    library=pilot.read_json(folder/'execution_skill_library.json')
    ids=set(cfg['named_ids'].values())|set(cfg['initial_support_by_object'].values())
    refs={i:f'r_{i}' for i in ids}
    goals=[{'constraint_id':f'on_{o}','subject_role':refs[int(o)],'relation':'on',
            'object_role':refs[s],'required':True} for o,s in desc['expected_support_by_object'].items() if s is not None]
    if desc['expected_end']['held_object_id'] is not None:
        held=desc['expected_end']['held_object_id']
        goals.extend({'constraint_id':relation,'subject_role':'robot','relation':relation,
                      'object_role':refs[held],'required':True} for relation in ('held','lifted'))
    else:
        goals.append({'constraint_id':'end_near','subject_role':'robot','relation':'near',
                      'object_role':refs[desc['expected_navigation_targets'][-1]],'required':True})
    initial=[]
    if state['held_object_id'] is not None:
        initial=[{'constraint_id':'init_'+r,'subject_role':'robot','relation':r,
                  'object_role':refs[state['held_object_id']],'required':True} for r in ('held','lifted')]
    alignment={'status':'success','context_bindings':{v:{'element_ids':[i]} for i,v in refs.items()},
        'task_decomposition':{'instruction':desc['instruction'],'initial_scene_constraints':initial,
            'subtasks':[{'subtask_id':'st_01','description':desc['instruction'],'dependencies':[]}],
            'goal_state':goals}}
    steps=[]
    for skill,obj,*surface in actions:
        key='object' if skill in (L,R) else 'target'
        args={key:{'role_id':refs[obj],'element_id':obj}}
        if surface: args['surface']={'role_id':refs[surface[0]],'element_id':surface[0]}
        steps.append({'skill_id':skill,'arguments':args})
    plan=build_result(alignment,library,{'subtask_plans':[{'subtask_id':'st_01','intent':desc['instruction'],'steps':steps}]})
    return dict(desc,alignment=alignment,plan=plan),library


class TwoSceneTests(unittest.TestCase):
    def test_eight_independent_cases_have_exact_two_plus_six_all_full_graphs(self):
        entries,_=suite.inputs()
        self.assertEqual(len(entries),8)
        self.assertEqual([cfg['scene_id'] for _,cfg in entries],['scene202']*2+['scene200']*6)
        self.assertEqual(sum(len(cfg['rounds']) for _,cfg in entries),19)
        all_text=[r['instruction'] for _,cfg in entries for r in cfg['rounds']]
        self.assertEqual(len(all_text),len(set(all_text)))
        from flat_stress_cases import catalog
        old_base=pilot.read_json(pilot.HERE/'scenarios/scene202_crossroom_6.json')
        old_text={r['instruction'] for r in old_base['rounds']}
        old_text|={r['instruction'] for _,cfg in catalog(old_base) for r in cfg['rounds']}
        self.assertFalse(set(all_text)&old_text)
        with tempfile.TemporaryDirectory() as temp:
            for name,cfg in entries:
                path=Path(temp)/(name+'.json')
                pilot.write_json(path,cfg)
                _,nodes,edges,_=pilot.load_case(path)
                self.assertEqual((len(nodes),len(edges)),(350,166) if cfg['scene_id']=='scene202' else (260,97))

    def test_all_new_tasks_feasible_and_all_intermediate_events_independently_checked(self):
        entries,_=suite.inputs()
        counts=[]; all_ids=set(); placements=0
        with tempfile.TemporaryDirectory() as temp:
            for (name,cfg),sequences in zip(entries,reference_sequences()):
                source=Path(temp)/(name+'.json')
                pilot.write_json(source,cfg)
                _,nodes,edges,_=pilot.load_case(source)
                state=pilot.initial_state(nodes,cfg)
                count=0
                self.assertIsNone(state['held_object_id'])  # independent reset for every task
                for desc,actions in zip(cfg['rounds'],sequences):
                    before=copy.deepcopy(state)
                    f,library=fixture(Path(temp)/name/f'round_{desc["round"]}'/'planning',cfg,desc,state,nodes,edges,actions)
                    validate_sequence(f['plan'])
                    projected,_=pilot.project_scene(nodes,edges,state,cfg)
                    review=review_symbolic(f,library,projected,state,1.5)
                    pilot.check_event_assertions(desc,f['plan'],review)
                    self.assertEqual(state,before)
                    self.assertEqual(len(review['case_event_assertions']['ordered_placement_events']),desc['expected_placement_count'])
                    ids={x['action_id'] for x in review['symbolic_trace']}
                    self.assertFalse(ids & all_ids)
                    self.assertEqual(len(ids),len(actions))
                    all_ids|=ids
                    state=review['predicted_end_state']
                    state['backend']['memory_version']=desc['round']
                    count+=len(actions)
                    placements+=desc['expected_placement_count']
                counts.append(count)
        self.assertEqual(counts,[39,63,7,6,17,24,31,90])
        self.assertEqual(len(all_ids),277)
        self.assertEqual(placements,35)

    def test_scene200_protected_tape_cannot_be_grasped_even_if_replaced(self):
        entries,_=suite.inputs(); name,cfg=entries[6]
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); path=root/'case.json'; pilot.write_json(path,cfg)
            _,nodes,edges,_=pilot.load_case(path)
            state=pilot.initial_state(nodes,cfg)
            actions=transfer(230,165)+reference_sequences()[6][0]
            f,library=fixture(root/'planning',cfg,cfg['rounds'][0],state,nodes,edges,actions)
            with self.assertRaisesRegex(ValueError,'PRESERVATION_MANIPULATION_FORBIDDEN'):
                review_symbolic(f,library,nodes,state,1.5)

    def test_free_preflight_all_eight_never_calls_model_or_encoder(self):
        entries,sources=suite.inputs()
        with tempfile.TemporaryDirectory() as temp:
            args=SimpleNamespace(output_dir=Path(temp)/'batch',case=None,repeats=1,preflight_only=True,
                                 model='offline',base_url='https://invalid.test')
            with patch.object(pilot,'LocalEncoder',side_effect=AssertionError('No encoder')), \
                    patch.object(pilot,'plan_flat',side_effect=AssertionError('No model')):
                self.assertEqual(run_cases(args,entries,sources,'offline fixture'),0)
            result=pilot.read_json(args.output_dir/'batch_summary.json')
            self.assertEqual(result['scene_task_counts'],{'scene200':6,'scene202':2})
            self.assertEqual(result['total_trials'],8)
            self.assertEqual(result['successful_trials'],0)
            self.assertTrue(result['all_expected_checks_passed'])


if __name__=='__main__': unittest.main()
