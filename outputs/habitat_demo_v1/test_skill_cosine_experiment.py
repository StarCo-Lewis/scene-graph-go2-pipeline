"""Offline checks with explicit synthetic vectors; no measured embedding accuracy."""
import copy
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parent))
import bounded_skill_planner as bounded
import plan_live_round as live
from skill_retrieval import enrich, candidates, cosine, check_membership, model_fingerprint
from run_navigation_demo import read_json, write_json
from run_json_long_horizon import initial_state, prepare_round
from test_json_long_horizon import nodes, row
from match_skills import build_result, compact_skills, request_plan
from run_skill_cosine_ab import report


class FakeEncoder:
    def __init__(self, query, profile, ambiguous=False):
        self.query,self.profile,self.ambiguous=query,profile,ambiguous
    def encode(self,texts):
        return [[1.,0.] if self.ambiguous or t==self.query or t==self.profile else [0.,1.] for t in texts]


class CosineTests(unittest.TestCase):
    def library(self):
        return read_json(Path(__file__).resolve().parents[1]/'quadruped_atomic_skill_library.v0.1.json')

    def test_metadata_covers_library_without_changing_parameters(self):
        library=self.library()
        old=copy.deepcopy(library)
        updated=enrich(library)
        self.assertEqual(library,old)
        self.assertEqual(len(updated['skills']),17)
        for original,new in zip(old['skills'],updated['skills']):
            for key in ('skill_id','parameters','preconditions','effects'):
                self.assertEqual(original[key],new[key])
            self.assertTrue(new['semantic_profile']['not_equivalent'])
        self.assertNotIn('semantic_profile',compact_skills(old)[0])
        self.assertIn('semantic_profile',compact_skills(updated)[0])

    def test_cosine_bounds_bad_vectors_and_zero(self):
        self.assertAlmostEqual(cosine([2,0],[1,0]),1)
        self.assertAlmostEqual(cosine([1,0],[0,1]),0)
        for a,b in (([],[]),([0,0],[1,0]),([1],[1,2]),([math.nan],[1])):
            with self.assertRaises(ValueError): cosine(a,b)

    def test_grasp_closure_not_lift_or_unsupported(self):
        fixture,library=row([('grasp_object',67)])
        library=enrich(library)
        query=fixture['alignment']['task_decomposition']['subtasks'][0]['description']
        profile=next(s for s in library['skills'] if s['skill_id']=='grasp_object')['description_en']
        retrieval=candidates(fixture['alignment'],library,FakeEncoder(query,profile),top_k=1)
        self.assertEqual(set(retrieval['subtasks'][0]['candidate_skill_ids']),
                         {'grasp_object','navigate_to_object','position_arm_for_object'})
        self.assertFalse(retrieval['score_is_probability'])

    def test_ambiguous_vectors_preserve_adapter_inventory(self):
        fixture,library=row([('navigate_to_object',13)])
        library=enrich(library)
        retrieval=candidates(fixture['alignment'],library,FakeEncoder('','',True))
        self.assertFalse(retrieval['subtasks'][0]['confident'])
        self.assertEqual(set(retrieval['subtasks'][0]['candidate_skill_ids']),
                         {s['skill_id'] for s in library['skills']})

    def test_conflict_not_silently_replaced(self):
        proposal={'subtask_plans':[{'subtask_id':'st_01','steps':[{'skill_id':'grasp_object'}]}]}
        retrieval={'subtasks':[{'subtask_id':'st_01','candidate_skill_ids':['navigate_to_object']}]}
        with self.assertRaisesRegex(ValueError,'SKILL_RETRIEVAL_CONFLICT'):
            check_membership(proposal,retrieval)
        result=check_membership(proposal,retrieval,widened=True)
        self.assertFalse(result['candidate_membership'])
        self.assertTrue(result['widened_inventory'])
        self.assertFalse(result['independent_agreement_claim'])

    def test_prompt_filter_and_expansion_keep_exact_parameters(self):
        fixture,library=row([('navigate_to_object',13)])
        library=enrich(library)
        subtask=fixture['alignment']['task_decomposition']['subtasks'][0]['subtask_id']
        retrieval={'subtasks':[{'subtask_id':subtask,'candidate_skill_ids':['navigate_to_object']}]}
        requests=[]
        def transport(req,*args,**kwargs):
            requests.append(json.loads(req.data))
            return {'choices':[{'message':{'content':json.dumps(fixture['plan'])},'finish_reason':'stop'}]}
        with patch.dict('os.environ',{'TEST_KEY':'fake'}),patch('match_skills.request_json_with_retry',transport):
            for widened in (False,True):
                request_plan(fixture['alignment'],library,'offline','https://invalid.test','TEST_KEY',
                             retrieval=retrieval,widen_inventory=widened)
        prompts=[json.loads(r['messages'][1]['content']) for r in requests]
        self.assertEqual([s['skill_id'] for s in prompts[0]['available_skills']],['navigate_to_object'])
        self.assertEqual(len(prompts[1]['available_skills']),len(library['skills']))
        self.assertEqual(prompts[0]['available_skills'][0]['parameters'],compact_skills(library)[0]['parameters'])

    def test_bounded_split_only_calls_llm_root_and_coverage(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            desc={'round':1,'instruction':'Visit ID 13, then visit ID 62.',
                'expected_end':{'held_object_id':None,'lifted':False},
                'expected_navigation_targets':[13,62], 'expected_support_id':None}
            prepare_round(root,desc,initial_state(nodes()),nodes(),'offline','https://invalid.test')
            stages,starts=[],[]
            def decide(a,folder,stage,system,payload,check):
                stages.append(stage)
                value=({'decision':'split','reason':'fixture','parts':['Visit ID 13,',' then visit ID 62.']}
                    if stage=='hierarchy_split' else dict(complete=True,order_preserved=True,
                    constraints_preserved=True,no_unrequested_actions=True,reason='fixture'))
                check(value)
                return value
            def planner(a,reviewer):
                self.assertEqual(a.skill_proposal_limit,2)
                self.assertTrue(a.compile_navigation)
                starts.append(read_json(a.output_dir/'runtime_state.json'))
                child=read_json(a.round_input)
                target=13 if '13' in child['instruction'] else 62
                fixture,library=row([('navigate_to_object',target)])
                fixture['alignment']['task_decomposition']['instruction']=child['instruction']
                write_json(a.output_dir/'object_alignment.json',fixture['alignment'])
                write_json(a.output_dir/'skill_plan_final.json',build_result(fixture['alignment'],library,fixture['plan']))
                return 0
            with patch.object(bounded,'request_decision',decide),patch.object(bounded,'plan_leaf',planner):
                self.assertEqual(bounded.plan_bounded(SimpleNamespace(model='offline',base_url='https://invalid.test',compile_navigation=True),root),0)
            self.assertEqual(stages,['hierarchy_split','hierarchy_coverage'])
            self.assertEqual(starts[1]['backend']['robot_position'],[0.,.1,0.])
            self.assertFalse(read_json(root/'planning_summary.json')['simulation_executed'])

    def test_model_fingerprint_detects_weight_changes(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)
            with self.assertRaises(ValueError): model_fingerprint(path)
            (path/'weights.safetensors').write_bytes(b'fake model fixture')
            first=model_fingerprint(path)
            (path/'weights.safetensors').write_bytes(b'changed fake fixture')
            self.assertNotEqual(first,model_fingerprint(path))

    def test_retrieval_worker_retries_once_with_full_inventory(self):
        fixture,library=row([('navigate_to_object',13)])
        library=enrich(library)
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            desc={'round':1,'instruction':fixture['alignment']['task_decomposition']['instruction'],
                  'expected_end':{},'expected_support_id':None}
            state=initial_state(nodes())
            for path,value in [('round_input.json',desc),('runtime_state.json',state),
                ('execution_skill_library.json',library),('input_memory/scene/scene_graph.json',nodes())]:
                write_json(root/path,value)
            q=fixture['alignment']['task_decomposition']['subtasks'][0]['description']
            profile=next(s for s in library['skills'] if s['skill_id']=='navigate_to_object')['description_en']
            args=SimpleNamespace(output_dir=root,round_input=root/'round_input.json',model='offline',
                base_url='https://invalid.test',reach=1.5,skill_proposal_limit=2,
                skill_retriever=FakeEncoder(q,profile),retrieval_top_k=1,retrieval_threshold=.35,retrieval_margin=.05)
            malformed=copy.deepcopy(fixture['plan'])
            malformed['subtask_plans'][0]['steps'][0]['arguments']={}
            with patch.object(live,'decompose_and_bind',return_value=fixture['alignment']), \
                 patch.object(live,'request_plan',side_effect=[malformed,fixture['plan']]) as propose, \
                 patch.object(live,'request_order',return_value={'invalid':'retain already validated original'}):
                from json_symbolic_review import review_symbolic
                self.assertEqual(live.plan(args,reviewer=review_symbolic),0)
            self.assertEqual(propose.call_count,2)
            self.assertFalse(propose.call_args_list[0].kwargs['widen_inventory'])
            self.assertTrue(propose.call_args_list[1].kwargs['widen_inventory'])
            self.assertEqual(read_json(root/'runtime_state.json'),state)

    def test_root_cannot_request_five_parts(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            desc={'round':1,'instruction':'A B C D E','expected_end':{}}
            prepare_round(root,desc,initial_state(nodes()),nodes(),'offline','https://invalid.test')
            def decide(a,folder,stage,system,payload,check):
                value={'decision':'split','reason':'fixture','parts':['A ','B ','C ','D ','E']}
                check(value)
                return value
            with patch.object(bounded,'request_decision',decide),patch.object(bounded,'plan_leaf') as leaf:
                self.assertEqual(bounded.plan_bounded(SimpleNamespace(model='offline',base_url='https://invalid.test'),root),2)
                leaf.assert_not_called()
            self.assertIn('BOUNDED_PART_LIMIT',read_json(root/'planning_summary.json')['error'])

    def test_failure_low_cost_not_counted_as_success(self):
        with tempfile.TemporaryDirectory() as d:
            s={'status':'completed','arms':['bounded_cosine'],'frozen_engine_unchanged':True,
               'case_inputs_unchanged':True,'trials':[{'case':'fixture','repeat':1,'arm':'bounded_cosine',
               'kind':'positive','evaluation_outcome':'final_failure','duration_seconds':1,
               'completed_round_count':0,'expected_round_count':18,'token_usage':{'total_tokens':10}}]}
            report(Path(d),s)
            value=read_json(Path(d)/'comparison_summary.json')['arms']['bounded_cosine']
            self.assertEqual(value['success_rate'],0)
            self.assertIsNone(value['tokens_per_successful_case_including_failures'])


if __name__=='__main__':
    unittest.main()
