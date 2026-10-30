"""Hand-authored feasibility fixtures only; never supplied to online planning."""
import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import run_crossroom_task as pilot
import run_flat_stress as batch
from flat_stress_cases import catalog
import flat_matrix_planner as flat
from test_crossroom_task import CrossroomTests
from json_symbolic_review import review_symbolic


NAV='navigate_to_object'; PREP='position_arm_for_object'; GRASP='grasp_object'
LIFT='lift_held_object'; PLACE='place_object_on_surface'


def transfer(obj,surface):
    return [(NAV,obj),(PREP,obj),(GRASP,obj),(LIFT,obj),(NAV,surface),(PREP,surface),(PLACE,obj,surface)]


def reference_sequences():
    return [
        [transfer(121,2)+transfer(86,101)+transfer(121,69)+[(NAV,2),(NAV,101),(NAV,2)]],
        [[(NAV,121),(PREP,121),(GRASP,121),(LIFT,121),(NAV,69),(NAV,2),(PREP,2),(PLACE,121,2)]
         +transfer(86,101)+transfer(121,101)+transfer(86,69)+[(NAV,2),(NAV,121),(NAV,2)]],
        [[(NAV,121),(PREP,121),(GRASP,121),(LIFT,121),(NAV,69),(NAV,2),(NAV,101)],
         [(PREP,101),(PLACE,121,101)]+transfer(86,2)+[(NAV,69),(NAV,101),(NAV,69)],
         transfer(86,101)+[(NAV,121),(PREP,121),(GRASP,121),(LIFT,121),(NAV,2)],
         [(PREP,2),(PLACE,121,2)]+transfer(86,69)+transfer(121,69)+[(NAV,101),(NAV,2),(NAV,86)]]]


class FlatStressTests(unittest.TestCase):
    def setUp(self):
        self.helper=CrossroomTests()
        self.helper.setUp()
        self.cases=catalog(self.helper.cfg)

    def test_new_instructions_and_manual_feasibility_all_events(self):
        old={r['instruction'] for r in self.helper.cfg['rounds']}
        totals=[]
        with tempfile.TemporaryDirectory() as temp:
            for (name,cfg),sequences in zip(self.cases,reference_sequences()):
                self.helper.cfg=cfg
                state=pilot.initial_state(self.helper.nodes,cfg)
                count=0
                for desc,actions in zip(cfg['rounds'],sequences):
                    self.assertNotIn(desc['instruction'],old)
                    before=copy.deepcopy(state)
                    fixture,library=self.helper.fixture(desc,actions,state,Path(temp)/name/str(desc['round']))
                    review=review_symbolic(fixture,library,self.helper.nodes,state,1.5)
                    pilot.check_event_assertions(desc,fixture['plan'],review)
                    self.assertTrue(review['case_event_assertions']['held_during_visits_checked'])
                    self.assertEqual(state,before)
                    state=review['predicted_end_state']
                    state['backend']['memory_version']=desc['round']
                    count+=len(review['symbolic_trace'])
                totals.append(count)
        self.assertEqual(totals,[24,32,50])

    def test_event_assertions_reject_wrong_intermediate_payload_and_pair(self):
        name,cfg=self.cases[0]
        self.helper.cfg=cfg
        with tempfile.TemporaryDirectory() as temp:
            desc=cfg['rounds'][0]
            fixture,library=self.helper.fixture(desc,reference_sequences()[0][0],self.helper.state,Path(temp))
            review=review_symbolic(fixture,library,self.helper.nodes,self.helper.state,1.5)
            bad=copy.deepcopy(desc)
            bad['expected_held_during_visits'][1]=86
            with self.assertRaisesRegex(ValueError,'Carried object'):
                pilot.check_event_assertions(bad,fixture['plan'],review)
            bad=copy.deepcopy(desc)
            bad['expected_placement_events'][0]=[121,69]
            with self.assertRaisesRegex(ValueError,'Ordered placement'):
                pilot.check_event_assertions(bad,fixture['plan'],review)

    def test_flat_calls_whole_planner_once_never_hierarchy_or_split(self):
        for accepted in (True,False):
            with self.subTest(accepted=accepted), tempfile.TemporaryDirectory() as temp:
                root=Path(temp)
                name,cfg=self.cases[0]
                self.helper.cfg=cfg
                desc=cfg['rounds'][0]
                fixture,library=self.helper.fixture(desc,reference_sequences()[0][0],self.helper.state,root)
                args=SimpleNamespace(model='offline',base_url='https://invalid.test',
                                     top_k=2,cosine_threshold=.35,cosine_margin=.05)
                def planner(a,reviewer):
                    self.assertEqual(a.round_input,root/'round_input.json')
                    self.assertEqual(a.skill_output_format,'matrix')
                    self.assertIsNotNone(a.skill_retriever)
                    result=reviewer(fixture,pilot.read_json(root/'execution_skill_library.json'),
                                   self.helper.nodes,self.helper.state,1.5)
                    pilot.write_json(root/'skill_plan_final.json',fixture['plan'])
                    pilot.write_json(root/'symbolic_review_final.json',result)
                    pilot.write_json(root/'planning_summary.json',{'status':'planning_passed'})
                    return 0
                def judge(a,d,stage,system,payload,check):
                    self.assertEqual(stage,'flat_whole_instruction_coverage')
                    self.assertEqual(payload['original_instruction'],desc['instruction'])
                    self.assertEqual(len(payload['segments']),1)
                    self.assertEqual(payload['segments'][0]['kind'],'whole_round_no_split')
                    value=dict(complete=accepted,order_preserved=True,constraints_preserved=True,
                               no_unrequested_actions=True,reason='Offline fixture')
                    check(value)
                    return value
                with patch.object(flat,'plan',side_effect=planner) as call, \
                        patch.object(flat,'request_decision',side_effect=judge) as audit, \
                        patch.object(pilot,'plan_bounded',side_effect=AssertionError('No hierarchy')):
                    self.assertEqual(flat.plan_flat(args,root,object()),0 if accepted else 2)
                self.assertEqual(call.call_count,1)
                self.assertEqual(audit.call_count,1)
                self.assertFalse((root/'hierarchy').exists())
                self.assertTrue(pilot.read_json(root/'round_input_integrity.json')['unchanged'])
                self.assertEqual(pilot.read_json(root/'experiment_profile.json')['max_split_levels'],0)

    def test_batch_preflight_no_paid_calls_and_new_case_configs(self):
        with tempfile.TemporaryDirectory() as temp:
            args=SimpleNamespace(output_dir=Path(temp)/'batch',case=None,repeats=1,preflight_only=True,
                                 model='offline',base_url='https://invalid.test')
            with patch.object(pilot,'LocalEncoder',side_effect=AssertionError('No embedding')), \
                    patch.object(pilot,'plan_flat',side_effect=AssertionError('No API')):
                self.assertEqual(batch.run(args),0)
            result=pilot.read_json(args.output_dir/'batch_summary.json')
            self.assertEqual(result['total_trials'],3)
            self.assertEqual(result['successful_trials'],0)  # preflight != model success
            self.assertTrue(result['all_expected_checks_passed'])
            for path in (args.output_dir/'cases').glob('*.json'):
                cfg,*_=pilot.load_case(path)
                self.assertTrue(cfg['case_id'].startswith('scene202_flat_stress_'))
                self.assertTrue(all('expected_placement_events' in r for r in cfg['rounds']))

    def test_independent_cases_continue_after_failure_without_replay(self):
        with tempfile.TemporaryDirectory() as temp:
            args=SimpleNamespace(output_dir=Path(temp)/'batch',case=None,repeats=1,preflight_only=False)
            calls=[]
            def runner(a):
                calls.append(a)
                self.assertEqual(a.planning_strategy,'flat')
                first=len(calls)==1
                pilot.write_json(a.output_dir/'regression_summary.json',{
                    'status':'failed' if first else 'json_symbolic_passed',
                    'rounds':[{'round':1,'status':'failed' if first else 'json_symbolic_passed'}],
                    'completed_round_count':0 if first else 1,'expected_round_count':1})
                return 2 if first else 0
            with patch.object(pilot,'run',side_effect=runner):
                self.assertEqual(batch.run(args),2)
            self.assertEqual(len(calls),3)
            summary=pilot.read_json(args.output_dir/'batch_summary.json')
            self.assertEqual(summary['status'],'completed')
            self.assertFalse(summary['all_expected_checks_passed'])
            self.assertEqual(summary['successful_trials'],2)
            self.assertEqual(summary['trials'][0]['failed_round'],1)


if __name__=='__main__': unittest.main()
