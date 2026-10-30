"""Offline synthetic fixtures; no model API or physical execution."""
import copy
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hierarchical_json_planner as h
import run_hierarchical_json_ab as ab
from test_json_long_horizon import nodes, row
from run_json_long_horizon import initial_state, prepare_round
from run_navigation_demo import read_json, write_json
from match_skills import build_result

NAV = 'navigate_to_object'


class HierarchicalTests(unittest.TestCase):
    def test_partition_rejects_drop_reorder_duplicate_empty_and_fake_leaf(self):
        for parts in (['A','C'], ['B','A'], ['A','AB'], ['','AB'], ['AB'], ['AB',' ']):
            with self.subTest(parts=parts), self.assertRaises(ValueError):
                h.validate_partition({'decision':'split','parts':parts,'reason':'test'}, 'AB')
        with self.assertRaises(ValueError):
            h.validate_partition({'decision':'leaf','reason':'test','parts':['A']}, 'A')
        self.assertEqual(h.validate_partition({'decision':'split','parts':['A','B'],'reason':'test'}, 'AB'), ['A','B'])

    def test_coverage_false_or_string_boolean_rejected(self):
        base = dict(complete=True,order_preserved=True,constraints_preserved=True,no_unrequested_actions=True,reason='test')
        self.assertEqual(h.validate_coverage(base), base)
        for value in (False, 'true', 1, None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                h.validate_coverage(dict(base,complete=value))

    def run_fixture(self, root, limits=None, coverage=True, fail_leaf=None, extra_nav=False, nested=False):
        text = 'Visit ID 13, then visit ID 62.'
        state = initial_state(nodes())
        desc = {'round':1,'instruction':text,'expected_end':{'held_object_id':None,'lifted':False},
                'expected_navigation_targets':[13,62], 'expected_support_id':None}
        prepare_round(root, desc, state, nodes(), 'offline-test', 'https://invalid.test')
        calls, starts = [], []
        def decide(args, folder, stage, system, payload, check):
            calls.append(copy.deepcopy(payload))
            if stage == 'hierarchy_coverage':
                value = dict(complete=coverage,order_preserved=True,constraints_preserved=True,
                             no_unrequested_actions=True,reason='offline fixture')
            elif payload['current_text'] == text:
                value = {'decision':'split','reason':'fixture','parts':['Visit ID 13,',' then visit ID 62.']}
            elif nested:
                t = payload['current_text']
                value = {'decision':'split','reason':'fixture','parts':[t[:len(t)//2],t[len(t)//2:]]}
            else:
                value = {'decision':'leaf','reason':'fixture'}
            check(value)
            return value
        def planner(args, reviewer):
            child = read_json(args.round_input)
            starts.append(read_json(args.output_dir/'runtime_state.json'))
            self.assertEqual(child['expected_end'], {})
            if len(starts) == fail_leaf:
                write_json(args.output_dir/'planning_summary.json', {'status':'failed','error':'fixture rejection'})
                return 2
            target = 13 if '13' in child['instruction'] else 62
            steps = [(NAV, target)]
            if extra_nav and len(starts)==1:
                steps.insert(0,(NAV,62))
            fixture, library = row(steps)
            alignment = fixture['alignment']
            alignment['task_decomposition']['instruction'] = child['instruction']
            plan = build_result(alignment,library,fixture['plan'])
            write_json(args.output_dir/'object_alignment.json',alignment)
            write_json(args.output_dir/'skill_plan_final.json',plan)
            return 0
        args = SimpleNamespace(model='offline-test',base_url='https://invalid.test')
        code = h.plan_hierarchical(args,root,limits=limits,decide=decide,leaf_planner=planner)
        return code,calls,starts

    def test_chain_uses_validated_full_state_and_scoped_goals(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            code,calls,starts=self.run_fixture(root)
            self.assertEqual(code,0)
            self.assertEqual(len(starts),2)
            # Second split request and second leaf both see the first leaf's endpoint.
            self.assertEqual(calls[2]['current_state'],starts[1])
            self.assertEqual(starts[1]['backend']['robot_position'],[0.,.1,0.])
            alignment=read_json(root/'object_alignment.json')
            self.assertEqual(len(alignment['milestone_scopes']),2)
            self.assertEqual(len(alignment['task_decomposition']['goal_state']),1)
            self.assertEqual(len(alignment['context_bindings']),6)
            plan=read_json(root/'skill_plan_final.json')
            self.assertEqual(plan['action_count'],2)
            self.assertEqual(plan['subtask_plans'][1]['dependencies'],['leaf_001_st_01'])
            review=read_json(root/'symbolic_review_final.json')
            self.assertEqual(review['navigation_targets'],[13,62])
            self.assertFalse(review['training_eligible'])

    def test_original_strict_visit_assertion_not_relaxed(self):
        with tempfile.TemporaryDirectory() as d:
            code,_,_=self.run_fixture(Path(d),extra_nav=True)
            self.assertEqual(code,2)
            self.assertIn('Explicit visit order',read_json(Path(d)/'planning_summary.json')['error'])

    def test_failed_leaf_stops_later_segments(self):
        with tempfile.TemporaryDirectory() as d:
            code,calls,starts=self.run_fixture(Path(d),fail_leaf=1)
            self.assertEqual(code,2)
            self.assertEqual(len(starts),1)
            self.assertEqual(len(calls),2)
            self.assertFalse((Path(d)/'skill_plan_final.json').exists())

    def test_semantic_judge_failure_is_not_pass(self):
        with tempfile.TemporaryDirectory() as d:
            code,_,_=self.run_fixture(Path(d),coverage=False)
            self.assertEqual(code,2)
            self.assertEqual(read_json(Path(d)/'planning_summary.json')['failed_stage'],'hierarchical_semantic_coverage')

    def test_bounded_tree_no_forced_success(self):
        variants=[({'max_depth':1,'max_nodes':23,'max_leaves':12},True,'DEPTH'),
                  ({'max_depth':3,'max_nodes':1,'max_leaves':12},False,'NODE'),
                  ({'max_depth':3,'max_nodes':23,'max_leaves':1},False,'LEAF')]
        for limits,nested,label in variants:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as d:
                code,_,_=self.run_fixture(Path(d),limits=limits,nested=nested)
                self.assertEqual(code,2)
                self.assertIn(f'HIERARCHY_{label}_LIMIT',read_json(Path(d)/'planning_summary.json')['error'])

    def test_no_empty_leaf_or_empty_combination_success(self):
        with self.assertRaises(ValueError):
            h.combine_leaves('test',[],{})

    def test_structured_request_attempts_are_bounded(self):
        response={'choices':[{'message':{'content':'{}'}}]}
        with tempfile.TemporaryDirectory() as d, patch.dict('os.environ',{'DASHSCOPE_API_KEY':'fixture-key-not-real'}), \
                patch.object(h,'request_json_with_retry',return_value=response) as transport:
            args=SimpleNamespace(model='offline-test',base_url='https://invalid.test')
            with self.assertRaises(ValueError):
                h.request_decision(args,Path(d),'hierarchy_split','test',{},lambda v:h.validate_partition(v,'test'))
            self.assertEqual(transport.call_count,3)
            self.assertTrue(list((Path(d)/'llm_response_audit').glob('*request.json')))

    def test_held_object_and_moved_position_pass_between_leaves(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            parts=['Pick up and lift kettle ID 67;', ' carry it to sideboard ID 13;', ' place it there.']
            text=''.join(parts)
            desc={'round':1,'instruction':text,'expected_end':{'held_object_id':None,'lifted':False},
                  'expected_support_id':13,'expected_navigation_targets':[67,13]}
            prepare_round(root,desc,initial_state(nodes()),nodes(),'offline','https://invalid.test')
            starts=[]
            def decide(args,folder,stage,system,payload,check):
                if stage=='hierarchy_coverage':
                    value=dict(complete=True,order_preserved=True,constraints_preserved=True,no_unrequested_actions=True,reason='fixture')
                elif payload['current_text']==text:
                    value=dict(decision='split',reason='fixture',parts=parts)
                else:
                    value=dict(decision='leaf',reason='fixture')
                check(value)
                return value
            def planner(args,reviewer):
                descriptor=read_json(args.round_input)
                starts.append(read_json(args.output_dir/'runtime_state.json'))
                sequences=[[(NAV,67),('position_arm_for_object',67),('grasp_object',67),('lift_held_object',67)],
                           [(NAV,13)],[('position_arm_for_object',13),('place_object_on_surface',67,13)]]
                i=len(starts)-1
                fixture,library=row(sequences[i],support=13 if i==2 else None,held=i<2)
                alignment=fixture['alignment']
                alignment['task_decomposition']['instruction']=descriptor['instruction']
                write_json(args.output_dir/'object_alignment.json',alignment)
                write_json(args.output_dir/'skill_plan_final.json',build_result(alignment,library,fixture['plan']))
                return 0
            result=h.plan_hierarchical(SimpleNamespace(model='offline',base_url='https://invalid.test'),root,
                                       decide=decide,leaf_planner=planner)
            self.assertEqual(result,0)
            self.assertEqual(starts[1]['held_object_id'],67)
            self.assertTrue(starts[1]['lifted'])
            self.assertEqual(starts[2]['backend']['objects']['67']['center'],[0.,1.35,0.])
            final=read_json(root/'symbolic_review_final.json')['predicted_end_state']
            self.assertIsNone(final['held_object_id'])
            self.assertEqual(final['backend']['objects']['67']['support_id'],13)

    def test_negative_judge_is_not_retried_until_acceptance(self):
        import json
        value=dict(complete=False,order_preserved=True,constraints_preserved=True,no_unrequested_actions=True,reason='missing intent')
        response={'choices':[{'message':{'content':json.dumps(value)}}]}
        with tempfile.TemporaryDirectory() as d, patch.dict('os.environ',{'DASHSCOPE_API_KEY':'fixture-key-not-real'}), \
                patch.object(h,'request_json_with_retry',return_value=response) as transport:
            with self.assertRaises(ValueError):
                h.request_decision(SimpleNamespace(model='test',base_url='https://invalid.test'),Path(d),'hierarchy_coverage','test',{},h.validate_coverage)
            self.assertEqual(transport.call_count,1)


class ComparisonTests(unittest.TestCase):
    def test_paired_runner_continues_and_uses_exact_case_bytes(self):
        from json_stability_cases import catalog
        # Use a compact local transport fixture with complete request/result evidence.
        with tempfile.TemporaryDirectory() as d:
            parent=Path(d)
            case=next(c for c in catalog(nodes()) if c['name']=='short_navigation_1')
            source=parent/'source'
            case_file=source/'cases/short_navigation_1.json'
            write_json(case_file,case)
            write_json(source/'batch_summary.json',{'case_input_hashes':{case['name']:ab.baseline.sha(case_file)}})
            write_json(source/'frozen_engine/engine_manifest.json',{'files':{}})
            args=SimpleNamespace(source_batch=source,output_dir=parent/'result',case=[case['name']],repeats=1,
                model='offline',base_url='https://invalid.test',max_depth=3,max_nodes=23,max_leaves=12,
                seconds_per_round=1800,preflight_only=False)
            calls=[]
            def worker(command,log,timeout):
                strategy=command[command.index('--strategy')+1]
                calls.append(strategy)
                root=Path(command[command.index('--output-dir')+1])
                result={'status':'json_symbolic_passed' if strategy=='baseline' else 'failed',
                        'kind':'positive','expected_round_count':1,'completed_round_count':1 if strategy=='baseline' else 0}
                write_json(root/'case_summary.json',result)
                for i in range(4):
                    record={'request_id_local':str(i),'stage':'offline_fixture','context':{}}
                    write_json(root/f'llm_response_audit/{i}_request.json',dict(record,model_input={'model':'offline'}))
                    write_json(root/f'llm_response_audit/{i}_http_1.json',dict(record,event='http_response',http_status=200))
                if strategy=='hierarchical':
                    raise ValueError('fixture worker rejected')
            with patch.object(ab.baseline,'execute_worker',side_effect=worker):
                self.assertEqual(ab.run(args),2)
            summary=read_json(args.output_dir/'batch_summary.json')
            self.assertEqual(summary['status'],'completed')
            self.assertEqual(calls,['baseline','hierarchical'])
            self.assertTrue(summary['case_inputs_unchanged'])
            self.assertEqual(case_file.read_bytes(),(args.output_dir/'cases'/case_file.name).read_bytes())
            results=read_json(args.output_dir/'comparison_summary.json')['strategies']
            self.assertEqual(results['baseline']['passed_positive_trials'],1)
            self.assertEqual(results['hierarchical']['passed_positive_trials'],0)

    def test_denominators_and_negative_controls(self):
        summary={'positive_trials_per_strategy':2,'frozen_engine_unchanged':True,'case_inputs_unchanged':True,
          'trials':[{'case':'a','repeat':1,'kind':'positive','strategy':'baseline','evaluation_outcome':'final_failure'},
                    {'case':'a','repeat':1,'kind':'positive','strategy':'hierarchical','evaluation_outcome':'clean_first_pass'},
                    {'case':'b','repeat':1,'kind':'positive','strategy':'baseline'},
                    {'case':'negative','repeat':1,'kind':'negative','strategy':'baseline','evaluation_outcome':'expected_refusal'}]}
        result=ab.comparison(summary)
        self.assertEqual(result['strategies']['baseline']['evaluated_positive_trials'],1)
        self.assertEqual(result['strategies']['baseline']['planned_positive_trials'],2)
        self.assertEqual(result['strategies']['hierarchical']['passed_positive_trials'],1)
        self.assertEqual(len(result['completed_pairs']),1)


if __name__=='__main__':
    unittest.main()
