"""Offline fixtures only; no live API, model accuracy or physical claims."""
import contextlib
import copy
import io
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parent))
import run_long_matrix_ab as ab
import plan_live_round as live
from run_navigation_demo import read_json,write_json
from test_json_long_horizon import nodes,row as fixture_row
from test_json_stability_batch import routes
from json_stability_cases import catalog
from verify_plan import flatten_steps


class FixtureEncoder:
    def __init__(self,*args): self.closed=False
    def encode(self,texts): return [[1.,0.] for _ in texts]
    def close(self): self.closed=True
    def stats(self): return {'backend':'unit_fixture_only','seconds_including_startup':0.}


class LongMatrixTests(unittest.TestCase):
    def case_source(self,root):
        cases=[c for c in catalog(nodes()) if c['name']!='baseline_12']
        hashes={}
        for c in cases:
            path=root/'cases'/f'{c["name"]}.json'
            write_json(path,c)
            hashes[c['name']]=ab.batch.sha(path)
        write_json(root/'batch_summary.json',{'case_input_hashes':hashes})

    def test_original_cases_and_twelve_round_source_are_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            self.case_source(root)
            cases,source=ab.select_cases(root,['baseline_12','extended_mixed_18'])
            self.assertEqual([len(c['rounds']) for c in cases],[12,18])
            self.assertEqual(cases[1],read_json(root/'cases/extended_mixed_18.json'))
            self.assertEqual(source['baseline_12']['sha256'],ab.batch.sha(root/'cases/extended_mixed_18.json'))
            self.assertEqual(ab.request_estimate(cases,1),240)
            for names in ([],['baseline_12','baseline_12'],['not_a_case']):
                with self.assertRaises(ValueError): ab.select_cases(root,names)

    def test_history_hash_mismatch_stops_before_requests(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            self.case_source(root)
            write_json(root/'cases/extended_mixed_18.json',{})
            with self.assertRaisesRegex(ValueError,'hash mismatch'):
                ab.select_cases(root,['baseline_12'])

    def test_all_difficulties_default_budget_and_shared_controls(self):
        selected=[c for c in catalog(nodes()) if c['name'] in ab.DEFAULT_SELECTION]
        self.assertEqual(len(selected),10)
        self.assertEqual(ab.request_estimate(selected,3),1218)
        self.assertEqual(sum(2 if c['kind']=='positive' else 1 for c in selected)*3,51)
        self.assertEqual(sorted(len(c['rounds']) for c in selected if c['kind']=='positive'),[1,2,3,6,8,12,18])

    def test_both_formats_full_eighteen_round_state_chain(self):
        case=next(c for c in catalog(nodes()) if c['name']=='extended_mixed_18')
        mapping=routes(case)
        for arm in ab.ARMS:
            with self.subTest(arm=arm),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp)
                write_json(root/'case.json',case)
                args=SimpleNamespace(worker_case=root/'case.json',arm=arm,output_dir=root/'result',
                    embedding_model_dir=root/'model',embedding_model_sha='fixture',embedding_python=sys.executable,
                    top_k=2,cosine_threshold=.35,cosine_margin=.05,model='offline',base_url='https://invalid.test')
                fixture={'reference':None}
                seen=[]
                def decompose(a,descriptor,summary,*unused):
                    reference,_=fixture_row(mapping[descriptor['round']],support=descriptor['expected_support_id'],
                        held=descriptor['expected_end']['held_object_id'] is not None)
                    reference['alignment']['task_decomposition']['instruction']=descriptor['instruction']
                    fixture['reference']=reference
                    write_json(a.output_dir/'object_alignment.json',reference['alignment'])
                    return reference['alignment']
                def nested(alignment,library,*unused,**options):
                    seen.append((options['audit_context']['skill_output_format'],options['widen_inventory']))
                    return {'subtask_plans':copy.deepcopy(fixture['reference']['plan']['subtask_plans'])}
                def matrix(alignment,library,book,*unused,**options):
                    seen.append((options['audit_context']['skill_output_format'],options['widen_inventory']))
                    obj={(o['role_id'],o['element_id']):o['code'] for o in book['objects']}
                    skill={s['skill_id']:s for s in book['skills']}
                    result=[]
                    for task_code,task in enumerate(fixture['reference']['plan']['subtask_plans'],1):
                        for step in task['steps']:
                            spec=skill[step['skill_id']]
                            codes=[obj[(step['arguments'][p['name']]['role_id'],step['arguments'][p['name']]['element_id'])] for p in spec['slots']]
                            result.append([task_code,spec['code'],*(codes+[0,0])[:2]])
                    return {'matrix':result}
                encoder=FixtureEncoder()
                before=ab.batch.plan_one_round
                with patch.object(ab,'model_fingerprint',return_value='fixture'), \
                     patch.object(ab,'LocalEncoder',return_value=encoder), \
                     patch.object(live,'decompose_and_bind',side_effect=decompose), \
                     patch.object(live,'request_plan',side_effect=nested), \
                     patch.object(live,'request_matrix',side_effect=matrix), \
                     patch.object(live,'request_order',side_effect=lambda plan,*a,**kw: {'ordered_step_ids':[s['step_id'] for s in flatten_steps(plan)],'reason':'retain fixture'}), \
                     contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(ab.worker(args),0)
                self.assertIs(ab.batch.plan_one_round,before)
                self.assertTrue(encoder.closed)
                summary=read_json(args.output_dir/'case_summary.json')
                self.assertEqual(summary['completed_round_count'],18)
                self.assertEqual(seen,[(arm,False)]*18)
                self.assertTrue(ab.format_metrics(args.output_dir)['round_inputs_unchanged'])
                self.assertFalse(summary['simulation_executed'])
                for previous,current in zip(summary['rounds'],summary['rounds'][1:]):
                    self.assertEqual(previous['end_state'],current['begin_state'])
                self.assertEqual(summary['rounds'][-1]['end_state']['backend']['objects']['67']['support_id'],13)

    def test_failure_prefix_not_counted_as_cost_savings(self):
        def trial(arm,status,tokens,seconds):
            return {'case':'long','repeat':1,'arm':arm,'kind':'positive','evaluation_outcome':status,
                'expected_round_count':18,'completed_round_count':1,'token_usage':{'total_tokens':tokens},
                'duration_seconds':seconds}
        summary={'status':'completed','trials':[trial('nested_json','clean_first_pass',100,10),trial('matrix','final_failure',1,1)],
            'frozen_engine_unchanged':True,'case_inputs_unchanged':True,'historical_sources_unchanged':True}
        result=ab.comparison(summary)
        self.assertEqual(result['paired_costs']['both_complete']['pair_count'],0)
        self.assertIsNone(result['paired_costs']['both_complete']['matrix_savings']['tokens'])
        self.assertEqual(result['arms']['matrix']['known_tokens_including_failures'],1)
        summary['status']='invalid_or_harness_failure'
        self.assertFalse(ab.comparison(summary)['evaluation_valid'])

    def test_preflight_freezes_without_api_or_embedding(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            self.case_source(root/'source')
            args=SimpleNamespace(source_batch=root/'source',case=['baseline_12','extended_mixed_18'],
                output_dir=root/'run',repeats=1,model='fixture',preflight_only=True,
                top_k=2,cosine_threshold=.35,cosine_margin=.05)
            with patch.object(ab,'LocalEncoder',side_effect=AssertionError('no model')), \
                 patch.object(ab.batch,'execute_worker',side_effect=AssertionError('no API')), \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(ab.run(args),0)
            summary=read_json(root/'run/batch_summary.json')
            self.assertEqual(summary['no_failure_estimated_logical_requests'],240)
            self.assertTrue(summary['frozen_engine_unchanged'])
            self.assertEqual(summary['trials'],[])

    def test_failed_trial_continues_other_arms_and_control_is_not_duplicated(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            self.case_source(root/'source')
            args=SimpleNamespace(source_batch=root/'source',case=['short_navigation_1','split_transfer_3','missing_apple'],
                output_dir=root/'run',repeats=1,model='fixture',preflight_only=False,
                embedding_model_dir=root/'model',embedding_python=Path(sys.executable),base_url='https://invalid.test',
                top_k=2,cosine_threshold=.35,cosine_margin=.05)
            calls=[]
            def execute(command,log,timeout):
                self.assertIn('frozen_engine',command[1])
                folder=Path(command[command.index('--output-dir')+1])
                case=read_json(Path(command[command.index('--worker-case')+1]))
                arm=command[command.index('--arm')+1]
                calls.append(arm)
                if len(calls)==1:
                    write_json(folder/'case_summary.json',{'status':'failed','failure_cause':'fixture failure'})
                    raise ValueError('fixture worker failed')
                write_json(folder/'case_summary.json',{'status':'expected_refusal' if case['kind']=='negative' else 'json_symbolic_passed',
                    'expected_round_count':len(case.get('rounds',[])),'completed_round_count':len(case.get('rounds',[]))})
            audit=dict.fromkeys(ab.batch.AUDIT_KEYS,0)
            audit.update(logical_request_count=20,http_attempt_count=20,program_rejections_by_stage={})
            with patch.object(ab,'LocalEncoder',return_value=FixtureEncoder()), \
                 patch.object(ab,'model_fingerprint',return_value='fixture'), \
                 patch.object(ab.batch,'execute_worker',side_effect=execute), \
                 patch.object(ab.batch,'build_audit_report',return_value=audit), \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(ab.run(args),2)
            summary=read_json(root/'run/batch_summary.json')
            self.assertEqual(summary['status'],'completed')
            self.assertEqual(len(summary['trials']),5)
            self.assertEqual(calls.count('shared_control'),1)
            self.assertEqual(summary['trials'][0]['evaluation_outcome'],'final_failure')
            self.assertTrue(all(r['evaluation_outcome']!='final_failure' for r in summary['trials'][1:]))
            self.assertTrue(read_json(root/'run/comparison_summary.json')['evaluation_valid'])


if __name__=='__main__':
    unittest.main()
