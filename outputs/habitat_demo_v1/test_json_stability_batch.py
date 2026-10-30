"""Offline fixtures only; no API and no claims about live-model reliability."""
import contextlib
import copy
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parent))
import run_json_stability_batch as batch
from json_stability_cases import catalog, validate_cases, DEFAULT_CASES
from test_json_long_horizon import nodes, row as unit_row
from run_navigation_demo import read_json, write_json


NAV='navigate_to_object'
PREP='position_arm_for_object'
GRASP='grasp_object'
LIFT='lift_held_object'
PLACE='place_object_on_surface'


def routes(case):
    baseline = {
        1:[(NAV,i) for i in (67,13,67)], 2:[(PREP,67),(GRASP,67),(LIFT,67)],
        3:[(NAV,13)], 4:[(PREP,13),(PLACE,67,13)],
        5:[(PREP,67),(GRASP,67),(LIFT,67),(NAV,62),(PREP,62),(PLACE,67,62)],
        6:[(NAV,13),(NAV,67)], 7:[(PREP,67),(GRASP,67),(LIFT,67)],
        8:[(NAV,13),(PREP,13),(PLACE,67,13)],
        9:[(PREP,67),(GRASP,67),(LIFT,67),(NAV,62)],
        10:[(PREP,62),(PLACE,67,62)],
        11:[(PREP,67),(GRASP,67),(LIFT,67),(NAV,13),(PREP,13),(PLACE,67,13)],
        12:[(NAV,i) for i in (67,13,67)]}
    complex_round=[(NAV,67),(PREP,67),(GRASP,67),(LIFT,67),(NAV,13),(PREP,13),
                   (PLACE,67,13),(NAV,62),(NAV,67)]
    if case['name']=='short_navigation_1':
        return {1:[(NAV,13)]}
    if case['name']=='split_transfer_3':
        return {1:[(NAV,67),(PREP,67),(GRASP,67),(LIFT,67)],2:[(NAV,13)],3:[(PREP,13),(PLACE,67,13)]}
    if case['name']=='compound_transfer_2':
        return {1:[(NAV,62),(NAV,67)],2:complex_round}
    if case['name']=='composed_6':
        return {1:[(NAV,i) for i in (62,13,67)],
            2:[(PREP,67),(GRASP,67),(LIFT,67),(NAV,13),(PREP,13),(PLACE,67,13)],
            3:[(NAV,62),(NAV,67)],4:[(PREP,67),(GRASP,67),(LIFT,67),(NAV,62)],
            5:[(NAV,13),(NAV,62)],6:[(PREP,62),(PLACE,67,62),(NAV,13),(NAV,67)]}
    if case['name']=='long_navigation_8':
        return {r['round']:[(NAV,i) for i in r['expected_navigation_targets']] for r in case['rounds']}
    if case['name']=='extended_mixed_18':
        baseline.update({13:[(NAV,67),(PREP,67),(GRASP,67),(LIFT,67),(NAV,62),(PREP,62),(PLACE,67,62)],
            14:[(NAV,i) for i in (13,67,13,67)],15:[(PREP,67),(GRASP,67),(LIFT,67)],
            16:[(NAV,13),(NAV,62)],17:[(PREP,62),(PLACE,67,62)],18:complex_round})
    return baseline


class StabilityTests(unittest.TestCase):
    def test_catalog_difficulty_and_no_prewritten_runtime_plans(self):
        cases=validate_cases(catalog(nodes()))
        selected=[c for c in cases if c['name'] in DEFAULT_CASES]
        positive=[c for c in selected if c['kind']=='positive']
        self.assertEqual([len(c['rounds']) for c in positive],[1,3,2,6,8,18])
        self.assertEqual([c['difficulty_rank'] for c in positive],list(range(1,7)))
        self.assertEqual(len(selected),9)
        self.assertEqual(2*sum(4*len(c['rounds']) if c['kind']=='positive' else 2 for c in selected),316)
        for case in positive:
            self.assertTrue(all('plan' not in r and 'steps' not in r for r in case['rounds']))
        duplicate=next(c for c in cases if c['name']=='ambiguous_sideboard')
        self.assertEqual(len(duplicate['nodes']),4)
        self.assertNotIn(113,[n['id'] for n in nodes()])

    def test_all_positive_assertions_are_satisfiable_by_offline_fixture(self):
        for case in catalog(nodes()):
            if case['kind']!='positive':
                continue
            with self.subTest(case=case['name']), tempfile.TemporaryDirectory() as directory:
                args=SimpleNamespace(output_dir=Path(directory),model='test',base_url='https://invalid.test')
                mapping=routes(case)
                def fixture(worker_args, root):
                    descriptor=read_json(root/'round_input.json')
                    output, _=unit_row(mapping[descriptor['round']], support=descriptor['expected_support_id'],
                        held=descriptor['expected_end']['held_object_id'] is not None)
                    output['alignment']['task_decomposition']['instruction']=descriptor['instruction']
                    output['plan']['instruction']=descriptor['instruction']
                    write_json(root/'object_alignment.json',output['alignment'])
                    write_json(root/'skill_plan_final.json',output['plan'])
                    write_json(root/'planning_summary.json',{'status':'planning_passed','validation_mode':'json_symbolic_test',
                        'input_memory_version':descriptor['round']-1,'fixture_only':True})
                    return 0
                with patch.object(batch,'plan_one_round',side_effect=fixture), contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(batch.positive_worker(args,case),0)
                result=read_json(args.output_dir/'case_summary.json')
                self.assertEqual(result['completed_round_count'],len(case['rounds']))
                for previous,current in zip(result['rounds'],result['rounds'][1:]):
                    self.assertEqual(previous['end_state'],current['begin_state'])
                self.assertFalse(result['simulation_executed'])

    def test_classification_separates_refusals_correction_and_transport(self):
        success={'status':'json_symbolic_passed','expected_round_count':1}
        audit=dict.fromkeys(batch.AUDIT_KEYS,0)
        audit['logical_request_count']=4
        self.assertEqual(batch.classify('positive',success,audit),'clean_first_pass')
        audit['transport_failure_count']=1
        self.assertEqual(batch.classify('positive',success,audit),'passed_after_transport_retry')
        audit['program_rejection_count']=2
        self.assertEqual(batch.classify('positive',success,audit),'passed_after_validation_and_transport_retry')
        audit['transport_failure_count']=0
        self.assertEqual(batch.classify('positive',success,audit),'passed_after_validation_correction')
        self.assertEqual(batch.classify('negative',{'status':'expected_refusal'},audit),'expected_refusal')
        self.assertEqual(batch.classify('negative',{'status':'failed'},audit),'final_failure')
        audit['logical_request_count']=0
        self.assertEqual(batch.classify('positive',success,audit),'evidence_incomplete')

    def args(self,directory):
        source=directory/'source.json'
        write_json(source,nodes())
        return SimpleNamespace(base_graph=source,verified_graph=source,case=None,repeats=2,
            preflight_only=False,output_dir=directory/'batch',model='test',base_url='https://invalid.test')

    def case_fixture(self,command,log):
        source=Path(command[command.index('--worker-case')+1])
        case=read_json(source)
        root=Path(command[command.index('--output-dir')+1])
        if case['kind']=='positive':
            result={'status':'json_symbolic_passed','kind':'positive','expected_round_count':len(case['rounds']),
                'completed_round_count':len(case['rounds']),'simulation_executed':False}
            count=4*len(case['rounds'])
        else:
            result={'status':'expected_refusal','kind':'negative','simulation_executed':False}
            count=2
        write_json(root/'case_summary.json',result)
        for i in range(count):
            record={'request_id_local':str(i),'stage':'fixture_only_not_real_llm','context':{}}
            write_json(root/f'llm_response_audit/{i}_request.json',dict(record,model_input={'model':'offline-test'}))
            write_json(root/f'llm_response_audit/{i}_http_1.json',dict(record,event='http_response',http_status=200))

    def test_batch_continues_after_case_failure_and_keeps_denominators(self):
        with tempfile.TemporaryDirectory() as directory:
            args=self.args(Path(directory))
            calls=[]
            def fixture(command,log,timeout):
                case=read_json(Path(command[command.index('--worker-case')+1]))
                calls.append(case['name'])
                self.case_fixture(command,log)
                if len(calls)==1:
                    root=Path(command[command.index('--output-dir')+1])
                    write_json(root/'case_summary.json',{'status':'failed','kind':'positive',
                        'completed_round_count':0,'expected_round_count':1,'failure_stage':'fixture_error'})
                    raise ValueError('Fixture case failure')
            with patch.object(batch,'execute_worker',side_effect=fixture), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(batch.run_batch(args),2)
            summary=read_json(args.output_dir/'batch_summary.json')
            report=read_json(args.output_dir/'stability_analysis.json')
            self.assertEqual(summary['status'],'completed')
            self.assertEqual(len(calls),18)
            self.assertEqual(calls[:9],list(DEFAULT_CASES))
            self.assertEqual(report['positive_trial_count'],12)
            self.assertEqual(report['positive_success_count'],11)
            self.assertEqual(report['negative_expected_refusal_count'],6)
            self.assertEqual(summary['no_failure_estimated_logical_requests'],316)
            self.assertTrue(summary['frozen_engine_unchanged'])
            self.assertTrue(summary['upstream_sources_unchanged'])
            self.assertEqual(len(list((args.output_dir/'trials').iterdir())),18)

    def test_frozen_worker_imports_without_habitat_and_hash_guard_detects_edits(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            manifest=batch.freeze_engine(root)
            worker=root/'outputs/habitat_demo_v1/run_json_stability_batch.py'
            process=subprocess.run([sys.executable,str(worker),'--help'],capture_output=True,text=True,timeout=30)
            self.assertEqual(process.returncode,0,process.stderr)
            self.assertTrue(batch.frozen_unchanged(root,manifest))
            # Deliberate edit in a disposable test snapshot, never in the actual engine.
            worker.write_text(worker.read_text(encoding='utf-8')+'\n# test-only tamper\n',encoding='utf-8')
            self.assertFalse(batch.frozen_unchanged(root,manifest))

    def test_network_failure_is_not_expected_negative_refusal(self):
        case=next(c for c in catalog(nodes()) if c['name']=='missing_apple')
        with tempfile.TemporaryDirectory() as directory:
            args=SimpleNamespace(output_dir=Path(directory),model='test',base_url='https://invalid.test')
            with patch.object(batch.subprocess,'run',side_effect=TimeoutError('fixture timeout')):
                self.assertEqual(batch.negative_worker(args,case),2)
            self.assertEqual(read_json(args.output_dir/'case_summary.json')['status'],'failed')


if __name__=='__main__':
    unittest.main()
