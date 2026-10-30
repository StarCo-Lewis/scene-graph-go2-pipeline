"""Offline checks: no measured model accuracy, paid calls or physical execution."""
import copy
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import run_crossroom_task as pilot
import run_contextual_ambiguity as contextual
import run_inference_comparison as comparison
import natural_inference_cases as suite
import inference_fact_probe as facts
import llm_graph_binding
import flat_matrix_planner


class InferenceComparisonTests(unittest.TestCase):
    def args(self,root,case_ids=None):
        return SimpleNamespace(output_dir=root,preflight_only=True,case_ids=case_ids,repeat=1,
            no_fact_probe=False,model='offline',base_url='https://invalid.test',
            top_k=2,cosine_threshold=.35,cosine_margin=.05)

    def test_twelve_paired_inputs_only_policy_and_run_identity_differ(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)/'comparison'
            with patch.object(pilot,'LocalEncoder') as encoder,patch.object(pilot,'plan_flat') as planner:
                self.assertEqual(comparison.run(self.args(root)),0)
                encoder.assert_not_called()
                planner.assert_not_called()
            summary = pilot.read_json(root/'comparison_summary.json')
            self.assertEqual(summary['status'],'preflight_completed')
            self.assertEqual(len(summary['pairs']),12)
            self.assertTrue(summary['all_paired_public_inputs_match'])
            for batch in summary['groups'].values():
                self.assertEqual(batch['finished_trial_count'],12)
                self.assertEqual(batch['symbolically_valid_trial_count'],0)
                self.assertFalse(batch['fresh_llm_calls'])
                for row in batch['trials']:
                    self.assertIn(row['fact_probe']['status'],('not_applicable','not_called_preflight'))
            sample = root/'bounded_inference/trials/H2_r01/round_1/planning/execution_contract.json'
            policy = pilot.read_json(sample)
            self.assertIn('never invent',policy['factual_boundary'].lower())
            self.assertNotIn('contents_status_reference',json.dumps(policy))

    def test_binding_treatment_replaces_rule_without_changing_scene_evidence(self):
        response = {'choices':[{'message':{'content':'{"bindings": []}'}}]}
        capabilities = {'graspable_ids':[121,86],'placeable_pairs':[[121,2],[86,2]]}
        policy = {'mode':'bounded_inference','rules':'Provisional interpretation only; never fabricate facts.'}
        bodies = []
        with patch.dict(os.environ,{'OFFLINE_TEST_KEY':'not-a-real-api-key'}):
            for treatment in (None,policy):
                with patch.object(llm_graph_binding,'request_json',return_value=response) as request:
                    llm_graph_binding.request_bindings([],[],'offline','https://invalid.test','OFFLINE_TEST_KEY',
                        inference_policy=treatment,scene_capabilities=capabilities)
                    bodies.append(json.loads(request.call_args.args[0].data))
        baseline,treatment = bodies
        self.assertIn('instances cannot be distinguished return ambiguous',baseline['messages'][0]['content'])
        self.assertNotIn('instances cannot be distinguished return ambiguous',treatment['messages'][0]['content'])
        a,b = [json.loads(body['messages'][1]['content']) for body in bodies]
        self.assertEqual(b.pop('inference_policy'),policy)
        self.assertEqual(a,b)
        self.assertEqual(a['scene_capabilities'],capabilities)

    def test_validators_remain_unchanged_in_both_contracts(self):
        cfg,nodes,edges,_ = contextual.scene_fixture('scene202')
        original = copy.deepcopy(cfg)
        with tempfile.TemporaryDirectory() as temporary:
            contracts = []
            for mode in ('baseline','bounded_inference'):
                cfg['inference_mode'] = mode
                root = Path(temporary)/mode
                pilot.prepare_round(root,{'round':1,'instruction':'Bring it here.'},
                    pilot.initial_state(nodes,cfg),nodes,'offline','https://invalid.test',cfg,edges)
                contracts.append(pilot.read_json(root/'execution_contract.json'))
            self.assertEqual(comparison.normalized_contract(contracts[0]),comparison.normalized_contract(contracts[1]))
            self.assertEqual(original['graspable_ids'],cfg['graspable_ids'])

    def test_coverage_judge_gets_treatment_not_private_reference(self):
        cfg,nodes,edges,_ = contextual.scene_fixture('scene202')
        cfg['inference_mode'] = 'bounded_inference'
        state = pilot.initial_state(nodes,cfg)
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)/'planning'
            pilot.prepare_round(folder,{'round':1,'instruction':'Bring it here.'},state,nodes,
                                'offline','https://invalid.test',cfg,edges)
            def fake_plan(args,reviewer):
                pilot.write_json(folder/'planning_summary.json',{'status':'planning_passed'})
                pilot.write_json(folder/'symbolic_review_final.json',{'predicted_end_state':state,
                    'status':'offline_fixture','goal_checks':[],'symbolic_trace':[]})
                pilot.write_json(folder/'skill_plan_final.json',{'subtask_plans':[]})
                return 0
            verdict = dict(complete=True,order_preserved=True,constraints_preserved=True,
                           no_unrequested_actions=True,reason='Offline policy propagation fixture only')
            with patch.object(flat_matrix_planner,'plan',side_effect=fake_plan), \
                    patch.object(flat_matrix_planner,'request_decision',return_value=verdict) as judge:
                self.assertEqual(flat_matrix_planner.plan_flat(self.args(Path(temporary)),folder,object()),0)
            payload = judge.call_args.args[4]
            self.assertEqual(payload['inference_policy']['mode'],'bounded_inference')
            self.assertNotIn('contents_status_reference',json.dumps(payload))

    def test_fact_probe_after_failed_plan_does_not_feed_back_or_change_memory(self):
        cfg,nodes,edges,_ = contextual.scene_fixture('scene202')
        cfg['inference_mode'] = 'bounded_inference'
        state = pilot.initial_state(nodes,cfg)
        instruction = next(c[3] for c in suite.cases() if c[0]=='H2')
        answer = {'contents_status':'unknown','contains_water':'unknown','thirst_satisfied':'not_established',
                  'evidence':[],'reason':'Offline synthetic answer only'}
        response = {'choices':[{'message':{'content':json.dumps(answer)}}]}
        with tempfile.TemporaryDirectory() as temporary:
            trial = Path(temporary)/'trial'
            folder = trial/'round_1/planning'
            pilot.prepare_round(folder,{'round':1,'instruction':instruction},state,nodes,
                                'offline','https://invalid.test',cfg,edges)
            pilot.write_json(folder/'skill_plan_final.json',{'subtask_plans':[],
                'inference_policy':{'mode':'bounded_inference','rules':'Must not reach probe'}})
            files = list(folder.rglob('*.json'))
            before = {str(p):pilot.sha(p) for p in files}
            args = self.args(trial)
            args.preflight_only = False
            row = {'case_id':'H2','instruction':instruction,'status':'planning_failed'}
            with patch.dict(os.environ,{'DASHSCOPE_API_KEY':'offline-test-not-a-real-key'}), \
                    patch.object(facts,'request_json',return_value=response) as request:
                result = facts.probe(args,trial,row)
            self.assertFalse(result['fed_back_to_planner'])
            self.assertTrue(all(result['authored_label_agreement'].values()))
            self.assertTrue(all(pilot.sha(p)==h for p,h in before.items()))
            body = json.loads(request.call_args.args[0].data)
            self.assertNotIn('inference_policy',body['messages'][1]['content'])
            self.assertNotIn('contents_status_reference',body['messages'][1]['content'])


if __name__ == '__main__':
    unittest.main()
