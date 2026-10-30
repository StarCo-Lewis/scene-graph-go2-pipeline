"""Offline isolation and public-input checks, not model capability evidence."""
import copy
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import run_contextual_ambiguity as experiment
import run_crossroom_task as pilot
import flat_matrix_planner
import llm_graph_binding
from planning_start_contract import public_round, conversation_context


class ContextualTests(unittest.TestCase):
    def test_binding_request_contains_public_context(self):
        context = {'user_location':{'room_name':'office','near_scene_element_id':133}}
        response = {'choices':[{'message':{'content':'{"bindings": []}'}}]}
        with patch.dict(os.environ,{'OFFLINE_BINDING_KEY':'not-a-real-key'}), \
                patch.object(llm_graph_binding,'request_json',return_value=response) as request:
            llm_graph_binding.request_bindings([],[],'offline','https://invalid.test','OFFLINE_BINDING_KEY',
                                               conversation_context=context)
        body = json.loads(request.call_args.args[0].data)
        payload = json.loads(body['messages'][1]['content'])
        self.assertEqual(payload['conversation_context'],context)

    def test_whole_instruction_judge_receives_same_context(self):
        cfg,nodes,edges,_ = experiment.scene_fixture('scene200')
        state = pilot.initial_state(nodes,cfg)
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)/'round_1/planning'
            pilot.prepare_round(folder,{'round':1,'instruction':'Bring it here.'},state,nodes,
                                'offline','https://invalid.test',cfg,edges)
            def fake_plan(args,reviewer):
                pilot.write_json(folder/'planning_summary.json',{'status':'planning_passed'})
                pilot.write_json(folder/'symbolic_review_final.json',{'predicted_end_state':state,
                    'status':'offline_fixture','goal_checks':[],'symbolic_trace':[]})
                pilot.write_json(folder/'skill_plan_final.json',{'subtask_plans':[]})
                return 0
            verdict = dict(complete=True,order_preserved=True,constraints_preserved=True,
                           no_unrequested_actions=True,reason='Offline context propagation fixture only')
            args = SimpleNamespace(top_k=2,cosine_threshold=.35,cosine_margin=.05)
            with patch.object(flat_matrix_planner,'plan',side_effect=fake_plan), \
                    patch.object(flat_matrix_planner,'request_decision',return_value=verdict) as judge:
                self.assertEqual(flat_matrix_planner.plan_flat(args,folder,object()),0)
            self.assertEqual(judge.call_args.args[4]['conversation_context'],cfg['conversation_context'])

    def test_round_input_is_allowlisted_not_test_answers(self):
        source = {'round':1,'instruction':'Bring it here.','expected_end':{'held_object_id':152},
                  'expected_support_by_object':{'152':133},'hidden_answer':'DO_NOT_SEND'}
        self.assertEqual(public_round(source),{'round':1,'instruction':'Bring it here.'})

    def test_context_is_validated_copied_and_does_not_move_robot(self):
        for scene in ('scene200','scene202'):
            cfg,nodes,_,_ = experiment.scene_fixture(scene)
            state = pilot.initial_state(nodes,cfg)
            before = copy.deepcopy(state)
            context = conversation_context(cfg,nodes)
            self.assertNotEqual(context['user_location']['room_name'],state['backend']['robot_room_hypothesis'])
            self.assertEqual(state,before)
            context['user_activity'] = 'Changed test copy'
            self.assertNotEqual(context,cfg['conversation_context'])
            cfg['conversation_context']['hidden_answer'] = 133
            with self.assertRaises(ValueError):
                conversation_context(cfg,nodes)

    def test_legacy_case_assertions_stay_outside_planning(self):
        cfg,nodes,edges,_ = pilot.load_case(pilot.HERE/'scenarios/scene202_crossroom_6.json')
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)/'round_1/planning'
            pilot.prepare_round(folder,cfg['rounds'][0],pilot.initial_state(nodes,cfg),nodes,
                                'offline','https://invalid.test',cfg,edges)
            self.assertTrue((folder.parent/'evaluation/task_assertions.json').exists())
            self.assertEqual(set(pilot.read_json(folder/'round_input.json')),{'round','instruction'})
            self.assertNotIn('expected_end_test_assertion',pilot.read_json(folder/'execution_contract.json'))

    def test_preflight_all_eight_has_no_models_no_private_answers(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)/'batch'
            args = SimpleNamespace(output_dir=root,preflight_only=True,case_ids=None,repeat=1,
                                   model='offline',base_url='https://invalid.test')
            with patch.object(pilot,'LocalEncoder') as encoder,patch.object(pilot,'plan_flat') as planner:
                self.assertEqual(experiment.run(args),0)
                encoder.assert_not_called()
                planner.assert_not_called()
            summary = pilot.read_json(root/'batch_summary.json')
            self.assertEqual(summary['status'],'preflight_completed')
            self.assertEqual(summary['finished_trial_count'],8)
            self.assertEqual(summary['symbolically_valid_trial_count'],0)
            self.assertIsNone(summary['semantic_success_rate'])
            for row in summary['trials']:
                folder = root/row['planning_path']
                contract = pilot.read_json(folder/'execution_contract.json')
                self.assertEqual(contract['conversation_context'],row['conversation_context'])
                def keys(value):
                    if isinstance(value,dict):
                        return set(value).union(*(keys(v) for v in value.values()))
                    if isinstance(value,list):
                        return set().union(*(keys(v) for v in value))
                    return set()
                self.assertFalse(keys(contract) & {'expected_end_test_assertion','expected_end',
                    'expected_navigation_targets','expected_support_by_object','expected_robot_room'})
                self.assertEqual(set(pilot.read_json(folder/'round_input.json')),{'round','instruction'})
                state = pilot.read_json(folder/'runtime_state.json')
                self.assertEqual(state['backend']['memory_version'],0)
                self.assertIsNone(state['held_object_id'])
                self.assertFalse((folder.parent/'symbolic_projection_after.json').exists())

    def test_failure_does_not_skip_independent_trials_or_commit_memory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)/'batch'
            args = SimpleNamespace(output_dir=root,preflight_only=False,case_ids=['c01','c02'],repeat=1,
                model='offline',base_url='https://invalid.test',embedding_python=Path(__file__),
                embedding_model_dir=Path(temporary))
            seen = []
            def fail_planning(options,folder,encoder):
                self.assertEqual(pilot.read_json(folder/'runtime_state.json')['backend']['memory_version'],0)
                seen.append(folder)
                if len(seen) == 1:
                    raise ValueError('Synthetic offline failure, not model evidence')
                pilot.write_json(folder/'planning_summary.json',{'status':'failed','error':'Synthetic rejection'})
                return 2
            with patch.dict(os.environ,{'DASHSCOPE_API_KEY':'offline-test-not-a-real-key'}), \
                    patch.object(pilot,'LocalEncoder') as encoder, \
                    patch.object(pilot,'model_fingerprint',return_value='offline-fingerprint'), \
                    patch.object(pilot,'plan_flat',side_effect=fail_planning):
                encoder.return_value.stats.return_value = {}
                self.assertEqual(experiment.run(args),0)
            summary = pilot.read_json(root/'batch_summary.json')
            self.assertEqual(len(seen),2)
            self.assertEqual([r['status'] for r in summary['trials']],['trial_error','planning_failed'])
            self.assertEqual(summary['symbolically_valid_trial_count'],0)
            self.assertIsNone(summary['semantic_success_rate'])


if __name__ == '__main__':
    unittest.main()
