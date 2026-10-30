"""Offline reference feasibility and transaction tests; never online model results."""
import copy
import json
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import run_dialogue_chain as runner
import run_crossroom_task as pilot
from dialogue_chain_cases import episodes, compact_state, evaluate
from json_symbolic_review import SymbolicBackend
from sequential_proxy_core import SerialExecutor
from planning_start_contract import conversation_context
import flat_matrix_planner
import llm_graph_binding


def reference_replay(n,state,nodes,cfg):
    """Test-only feasible actions. Never imported by the paid entry point."""
    backend = SymbolicBackend(state,nodes,1.5,{'symbolic_scene_capabilities':{
        'graspable_ids':cfg['graspable_ids'],'placeable_pairs':cfg['placeable_pairs']}})
    executor = SerialExecutor(backend)
    for key in ('held_object_id','lifted','prepared_target_id'):
        setattr(executor,key,state[key])
    actions = []
    bindings = {f'id_{i}':{'element_ids':[i]} for i in [86,121,101,69,2]}
    def do(skill,**args):
        action = dict(step_id=f's_{len(actions)+1}',skill_id=skill,
            arguments={key:dict(role_id=f'id_{i}',element_id=i) for key,i in args.items()})
        executor.execute(action,{'context_bindings':bindings})
        actions.append(action)
    def nav(i):
        do('navigate_to_object',target=i)
    def pickup(i):
        nav(i)
        do('position_arm_for_object',target=i)
        do('grasp_object',target=i)
        do('lift_held_object',object=i)
    def place(i,destination):
        nav(destination)
        do('position_arm_for_object',target=destination)
        do('place_object_on_surface',object=i,surface=destination)
    def move(i,destination):
        pickup(i)
        place(i,destination)
    if n==1:
        move(121,2)
    elif n==2:
        if executor.held_object_id==121:
            place(121,2)
        move(86,2)
    elif n==3:
        move(86,69)
        pickup(121)
        nav(2)
    elif n==4:
        place(121,2)
        move(86,2)
    elif n==5:
        move(121,101)
    elif n==6:
        move(121,2)
    elif n==7:
        move(86,69)
        move(121,69)
    elif n==8:
        move(86,2)
    result = executor.snapshot()
    result['state_source'] = pilot.STATE_SOURCE
    plan = dict(action_count=len(actions),subtask_plans=[dict(subtask_id='offline',steps=actions)])
    return result,plan


class DialogueTests(unittest.TestCase):
    def args(self,**changes):
        values = dict(model='offline',base_url='https://invalid.test',preflight_only=False,
            inference_mode='bounded_inference',repeat=1,episode_ids=None,top_k=2,
            cosine_threshold=.35,cosine_margin=.05)
        values.update(changes)
        return SimpleNamespace(**values)

    def test_two_episodes_differ_only_first_utterance(self):
        a,b = episodes()
        self.assertEqual(a['rounds'][1:],b['rounds'][1:])
        self.assertNotEqual(a['rounds'][0],b['rounds'][0])
        self.assertEqual(len(a['rounds']),8)
        self.assertTrue(all('ID ' not in r['instruction'] for e in episodes() for r in e['rounds']))

    def test_all_eight_rounds_are_symbolically_feasible_with_inheritance(self):
        cfg,nodes,_,_ = runner.scene_fixture('scene202')
        state = pilot.initial_state(nodes,cfg)
        for n in range(1,9):
            end,plan = reference_replay(n,state,nodes,cfg)
            diagnostic = evaluate(n,state,end,plan,'explicit_start')
            self.assertTrue(diagnostic['reference_checks_all_passed'],diagnostic)
            end['backend']['memory_version']=n
            state = end
        self.assertEqual(state['backend']['objects']['86']['support_id'],2)
        self.assertEqual(state['backend']['objects']['121']['support_id'],69)

    def fake_plan(self,args,folder,encoder):
        cfg,nodes,_,_ = runner.scene_fixture('scene202')
        descriptor = pilot.read_json(folder/'round_input.json')
        state = pilot.read_json(folder/'runtime_state.json')
        end,plan = reference_replay(descriptor['round'],state,nodes,cfg)
        pilot.write_json(folder/'skill_plan_final.json',plan)
        pilot.write_json(folder/'object_alignment.json',{})
        pilot.write_json(folder/'llm_binding_proposal.json',{'bindings':[]})
        pilot.write_json(folder/'symbolic_review_final.json',{'predicted_end_state':end})
        pilot.write_json(folder/'planning_summary.json',{'status':'planning_passed'})
        return 0

    def test_history_versions_future_isolation_and_full_graph_projection(self):
        ep = episodes()[1]
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)/'ep'
            def replay(row,library,nodes,state,reach):
                cfg,_,_,_ = runner.scene_fixture('scene202')
                return {'predicted_end_state':reference_replay(row['round'],state,nodes,cfg)[0]}
            with patch.object(pilot,'plan_flat',side_effect=self.fake_plan), \
                    patch.object(pilot,'review_symbolic',side_effect=replay):
                summary = runner.run_episode(self.args(),ep,folder,object())
            self.assertEqual(summary['status'],'symbolic_chain_completed_semantics_unreviewed')
            self.assertEqual(summary['completed_round_count'],8)
            for n in range(1,9):
                planning = folder/f'round_{n}/planning'
                contract = pilot.read_json(planning/'execution_contract.json')
                history = contract['conversation_context']['symbolic_dialogue_history']
                self.assertEqual(len(history),n-1)
                self.assertEqual([e['round'] for e in history],list(range(1,n)))
                self.assertEqual(contract['current_runtime_state']['backend']['memory_version'],n-1)
                self.assertEqual(set(pilot.read_json(planning/'round_input.json')),{'round','instruction'})
                serialized = json.dumps(contract)
                self.assertNotIn('reference_checks',serialized)
                for future in ep['rounds'][n:]:
                    self.assertNotIn(future['instruction'],serialized)
                graph = pilot.read_json(planning/'input_memory/scene/scene_graph.json')
                self.assertEqual(len(graph),350)
            context = pilot.read_json(folder/'round_6/planning/execution_contract.json')['conversation_context']
            self.assertEqual(context['symbolic_dialogue_history'][0]['begin_state']['objects']['121']['support_id'],101)
            self.assertEqual(context['symbolic_dialogue_history'][-1]['end_state']['objects']['121']['support_id'],101)

    def test_failed_second_turn_does_not_commit_or_prepare_later_turns(self):
        ep = episodes()[1]
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)/'ep'
            def planner(args,p,encoder):
                if pilot.read_json(p/'round_input.json')['round']==2:
                    pilot.write_json(p/'planning_summary.json',{'status':'failed','error':'offline rejection'})
                    return 2
                return self.fake_plan(args,p,encoder)
            cfg,nodes,_,_ = runner.scene_fixture('scene202')
            with patch.object(pilot,'plan_flat',side_effect=planner),patch.object(pilot,'review_symbolic',
                side_effect=lambda row,lib,ns,st,reach:{'predicted_end_state':reference_replay(row['round'],st,ns,cfg)[0]}):
                summary = runner.run_episode(self.args(),ep,folder,object())
            self.assertEqual(summary['status'],'stopped_on_failed_round')
            self.assertEqual(summary['completed_round_count'],1)
            self.assertEqual(summary['later_rounds_skipped'],6)
            self.assertEqual(len(pilot.read_json(folder/'accepted_symbolic_history.json')),1)
            self.assertEqual(pilot.read_json(folder/'committed_symbolic_state.json')['backend']['memory_version'],1)
            self.assertFalse((folder/'round_3').exists())

    def test_diagnostic_mismatch_is_not_oracle_repair_or_commit_gate(self):
        ep = copy.deepcopy(episodes()[1])
        ep['rounds'] = ep['rounds'][:2]
        cfg,nodes,_,_ = runner.scene_fixture('scene202')
        with tempfile.TemporaryDirectory() as temporary:
            with patch.object(pilot,'plan_flat',side_effect=self.fake_plan),patch.object(pilot,'review_symbolic',
                side_effect=lambda row,lib,ns,st,reach:{'predicted_end_state':reference_replay(row['round'],st,ns,cfg)[0]}), \
                patch.object(runner,'evaluate',return_value={'reference_checks_all_passed':False,'ordered_placement_events':[]}):
                summary = runner.run_episode(self.args(),ep,Path(temporary)/'ep',object())
            self.assertEqual(summary['completed_round_count'],2)
            self.assertEqual(summary['first_diagnostic_mismatch_round'],1)

    def test_public_history_validation_rejects_future_and_observation_claims(self):
        cfg,nodes,edges,_ = runner.scene_fixture('scene202')
        state = pilot.initial_state(nodes,cfg)
        end,plan = reference_replay(1,state,nodes,cfg)
        end['backend']['memory_version']=1
        entry = runner.history_entry(episodes()[1]['rounds'][0],state,end,plan,{'bindings':[]})
        cfg['conversation_context']['symbolic_dialogue_history']=[entry]
        self.assertEqual(len(conversation_context(cfg,nodes)['symbolic_dialogue_history']),1)
        bad = copy.deepcopy(cfg)
        bad['conversation_context']['symbolic_dialogue_history'][0]['source']='robot_observation'
        with self.assertRaises(ValueError):
            conversation_context(bad,nodes)
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError,'version mismatch'):
                runner.prepare(Path(temporary),{'round':2,'instruction':'current'},state,[entry],cfg,nodes,edges,self.args())

    def test_preflight_does_not_call_planners_or_models(self):
        with tempfile.TemporaryDirectory() as temporary:
            with patch.object(pilot,'LocalEncoder') as encoder,patch.object(pilot,'plan_flat') as planner:
                code = runner.run(self.args(output_dir=Path(temporary)/'batch',preflight_only=True))
            self.assertEqual(code,0)
            encoder.assert_not_called()
            planner.assert_not_called()
            summary = pilot.read_json(Path(temporary)/'batch/dialogue_summary.json')
            self.assertEqual(summary['planned_round_count'],16)
            self.assertEqual(summary['completed_round_count'],0)
            self.assertEqual(len(summary['episodes']),2)

    def test_failed_episode_does_not_skip_independent_control(self):
        cfg,nodes,_,_ = runner.scene_fixture('scene202')
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)/'batch'
            def planner(args,p,encoder):
                if 'natural_start_r01' in str(p):
                    pilot.write_json(p/'planning_summary.json',{'status':'failed','error':'offline ambiguous'})
                    return 2
                return self.fake_plan(args,p,encoder)
            args = self.args(output_dir=root,embedding_python=Path(__file__),embedding_model_dir=Path(temporary))
            with patch.dict(os.environ,{'DASHSCOPE_API_KEY':'offline-fixture-not-a-real-key'}), \
                    patch.object(pilot,'LocalEncoder') as encoder,patch.object(pilot,'model_fingerprint',return_value='offline'), \
                    patch.object(pilot,'plan_flat',side_effect=planner),patch.object(pilot,'review_symbolic',
                        side_effect=lambda row,lib,ns,st,reach:{'predicted_end_state':reference_replay(row['round'],st,ns,cfg)[0]}):
                encoder.return_value.stats.return_value={}
                self.assertEqual(runner.run(args),0)
            result = pilot.read_json(root/'dialogue_summary.json')
            self.assertEqual(len(result['episodes']),2)
            self.assertEqual(result['episodes'][0]['completed_round_count'],0)
            self.assertEqual(result['episodes'][0]['later_rounds_skipped'],7)
            self.assertEqual(result['episodes'][1]['completed_round_count'],8)
            self.assertEqual(result['completed_round_count'],8)

    def test_planner_input_mutation_never_commits(self):
        ep = episodes()[1]
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)/'ep'
            def mutate(args,p,encoder):
                code = self.fake_plan(args,p,encoder)
                state = pilot.read_json(p/'runtime_state.json')
                state['held_object_id']=86
                pilot.write_json(p/'runtime_state.json',state)
                return code
            with patch.object(pilot,'plan_flat',side_effect=mutate):
                summary = runner.run_episode(self.args(),ep,folder,object())
            self.assertEqual(summary['completed_round_count'],0)
            self.assertEqual(pilot.read_json(folder/'committed_symbolic_state.json')['backend']['memory_version'],0)
            self.assertFalse((folder/'round_2').exists())

    def test_history_forwarded_to_binding_and_whole_judge(self):
        cfg,nodes,edges,_ = runner.scene_fixture('scene202')
        initial = pilot.initial_state(nodes,cfg)
        state,plan = reference_replay(1,initial,nodes,cfg)
        state['backend']['memory_version']=1
        history = [runner.history_entry(episodes()[1]['rounds'][0],initial,state,plan,{'bindings':[]})]
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)/'round_2/planning'
            runner.prepare(folder,episodes()[1]['rounds'][1],state,history,cfg,nodes,edges,self.args())
            context = pilot.read_json(folder/'execution_contract.json')['conversation_context']
            response = {'choices':[{'message':{'content':'{"bindings": []}'}}]}
            with patch.dict(os.environ,{'OFFLINE_KEY':'offline-key'}), \
                    patch.object(llm_graph_binding,'request_json',return_value=response) as request:
                llm_graph_binding.request_bindings([],[],'offline','https://invalid.test','OFFLINE_KEY',
                                                   conversation_context=context)
            payload = json.loads(json.loads(request.call_args.args[0].data)['messages'][1]['content'])
            self.assertEqual(payload['conversation_context']['symbolic_dialogue_history'],history)
            def fake_plan(args,reviewer):
                pilot.write_json(folder/'planning_summary.json',{'status':'planning_passed'})
                pilot.write_json(folder/'symbolic_review_final.json',{'predicted_end_state':state,
                    'status':'offline_fixture','goal_checks':[],'symbolic_trace':[]})
                pilot.write_json(folder/'skill_plan_final.json',{'subtask_plans':[]})
                return 0
            verdict = dict(complete=True,order_preserved=True,constraints_preserved=True,
                no_unrequested_actions=True,reason='Offline propagation fixture only')
            with patch.object(flat_matrix_planner,'plan',side_effect=fake_plan), \
                    patch.object(flat_matrix_planner,'request_decision',return_value=verdict) as judge:
                self.assertEqual(flat_matrix_planner.plan_flat(self.args(),folder,object()),0)
            self.assertEqual(judge.call_args.args[4]['conversation_context']['symbolic_dialogue_history'],history)


if __name__=='__main__':
    unittest.main()
