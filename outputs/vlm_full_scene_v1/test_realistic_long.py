"""Offline feasibility/inheritance checks, NEVER imported by the online runner."""
import copy
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import run_realistic_long as runner
import dialogue_execution_feedback as feedback
import run_crossroom_task as pilot
from run_contextual_ambiguity import scene_fixture
from realistic_long_cases import episodes,evaluate
from json_symbolic_review import SymbolicBackend
from sequential_proxy_core import SerialExecutor


def reference(ep,row,state,nodes,cfg):
    """Test-only witness, not a required model answer or online action generator."""
    backend=SymbolicBackend(state,nodes,1.5,{'symbolic_scene_capabilities':{
        'graspable_ids':cfg['graspable_ids'],'placeable_pairs':cfg['placeable_pairs']}})
    executor=SerialExecutor(backend)
    for key in ('held_object_id','lifted','prepared_target_id'): setattr(executor,key,state[key])
    bound=ep['tracked_object_ids']+[p[1] for p in cfg['placeable_pairs']]
    bindings={f'i{i}':{'element_ids':[i]} for i in set(bound)}
    alignment={'context_bindings':bindings}
    actions=[]
    visits=copy.deepcopy(row['expected_visits'])
    def do(skill,**arguments):
        step=dict(step_id=f'offline_{len(actions)}',skill_id=skill,
            arguments={k:{'role_id':f'i{i}','element_id':i} for k,i in arguments.items()})
        executor.execute(step,alignment)
        actions.append(step)
    def nav(i):
        if visits and visits[0]==[i,executor.held_object_id]: visits.pop(0)
        do('navigate_to_object',target=i)
    def pickup(i):
        if executor.held_object_id==i: return
        nav(i)
        do('position_arm_for_object',target=i)
        do('grasp_object',target=i)
        do('lift_held_object',object=i)
    for obj,surface in row['expected_placement_events']:
        pickup(obj)
        while visits and visits[0][1]==obj: nav(visits[0][0])
        nav(surface)
        do('position_arm_for_object',target=surface)
        do('place_object_on_surface',object=obj,surface=surface)
    if row['expected_end']['held_object_id'] is not None:
        pickup(row['expected_end']['held_object_id'])
    while visits:
        if visits[0][1]!=executor.held_object_id: raise AssertionError('Infeasible authored visit payload')
        nav(visits[0][0])
    end=executor.snapshot()
    end['state_source']=pilot.STATE_SOURCE
    plan={'subtask_plans':[{'subtask_id':'offline_witness','steps':actions}],'action_count':len(actions)}
    return end,plan,alignment


class RealisticLongTests(unittest.TestCase):
    def args(self,root,preflight=False):
        return SimpleNamespace(output_dir=root,preflight_only=preflight,episode_ids=None,repeat=1,
            inference_mode='bounded_inference',model='offline',base_url='https://invalid.test',
            embedding_python=Path(__file__),embedding_model_dir=root.parent,
            top_k=2,cosine_threshold=.35,cosine_margin=.05)

    def test_manifest_covers_two_scenes_six_locations_and_sixty_rounds(self):
        suite=episodes()
        self.assertEqual(len(suite),10)
        self.assertEqual(sum(len(e['rounds']) for e in suite),60)
        self.assertEqual(len({(e['scene'],e['user_location']['room_name']) for e in suite}),6)
        for scene in ('scene200','scene202'):
            self.assertEqual(sum(e['scene']==scene for e in suite),5)
        self.assertTrue(all('ID ' not in r['instruction'] for e in suite for r in e['rounds']))
        self.assertEqual(len({r['instruction'] for e in suite for r in e['rounds']}),60)

    def test_all_ten_authored_chains_have_proxy_feasible_witnesses(self):
        for ep in episodes():
            cfg,nodes,_,_=scene_fixture(ep['scene'])
            state=pilot.initial_state(nodes,cfg)
            for row in ep['rounds']:
                end,plan,_=reference(ep,row,state,nodes,cfg)
                audit=evaluate(row['round'],state,end,plan,ep['episode_id'])
                self.assertTrue(audit['reference_checks_all_passed'],(ep['episode_id'],row['round'],audit))
                self.assertTrue(audit['dependency_reference_matched'])
                self.assertTrue(audit['strict_id_visit_sequence_matched'],(ep['episode_id'],row['round'],audit))
                end['backend']['memory_version']=row['round']
                state=end

    def test_ten_first_inputs_have_correct_user_locations_and_no_private_or_future_input(self):
        with tempfile.TemporaryDirectory() as temporary:
            args=self.args(Path(temporary)/'pf',True)
            with patch.object(feedback,'propose_once') as proposer:
                self.assertEqual(runner.run(args),0)
            proposer.assert_not_called()
            result=pilot.read_json(args.output_dir/'dialogue_summary.json')
            self.assertEqual(result['planned_round_count'],60)
            self.assertEqual(result['processed_round_count'],0)
            for ep in episodes():
                folder=args.output_dir/'episodes'/f"{ep['episode_id']}_r01/round_1/planning"
                contract=pilot.read_json(folder/'execution_contract.json')
                self.assertEqual(contract['conversation_context']['user_location'],ep['user_location'])
                public=json.dumps(contract)
                self.assertNotIn('expected_placement_events',public)
                self.assertNotIn('expected_begin',public)
                for future in ep['rounds'][1:]: self.assertNotIn(future['instruction'],public)
                self.assertEqual(len(pilot.read_json(folder/'input_memory/scene/scene_graph.json')),
                                 260 if ep['scene']=='scene200' else 350)

    def test_sixty_turn_mock_rollout_commits_scene_specific_objects_and_audits_afterward(self):
        suite={e['episode_id']:e for e in episodes()}
        def proposer(args,folder,encoder):
            eid=folder.parent.parent.name.rsplit('_r',1)[0]
            ep=suite[eid]
            row=pilot.read_json(folder/'round_input.json')
            cfg,nodes,_,_=scene_fixture(ep['scene'])
            state=pilot.read_json(folder/'runtime_state.json')
            context=pilot.read_json(folder/'execution_contract.json')['conversation_context']
            self.assertEqual(len(context['proxy_execution_history']),row['round']-1)
            serialized=json.dumps(context)
            for future in ep['rounds'][row['round']:]: self.assertNotIn(future['instruction'],serialized)
            _,plan,alignment=reference(ep,ep['rounds'][row['round']-1],state,nodes,cfg)
            pilot.write_json(folder/'skill_plan_proposed.json',plan)
            return {'status':'proposal_ready_not_semantically_validated'},plan,alignment
        with tempfile.TemporaryDirectory() as temporary:
            args=self.args(Path(temporary)/'batch')
            with patch.dict(os.environ,{'DASHSCOPE_API_KEY':'offline-not-a-key'}), \
                    patch.object(pilot,'LocalEncoder') as encoder, \
                    patch.object(pilot,'model_fingerprint',return_value='offline'), \
                    patch.object(feedback,'propose_once',side_effect=proposer):
                encoder.return_value.stats.return_value={}
                self.assertEqual(runner.run(args),0)
            result=pilot.read_json(args.output_dir/'evaluation/error_localization.json')
            self.assertEqual(result['processed_round_count'],60)
            self.assertFalse(any(r['observed_deviation'] for r in result['rows']))
            for ep in suite.values():
                checkpoint=pilot.read_json(args.output_dir/'episodes'/f"{ep['episode_id']}_r01/memory_checkpoint.json")
                self.assertEqual(set(checkpoint['history'][-1]['end_state']['objects']),
                                 {str(i) for i in ep['tracked_object_ids']})
                self.assertEqual(checkpoint['state']['backend']['memory_version'],6)

    def test_failed_history_dependency_is_flagged_not_treated_as_independent_ground_truth(self):
        ep=episodes()[0]
        cfg,nodes,_,_=scene_fixture(ep['scene'])
        state=pilot.initial_state(nodes,cfg)
        diagnostic=evaluate(2,state,state,{'subtask_plans':[]},ep['episode_id'])
        self.assertFalse(diagnostic['dependency_reference_matched'])
        self.assertFalse(diagnostic['reference_checks_all_passed'])

    def test_failure_location_keeps_action_parameters_and_next_turn_runs(self):
        ep=episodes()[3]
        ep=copy.deepcopy(ep)
        ep['rounds']=ep['rounds'][:2]
        cfg,nodes,_,_=scene_fixture(ep['scene'])
        def proposer(args,folder,encoder):
            row=pilot.read_json(folder/'round_input.json')
            state=pilot.read_json(folder/'runtime_state.json')
            if row['round']==1:
                _,plan,alignment=reference(ep,ep['rounds'][0],state,nodes,cfg)
            else:
                # Deliberately omit placing the held bottle before grasping the book.
                alignment={'context_bindings':{'book':{'element_ids':[86]}}}
                steps=[{'step_id':f'bad_{n}','action_id':f'bad_{n}','action_ordinal':n,'skill_id':skill,
                    'arguments':{'target':{'role_id':'book','element_id':86}}}
                    for n,skill in enumerate(('navigate_to_object','position_arm_for_object','grasp_object'),1)]
                # The planned visit below must not appear as an executed visit after the failure.
                steps.append({'step_id':'skipped','action_id':'bad_4','action_ordinal':4,
                    'skill_id':'navigate_to_object','arguments':{'target':{'role_id':'book','element_id':86}}})
                plan={'subtask_plans':[{'steps':steps}],'action_count':4}
            pilot.write_json(folder/'skill_plan_proposed.json',plan)
            return {'status':'proposal_ready_not_semantically_validated'},plan,alignment
        with tempfile.TemporaryDirectory() as temporary:
            args=self.args(Path(temporary)/'ep')
            args.feedback_evaluator=evaluate
            with patch.object(feedback,'propose_once',side_effect=proposer):
                result=feedback.run_episode_feedback(args,ep,args.output_dir,object())
            self.assertEqual(result['processed_round_count'],2)
            failure=pilot.read_json(args.output_dir/'round_2/execution_feedback.json')['failure']
            self.assertEqual(failure['action_id'],'bad_3')
            self.assertEqual(failure['arguments']['target']['element_id'],86)
            self.assertEqual(failure['error'],'GRIPPER_NOT_EMPTY')
            state=pilot.read_json(args.output_dir/'memory_checkpoint.json')['state']
            self.assertEqual(state['held_object_id'],121)
            self.assertEqual(state['backend']['memory_version'],2)
            audit=pilot.read_json(args.output_dir/'round_2/evaluation/post_rollout_diagnostics.json')
            self.assertEqual(len(audit['navigation_visits']),1)
            self.assertEqual(len(audit['proposed_navigation_visits']),2)


if __name__=='__main__':
    unittest.main()
