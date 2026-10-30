"""Offline proxy rollout tests; no paid calls or robot execution."""
import copy
import json
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import dialogue_execution_feedback as feedback
import run_dialogue_chain as dialogue
import run_crossroom_task as pilot
import test_dialogue_state_recovery as fixtures
import test_dialogue_chain as chain_fixtures
from dialogue_chain_cases import episodes,compact_state
from planning_start_contract import conversation_context
from sequential_proxy_core import ExecutionRejected
from state_aware_planning import close_runtime_objects
from action_matrix import make_codebook,decode_matrix
from match_skills import build_result


class FeedbackTests(unittest.TestCase):
    def setUp(self):
        self.fixture=fixtures.RecoveryTests()
        self.cfg,self.nodes,self.edges,self.held,self.instruction,self.args=self.fixture.inputs()
        self.args.preflight_only=False
        self.args.inference_mode='bounded_inference'
        self.args.memory_mode='execution_feedback'

    def library(self,root,state=None):
        feedback.prepare_feedback(root,dict(round=1,instruction=self.instruction),state or self.held,
            [],self.cfg,self.nodes,self.edges,self.args)
        return pilot.read_json(root/'execution_skill_library.json')

    def plan(self,library,recovered=False):
        alignment=self.fixture.alignment(self.instruction,recovered)
        alignment,_=close_runtime_objects(alignment,library,self.held,self.nodes)
        book=make_codebook(alignment,library)
        proposal=decode_matrix(self.fixture.matrix(book,recovered),book,alignment,library)
        return build_result(alignment,library,proposal),alignment

    def test_failure_keeps_navigation_effects_and_does_not_grasp_new_object(self):
        with tempfile.TemporaryDirectory() as temporary:
            library=self.library(Path(temporary))
            plan,alignment=self.plan(library)
            before=copy.deepcopy(self.held)
            result=feedback.execute_proxy(plan,alignment,library,self.nodes,self.held)
            self.assertEqual(result['status'],'proxy_action_failed')
            self.assertEqual(result['error'],'GRIPPER_NOT_EMPTY')
            self.assertEqual(result['attempted_action_count'],3)
            self.assertEqual(result['successful_action_count'],2)
            self.assertEqual(result['skipped_action_count'],4)
            end=result['resulting_state']
            self.assertEqual(end['held_object_id'],121)
            self.assertEqual(end['prepared_target_id'],86)
            self.assertEqual(end['backend']['robot_room_hypothesis'],'living_room')
            self.assertEqual(end['backend']['objects']['121']['room_hypothesis'],'living_room')
            self.assertEqual(self.held,before)

    def test_successful_proxy_actions_not_forced_goal_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            library=self.library(Path(temporary))
            plan,alignment=self.plan(library,True)
            # A false desired goal must not stop action dispatch or assign its imagined state.
            alignment['task_decomposition']['goal_state'][0]['object_role']='book'
            result=feedback.execute_proxy(plan,alignment,library,self.nodes,self.held)
            self.assertEqual(result['status'],'proxy_actions_completed')
            self.assertIsNone(result['resulting_state']['held_object_id'])
            self.assertEqual(result['resulting_state']['backend']['objects']['86']['support_id'],2)

    def test_partial_side_effect_of_failed_place_is_retained(self):
        with tempfile.TemporaryDirectory() as temporary:
            library=self.library(Path(temporary))
            plan,alignment=self.plan(library,True)
            def failed_place(backend,obj,surface):
                backend.state['objects'][str(obj)]['support_id']=None
                return {'status':'failed_support'}
            with patch.object(feedback.SymbolicBackend,'place',failed_place):
                result=feedback.execute_proxy(plan,alignment,library,self.nodes,self.held)
            self.assertEqual(result['status'],'proxy_action_failed')
            self.assertEqual(result['error'],'PLACEMENT_SUPPORT_NOT_VERIFIED')
            self.assertIsNone(result['resulting_state']['held_object_id'])
            self.assertFalse(result['resulting_state']['lifted'])

    def test_proposal_path_has_one_matrix_no_review_replan_or_critique(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            self.library(root)
            def request(alignment,library,book,*a,**kw):
                self.assertNotIn('validation_error',kw)
                return self.fixture.matrix(book)
            with patch.object(feedback,'decompose_and_bind',return_value=self.fixture.alignment(self.instruction)) as decompose, \
                    patch.object(feedback,'request_matrix',side_effect=request) as matrix, \
                    patch.object(pilot,'review_symbolic',side_effect=AssertionError('No preexecution review')):
                summary,plan,alignment=feedback.propose_once(self.args,root,fixtures.Encoder())
            self.assertEqual(summary['status'],'proposal_ready_not_semantically_validated')
            self.assertEqual(decompose.call_count,1)
            self.assertEqual(decompose.call_args.args[0].alignment_validation_attempts,1)
            self.assertEqual(matrix.call_count,1)
            self.assertEqual(summary['runtime_redecomposition_count'],0)
            self.assertFalse(summary['whole_instruction_review_requested'])
            self.assertFalse(summary['redundancy_review_requested'])
            self.assertIsNotNone(plan)

    def test_cosine_conflict_recorded_but_does_not_block_valid_protocol(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            self.library(root)
            def retrieval(alignment,*a,**kw):
                return {'subtasks':[{'subtask_id':s['subtask_id'],
                    'candidate_skill_ids':['navigate_to_object']} for s in alignment['task_decomposition']['subtasks']]}
            with patch.object(feedback,'decompose_and_bind',return_value=self.fixture.alignment(self.instruction)), \
                    patch.object(feedback,'candidates',side_effect=retrieval),patch.object(feedback,'request_matrix',
                        side_effect=lambda a,l,b,*p,**k:self.fixture.matrix(b)):
                summary,plan,_=feedback.propose_once(self.args,root,fixtures.Encoder())
            self.assertIsNotNone(plan)
            self.assertTrue(summary['cosine_conflicts'])

    def test_malformed_matrix_records_no_plan_no_corrected_proposal(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            self.library(root)
            with patch.object(feedback,'decompose_and_bind',return_value=self.fixture.alignment(self.instruction)), \
                    patch.object(feedback,'request_matrix',return_value={'matrix':[[1,999,1,0]]}) as matrix:
                summary,plan,_=feedback.propose_once(self.args,root,fixtures.Encoder())
            self.assertIsNone(plan)
            self.assertEqual(summary['status'],'proposal_unavailable')
            self.assertEqual(matrix.call_count,1)

    def test_alignment_transport_failure_remains_infrastructure_error(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            self.library(root)
            def unavailable(args,descriptor,summary):
                summary['upstream_error']={'category':'infrastructure_error','message':'offline HTTP 401'}
                raise ValueError('LLM decomposition/object binding failed: offline HTTP 401')
            with patch.object(feedback,'decompose_and_bind',side_effect=unavailable), \
                    patch.object(feedback,'request_matrix') as matrix:
                summary,plan,_=feedback.propose_once(self.args,root,fixtures.Encoder())
            self.assertEqual(summary['status'],'infrastructure_error')
            self.assertIsNone(plan)
            matrix.assert_not_called()

    def test_failure_commits_and_next_turn_continues_from_result_not_intended_goal(self):
        episode=dict(episode_id='natural_start',scene='scene202',rounds=[
            {'round':1,'instruction':'First instruction'},{'round':2,'instruction':'Second instruction'}])
        observations=[]
        def proposer(args,folder,encoder):
            state=pilot.read_json(folder/'runtime_state.json')
            contract=pilot.read_json(folder/'execution_contract.json')
            observations.append((copy.deepcopy(state),contract))
            self.assertNotIn('Second instruction',json.dumps(contract) if len(observations)==1 else '')
            if len(observations)==2:
                self.assertEqual(state['held_object_id'],121)
                self.assertEqual(state['backend']['robot_room_hypothesis'],'living_room')
                self.assertEqual(contract['conversation_context']['proxy_execution_history'][0]['error'],'GRIPPER_NOT_EMPTY')
                self.assertEqual(len(contract['conversation_context']['proxy_execution_history'][0]['actions']),2)
                return {'status':'proposal_unavailable','error':'offline unresolved'},None,None
            lib=pilot.read_json(folder/'execution_skill_library.json')
            plan,alignment=self.plan(lib)
            pilot.write_json(folder/'skill_plan_proposed.json',plan)
            return {'status':'proposal_ready_not_semantically_validated'},plan,alignment
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)/'episode'
            with patch.object(pilot,'initial_state',return_value=copy.deepcopy(self.held)), \
                    patch.object(feedback,'propose_once',side_effect=proposer):
                summary=feedback.run_episode_feedback(self.args,episode,root,object())
            self.assertEqual(summary['status'],'dialogue_rollout_completed_not_success_claim')
            self.assertEqual(summary['processed_round_count'],2)
            self.assertEqual(summary['failed_round_count'],2)
            self.assertEqual(summary['successful_execution_round_count'],0)
            checkpoint=pilot.read_json(root/'memory_checkpoint.json')
            self.assertEqual(checkpoint['state']['backend']['memory_version'],2)
            self.assertEqual(checkpoint['state']['held_object_id'],121)
            self.assertEqual(len(checkpoint['history']),2)
            self.assertEqual(checkpoint['history'][1]['actions'],[])
            self.assertTrue((root/'evaluation/final_audit.json').is_file())
            self.assertFalse((root/'round_1/evaluation/private_diagnostics.json').exists())

    def test_reference_audit_runs_only_after_all_turns_and_cannot_change_memory(self):
        episode=dict(episode_id='natural_start',scene='scene202',rounds=[
            {'round':1,'instruction':'first'},{'round':2,'instruction':'second'}])
        seen=[]
        def unavailable(*a):
            seen.append('propose')
            return {'status':'proposal_unavailable','error':'offline'},None,None
        def audit(*a):
            self.assertEqual(seen,['propose','propose'])
            raise ValueError('offline private audit error')
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)/'episode'
            with patch.object(feedback,'propose_once',side_effect=unavailable), \
                    patch.object(feedback,'evaluate',side_effect=audit):
                summary=feedback.run_episode_feedback(self.args,episode,root,object())
            self.assertEqual(summary['processed_round_count'],2)
            state=pilot.read_json(root/'memory_checkpoint.json')['state']
            self.assertEqual(state['backend']['memory_version'],2)
            self.assertIsNone(state['held_object_id'])

    def test_preflight_no_calls_and_unambiguous_mode_counters(self):
        with tempfile.TemporaryDirectory() as temporary:
            self.args.output_dir=Path(temporary)/'batch'
            self.args.preflight_only=True
            self.args.episode_ids=None
            self.args.repeat=1
            with patch.object(feedback,'propose_once') as proposer,patch.object(pilot,'LocalEncoder') as encoder:
                self.assertEqual(dialogue.run(self.args),0)
            proposer.assert_not_called()
            encoder.assert_not_called()
            summary=pilot.read_json(self.args.output_dir/'dialogue_summary.json')
            self.assertEqual(summary['processed_round_count'],0)
            self.assertNotIn('completed_round_count',summary)
            self.assertEqual(summary['memory_mode'],'execution_feedback')
            self.assertFalse(summary['planning_semantic_gate'])
            self.assertEqual(summary['planned_round_count'],16)

    def test_full_batch_counter_and_final_audit_with_mock_planner(self):
        def proposer(args,folder,encoder):
            cfg,nodes,_,_=feedback.scene_fixture('scene202')
            descriptor=pilot.read_json(folder/'round_input.json')
            state=pilot.read_json(folder/'runtime_state.json')
            _,plan=chain_fixtures.reference_replay(descriptor['round'],state,nodes,cfg)
            alignment={'context_bindings':{f'id_{i}':{'element_ids':[i]} for i in (86,121,101,69,2)}}
            pilot.write_json(folder/'skill_plan_proposed.json',plan)
            return {'status':'proposal_ready_not_semantically_validated'},plan,alignment
        with tempfile.TemporaryDirectory() as temporary:
            self.args.output_dir=Path(temporary)/'batch'
            self.args.episode_ids=None
            self.args.repeat=1
            self.args.embedding_python=Path(__file__)
            self.args.embedding_model_dir=Path(temporary)
            with patch.dict(os.environ,{'DASHSCOPE_API_KEY':'offline-not-a-key'}), \
                    patch.object(pilot,'LocalEncoder') as encoder, \
                    patch.object(pilot,'model_fingerprint',return_value='offline'), \
                    patch.object(feedback,'propose_once',side_effect=proposer):
                encoder.return_value.stats.return_value={}
                self.assertEqual(dialogue.run(self.args),0)
            summary=pilot.read_json(self.args.output_dir/'dialogue_summary.json')
            self.assertEqual(summary['processed_round_count'],16)
            self.assertEqual(summary['successful_execution_round_count'],16)
            self.assertEqual(summary['processed_episode_count'],2)
            self.assertNotIn('completed_episode_count',summary)
            for ep in ('natural_start','explicit_start'):
                root=self.args.output_dir/'episodes'/f'{ep}_r01'
                checkpoint=pilot.read_json(root/'memory_checkpoint.json')
                self.assertEqual(checkpoint['state']['backend']['memory_version'],8)
                self.assertEqual(checkpoint['state']['backend']['objects']['121']['support_id'],69)
                audit=pilot.read_json(root/'evaluation/final_audit.json')
                self.assertEqual(len(audit['rows']),8)
                self.assertTrue(all(r['reference_checks_all_passed'] for r in audit['rows']))


if __name__=='__main__':
    unittest.main()
