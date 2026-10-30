"""Offline mocks verify engineering recovery, not LLM task success."""
import copy
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import run_crossroom_task as pilot
import run_dialogue_chain as dialogue
import plan_live_round as worker
from json_symbolic_review import review_symbolic
from state_aware_planning import (VERSION,close_runtime_objects,route_runtime_error)
from action_matrix import make_codebook,decode_matrix


class Encoder:
    def encode(self,texts):
        return [[1.,0.] for _ in texts]


class RecoveryTests(unittest.TestCase):
    def inputs(self):
        cfg,nodes,edges,_ = dialogue.scene_fixture('scene202')
        state = pilot.initial_state(nodes,cfg)
        state.update(held_object_id=121,lifted=True,prepared_target_id=None)
        desk = state['backend']['objects']['2']
        state['backend'].update(robot_position=[desk['center'][0],.1,desk['center'][2]],
            robot_room_hypothesis='office_B',symbolically_updated_object_ids=[121])
        state['backend']['objects']['121'].update(center=[desk['center'][0],1.35,desk['center'][2]],
            support_id=None,room_hypothesis='office_B')
        instruction = 'Bring the black hardcover book from the living room to my desk and leave it there.'
        args = SimpleNamespace(model='offline',base_url='https://invalid.test',top_k=2,
            cosine_threshold=.35,cosine_margin=.05,reach=1.5,skill_output_format='matrix',
            retrieval_top_k=2,retrieval_threshold=.35,retrieval_margin=.05,
            skill_retriever=Encoder(),skill_proposal_limit=2)
        return cfg,nodes,edges,state,instruction,args

    def alignment(self,instruction,recovered=False):
        tasks = ['Pick up the black hardcover book.','Carry the book to my desk.','Place the book on my desk.']
        bindings = {'book':{'element_ids':[86]},'desk':{'element_ids':[2]}}
        if recovered:
            tasks.insert(0,'First place the currently held bottle on my desk to free the gripper.')
            bindings['held_bottle']={'element_ids':[121]}
        return dict(status='success',context_bindings=bindings,task_decomposition=dict(instruction=instruction,
            subtasks=[dict(subtask_id=f'st_{i}',description=t,dependencies=[] if i==1 else [f'st_{i-1}'])
                      for i,t in enumerate(tasks,1)],initial_scene_constraints=[],goal_state=[
                dict(constraint_id='g1',subject_role='book',relation='on',object_role='desk',required=True),
                dict(constraint_id='g2',subject_role='robot',relation='held',object_role='book',
                     expected_value=False,required=True)]))

    def matrix(self,book,recovered=False):
        skills = {s['skill_id']:s['code'] for s in book['skills']}
        objects = {o['element_id']:o['code'] for o in book['objects']}
        rows = []
        def step(task,skill,a,b=0):
            rows.append([task,skills[skill],objects[a],objects[b] if b else 0])
        offset=0
        if recovered:
            step(1,'position_arm_for_object',2)
            step(1,'place_object_on_surface',121,2)
            offset=1
        step(offset+1,'navigate_to_object',86)
        step(offset+1,'position_arm_for_object',86)
        step(offset+1,'grasp_object',86)
        step(offset+1,'lift_held_object',86)
        step(offset+2,'navigate_to_object',2)
        step(offset+3,'position_arm_for_object',2)
        step(offset+3,'place_object_on_surface',86,2)
        return {'matrix':rows}

    def exercise(self,root,fix=True,policy=True):
        cfg,nodes,edges,state,instruction,args = self.inputs()
        descriptor = dict(round=1,instruction=instruction)
        dialogue.prepare(root,descriptor,state,[],cfg,nodes,edges,args)
        library = pilot.read_json(root/'execution_skill_library.json')
        if not policy:
            library['execution_contract'].pop('state_aware_planning')
            pilot.write_json(root/'execution_skill_library.json',library)
            pilot.write_json(root/'execution_contract.json',library['execution_contract'])
        args.output_dir,args.round_input = root,root/'round_input.json'
        observed=[]
        def decompose(a,d,summary,semantic_attempt=1,feedback=None):
            observed.append(dict(attempt=semantic_attempt,feedback=copy.deepcopy(feedback),
                                 runtime=pilot.read_json(root/'runtime_state.json')))
            value=self.alignment(instruction,recovered=fix and semantic_attempt==2)
            pilot.write_json(root/f'object_alignment_semantic_attempt_{semantic_attempt}.json',value)
            return value
        calls=[]
        def request(alignment,lib,book,*pos,**kw):
            calls.append(copy.deepcopy(book))
            return self.matrix(book,recovered=fix and len(observed)==2)
        with patch.object(worker,'decompose_and_bind',side_effect=decompose), \
                patch.object(worker,'request_matrix',side_effect=request), \
                patch.object(worker,'request_order',return_value={'reason':'Offline keep original'}), \
                patch.object(worker,'candidate_plan',side_effect=lambda p,c:copy.deepcopy(p)):
            code=worker.plan(args,reviewer=review_symbolic)
        self.assertEqual(pilot.read_json(root/'runtime_state.json'),state)
        return code,pilot.read_json(root/'planning_summary.json'),observed,calls

    def test_closure_adds_only_exact_held_object_no_actions_or_goal_changes(self):
        cfg,nodes,edges,state,instruction,args=self.inputs()
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            dialogue.prepare(root,dict(round=1,instruction=instruction),state,[],cfg,nodes,edges,args)
            library=pilot.read_json(root/'execution_skill_library.json')
            raw=self.alignment(instruction)
            before=copy.deepcopy(raw)
            effective,record=close_runtime_objects(raw,library,state,nodes)
            self.assertEqual(raw,before)
            self.assertEqual(effective['task_decomposition'],before['task_decomposition'])
            self.assertEqual(set(effective['context_bindings'])-set(raw['context_bindings']),{'runtime_held_object'})
            self.assertEqual(effective['context_bindings']['runtime_held_object']['element_ids'],[121])
            self.assertFalse(record['actions_inserted'])
            book=make_codebook(effective,library)
            self.assertIn(121,[o['element_id'] for o in book['objects']])
            self.assertNotIn(101,[o['element_id'] for o in book['objects']])

    def test_existing_binding_reused_not_rebound(self):
        cfg,nodes,edges,state,instruction,args=self.inputs()
        library={'execution_contract':{'state_aware_planning':dict(version=VERSION,enabled=True,max_runtime_redecompositions=1)}}
        raw=self.alignment(instruction,True)
        effective,note=close_runtime_objects(raw,library,state,nodes)
        self.assertEqual(effective,raw)
        self.assertEqual(note['status'],'already_bound')

    def test_missing_or_inactive_held_id_is_not_fabricated(self):
        cfg,nodes,edges,state,instruction,args=self.inputs()
        library={'execution_contract':{'state_aware_planning':dict(version=VERSION,enabled=True,max_runtime_redecompositions=1)}}
        state['held_object_id']=999999
        with self.assertRaisesRegex(ValueError,'missing or inactive'):
            close_runtime_objects(self.alignment(instruction),library,state,nodes)

    def test_capacity_error_redecomposes_once_rebuilds_book_and_retains_original_error(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            code,summary,observed,books=self.exercise(root)
            self.assertEqual(code,0)
            self.assertEqual(len(observed),2)
            self.assertEqual(len(books),2)
            self.assertEqual(summary['runtime_redecomposition_count'],1)
            self.assertFalse(summary['initial_skill_proposal_passed'])
            self.assertEqual(summary['selected_semantic_attempt'],2)
            self.assertEqual(observed[0]['runtime'],observed[1]['runtime'])
            self.assertEqual(observed[1]['feedback']['code'],'GRIPPER_NOT_EMPTY')
            self.assertNotEqual(books[0]['sha256'],books[1]['sha256'])
            self.assertTrue((root/'llm_matrix_proposal_semantic_1_attempt_1.json').is_file())
            self.assertEqual(pilot.read_json(root/'runtime_rejection_semantic_1.json')['code'],'GRIPPER_NOT_EMPTY')
            self.assertEqual(pilot.read_json(root/'object_alignment_raw_semantic_1.json')['context_bindings'].keys(),
                             {'book','desk'})
            first=pilot.read_json(root/'skill_plan_proposal_semantic_1_attempt_1.json')
            final=pilot.read_json(root/'skill_plan_final.json')
            self.assertNotEqual(first['action_index'][0]['action_id'],final['action_index'][0]['action_id'])
            end=pilot.read_json(root/'symbolic_review_final.json')['predicted_end_state']
            self.assertIsNone(end['held_object_id'])
            self.assertEqual(end['backend']['objects']['86']['support_id'],2)
            self.assertEqual(end['backend']['objects']['121']['support_id'],2)

    def test_bad_redecomposition_stops_at_budget_no_inserted_release(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            code,summary,observed,books=self.exercise(root,fix=False)
            self.assertEqual(code,2)
            self.assertEqual(len(observed),2)
            self.assertEqual(summary['runtime_redecomposition_count'],1)
            self.assertEqual(len(summary['recovery_events']),2)
            self.assertTrue(summary['recovery_events'][-1]['budget_exhausted'])
            self.assertFalse((root/'skill_plan_final.json').exists())
            for path in root.glob('decoded_matrix_semantic_*_attempt_1.json'):
                data=pilot.read_json(path)
                self.assertFalse(any(s['skill_id']=='place_object_on_surface' and
                    s['arguments']['object']['element_id']==121 for t in data['subtask_plans'] for s in t['steps']))

    def test_legacy_policy_off_retains_matrix_only_retry(self):
        with tempfile.TemporaryDirectory() as temporary:
            code,summary,observed,books=self.exercise(Path(temporary),policy=False)
            self.assertEqual(code,2)
            self.assertEqual(len(observed),1)
            self.assertEqual(len(books),2)
            self.assertEqual(summary['runtime_redecomposition_count'],0)

    def test_network_schema_and_other_runtime_errors_do_not_route(self):
        cfg,nodes,edges,state,instruction,args=self.inputs()
        library={'execution_contract':{'state_aware_planning':dict(version=VERSION,enabled=True,max_runtime_redecompositions=1)}}
        for error in ['HTTP 400','timeout','Unknown object code','PROXY_PREPARATION_REQUIRED',
                      'SURFACE_PROXY_PREPARATION_REQUIRED','UNSUPPORTED_SYMBOLIC_GRASP_ID']:
            self.assertIsNone(route_runtime_error(ValueError(error),library,state,self.alignment(instruction),'raw.json'))

    def test_keep_holding_is_not_overridden_by_recovery_hint(self):
        cfg,nodes,edges,state,instruction,args=self.inputs()
        library={'execution_contract':{'state_aware_planning':dict(version=VERSION,enabled=True,max_runtime_redecompositions=1)}}
        error=route_runtime_error(ValueError('GRIPPER_NOT_EMPTY'),library,state,self.alignment(instruction),'raw.json')
        self.assertIn('Do not release',error['correction'])
        self.assertIn('no automatic placement target',error['correction'])
        self.assertNotIn('desk',error['correction'])

    def test_actual_decomposition_receives_error_original_instruction_and_state(self):
        cfg,nodes,edges,state,instruction,args=self.inputs()
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            dialogue.prepare(root,dict(round=1,instruction=instruction),state,[],cfg,nodes,edges,args)
            original=pilot.read_json(root/'execution_contract.json')
            feedback=route_runtime_error(ValueError('GRIPPER_NOT_EMPTY'),
                {'execution_contract':original},state,self.alignment(instruction),'first_raw.json')
            def launch(command,**kw):
                policy=pilot.read_json(Path(command[command.index('--execution-contract')+1]))
                self.assertEqual(policy['current_runtime_state'],state)
                self.assertEqual(policy['semantic_correction'],feedback['correction'])
                self.assertEqual(command[command.index('--instruction')+1],instruction)
                pilot.write_json(Path(command[command.index('--output')+1]),self.alignment(instruction,True))
                return SimpleNamespace(stdout=[],wait=lambda:0)
            args.output_dir=root
            with patch.object(worker.subprocess,'Popen',side_effect=launch):
                worker.decompose_and_bind(args,dict(instruction=instruction),{'stages':[]},2,feedback)
            self.assertEqual(pilot.read_json(root/'execution_contract.json'),original)


if __name__=='__main__':
    unittest.main()
