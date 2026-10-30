"""Offline adapter tests. Hand-authored actions are NOT LLM test results."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

import run_crossroom_task as pilot
from json_symbolic_review import SymbolicBackend,review_symbolic
from sequential_proxy_core import ExecutionRejected
from match_skills import build_result
from bounded_skill_planner import plan_bounded
from action_matrix import make_codebook, decode_matrix
from match_skills import PlanError
import hierarchical_json_planner as hierarchy


class CrossroomTests(unittest.TestCase):
    def setUp(self):
        self.cfg,self.nodes,self.edges,self.hashes = pilot.load_case(pilot.HERE/'scenarios/scene202_crossroom_6.json')
        self.state = pilot.initial_state(self.nodes,self.cfg)

    def policy(self):
        return {'symbolic_scene_capabilities':{'graspable_ids':self.cfg['graspable_ids'],
                'placeable_pairs':self.cfg['placeable_pairs']}}

    def test_complete_new_graph_and_original_ids(self):
        self.assertEqual(len(self.nodes),350)
        projected,edges = pilot.project_scene(self.nodes,self.edges,self.state,self.cfg)
        self.assertEqual([n['id'] for n in projected],[n['id'] for n in self.nodes])
        self.assertEqual(edges,self.edges)
        for identity,room in self.cfg['control_node_rooms'].items():
            self.assertEqual(self.state['backend']['objects'][identity]['room_hypothesis'],room)

    def test_generic_grasp_and_place_restrictions(self):
        backend = SymbolicBackend(self.state,self.nodes,1.5,self.policy())
        backend.require_graspable_instance(121)
        backend.require_graspable_instance(86)
        backend.require_placeable_pair(121,69)
        with self.assertRaises(ExecutionRejected): backend.require_graspable_instance(2)
        with self.assertRaises(ExecutionRejected): backend.require_placeable_pair(121,85)
        with self.assertRaises(ExecutionRejected): backend.require_placeable_pair(86,86)

    def test_missing_explicit_policy_does_not_enable_new_ids(self):
        backend = SymbolicBackend(self.state,self.nodes,1.5)
        with self.assertRaises(ExecutionRejected): backend.require_graspable_instance(121)

    def test_moving_one_object_only_replaces_its_outgoing_edges(self):
        backend = SymbolicBackend(self.state,self.nodes,1.5,self.policy())
        backend.navigate(69,None)
        backend.attach(121)
        backend.place(121,69)
        snapshot = copy.deepcopy(self.state)
        snapshot['backend'] = backend.snapshot()
        projected,edges = pilot.project_scene(self.nodes,self.edges,snapshot,self.cfg)
        self.assertEqual([e for e in edges if e['source']!=121],[e for e in self.edges if e['source']!=121])
        changed = [e for e in edges if e['source']==121]
        self.assertEqual(len(changed),1)
        self.assertEqual(changed[0]['target'],69)
        self.assertFalse(changed[0]['verified'])
        self.assertEqual(next(n for n in projected if n['id']==121)['json_test_context']['room_hypothesis'],'living_room')
        self.assertEqual(self.state['backend']['symbolically_updated_object_ids'],[])

    def test_hypothetical_pose_round_trip(self):
        projected,_ = pilot.project_scene(self.nodes,self.edges,self.state,self.cfg)
        for original,updated in zip(self.nodes,projected):
            self.assertEqual(original['bbox_center'],updated['bbox_center'])

    def test_five_subtasks_six_actions_prepare_and_place_share_owner(self):
        # Reproduce the failed segment's shape using a hand-authored unit fixture.
        # This is NOT a new LLM run, resumed chain, or physical execution result.
        descriptor = copy.deepcopy(self.cfg['rounds'][0])
        descriptor['expected_navigation_targets'] = [69]
        actions = [('position_arm_for_object',121),('grasp_object',121),
                   ('lift_held_object',121),('navigate_to_object',69),
                   ('position_arm_for_object',69),('place_object_on_surface',121,69)]
        with tempfile.TemporaryDirectory() as temporary:
            fixture, library = self.fixture(descriptor,actions,self.state,Path(temporary))
            alignment = fixture['alignment']
            alignment['task_decomposition']['subtasks'] = [
                {'subtask_id':f'st_{i:02}', 'description':text,
                 'dependencies':[] if i==1 else [f'st_{i-1:02}']}
                for i,text in enumerate(('Prepare bottle','Grasp bottle','Lift bottle',
                                         'Carry to coffee table','Place and release'),1)]
            book = make_codebook(alignment,library)
            skills = {s['skill_id']:s['code'] for s in book['skills']}
            objects = {o['element_id']:o['code'] for o in book['objects']}
            matrix = {'matrix':[
                [1,skills['position_arm_for_object'],objects[121],0],
                [2,skills['grasp_object'],objects[121],0],
                [3,skills['lift_held_object'],objects[121],0],
                [4,skills['navigate_to_object'],objects[69],0],
                [5,skills['position_arm_for_object'],objects[69],0],
                [5,skills['place_object_on_surface'],objects[121],objects[69]] ]}
            invalid = copy.deepcopy(matrix)
            invalid['matrix'][-1][0] = 6
            with self.assertRaisesRegex(PlanError,'unknown subtask code 6'):
                decode_matrix(invalid,book,alignment,library)
            plan = build_result(alignment,library,decode_matrix(matrix,book,alignment,library))
            self.assertEqual(len(plan['subtask_plans'][-1]['steps']),2)
            before = copy.deepcopy(self.state)
            review = review_symbolic(dict(fixture,plan=plan),library,self.nodes,self.state,1.5)
            self.assertEqual(review['predicted_end_state']['backend']['objects']['121']['support_id'],69)
            self.assertIsNone(review['predicted_end_state']['held_object_id'])
            self.assertFalse(review['simulation_executed'])
            self.assertEqual(self.state,before)
            # Deleting preparation still fails the unchanged executor gate.
            missing = copy.deepcopy(matrix)
            del missing['matrix'][-2]
            plan = build_result(alignment,library,decode_matrix(missing,book,alignment,library))
            with self.assertRaisesRegex(ExecutionRejected,'SURFACE_PROXY_PREPARATION_REQUIRED'):
                review_symbolic(dict(fixture,plan=plan),library,self.nodes,self.state,1.5)
            self.assertEqual(self.state,before)

    def test_preflight_has_no_api_or_embedding(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = SimpleNamespace(case_config=pilot.HERE/'scenarios/scene202_crossroom_6.json',
                preflight_only=True,output_dir=Path(temporary)/'preflight',model='fake',base_url='https://example.invalid')
            with patch.object(pilot,'LocalEncoder') as encoder,patch.object(pilot,'plan_bounded') as planner:
                self.assertEqual(pilot.run(args),0)
                encoder.assert_not_called()
                planner.assert_not_called()

    def state_after_first_round(self, folder):
        actions = [('navigate_to_object',101),('position_arm_for_object',121),
                   ('grasp_object',121),('lift_held_object',121),('navigate_to_object',69),
                   ('position_arm_for_object',69),('place_object_on_surface',121,69)]
        fixture, library = self.fixture(self.cfg['rounds'][0],actions,self.state,folder)
        state = review_symbolic(fixture,library,self.nodes,self.state,1.5)['predicted_end_state']
        state['backend']['memory_version'] = 1
        return state

    def test_second_round_constraint_leaf_has_no_calls_and_preserves_visit_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            state = self.state_after_first_round(root/'first_round')
            descriptor = self.cfg['rounds'][1]
            folder = root/'second_round'
            pilot.prepare_round(folder,descriptor,state,self.nodes,'fake','https://invalid.test',self.cfg,self.edges)
            calls = []
            def decide(args, directory, stage, system, payload, check):
                if stage == 'hierarchy_coverage':
                    value = dict(complete=True,order_preserved=True,constraints_preserved=True,
                                 no_unrequested_actions=True,reason='Offline test fixture, not LLM evidence')
                else:
                    text = payload['current_text']
                    if 'Leave the water bottle' in text:
                        cut = text.index('Leave the water bottle')
                        value = {'decision':'split','reason':'Offline exact partition',
                                 'parts':[text[:cut],text[cut:]]}
                    else:
                        value = {'decision':'leaf','reason':'Offline action leaf'}
                check(value)
                return value
            def leaf(args, reviewer):
                child = pilot.read_json(args.round_input)
                calls.append(child['instruction'])
                self.assertNotIn('Leave the water bottle', child['instruction'])
                desc = copy.deepcopy(descriptor)
                desc['instruction'] = child['instruction']
                actions = [('navigate_to_object',86),('position_arm_for_object',86),
                           ('grasp_object',86),('lift_held_object',86),('navigate_to_object',2),
                           ('position_arm_for_object',2),('place_object_on_surface',86,2)]
                fixture, _ = self.fixture(desc,actions,state,root/'unit_leaf_fixture')
                library = pilot.read_json(args.output_dir/'execution_skill_library.json')
                reviewer(fixture,library,self.nodes,state,1.5)
                pilot.write_json(args.output_dir/'object_alignment.json',fixture['alignment'])
                pilot.write_json(args.output_dir/'skill_plan_final.json',fixture['plan'])
                return 0
            args = SimpleNamespace(model='offline',base_url='https://invalid.test')
            with patch.object(hierarchy,'request_decision',side_effect=AssertionError('No API')):
                code = hierarchy.plan_hierarchical(args,folder,decide=decide,leaf_planner=leaf,
                    prepare=lambda f,d,s,n,m,u: pilot.prepare_round(f,d,s,n,m,u,self.cfg,self.edges))
            self.assertEqual(code,0)
            self.assertEqual(len(calls),1)
            review = pilot.read_json(folder/'symbolic_review_final.json')
            self.assertEqual(review['navigation_targets'],[86,2])
            self.assertEqual(review['predicted_end_state']['backend']['objects']['121'],state['backend']['objects']['121'])
            self.assertEqual(review['predicted_end_state']['backend']['objects']['86']['support_id'],2)
            summary = pilot.read_json(folder/'hierarchy_summary.json')
            constraint = next(n for n in summary['nodes'] if n['status']=='constraint_validated_no_actions')
            self.assertEqual(constraint['constraint_report']['action_count'],0)
            self.assertFalse(constraint['constraint_report']['fresh_llm_calls'])
            self.assertEqual(len(review['symbolic_trace']),7)

    def test_pick_up_and_replace_protected_bottle_fails_even_with_same_final_support(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            state = self.state_after_first_round(root/'first_round')
            desc = copy.deepcopy(self.cfg['rounds'][1])
            desc['instruction'] = 'Leave the water bottle (ID 121) on the coffee table (ID 69).'
            desc['expected_navigation_targets'] = [69]
            desc['expected_support_by_object'] = {'121':69}
            actions = [('navigate_to_object',69),('position_arm_for_object',121),
                       ('grasp_object',121),('lift_held_object',121),
                       ('position_arm_for_object',69),('place_object_on_surface',121,69)]
            fixture,library = self.fixture(desc,actions,state,root/'bad_plan')
            before = copy.deepcopy(state)
            with self.assertRaisesRegex(ValueError,'PRESERVATION_MANIPULATION_FORBIDDEN'):
                review_symbolic(fixture,library,self.nodes,state,1.5)
            self.assertEqual(state,before)

    def test_round_four_carry_contract_is_sent_but_negative_judge_still_stops(self):
        # Manual fixture reproduces the reported round-4 state, not a paid replay.
        for accepted in (True, False):
            with self.subTest(accepted=accepted), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                state = self.state_after_first_round(root/'first')
                second_actions = [('navigate_to_object',86),('position_arm_for_object',86),
                                  ('grasp_object',86),('lift_held_object',86),('navigate_to_object',2),
                                  ('position_arm_for_object',2),('place_object_on_surface',86,2)]
                fixture,library = self.fixture(self.cfg['rounds'][1],second_actions,state,root/'second')
                state = review_symbolic(fixture,library,self.nodes,state,1.5)['predicted_end_state']
                descriptor = self.cfg['rounds'][3]
                text = descriptor['instruction']
                cut = text.index('carry')
                parts = [text[:cut],text[cut:]]
                folder = root/'fourth'
                pilot.prepare_round(folder,descriptor,state,self.nodes,'offline','https://invalid.test',self.cfg,self.edges)
                starts, coverage_calls = [], []
                def decide(args, directory, stage, system, payload, check):
                    if stage == 'hierarchy_coverage':
                        coverage_calls.append(copy.deepcopy(payload))
                        self.assertIn('NOT a holding flag', system)
                        contract = payload['executor_state_contract']
                        self.assertEqual(contract['successful_action_effects']['navigate_to_object']['held_object_id'], 'unchanged')
                        end = payload['predicted_end_state']
                        self.assertEqual((end['held_object_id'],end['lifted'],end['prepared_target_id']), (121,True,None))
                        trace = payload['symbolic_validation_evidence']['state_transitions']
                        self.assertEqual(trace[-1]['executor_state_before']['prepared_target_id'],121)
                        self.assertIsNone(trace[-1]['executor_state_after']['prepared_target_id'])
                        self.assertEqual(trace[-1]['executor_state_after']['held_object_id'],121)
                        value = dict(complete=accepted,order_preserved=True,constraints_preserved=accepted,
                                     no_unrequested_actions=True,reason='Offline mock judge, NOT a real LLM result')
                    elif payload['current_text'] == text:
                        value = dict(decision='split',reason='Offline exact partition',parts=parts)
                    else:
                        value = dict(decision='leaf',reason='Offline fixture')
                    check(value)
                    return value
                def leaf(args, reviewer):
                    child = pilot.read_json(args.round_input)
                    start = pilot.read_json(args.output_dir/'runtime_state.json')
                    starts.append(copy.deepcopy(start))
                    actions = ([('navigate_to_object',69),('position_arm_for_object',121),
                                ('grasp_object',121),('lift_held_object',121)] if len(starts)==1
                               else [('navigate_to_object',101)])
                    desc = dict(descriptor,instruction=child['instruction'],
                                expected_navigation_targets=[69] if len(starts)==1 else [101])
                    fixture,_ = self.fixture(desc,actions,start,root/f'fixture_{len(starts)}')
                    pilot.write_json(args.output_dir/'object_alignment.json',fixture['alignment'])
                    pilot.write_json(args.output_dir/'skill_plan_final.json',fixture['plan'])
                    return 0
                original = copy.deepcopy(state)
                with patch.object(hierarchy,'request_decision',side_effect=AssertionError('No API')):
                    code = hierarchy.plan_hierarchical(SimpleNamespace(model='offline',base_url='https://invalid.test'),
                        folder,decide=decide,leaf_planner=leaf,
                        prepare=lambda f,d,s,n,m,u: pilot.prepare_round(f,d,s,n,m,u,self.cfg,self.edges))
                self.assertEqual(code,0 if accepted else 2)
                self.assertEqual(len(coverage_calls),1)
                self.assertEqual(state,original)
                self.assertEqual(pilot.read_json(folder/'coverage_input.json'),coverage_calls[0])
                if not accepted:
                    self.assertEqual(pilot.read_json(folder/'planning_summary.json')['failed_stage'], 'hierarchical_semantic_coverage')

    def test_pure_constraint_round_passes_only_with_checked_state_and_no_leaf_model(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            state = self.state_after_first_round(root/'first_round')
            descriptor = {'round':2,'instruction':'Leave the water bottle (ID 121) on the coffee table (ID 69).',
                          'expected_end':{'held_object_id':None,'lifted':False},
                          'expected_navigation_targets':[], 'expected_support_by_object':{'121':69}}
            folder = root/'constraint_round'
            pilot.prepare_round(folder,descriptor,state,self.nodes,'fake','https://invalid.test',self.cfg,self.edges)
            def decide(a,d,stage,system,payload,check):
                self.assertEqual(stage,'hierarchy_coverage')
                value = dict(complete=True,order_preserved=True,constraints_preserved=True,
                             no_unrequested_actions=True,reason='Offline fixture')
                check(value)
                return value
            with patch.object(hierarchy,'plan_leaf',side_effect=AssertionError('Constraint must not generate actions')):
                code = hierarchy.plan_hierarchical(SimpleNamespace(model='offline',base_url='https://invalid.test'),
                                                   folder,decide=decide)
            self.assertEqual(code,0)
            review = pilot.read_json(folder/'symbolic_review_final.json')
            self.assertEqual(review['predicted_end_state'],state)
            self.assertEqual(review['symbolic_trace'],[])
            self.assertTrue(review['preservation_constraint_checks'])

    def test_bounded_hierarchy_forwards_scene_preparer(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary)
            pilot.prepare_round(folder,self.cfg['rounds'][0],self.state,self.nodes,'fake','https://example.invalid',self.cfg,self.edges)
            args=SimpleNamespace(skill_output_format='matrix',compile_navigation=False)
            hook=lambda *unused: None
            with patch('bounded_skill_planner.plan_hierarchical',return_value=0) as hierarchy:
                self.assertEqual(plan_bounded(args,folder,encoder=object(),max_parts=4,prepare=hook),0)
                self.assertIs(hierarchy.call_args.kwargs['prepare'],hook)

    def fixture(self,descriptor,actions,state,folder):
        pilot.prepare_round(folder,descriptor,state,self.nodes,'fake','https://example.invalid',self.cfg,self.edges)
        library=pilot.read_json(folder/'execution_skill_library.json')
        refs={i:'r_'+str(i) for i in (2,69,85,86,101,121)}
        goals=[]
        for identity,support in descriptor['expected_support_by_object'].items():
            if support is not None:
                goals.append({'constraint_id':'on_'+identity,'subject_role':refs[int(identity)],
                              'object_role':refs[support],'relation':'on','required':True})
        if descriptor['expected_end']['held_object_id'] is not None:
            goals.extend({'constraint_id':relation,'subject_role':'robot','object_role':refs[121],
                          'relation':relation,'required':True} for relation in ('held','lifted'))
        else:
            goals.append({'constraint_id':'near','subject_role':'robot',
                          'object_role':refs[descriptor['expected_navigation_targets'][-1]],
                          'relation':'near','required':True})
        alignment={'status':'success','context_bindings':{role:{'element_ids':[i]} for i,role in refs.items()},
                   'task_decomposition':{'instruction':descriptor['instruction'],'initial_scene_constraints':[],
                       'subtasks':[{'subtask_id':'st_01','description':descriptor['instruction'],'dependencies':[]}],
                       'goal_state':goals}}
        steps=[]
        for skill,identity,*surface in actions:
            parameter='object' if skill in ('lift_held_object','place_object_on_surface') else 'target'
            arguments={parameter:{'role_id':refs[identity],'element_id':identity}}
            if surface:
                arguments['surface']={'role_id':refs[surface[0]],'element_id':surface[0]}
            steps.append({'skill_id':skill,'arguments':arguments})
        proposal={'subtask_plans':[{'subtask_id':'st_01','intent':descriptor['instruction'],'steps':steps}]}
        plan=build_result(alignment,library,proposal)
        return dict(descriptor,alignment=alignment,plan=plan),library

    def test_six_round_manual_fixture_assertions_and_state_inheritance(self):
        # Explicit synthetic unit actions, never written to a live experiment.
        nav='navigate_to_object'; prep='position_arm_for_object'; grasp='grasp_object'
        lift='lift_held_object'; place='place_object_on_surface'
        sequences=[
            [(nav,101),(prep,121),(grasp,121),(lift,121),(nav,69),(prep,69),(place,121,69)],
            [(nav,86),(prep,86),(grasp,86),(lift,86),(nav,2),(prep,2),(place,86,2)],
            [(nav,101),(nav,69),(nav,2)],
            [(nav,69),(prep,121),(grasp,121),(lift,121),(nav,101)],
            [(prep,101),(place,121,101),(nav,86),(prep,86),(grasp,86),(lift,86),(nav,69),(prep,69),(place,86,69)],
            [(nav,86),(prep,86),(grasp,86),(lift,86),(nav,2),(prep,2),(place,86,2),(nav,121)]
        ]
        state=copy.deepcopy(self.state)
        total=0
        with tempfile.TemporaryDirectory() as temporary:
            for descriptor,actions in zip(self.cfg['rounds'],sequences):
                row,library=self.fixture(descriptor,actions,state,Path(temporary)/str(descriptor['round']))
                projected,_=pilot.project_scene(self.nodes,self.edges,state,self.cfg)
                review=review_symbolic(row,library,projected,state,1.5)
                self.assertFalse(review['simulation_executed'])
                total+=len(review['symbolic_trace'])
                state=review['predicted_end_state']
                state['backend']['memory_version']=descriptor['round']
                if descriptor['round']==4:
                    self.assertEqual(state['held_object_id'],121)
                if descriptor['round']==5:
                    self.assertEqual(state['backend']['objects']['121']['support_id'],101)
            self.assertEqual(total,39)
            self.assertEqual(state['backend']['objects']['86']['support_id'],2)

    def test_wrong_requested_support_rejected(self):
        descriptor=copy.deepcopy(self.cfg['rounds'][2])
        descriptor['expected_support_by_object']={'121':2}
        actions=[('navigate_to_object',i) for i in (101,69,2)]
        with tempfile.TemporaryDirectory() as temporary:
            row,library=self.fixture(descriptor,actions,self.state,Path(temporary))
            with self.assertRaisesRegex(ValueError,'Wrong requested symbolic support'):
                review_symbolic(row,library,self.nodes,self.state,1.5)


if __name__=='__main__': unittest.main()
