import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
from state_boundary_guards import attach_constraints, holding_constraint, state_facts, repair_placement_approach, check_holding_invariant, clean_empty_annotations
from test_json_long_horizon import nodes, row
from run_json_long_horizon import initial_state, prepare_round
from run_navigation_demo import write_json, read_json
from match_skills import build_result
from hierarchical_json_planner import plan_hierarchical
from token_usage_audit import usage


class BoundaryTests(unittest.TestCase):
    def test_cleanup_only_annotations_preserves_blank_actions_and_state(self):
        plan={'held_object_id':None, 'lifted':False, 'memory_version':0,
              'subtask_plans':[{'intent':'', 'dependencies':[], 'notes':'  ',
                  'steps':[{}, {'skill_id':'', 'arguments':{'target':None},'comment':None}]}]}
        original=copy.deepcopy(plan)
        candidate,paths=clean_empty_annotations(plan)
        self.assertEqual(len(paths),2)
        self.assertEqual(candidate['subtask_plans'][0]['steps'][0],{})
        self.assertEqual(candidate['subtask_plans'][0]['steps'][1],{'skill_id':'','arguments':{'target':None}})
        self.assertIsNone(candidate['held_object_id'])
        self.assertEqual(candidate['memory_version'],0)
        self.assertEqual(plan,original)

    def test_constraints_attach_verbatim(self):
        parts = ['Visit sideboard, ', 'then cabinet. ', 'Do not place it; keep holding it.']
        merged, events = attach_constraints(parts)
        self.assertEqual(''.join(parts), ''.join(merged))
        self.assertEqual(len(merged), 2)
        self.assertEqual(len(events), 1)

    def test_unknown_constraints_not_implicitly_accepted(self):
        for text in ['Do not place it; visit ID 13.', 'Keep holding it until I arrive.',
                     'Keep holding it and release it.', 'Keep holding the apple.']:
            self.assertFalse(holding_constraint(text))
        with self.assertRaises(ValueError):
            attach_constraints(['Keep holding it.'])

    def fixture(self, text='and place it on that cabinet.'):
        state = initial_state(nodes())
        state.update(held_object_id=67, lifted=True)
        state['backend']['robot_position'] = [0., .1, -7.]
        state['backend']['objects']['67']['center'] = [0.,1.35,-7.]
        fixture, library = row([('navigate_to_object',62), ('position_arm_for_object',62),
                                ('place_object_on_surface',67,62)], support=62)
        fixture['instruction'] = text
        fixture['alignment']['task_decomposition']['instruction'] = text
        fixture['plan'] = build_result(fixture['alignment'], library, fixture['plan'])
        return fixture, library, state

    def repair(self, f, lib, st):
        return repair_placement_approach(f, f['alignment'], f['plan'], lib, nodes(), st)

    def test_redundant_approach_removed_with_identical_end_state(self):
        f, lib, st = self.fixture()
        untouched = copy.deepcopy((f, st))
        plan, review, report = self.repair(f,lib,st)
        self.assertTrue(report['changed'])
        self.assertEqual(plan['action_count'],2)
        self.assertEqual(review['navigation_targets'],[])
        self.assertEqual((f,st),untouched)

    def test_explicit_visit_preserved(self):
        for text in ['Visit cabinet ID 62 again and place it there.',
                     'First go to that cabinet, then place it.', 'and place it on that cabinet, then return.']:
            f, lib, st = self.fixture(text)
            self.assertFalse(self.repair(f,lib,st)[2]['changed'])

    def test_far_and_near_but_different_end_pose_preserved(self):
        for offset in (0.4, 4.):
            f, lib, st = self.fixture()
            st['backend']['robot_position'][0] += offset
            self.assertFalse(self.repair(f,lib,st)[2]['changed'])

    def test_invalid_initial_state_not_repaired(self):
        f,lib,st = self.fixture()
        st.update(held_object_id=None, lifted=False)
        with self.assertRaises(Exception):
            self.repair(f,lib,st)

    def test_invalid_coordinate_not_used(self):
        st = initial_state(nodes())
        st['backend']['robot_position'][0] = float('nan')
        with self.assertRaises(ValueError):
            state_facts(st)

    def test_holding_invariant_rejects_release(self):
        _,_,st = self.fixture()
        after = copy.deepcopy(st)
        after['held_object_id'] = None
        with self.assertRaises(ValueError):
            check_holding_invariant('Do not place it; keep holding it.',st,
                                    {'predicted_end_state':after,'symbolic_trace':[]})

    def test_integrated_constraint_merge_bypasses_no_empty_guard(self):
        # Real symbolic reviews, fixture LLM responses. The source bytes are retained.
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            text='Carry it to cabinet. Do not place it; keep holding it.'
            _,lib,st=self.fixture()
            st['backend']['robot_position']=[0.,.1,0.]
            desc={'round':1,'instruction':text,'expected_end':{'held_object_id':67,'lifted':True},
                  'expected_navigation_targets':[62]}
            prepare_round(root,desc,st,nodes(),'offline','https://invalid.test')
            def decide(a,d,stage,system,payload,check):
                value=({'decision':'split','reason':'fixture',
                        'parts':['Carry it to cabinet. ','Do not place it; keep holding it.']}
                       if stage=='hierarchy_split' else
                       dict(complete=True,order_preserved=True,constraints_preserved=True,
                            no_unrequested_actions=True,reason='fixture'))
                check(value)
                return value
            def leaf(a,reviewer):
                child=read_json(a.round_input)
                self.assertEqual(child['instruction'],text)
                self.assertTrue((a.output_dir/'robot_state_snapshot.json').exists())
                f,library=row([('navigate_to_object',62)],held=True)
                f['alignment']['task_decomposition']['instruction']=text
                write_json(a.output_dir/'object_alignment.json',f['alignment'])
                write_json(a.output_dir/'skill_plan_final.json',build_result(f['alignment'],library,f['plan']))
                return 0
            code=plan_hierarchical(SimpleNamespace(model='offline',base_url='https://invalid.test'),
                                   root,decide=decide,leaf_planner=leaf,state_guards=True)
            self.assertEqual(code,0)
            self.assertEqual(read_json(root/'planning_summary.json')['leaf_count'],1)

    def test_usage_missing_and_timeout_not_zero_cost(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'llm_response_audit'
            root.mkdir()
            write_json(root/'a_http_1.json',{'event':'http_response','request_id_local':'a','http_attempt':1,
                'response_body':json.dumps({'usage':dict(prompt_tokens=10,completion_tokens=2,total_tokens=12)})})
            write_json(root/'b_http_1.json',{'event':'http_response','request_id_local':'b','http_attempt':1,
                'response_body':'{}'})
            write_json(root/'c_transport_1.json',{'event':'transport_failure','request_id_local':'c','http_attempt':1})
            value=usage(Path(temp))
            self.assertEqual(value['total_tokens'],12)
            self.assertEqual(value['responses_missing_usage'],1)
            self.assertEqual(value['transport_failures_unknown_usage'],1)


if __name__=='__main__':
    unittest.main()
