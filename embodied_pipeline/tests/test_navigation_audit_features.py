"""Offline regressions for temporal goals, adjustable clearance and GO audit display data."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from embodied_pipeline._legacy import ROOT,load
from embodied_pipeline.orchestration import real_scene_plan as planner
from embodied_pipeline.orchestration.repair_ordered_navigation import correct
from embodied_pipeline.execution.navigation_outcome import outcome
from embodied_pipeline.execution.navigation_points import bbox_approach,bbox_distance,validate_clearance
from embodied_pipeline.ui.navigation_plot import Grid,Evidence,Renderer
from embodied_pipeline.tests.test_multi_navigation import terminal
from embodied_pipeline.tests import test_auto_navigation as auto_tests

SOURCE=ROOT/'outputs/real_scene_runs/field_ui_20261030_132625_030a80db/planning'


class TemporalTests(unittest.TestCase):
    def setUp(self):
        self.alignment=planner.read(SOURCE/'object_alignment.json')
        self.plan=planner.read(SOURCE/'skill_plan.json')

    def test_real_failure_repaired_without_changing_actions_or_raw_output(self):
        original=copy.deepcopy(self.plan)
        a,p,note=correct(self.alignment,self.plan,[74,123])
        self.assertEqual(self.plan,original)
        self.assertEqual(p['subtask_plans'],original['subtask_plans'])
        self.assertEqual(p['action_sequence'],original['action_sequence'])
        self.assertEqual(len(a['task_decomposition']['goal_state']),1)
        self.assertEqual(len(note['milestones']),2)
        graph=SOURCE/'input_memory/scene'
        args=[planner.read(SOURCE/'execution_skill_library.json'),planner.read(graph/'scene_graph.json'),planner.read(graph/'scene_graph_edges.json')]
        self.assertEqual(load('verify_plan').review(self.alignment,self.plan,*args)['status'],'invalid_symbolic_plan')
        self.assertEqual(load('verify_plan').review(a,p,*args)['status'],'conditionally_valid_symbolic_plan')

    def test_correction_cannot_reorder_remove_action_or_invent_goal(self):
        with self.assertRaises(ValueError):correct(self.alignment,self.plan,[123,74])
        p=copy.deepcopy(self.plan);p['subtask_plans'][1]['dependencies']=[]
        with self.assertRaises(ValueError):correct(self.alignment,p,[74,123])
        a=copy.deepcopy(self.alignment);p=copy.deepcopy(self.plan)
        a['task_decomposition']['goal_state']=a['task_decomposition']['goal_state'][:1]
        p['deferred']['goal_state']=copy.deepcopy(a['task_decomposition']['goal_state'])
        with self.assertRaises(ValueError):correct(a,p,[74,123])

    def test_check_explains_temporal_error(self):
        task=planner.read(SOURCE/'task_input.json')
        bundle={'nodes':planner.read(SOURCE/'input_memory/scene/scene_graph.json')}
        check=planner.check_navigation(self.plan,task,bundle,self.alignment)
        self.assertFalse(check['passed'])
        self.assertEqual(check['issues'][0]['code'],'intermediate_visit_declared_as_terminal_goal')

    def test_prompt_contract_is_in_actual_decomposition_input(self):
        with tempfile.TemporaryDirectory() as d:
            task,bundle,library,_,_=planner.prepare(SOURCE/'task_input.json',Path(d)/'run')
            contract=planner.read(Path(d)/'run/execution_contract.json')
            self.assertIn(planner.ORDERED_GOAL_POLICY,contract['factual_boundary'])
            self.assertEqual(contract,library['execution_contract'])
            self.assertIn('A -> B -> A',contract['factual_boundary'])


class ClearanceTests(unittest.TestCase):
    def test_geometry_and_validation_use_selected_value(self):
        node={'bbox_center':[2,3,0],'bbox_extent':[2,2,0]}
        for distance in (.2,.4,.8,1.2):
            point=bbox_approach(node,{'x':-2,'y':3,'yaw':0},distance)
            self.assertAlmostEqual(bbox_distance(node,point),distance)
            validate_clearance(node,point,distance)
            with self.assertRaises(ValueError):validate_clearance(node,point,distance+.1)
        for distance in (0,-1,float('nan'),float('inf'),True):
            with self.assertRaises(ValueError):bbox_approach(node,{'x':-2,'y':3,'yaw':0},distance)

    def test_ui_apply_recomputes_unconfirmed_candidate_and_rejects_active_edit(self):
        fixture=auto_tests.AutoMappingTests();fixture.setUp();self.addCleanup(fixture.doCleanups);fixture.fill_start();p=fixture.panel
        p.maybe_auto_map();before=p.mapping_key()
        p.confirmations['approach_pose_verified'].set(True)
        p.clearance_input.set('.8');p.apply_clearance()
        self.assertNotEqual(before,p.mapping_key())
        self.assertAlmostEqual(bbox_distance(fixture.node,{k:float(v.get()) for k,v in p.goal.items()}),.8)
        self.assertFalse(p.confirmations['approach_pose_verified'].get())
        p.active=True;p.clearance_input.set('.2')
        with self.assertRaises(ValueError):p.apply_clearance()
        self.assertEqual(p.clearance(),.8)


class OutcomeTests(unittest.TestCase):
    def test_go_status_not_distance_or_plan_controls_success(self):
        for status,success,reason,expected in [(4,True,'arrived','succeeded'),(6,False,'position_out_of_tol','failed'),
             (5,False,'cancelled','cancelled'),(4,True,'other','failed'),(2,True,'arrived','unknown')]:
            d=terminal();d['ros_goal_status']=status;d['result'].update(success=success,reason=reason)
            self.assertEqual(outcome({'type':'result','data':d})['code'],expected)
        self.assertEqual(outcome({'type':'dispatch_reserved'})['code'],'pending')
        self.assertEqual(outcome({'type':'connection_lost'})['code'],'unknown')

    def test_target_and_failure_visible_data_matches_per_task_log(self):
        with tempfile.TemporaryDirectory() as temp:
            e=Evidence(Grid(),Path(temp)/'session');rid='case-A'
            e.consume({'type':'candidate_goals','data':{'goals':[{'target_element_id':74,'goal':{'x':1,'y':2,'yaw':0}}]}})
            self.assertEqual(e.previews[0]['state'],'候选 / 未执行')
            e.consume({'type':'dispatch_reserved','request_id':rid,'data':{'target_element_id':74,'goal':{'g_x':1,'g_y':2,'g_yaw':0}}})
            self.assertEqual(len(e.previews),0)
            d=terminal(.9,2,.1);d['ros_goal_status']=6;d['result'].update(success=False,reason='position_out_of_tol')
            e.consume({'type':'result','request_id':rid,'data':d})
            self.assertIn('失败',e.goals[0]['state'])
            self.assertEqual(e.goals[0]['pose'],[1,2,0]);self.assertEqual(e.pose[:2],(.9,2))
            e.save(Renderer(e.grid,400,300).render(e),True)
            folder=e.task_folders[rid]
            saved=json.loads((folder/'outcome.json').read_text(encoding='utf-8'))
            self.assertEqual(saved['execution_outcome'],e.goals[0]['outcome'])
            self.assertTrue((folder/'trajectory.png').exists());self.assertTrue((folder/'task.json').exists())
            self.assertEqual(len((folder/'events.jsonl').read_text().splitlines()),2)
