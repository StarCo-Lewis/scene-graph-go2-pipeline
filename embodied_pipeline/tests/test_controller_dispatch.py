"""No-network contract and lifecycle tests for controller dispatch scaffolding."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from embodied_pipeline.execution import controller_dispatch as d


def fixture():
    return {'schema_version':'real_robot_proposal_envelope_v1','task_id':'synthetic_dispatch_test',
        'scene_id':'test_only','execution_ready':False,'coordinate_frame':'test_frame','unit':'metre',
        'plan':{'subtask_plans':[
            {'subtask_id':'s1','dependencies':[],'steps':[
                {'step_id':'a1','skill_id':'navigate_to_object','arguments':{'target':{'role_id':'target','element_id':13}}}]},
            {'subtask_id':'s2','dependencies':['s1'],'steps':[
                {'step_id':'a2','skill_id':'grasp_object','arguments':{'target':{'role_id':'target','element_id':13}}}]}],
            'unplanned_subtasks':[]}}


class DispatchTests(unittest.TestCase):
    def compile(self, envelope=None):
        test_root=(d.ROOT/'tmp/controller_dispatch_tests').resolve()
        test_root.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=test_root) as folder:
            self.assertEqual(Path(folder).resolve().parent,test_root)
            p=Path(folder)/'engineering_plan.json'
            p.write_text(json.dumps(fixture() if envelope is None else envelope),encoding='utf-8')
            return d.compile_requests(p)

    def test_routes_cover_current_catalogue_without_claiming_interfaces(self):
        routes=d.read(d.DEFAULT_ROUTES);lib=d.read(d.DEFAULT_LIBRARY)
        self.assertEqual(set(routes['skills']),{s['skill_id'] for s in lib['skills']})
        self.assertEqual(len(routes['modules']),5)
        for mid in routes['skills'].values():
            self.assertEqual(routes['modules'][mid]['adapter_status'],'interface_pending')
        self.assertEqual(routes['modules']['navigation']['known_interface']['action_name'],'/nav/navigate_to')

    def test_actual_engineering_json_preserves_source_and_arguments(self):
        path=d.ROOT/'outputs/real_scene_runs/d3_619_plan_20261022_132214_282023/engineering_plan.json'
        original=path.read_bytes();source=json.loads(original)
        batch=d.compile_requests(path)
        self.assertEqual(path.read_bytes(),original)
        self.assertEqual(batch['source_plan_sha256'],hashlib.sha256(original).hexdigest())
        before=[s for t in source['plan']['subtask_plans'] for s in t['steps']]
        for command,step in zip(batch['commands'],before):
            self.assertEqual(command['skill_arguments'],step['arguments'])
            self.assertEqual(command['source_action_id'],step['action_id'])
            self.assertIsNone(command['controller_payload'])
            self.assertFalse(command['execution_authorized'])
        self.assertEqual(batch['commands'],d.compile_requests(path)['commands'])

    def test_structural_errors_are_not_silently_skipped(self):
        for mutation in ('unknown_skill','forward_dependency','duplicate_id','unplanned','empty','bad_binding','extra_argument'):
            with self.subTest(mutation=mutation):
                e=fixture();steps=e['plan']['subtask_plans'];s=steps[0]['steps'][0]
                if mutation=='unknown_skill':s['skill_id']='invented'
                if mutation=='forward_dependency':steps[0]['dependencies']=['s2']
                if mutation=='duplicate_id':steps[1]['steps'][0]['step_id']='a1'
                if mutation=='unplanned':e['plan']['unplanned_subtasks']=['s3']
                if mutation=='empty':steps[1]['steps']=[]
                if mutation=='bad_binding':s['arguments']['target']['element_id']=True
                if mutation=='extra_argument':s['arguments']['invented_speed']=99
                with self.assertRaises(ValueError):self.compile(e)

    def test_preview_is_pending_and_bindings_are_checked(self):
        batch=self.compile();original=copy.deepcopy(batch)
        with patch('socket.socket.connect',side_effect=AssertionError('Network forbidden')):
            result=d.preview(batch)
        self.assertEqual(batch,original)
        self.assertEqual(result['status'],'interfaces_pending_not_executed')
        self.assertTrue(all(p['payload'] is None for p in result['preparations']))
        with self.assertRaises(ValueError):d.preview(batch,[])
        with self.assertRaises(ValueError):d.preview(batch,{'stale_command':{}})
        adapter=d.PendingGroupAdapter('navigation',batch['modules']['navigation'])
        with self.assertRaises(NotImplementedError):adapter.submit(batch['commands'][0],{})

    def test_navigation_slot_reuses_converter_and_checks_source_identity(self):
        batch=self.compile();command=batch['commands'][0]
        command['source_action_id']='source_action'
        adapter=d.NavigationAdapterSlot('navigation',batch['modules']['navigation'])
        converted={'source_plan_sha256':command['source_plan_sha256'],
                   'source_action_id':'source_action','goal':{'unchanged_supplied_value':0.4}}
        binding={'navigation_replay_config':{'sentinel':'supplied_config'}}
        with patch('embodied_pipeline.execution.navigation_contract.prepare',return_value=converted) as converter:
            result=adapter.prepare(command,binding)
            converter.assert_called_once_with(binding['navigation_replay_config'])
            self.assertEqual(result['payload'],converted['goal'])
            self.assertEqual(result['status'],'interface_pending')
        for field in ('source_plan_sha256','source_action_id'):
            wrong={**converted,field:'wrong'}
            with patch('embodied_pipeline.execution.navigation_contract.prepare',return_value=wrong):
                with self.assertRaises(ValueError):adapter.prepare(command,binding)

    def event(self,command,status):
        return {'command_id':command['command_id'],'module_id':command['module_id'],
                'mode':'simulation','status':status}

    def test_acknowledgement_does_not_advance_and_feedback_is_correlated(self):
        state=d.SimulationState(self.compile());command=state.reserve_next()
        self.assertIsNone(state.reserve_next())
        state.record(self.event(command,'accepted'))
        self.assertIsNone(state.reserve_next())
        state.record(self.event(command,'running'))
        with self.assertRaises(ValueError):state.record({**self.event(command,'succeeded'),'command_id':'other'})
        with self.assertRaises(ValueError):state.record({**self.event(command,'succeeded'),'module_id':'other'})
        state.record(self.event(command,'succeeded'))
        next_command=state.reserve_next()
        self.assertEqual(next_command['skill_id'],'grasp_object')
        with self.assertRaises(ValueError):state.record(self.event(command,'succeeded'))

    def test_failure_unknown_rejection_and_cancellation_stop_dependents(self):
        for terminal in ('failed','unknown','rejected','cancelled'):
            with self.subTest(terminal=terminal):
                state=d.SimulationState(self.compile());command=state.reserve_next()
                if terminal=='cancelled':
                    state.record(self.event(command,'cancel_requested'))
                    self.assertIsNone(state.reserve_next())
                state.record(self.event(command,terminal))
                self.assertIsNone(state.reserve_next())
                result=state.report()
                self.assertEqual(result['status'],'simulation_halted')
                self.assertEqual(len(result['unstarted_command_ids']),1)
                self.assertFalse(result['robot_executed'])

    def test_complete_simulation_has_explicit_synthetic_provenance(self):
        result=d.simulate(self.compile())
        self.assertEqual(result['status'],'simulation_completed')
        self.assertEqual([e['status'] for e in result['events']],['accepted','running','succeeded']*2)
        self.assertFalse(result['robot_executed']);self.assertFalse(result['memory_committed'])
        self.assertIsNone(result['observed_robot_state'])

    def test_adapter_timeout_and_exception_never_retry_or_run_next(self):
        class Silent(d.SimulatedGroupAdapter):
            def __init__(self,mid):super().__init__(mid);self.calls=0
            def submit(self,c,p):self.calls+=1;return super().submit(c,p)
            def poll(self,c):return None
        class Broken(Silent):
            def submit(self,c,p):self.calls+=1;raise TimeoutError('Delivery outcome unknown')
        for cls in (Silent,Broken):
            nav=cls('navigation');manip=Silent('manipulation')
            result=d.simulate(self.compile(),{'navigation':nav,'manipulation':manip},max_polls=2)
            self.assertEqual(nav.calls,1);self.assertEqual(manip.calls,0)
            self.assertEqual(result['status'],'simulation_halted')
            self.assertEqual(result['events'][-1]['status'],'unknown')

    def test_non_simulation_adapter_is_rejected_before_any_submit(self):
        nav=d.SimulatedGroupAdapter('navigation');manip=d.SimulatedGroupAdapter('manipulation')
        manip.mode='live'
        with self.assertRaises(ValueError):d.simulate(self.compile(),{'navigation':nav,'manipulation':manip})
        self.assertEqual(nav.pending,{})

    def test_stale_derived_order_is_rejected(self):
        envelope=fixture()
        envelope['plan']=d.load('action_sequence').refresh_sequence(envelope['plan'])
        envelope['plan']['action_sequence'].reverse()
        with self.assertRaises(ValueError):self.compile(envelope)


if __name__=='__main__':unittest.main()
