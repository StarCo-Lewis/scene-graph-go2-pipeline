"""Offline map, log and wire observation tests; no ROS/network or robot commands."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock
import zipfile

from embodied_pipeline.ui.navigation_plot import Grid,Evidence,Renderer,read_npz,npy
from embodied_pipeline.execution.tcp_bridge import BridgeCore,Journal
from embodied_pipeline.tests.test_tcp_navigation import packet
from embodied_pipeline.tests.test_multi_navigation import terminal


class ViewTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.grid=Grid()

    def row(self,x=1,request='A'):
        d=terminal(x,2,.3)['result']
        return {'type':'feedback','coordinate_frame':'test_scene','request_id':request,
                'received_utc':'2026-10-30T05:00:00+00:00',
                'data':{'current_pose':d['final_pose'],'stage':'moving','remaining_dist':1.2}}

    def test_supplied_grid_uses_lower_origin_and_full_dimensions(self):
        self.assertEqual((self.grid.h,self.grid.w),(400,474))
        self.assertEqual(self.grid.origin,(-12.851,-12.56))
        self.assertAlmostEqual(self.grid.res,.05)
        self.assertEqual(len(self.grid.occ),400*474)

    def test_roundtrip_and_failed_terminal_record_not_invented_target(self):
        e=Evidence(self.grid,self.root/'session');e.consume(self.row())
        d=terminal(1.03,2.01,.31);d['ros_goal_status']=6
        d['result'].update(success=False,reason='position_out_of_tol')
        e.consume({'type':'result','request_id':'A','coordinate_frame':'test_scene','data':d})
        self.assertEqual(len(e.traj),2);self.assertEqual(e.pose[:2],(1.03,2.01))
        self.assertEqual(e.paths,[])
        e.save(Renderer(self.grid,500,350).render(e),terminal=True)
        stored=read_npz(e.directory/'trajectory.npz')
        self.assertEqual(stored['traj'][0],(2,3))
        self.assertEqual(stored['n_path'][1],[0])
        self.assertEqual(stored['stage_names'][1][-1],'position_out_of_tol')
        manifest=json.loads((e.directory/'manifest.json').read_text(encoding='utf-8'))
        self.assertFalse(manifest['results'][0]['data']['result']['success'])
        self.assertEqual(manifest['map_sha256'],self.grid.sha256)
        self.assertTrue((e.directory/'terminal_001.png').is_file())

    def test_mismatched_frame_invalid_pose_and_path_retained_as_evidence(self):
        e=Evidence(self.grid,self.root/'session');e.consume(self.row())
        other=self.row(99);other['coordinate_frame']='other';e.consume(other)
        bad=self.row(9);bad['data']['current_pose']['pose']['orientation']={};e.consume(bad)
        e.consume({'type':'planned_path','data':{'waypoints':[[float('nan'),0,0]]}})
        self.assertEqual(len(e.traj),1);self.assertEqual(len(e.errors),2)
        self.assertEqual(len((e.directory/'events.jsonl').read_text().splitlines()),4)

    def test_segments_do_not_draw_connection_after_disconnect(self):
        e=Evidence(self.grid);e.consume(self.row())
        e.consume({'type':'connection_lost'});self.assertIsNone(e.last_pose_time)
        e.consume(self.row(5));e.consume(self.row(6,'B'))
        self.assertEqual(len(set(e.segments)),3)

    def test_path_is_exact_observation_and_not_sent_as_a_command(self):
        e=Evidence(self.grid);p=[[0,0,0],[1,.5,.2]]
        e.consume({'type':'planned_path','request_id':'A','data':{'waypoints':p}})
        self.assertEqual(e.paths,[p]);self.assertEqual(e.traj,[])
        self.assertEqual(e.path_meta[0]['association'],'unverified_topic_association')

    def test_npz_object_pickle_is_rejected(self):
        f=self.root/'bad.npz'
        content=npy([0],(1,)).replace(b"'<f8'",b"'|O8'")
        with zipfile.ZipFile(f,'w') as z:z.writestr('traj.npy',content)
        with self.assertRaises(ValueError):read_npz(f)

    def test_bridge_observation_does_not_modify_control_state(self):
        driver=Mock();driver.ready.return_value=True
        core=BridgeCore(driver,'test_scene',True,Journal(self.root/'bridge'))
        events=[];core.navigate('owner',packet(),events.append)
        core.visualization({'waypoints':[[0,0,0],[1,1,0]]})
        self.assertEqual(events[-1]['type'],'planned_path')
        self.assertFalse(core.active['cancel_requested']);driver.start.assert_called_once()
        driver.cancel.assert_not_called()
        core.visualization({'waypoints':[[0,0,0]]*20000})
        self.assertFalse(core.active['cancel_requested'])
        core.active=None;before=len(events)
        core.visualization({'waypoints':[[0,0,0]]})
        self.assertEqual(len(events),before)
