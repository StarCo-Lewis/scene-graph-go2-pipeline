import unittest
from robot_motion_interface import compile_upstream, dispatch, apply_feedback


class InterfaceTests(unittest.TestCase):
    def setUp(self):
        self.plan = {'query_achievable': True, 'final_relevant_objects': [4],
                     'actions': [{'action_type': 'move_to', 'params': {'x': 2}}]}
        self.goals = {'0': {'frame_id': 'world', 'validated': True, 'goal_pose': [2, 0, 0, 0, 0, 3.13]}}
        self.state = {'robot': {'frame_id': 'world', 'pose': [0]*6,
                               'localization_valid': True, 'pose_fresh': True}, 'objects': {'4': {'active': True}}}

    def request(self):
        return dispatch(compile_upstream(self.plan, [{'id': 4}], self.goals, 'world')['motion_templates'][0], self.state)

    def test_order(self):
        self.assertEqual(self.request()['parameters'], [0]*6 + self.goals['0']['goal_pose'])

    def test_success_wrap_and_no_object_update(self):
        req = self.request()
        fb = {'command_id': 'move_0', 'frame_id': 'world', 'status': 'succeeded', 'actual_pose': [2.1, 0, 0, 0, 0, -3.13]}
        out = apply_feedback(self.state, req, fb)
        self.assertFalse(out['scene_update_required'])
        self.assertEqual(out['objects'], self.state['objects'])
        self.assertEqual(apply_feedback(out, req, fb), out)

    def test_bad_arrival(self):
        req = self.request()
        out = apply_feedback(self.state, req, {'command_id': 'move_0', 'frame_id': 'world', 'status': 'succeeded', 'actual_pose': [0]*6})
        self.assertTrue(out['scene_update_required'])

    def test_missing_goal(self):
        with self.assertRaises(ValueError):
            compile_upstream(self.plan, [{'id': 4}], {}, 'world')

    def test_lock_not_silently_dropped(self):
        self.plan['actions'].append({'action_type': 'approach_and_lock', 'params': {'target_id': 4, 'posture': 'crouch'}})
        with self.assertRaisesRegex(ValueError, 'UNSUPPORTED_CAPABILITY'):
            compile_upstream(self.plan, [{'id': 4}], self.goals, 'world')

    def test_missing_object(self):
        with self.assertRaisesRegex(ValueError, 'NO object'):
            compile_upstream(self.plan, [], self.goals, 'world')

    def test_stale_pose(self):
        self.state['robot']['pose_fresh'] = False
        with self.assertRaises(ValueError):
            self.request()


if __name__ == '__main__':
    unittest.main()
