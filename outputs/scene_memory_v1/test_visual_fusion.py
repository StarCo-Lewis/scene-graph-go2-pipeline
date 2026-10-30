import copy
import unittest
from update_memory import initialize
from visual_fusion import project_pixel, fuse_frame


class VisualFusionTests(unittest.TestCase):
    def setUp(self):
        self.state = initialize([{'id': 67, 'bbox_center': [0., 0., 0.], 'bbox_extent': [1., 1., 1.]}], [], 'fixture')
        self.points = {'67': [[.1, .1, .1, 100., 120., 140.] for _ in range(25)]}

    def test_projection_coordinate_convention(self):
        identity = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        point = project_pixel(0, 0, 2., [1., 1., .5, .5], identity, [1., 3., 4.])
        self.assertEqual(point, [1., -2., 3.])

    def test_pose_rotation(self):
        rotation = [[0, 0, 1], [0, 1, 0], [-1, 0, 0]]
        self.assertEqual(project_pixel(0, 0, 2, [1, 1, .5, .5], rotation, [0, 0, 0]), [-2., -0., 0.])

    def test_overlap_and_idempotent_frames(self):
        layer = fuse_frame(None, self.state, 'f1', self.points)
        self.assertEqual(fuse_frame(layer, self.state, 'f1', self.points), layer)
        second = fuse_frame(layer, self.state, 'f2', self.points)
        self.assertEqual(len(second['objects']['67']['voxels']), 1)
        self.assertEqual(next(iter(second['objects']['67']['voxels'].values()))['count'], 50)
        self.assertEqual(next(iter(layer['objects']['67']['voxels'].values()))['count'], 25)

    def test_move_invalidates_old_cloud(self):
        layer = fuse_frame(None, self.state, 'f1', self.points)
        changed = copy.deepcopy(self.state)
        changed['entities']['67']['node']['bbox_center'] = [5., 5., 5.]
        second = fuse_frame(layer, changed, 'f2', {})
        self.assertEqual(second['objects'], {})

    def test_unseen_is_not_deleted(self):
        layer = fuse_frame(None, self.state, 'f1', self.points)
        second = fuse_frame(layer, self.state, 'f2', {})
        self.assertEqual(second['objects'], layer['objects'])
        self.assertTrue(self.state['entities']['67']['active'])

    def test_unknown_id_is_not_added(self):
        with self.assertRaisesRegex(ValueError, 'NO object'):
            fuse_frame(None, self.state, 'f1', {'99': self.points['67']})

    def test_geometry_conflict_not_accepted(self):
        far = {'67': [[5., 5., 5., 0., 0., 0.] for _ in range(25)]}
        self.assertEqual(fuse_frame(None, self.state, 'f1', far)['objects'], {})

    def test_conflicting_frame_id_rejected(self):
        layer = fuse_frame(None, self.state, 'f1', self.points)
        with self.assertRaisesRegex(ValueError, 'Conflicting'):
            fuse_frame(layer, self.state, 'f1', {})
