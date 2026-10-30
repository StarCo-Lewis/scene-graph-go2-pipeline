import unittest
from replay_vlm_rgbd import observation_record


class ObservationTests(unittest.TestCase):
    def test_category_only_never_commits_position(self):
        record = observation_record({'frame': 'view_0', 'category_recognized': True,
            'status': 'rejected', 'reason': 'Insufficient valid depth'})
        self.assertEqual(record['observation_state'], 'category_only_position_unresolved')
        self.assertFalse(record['position_update_allowed'])
        self.assertFalse(record['delete_allowed'])

    def test_no_detection_not_deletion(self):
        record = observation_record({'frame': 'view_0', 'status': 'rejected'})
        self.assertEqual(record['observation_state'], 'not_confirmed')
        self.assertFalse(record['delete_allowed'])

    def test_surface_not_full_object_pose(self):
        record = observation_record({'frame': 'view_0', 'category_recognized': True, 'status': 'accepted'})
        self.assertTrue(record['surface_evidence_accepted'])
        self.assertFalse(record['position_update_allowed'])
